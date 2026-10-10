#!/usr/bin/env python
# coding: utf-8
"""
DANN + Adapter (converted from dann_adapter.ipynb).

Every notebook CFG value can be overridden from the command line, e.g.:

    python dann_adapter.py                                   # quiet: only header + result summary
    python dann_adapter.py --use-verbose                     # full training logs
    python dann_adapter.py --epochs 5 --batch_size 64 --learning_rate 2e-5
    python dann_adapter.py --no_use_adapters --adapter_dim 64
    python dann_adapter.py --skip_train --model_dir outputs/best
    python dann_adapter.py --skip_train --skip_visualize --text "Barangnya bagus sekali"

Verbosity (for seed sweeps / shell loops, see run_seeds.sh):

    default            quiet — progress bars and per-step logs are off, only the
                       compact header + final result summary is printed
    --use-verbose      (alias --use_verbose) show the full training logs
    --use-verbose false   same as the default (quiet), explicit form

Run `python dann_adapter.py --help` for the full list of arguments.
"""

import argparse
import json
import math
import os
import random
import sys

# Silence the "process just got forked ... Disabling parallelism" HF warning
# (must be set before transformers/tokenizers is imported).
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.autograd import Function
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Headless-safe backend (no display -> save figures only); override with MPLBACKEND=...
import matplotlib
if os.environ.get('DISPLAY', '') == '' and os.environ.get('MPLBACKEND') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Same plot style as ADAPT-DANN (light-gray spines, seaborn fonts)
sns.set(style="whitegrid")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------- Verbose control ---------------------------------------------------
# Default: quiet — only the compact header + final result summary is printed.
# Pass --use-verbose to see the full training logs (progress bars, epochs, ...).
VERBOSE = False
_ORIG_STDOUT = sys.stdout


def str2bool(value) -> bool:
    """Accept true/false, 1/0, yes/no, on/off (case-insensitive)."""
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ('true', 't', 'yes', 'y', '1', 'on'):
        return True
    if v in ('false', 'f', 'no', 'n', '0', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean, got {value!r}')


def set_verbose(verbose: bool) -> None:
    """Silence stdout + library logs/progress bars unless verbose is set."""
    global VERBOSE
    VERBOSE = bool(verbose)
    if VERBOSE:
        return
    sys.stdout = open(os.devnull, 'w')
    try:  # transformers warnings/info go to stderr
        from transformers.utils.logging import set_verbosity_error
        set_verbosity_error()
    except Exception:
        pass


def always_print(*args, **kwargs) -> None:
    """print() that survives quiet mode (writes to the real stdout)."""
    print(*args, **kwargs, file=_ORIG_STDOUT)
    _ORIG_STDOUT.flush()


# =========================
# 2) Konfigurasi utama
# =========================
CFG = {
    'model_name': 'indolem/indobert-base-uncased',
    'source_csv': '../../datasets/telemedicine_preprocessed/train.csv',
    'target_csv': '../../datasets/lazada_train_preprocessed.csv',
    'eval_csv': '../../datasets/lazada_test_preprocessed.csv',  # boleh None
    'output_dir': 'outputs',

    'epochs': 3,
    'batch_size': 32,
    'max_length': 128,
    'learning_rate': 1e-5,
    'weight_decay': 0.02,
    'warmup_ratio': 0.1,
    'dropout': 0.2,
    'domain_loss_weight': 0.1,
    'max_grad_norm': 1.0,
    'seed': 42,

    # Adapter config: bottleneck layer untuk parameter-efficient tuning
    # Adapter diletakkan di antara encoder -> task heads agar encoder tetap stabil
    # tapi representasi bisa diadaptasi untuk klasifikasi sentimen dan domain alignment
    'use_adapters': True,
    'adapter_dim': 16,
}


# ---------- CLI overrides: python dann_adapter.py --batch_size 64 --epochs 5 ... ----------
def _get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='DANN + Adapter: train / inference / t-SNE visualization / EDA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Verbosity (default: quiet — header + result summary only)
    p.add_argument('--use-verbose', '--use_verbose', dest='use_verbose',
                   type=str2bool, nargs='?', const=True, default=False,
                   help='true: show the full training logs. false (default): '
                        'only the compact header + final result summary')
    # Paths
    p.add_argument('--model_name', type=str, default=None, help='HuggingFace model name or local dir')
    p.add_argument('--source_csv', type=str, default=None, help='Labeled source CSV')
    p.add_argument('--target_csv', type=str, default=None, help='Unlabeled target (train) CSV')
    p.add_argument('--eval_csv', type=str, default=None, help='Labeled target eval CSV ("" or "none" to disable)')
    p.add_argument('--output_dir', type=str, default=None, help='Where checkpoints + metrics are written')
    p.add_argument('--model_dir', type=str, default=None,
                   help='Checkpoint to load for inference/visualization (default: <output_dir>/best)')
    # Training
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--max_length', type=int, default=None)
    p.add_argument('--learning_rate', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--warmup_ratio', type=float, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--domain_loss_weight', type=float, default=None)
    p.add_argument('--max_grad_norm', type=float, default=None)
    p.add_argument('--seed', type=int, default=None)
    # Adapter
    p.add_argument('--adapter_dim', type=int, default=None)
    try:  # Python >= 3.9
        p.add_argument('--use_adapters', action=argparse.BooleanOptionalAction, default=None)
    except AttributeError:
        p.add_argument('--use_adapters', dest='use_adapters', action='store_true', default=None)
        p.add_argument('--no_use_adapters', dest='use_adapters', action='store_false')
    # Stage toggles
    p.add_argument('--skip_train', action='store_true', help='Skip the training stage')
    p.add_argument('--skip_inference', action='store_true', help='Skip the example inference stage')
    p.add_argument('--skip_visualize', action='store_true', help='Skip the embedding visualization stage')
    p.add_argument('--skip_eda', action='store_true', help='Skip the EDA stage')
    p.add_argument('--no_show', action='store_true', help='Save plots but do not open a window')
    # Inference / visualization options
    p.add_argument('--text', action='append', default=None,
                   help='Text to classify (repeatable). Replaces the default example sentences.')
    p.add_argument('--vis_method', type=str, default=None, choices=['tsne', 'pca'],
                   help='Projection method for the embedding plot')
    p.add_argument('--vis_stage', type=str, default=None,
                   choices=['encoder', 'encoder+domain', 'encoder+task', 'encoder+domain+task'],
                   help='Which adapted features to visualize')
    p.add_argument('--max_per_domain', type=int, default=None, help='Max samples per domain for visualization')
    return p


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Only non-None CLI values overwrite the notebook defaults."""
    for key, value in vars(args).items():
        if key in ('skip_train', 'skip_inference', 'skip_visualize', 'skip_eda',
                   'no_show', 'text', 'model_dir', 'use_verbose'):
            continue
        if value is None:
            continue
        if key == 'eval_csv' and str(value).strip().lower() in ('', 'none'):
            cfg[key] = None
        else:
            cfg[key] = value
    return cfg


def print_config(cfg: dict, args: argparse.Namespace):
    if not VERBOSE:  # quiet mode: the compact header is printed instead
        return
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU   :', torch.cuda.get_device_name(0))
    for k in ['source_csv', 'target_csv']:
        print(f'{k} => {cfg[k]} | exists: {Path(cfg[k]).exists()}')
    print('eval_csv =>', cfg['eval_csv'],
          '| exists:', Path(cfg['eval_csv']).exists() if cfg['eval_csv'] else False)
    print('use_adapters =>', cfg['use_adapters'], '| adapter_dim =>', cfg['adapter_dim'])
    print('Active config:')
    for k, v in cfg.items():
        print(f'  {k:20s} = {v}')
    stages = [name for name, flag in [
        ('train', args.skip_train), ('inference', args.skip_inference),
        ('visualize', args.skip_visualize), ('eda', args.skip_eda)] if not flag]
    print('Stages to run:', ', '.join(stages) if stages else '(none)')


def print_header(cfg: dict) -> None:
    """Compact header — always printed, even in quiet mode."""
    adapter_info = (f'adapter_dim={cfg["adapter_dim"]}' if cfg['use_adapters'] else 'no adapter')
    always_print(f'Model: DANN + Adapter ({cfg["model_name"]})  seed={cfg["seed"]}')
    always_print(
        f'batch={cfg["batch_size"]}  max_len={cfg["max_length"]}  epochs={cfg["epochs"]}  '
        f'lr={cfg["learning_rate"]:g}  wd={cfg["weight_decay"]:g}  {adapter_info}'
    )
    always_print(f'output_dir={cfg["output_dir"]}')


def print_result_summary(cfg: dict) -> None:
    """Final metrics — always printed, even in quiet mode."""
    metrics_path = Path(cfg['output_dir']) / 'comparison_metrics.json'
    if not metrics_path.exists():
        return
    with open(metrics_path, 'r', encoding='utf-8') as f:
        m = json.load(f)

    src = m.get('source_val', {})
    always_print('=' * 60)
    always_print(f'TRAINING RESULT SUMMARY  (seed={cfg["seed"]}, split={m.get("primary_split", "?")})')
    always_print('=' * 60)
    for key in ('accuracy', 'precision', 'recall', 'f1'):
        if key not in src:
            continue
        line = f'{key}: source_val={src[key]:.4f}'
        tgt = m.get('target_eval', {}).get(key)
        if tgt is not None:
            line += f'  target_eval={tgt:.4f}  gap={tgt - src[key]:+.4f}'
        always_print(line)
    always_print(f'best_checkpoint: {m.get("best_checkpoint", "")}')
    always_print(f'metrics_json:    {metrics_path}')
    always_print('=' * 60)


LABEL2ID = {'negative': 0, 'positive': 1}
ID2LABEL = {0: 'negative', 1: 'positive'}


def _resolve_text_column(df: pd.DataFrame) -> str:
    for col in ['content', 'reviewContent', 'text', 'review']:
        if col in df.columns:
            return col
    raise ValueError(f'Cannot find text column. Available columns: {df.columns.tolist()}')


def _resolve_label_column(df: pd.DataFrame) -> str:
    for col in ['label', 'sentiment', 'target']:
        if col in df.columns:
            return col
    raise ValueError(f'Cannot find label column. Available columns: {df.columns.tolist()}')


def _normalize_labels(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.replace({'neg': 'negative', 'pos': 'positive'})


def clean_dataframe(df: pd.DataFrame, text_col: str, label_col: Optional[str] = None) -> pd.DataFrame:
    use_cols = [text_col] + ([label_col] if label_col else [])
    out = df[use_cols].copy()
    out = out.dropna(subset=[text_col])
    out[text_col] = out[text_col].astype(str).str.strip()
    out = out[out[text_col] != '']

    if label_col:
        out = out.dropna(subset=[label_col])
        out[label_col] = _normalize_labels(out[label_col])
        out = out[out[label_col].isin(LABEL2ID)]

    return out.reset_index(drop=True)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128, labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def make_loaders(
    tokenizer,
    source_path: Path,
    target_path: Path,
    eval_path: Optional[Path],
    batch_size: int,
    max_length: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DataLoader], dict]:
    source_df = pd.read_csv(source_path)
    source_text_col = _resolve_text_column(source_df)
    source_label_col = _resolve_label_column(source_df)
    source_df = clean_dataframe(source_df, source_text_col, source_label_col)
    source_df['label_id'] = source_df[source_label_col].map(LABEL2ID)

    train_df, val_df = train_test_split(
        source_df,
        test_size=0.1,
        random_state=seed,
        stratify=source_df['label_id'],
    )

    target_df = pd.read_csv(target_path)
    target_text_col = _resolve_text_column(target_df)
    target_df = clean_dataframe(target_df, target_text_col)

    source_train_ds = TextDataset(
        texts=train_df[source_text_col].tolist(),
        labels=train_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    source_val_ds = TextDataset(
        texts=val_df[source_text_col].tolist(),
        labels=val_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    target_train_ds = TextDataset(
        texts=target_df[target_text_col].tolist(),
        labels=None,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    source_train_loader = DataLoader(source_train_ds, batch_size=batch_size, shuffle=True)
    source_val_loader = DataLoader(source_val_ds, batch_size=batch_size, shuffle=False)
    target_train_loader = DataLoader(target_train_ds, batch_size=batch_size, shuffle=True)

    eval_loader = None
    eval_samples = 0

    if eval_path is not None and eval_path.exists():
        eval_df = pd.read_csv(eval_path)
        eval_text_col = _resolve_text_column(eval_df)
        eval_label_col = _resolve_label_column(eval_df)

        if 'origin' in eval_df.columns:
            origin = eval_df['origin'].astype(str).str.lower()
            target_only = eval_df[origin == 'target']
            if len(target_only) > 0:
                eval_df = target_only

        eval_df = clean_dataframe(eval_df, eval_text_col, eval_label_col)
        eval_df['label_id'] = eval_df[eval_label_col].map(LABEL2ID)

        eval_ds = TextDataset(
            texts=eval_df[eval_text_col].tolist(),
            labels=eval_df['label_id'].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
        )
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
        eval_samples = len(eval_ds)

    stats = {
        'source_train': len(source_train_ds),
        'source_val': len(source_val_ds),
        'target_train': len(target_train_ds),
        'target_eval': eval_samples,
    }

    return source_train_loader, source_val_loader, target_train_loader, eval_loader, stats


# =========================
# Model
# =========================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, inputs, alpha):
        ctx.alpha = alpha
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, inputs, alpha=1.0):
        return GradientReversalFunction.apply(inputs, alpha)


class PfeifferAdapter(nn.Module):
    def __init__(self, hidden_size: int, adapter_dim: int, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Residual adapter: mempertahankan fitur asli sambil menambahkan delta kecil
        # yang mudah dipelajari. Ini membantu fine-tuning lebih stabil dan efisien.
        return x + self.dropout(self.up(self.act(self.down(x))))


class DANNModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        dropout: float = 0.1,
        use_adapters: bool = True,
        adapter_dim: int = 128,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        print("Hidden Size: ", hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.use_adapters = use_adapters

        # Adapter placement (exact location):
        # 1) label_adapter dipakai setelah encoder feature dan sebelum label classifier
        #    supaya representasi lebih fokus ke sentiment task.
        # 2) domain_adapter dipakai sebelum GRL + domain classifier
        #    supaya representasi domain dapat disesuaikan untuk domain confusion.
        # Jika use_adapters=False, model kembali ke arsitektur DANN standar.
        if self.use_adapters:
            self.label_adapter = PfeifferAdapter(hidden_size, adapter_dim, dropout=dropout)
            self.domain_adapter = PfeifferAdapter(hidden_size, adapter_dim, dropout=dropout)
        else:
            self.label_adapter = nn.Identity()
            self.domain_adapter = nn.Identity()

        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_labels),
        )

        self.grl = GradientReversalLayer()

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def extract_features(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask, alpha=1.0):
        features = self.extract_features(input_ids, attention_mask)
        features = self.dropout(features)

        # Branch 1: Sentiment classification adapter path
        label_features = self.label_adapter(features)
        class_logits = self.label_classifier(label_features)

        # Branch 2: Domain classifier adapter path
        domain_features = self.domain_adapter(features)
        reversed_features = self.grl(domain_features, alpha=alpha)
        domain_logits = self.domain_classifier(reversed_features)

        return class_logits, domain_logits


# =========================
# Training / evaluation
# =========================
@torch.no_grad()
def evaluate_label_head(model, dataloader, device):
    model.eval()

    all_preds, all_labels = [], []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        class_logits, _ = model(input_ids, attention_mask, alpha=0.0)
        loss = loss_fn(class_logits, labels)

        total_loss += loss.item()
        all_preds.extend(torch.argmax(class_logits, dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return {
        'loss': avg_loss,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'preds': all_preds,
        'labels': all_labels,
    }


def _pretty_metrics(title: str, metrics: dict):
    print(title)
    print('=' * len(title))
    print(
        f"loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
    )


def _metrics_dict(metrics: dict) -> dict:
    return {
        'loss': float(metrics['loss']),
        'accuracy': float(metrics['acc']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
    }


def train_dann(cfg: dict):
    set_seed(cfg['seed'])

    source_path = Path(cfg['source_csv'])
    target_path = Path(cfg['target_csv'])
    eval_path = Path(cfg['eval_csv']) if cfg.get('eval_csv') else None

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

    source_train_loader, source_val_loader, target_train_loader, target_eval_loader, stats = make_loaders(
        tokenizer=tokenizer,
        source_path=source_path,
        target_path=target_path,
        eval_path=eval_path,
        batch_size=cfg['batch_size'],
        max_length=cfg['max_length'],
        seed=cfg['seed'],
    )

    print('Dataset summary:', stats)

    model = DANNModel(
        model_name=cfg['model_name'],
        num_labels=2,
        dropout=cfg['dropout'],
        use_adapters=cfg.get('use_adapters', True),
        adapter_dim=cfg.get('adapter_dim', 128),
    ).to(device)

    class_loss_fn = nn.CrossEntropyLoss()
    domain_loss_fn = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    steps_per_epoch = min(len(source_train_loader), len(target_train_loader))
    total_steps = cfg['epochs'] * steps_per_epoch
    warmup_steps = int(cfg['warmup_ratio'] * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_dir = Path(cfg['output_dir']) / 'best'
    best_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_metrics = None

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()

        running_total = 0.0
        running_cls = 0.0
        running_domain = 0.0

        source_iter = iter(source_train_loader)
        target_iter = iter(target_train_loader)

        progress = tqdm(range(steps_per_epoch), desc=f'Epoch {epoch}/{cfg["epochs"]}',
                        disable=not VERBOSE)
        for step_idx in progress:
            source_batch = next(source_iter)
            target_batch = next(target_iter)

            p = float(global_step) / max(1, total_steps - 1)
            alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0

            source_input_ids = source_batch['input_ids'].to(device)
            source_attention_mask = source_batch['attention_mask'].to(device)
            source_labels = source_batch['labels'].to(device)

            target_input_ids = target_batch['input_ids'].to(device)
            target_attention_mask = target_batch['attention_mask'].to(device)

            class_logits, source_domain_logits = model(source_input_ids, source_attention_mask, alpha=alpha)
            _, target_domain_logits = model(target_input_ids, target_attention_mask, alpha=alpha)

            cls_loss = class_loss_fn(class_logits, source_labels)

            source_domain_labels = torch.zeros(source_domain_logits.size(0), dtype=torch.long, device=device)
            target_domain_labels = torch.ones(target_domain_logits.size(0), dtype=torch.long, device=device)

            domain_source_loss = domain_loss_fn(source_domain_logits, source_domain_labels)
            domain_target_loss = domain_loss_fn(target_domain_logits, target_domain_labels)
            domain_loss = 0.5 * (domain_source_loss + domain_target_loss)

            total_loss = cls_loss + cfg['domain_loss_weight'] * domain_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            optimizer.step()
            scheduler.step()

            running_total += total_loss.item()
            running_cls += cls_loss.item()
            running_domain += domain_loss.item()
            global_step += 1

            seen = step_idx + 1
            progress.set_postfix(
                total=f'{running_total/seen:.4f}',
                cls=f'{running_cls/seen:.4f}',
                dom=f'{running_domain/seen:.4f}',
                alpha=f'{alpha:.3f}',
            )

        val_metrics = evaluate_label_head(model, source_val_loader, device)
        print(
            f'Epoch {epoch} | val_loss={val_metrics["loss"]:.4f} val_acc={val_metrics["acc"]:.4f} '
            f'val_precision={val_metrics["precision"]:.4f} val_recall={val_metrics["recall"]:.4f} val_f1={val_metrics["f1"]:.4f}'
        )

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_metrics = val_metrics
            model.encoder.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            torch.save(
                {
                    'label_classifier': model.label_classifier.state_dict(),
                    'domain_classifier': model.domain_classifier.state_dict(),
                    'label_adapter': model.label_adapter.state_dict(),
                    'domain_adapter': model.domain_adapter.state_dict(),
                    'args': cfg,
                },
                best_dir / 'dann_heads.pt',
            )
            print('  Saved new best model to:', best_dir)

    print(f'Training complete. Best source-val F1: {best_val_f1:.4f}')

    # Reload the actual best checkpoint and evaluate from there
    print('\nReloading best checkpoint for final evaluation...')
    best_model = DANNModel(
        model_name=str(best_dir),
        num_labels=2,
        dropout=cfg['dropout'],
        use_adapters=cfg.get('use_adapters', True),
        adapter_dim=cfg.get('adapter_dim', 128),
    ).to(device)
    heads = torch.load(best_dir / 'dann_heads.pt', map_location=device)
    best_model.label_classifier.load_state_dict(heads['label_classifier'])
    best_model.domain_classifier.load_state_dict(heads['domain_classifier'])
    if 'label_adapter' in heads and 'domain_adapter' in heads:
        best_model.label_adapter.load_state_dict(heads['label_adapter'])
        best_model.domain_adapter.load_state_dict(heads['domain_adapter'])

    source_best_metrics = evaluate_label_head(best_model, source_val_loader, device)
    _pretty_metrics('Source Evaluation (best checkpoint)', source_best_metrics)

    comparison_metrics = {
        'framework': 'dann',
        'split': 'source_val',
        'best_checkpoint': str(best_dir),
        'source_val': _metrics_dict(source_best_metrics),
    }

    if target_eval_loader is not None:
        print('\nEvaluating best checkpoint on target labeled set...')
        target_best_metrics = evaluate_label_head(best_model, target_eval_loader, device)
        _pretty_metrics('Target Evaluation (best checkpoint)', target_best_metrics)
        print('\nClassification report (target eval):')
        print(classification_report(target_best_metrics['labels'], target_best_metrics['preds'],
                                    target_names=[ID2LABEL[0], ID2LABEL[1]], digits=4))

        comparison_metrics['target_eval'] = _metrics_dict(target_best_metrics)
        comparison_metrics['primary_split'] = 'target_eval'
        comparison_metrics['loss'] = comparison_metrics['target_eval']['loss']
        comparison_metrics['accuracy'] = comparison_metrics['target_eval']['accuracy']
        comparison_metrics['precision'] = comparison_metrics['target_eval']['precision']
        comparison_metrics['recall'] = comparison_metrics['target_eval']['recall']
        comparison_metrics['f1'] = comparison_metrics['target_eval']['f1']
    else:
        comparison_metrics['primary_split'] = 'source_val'
        comparison_metrics['loss'] = comparison_metrics['source_val']['loss']
        comparison_metrics['accuracy'] = comparison_metrics['source_val']['accuracy']
        comparison_metrics['precision'] = comparison_metrics['source_val']['precision']
        comparison_metrics['recall'] = comparison_metrics['source_val']['recall']
        comparison_metrics['f1'] = comparison_metrics['source_val']['f1']

    print('\nSource Evaluation')
    print('===========================')
    print(comparison_metrics['source_val'])

    if 'target_eval' in comparison_metrics:
        print('\nTarget Evaluation')
        print('===========================')
        print(comparison_metrics['target_eval'])

    comparison_path = Path(cfg['output_dir']) / 'comparison_metrics.json'
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_metrics, f, indent=2)

    print('Saved comparison metrics to:', comparison_path)
    return best_dir


# =========================
# 3) Inference utilities
# =========================
def load_model(model_dir: Path, heads_path: Optional[Path] = None, dropout: float = 0.1,
               device_override: Optional[torch.device] = None):
    if device_override is None:
        device_override = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    heads_path = heads_path or (model_dir / 'dann_heads.pt')
    if not model_dir.exists():
        raise FileNotFoundError(f'Model directory not found: {model_dir}')
    if not heads_path.exists():
        raise FileNotFoundError(f'Heads checkpoint not found: {heads_path}')

    heads = torch.load(heads_path, map_location=device_override)
    train_args = heads.get('args', {})
    model_dropout = float(train_args.get('dropout', dropout))
    model_use_adapters = bool(train_args.get('use_adapters', True))
    model_adapter_dim = int(train_args.get('adapter_dim', 128))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = DANNModel(
        model_name=str(model_dir),
        num_labels=2,
        dropout=model_dropout,
        use_adapters=model_use_adapters,
        adapter_dim=model_adapter_dim,
    ).to(device_override)
    model.label_classifier.load_state_dict(heads['label_classifier'])
    model.domain_classifier.load_state_dict(heads['domain_classifier'])
    if 'label_adapter' in heads and 'domain_adapter' in heads:
        model.label_adapter.load_state_dict(heads['label_adapter'])
        model.domain_adapter.load_state_dict(heads['domain_adapter'])
    model.eval()

    return model, tokenizer, device_override


@torch.no_grad()
def predict_texts(model: DANNModel, tokenizer, texts: List[str], device_infer: torch.device,
                  max_length: int = 128, batch_size: int = 16):
    rows = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        encoding = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length,
                             return_tensors='pt')

        input_ids = encoding['input_ids'].to(device_infer)
        attention_mask = encoding['attention_mask'].to(device_infer)

        class_logits, domain_logits = model(input_ids, attention_mask, alpha=0.0)

        class_probs = F.softmax(class_logits, dim=1).cpu()
        domain_probs = F.softmax(domain_logits, dim=1).cpu()

        for i, text in enumerate(batch_texts):
            sentiment_id = int(torch.argmax(class_probs[i]).item())
            domain_id = int(torch.argmax(domain_probs[i]).item())
            rows.append({
                'text': text,
                'sentiment_id': sentiment_id,
                'sentiment_label': ID2LABEL[sentiment_id],
                'sentiment_negative_prob': float(class_probs[i][0].item()),
                'sentiment_positive_prob': float(class_probs[i][1].item()),
                'domain_pred_id': domain_id,
                'domain_pred_label': 'source' if domain_id == 0 else 'target',
                'domain_source_prob': float(domain_probs[i][0].item()),
                'domain_target_prob': float(domain_probs[i][1].item()),
            })

    return rows


def print_results(rows: List[dict], max_rows: int = 20):
    preview = rows[:max_rows]
    if not preview:
        print('No predictions.')
        return

    print(f'Showing {len(preview)} / {len(rows)} predictions')
    print('=' * 80)
    for i, row in enumerate(preview, 1):
        print(f'[{i}] sentiment={row["sentiment_label"]} '
              f'(neg={row["sentiment_negative_prob"]:.4f}, pos={row["sentiment_positive_prob"]:.4f})')
        print(f'    domain={row["domain_pred_label"]} '
              f'(source={row["domain_source_prob"]:.4f}, target={row["domain_target_prob"]:.4f})')
        print(f'    text={row["text"]}')
        print('-' * 80)


DEFAULT_INFER_TEXTS = [
    "Produk ini kualitasnya sangat bagus, pengiriman cepat dan sesuai deskripsi.",
    "Barang yang saya terima rusak dan tidak berfungsi dengan baik, sangat mengecewakan.",
    "Harga cukup terjangkau tapi kualitasnya kurang memuaskan untuk pemakaian jangka panjang.",
    "Pantainya sangat indah, airnya jernih dan suasananya tenang cocok untuk liburan.",
    "Tempatnya kotor dan banyak sampah, fasilitas juga kurang terawat.",
    "Pemandangan cukup bagus tapi akses menuju lokasi cukup sulit dan jalan rusak.",
]


# =========================
# Embedding visualization helpers (DANN adapted features)
# =========================
def extract_embeddings_from_dataloader(model, dataloader, device, stage="encoder+domain+task"):
    """Extract embeddings from a dataloader for a given stage."""
    model.eval()
    Xs = []
    Ys = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            feats = model.extract_features(input_ids=input_ids, attention_mask=attention_mask)

            if stage in ('encoder+domain', 'encoder+domain+task'):
                feats = model.domain_adapter(feats)
            if stage in ('encoder+task', 'encoder+domain+task'):
                feats = model.label_adapter(feats)

            Xs.append(feats.cpu().numpy())
            if 'labels' in batch:
                Ys.append(batch['labels'].cpu().numpy())
            else:
                Ys.append(np.full((feats.size(0),), -1, dtype=int))

    if len(Xs) == 0:
        return np.zeros((0, model.encoder.config.hidden_size)), np.array([])

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def project_embeddings(X, method="pca", n_components=2, random_state=42):
    if X.shape[0] == 0:
        return X

    if method == 'tsne':
        if X.shape[1] > 50:
            p = PCA(n_components=50, random_state=random_state).fit_transform(X)
        else:
            p = X
        ts = TSNE(n_components=n_components, random_state=random_state, init='pca', perplexity=30)
        return ts.fit_transform(p)

    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X)


def plot_single_panel(proj, labels, domains, title, outpath=None, figsize=(7, 6), show: bool = True):
    """Plot a single projection panel with legend inside plot."""
    palette = {
        ('source', 0): '#d62728',
        ('source', 1): '#1f77b4',
        ('target', 0): '#2ca02c',
        ('target', 1): '#9467bd',
    }
    display_names = {
        ('source', 0): 'source negative',
        ('source', 1): 'source positive',
        ('target', 0): 'target negative',
        ('target', 1): 'target positive',
    }
    plot_order = [('source', 0), ('source', 1), ('target', 0), ('target', 1)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    domains_np = np.array(domains)

    plt.axis(True)

    xs, ys = proj[:, 0], proj[:, 1]
    for dom, lab in plot_order:
        mask = (domains_np == dom) & (labels == lab)
        if mask.sum() == 0:
            continue
        ax.scatter(
            xs[mask],
            ys[mask],
            s=10,
            c=palette[(dom, lab)],
            label=display_names[(dom, lab)],
            alpha=0.8,
        )

    ax.set_title(title, fontsize=24)
    ax.grid(False)

    ax.legend(loc='best', fontsize=18, frameon=True, markerscale=2)

    fig.tight_layout()
    if outpath:
        outdir = os.path.dirname(str(outpath))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.savefig(outpath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def make_labeled_loader(tokenizer, csv_path: Path, batch_size: int, max_length: int,
                        filter_origin: Optional[str] = None):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    text_col = _resolve_text_column(df)
    label_col = _resolve_label_column(df)

    if filter_origin is not None and 'origin' in df.columns:
        origin = df['origin'].astype(str).str.lower()
        df = df[origin == filter_origin.lower()]

    df = clean_dataframe(df, text_col, label_col)
    df['label_id'] = df[label_col].map(LABEL2ID)

    ds = TextDataset(
        texts=df[text_col].tolist(),
        labels=df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def limited_batches(loader, max_samples=4000):
    cnt = 0
    for batch in loader:
        bsize = batch['input_ids'].size(0)
        if cnt + bsize > max_samples:
            remain = max_samples - cnt
            if remain <= 0:
                break
            trimmed = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    trimmed[k] = v[:remain]
                else:
                    trimmed[k] = v
            yield trimmed
            break
        yield batch
        cnt += bsize


def batches_iter(batches):
    for b in batches:
        yield b


# Helper: Create target eval loader independently
def make_target_eval_loader(
    tokenizer,
    eval_path: Optional[Path],
    batch_size: int,
    max_length: int,
    filter_origin: str = 'target',
) -> Tuple[Optional[DataLoader], int]:
    """
    Load labeled target eval data independently.

    Args:
        tokenizer: HuggingFace tokenizer
        eval_path: Path to eval CSV
        batch_size: Batch size for DataLoader
        max_length: Max sequence length
        filter_origin: Filter by 'origin' column value ('target', 'source', or None)

    Returns:
        (eval_loader, num_samples) or (None, 0) if eval_path doesn't exist
    """
    if eval_path is None or not eval_path.exists():
        return None, 0

    eval_df = pd.read_csv(eval_path)
    eval_text_col = _resolve_text_column(eval_df)
    eval_label_col = _resolve_label_column(eval_df)

    # Filter by origin if column exists
    if filter_origin is not None and 'origin' in eval_df.columns:
        origin = eval_df['origin'].astype(str).str.lower()
        eval_df = eval_df[origin == filter_origin.lower()]
        print(f"Filtered to {filter_origin} origin: {len(eval_df)} samples")

    eval_df = clean_dataframe(eval_df, eval_text_col, eval_label_col)
    eval_df['label_id'] = eval_df[eval_label_col].map(LABEL2ID)

    eval_ds = TextDataset(
        texts=eval_df[eval_text_col].tolist(),
        labels=eval_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    return eval_loader, len(eval_ds)


# =========================
# Stages
# =========================
def run_inference(cfg: dict, model_dir: Path, texts: Optional[List[str]] = None):
    print(f'\n[Inference] loading model from {model_dir}')
    model, tokenizer, infer_device = load_model(model_dir=model_dir)
    print('Inference device:', infer_device)

    texts = texts or DEFAULT_INFER_TEXTS
    rows = predict_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device_infer=infer_device,
        max_length=cfg['max_length'],
        batch_size=cfg['batch_size'],
    )

    print_results(rows)

    pred_path = Path(cfg['output_dir']) / 'inference_predictions.json'
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print('Saved predictions to:', pred_path)
    return rows


def run_visualization(cfg: dict, model_dir: Path, method: str = 'tsne',
                      stage: str = 'encoder+domain+task', max_per_domain: int = 2000,
                      show: bool = True):
    print(f'\n[Visualize] loading model from {model_dir} | method={method} | stage={stage}')
    device_vis = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, _ = load_model(model_dir=model_dir, device_override=device_vis)

    src_loader = make_labeled_loader(
        tokenizer=tokenizer,
        csv_path=Path(cfg['source_csv']),
        batch_size=cfg['batch_size'],
        max_length=cfg['max_length'],
    )

    eval_path = Path(cfg['eval_csv']) if cfg.get('eval_csv') else None
    if eval_path is None or not eval_path.exists():
        raise FileNotFoundError("eval_csv missing; provide a labeled target set to visualize.")

    tgt_loader = make_labeled_loader(
        tokenizer=tokenizer,
        csv_path=eval_path,
        batch_size=cfg['batch_size'],
        max_length=cfg['max_length'],
        filter_origin='target',
    )

    src_batches = list(limited_batches(src_loader, max_samples=max_per_domain))
    tgt_batches = list(limited_batches(tgt_loader, max_samples=max_per_domain))

    Xs_src, Ys_src = extract_embeddings_from_dataloader(model, batches_iter(src_batches), device_vis, stage=stage)
    Xs_tgt, Ys_tgt = extract_embeddings_from_dataloader(model, batches_iter(tgt_batches), device_vis, stage=stage)

    X = np.concatenate([Xs_src, Xs_tgt], axis=0)
    Y = np.concatenate([Ys_src, Ys_tgt], axis=0)
    domains = ['source'] * Xs_src.shape[0] + ['target'] * Xs_tgt.shape[0]

    proj = project_embeddings(X, method=method, n_components=2, random_state=cfg['seed'])

    outpath = Path(cfg['output_dir']) / f'embeddings_visualization_dann_{method}.png'
    plot_single_panel(
        proj,
        Y,
        domains,
        title='DANN + Adapter',
        outpath=str(outpath),
        figsize=(7, 6),
        show=show,
    )
    print(f'Saved DANN-only visualization to: {outpath}')
    return outpath


def run_eda(cfg: dict, show_loader_check: bool = True):
    print('\n[EDA] class balance')
    source_df = pd.read_csv(cfg['source_csv'])
    source_text_col = _resolve_text_column(source_df)
    source_label_col = _resolve_label_column(source_df)
    source_df_clean = clean_dataframe(source_df, source_text_col, source_label_col)
    source_df_clean['label_id'] = source_df_clean[source_label_col].map(LABEL2ID)

    print("=== SOURCE CLASS DISTRIBUTION ===")
    print(source_df_clean['label_id'].value_counts().sort_index())
    print(source_df_clean['label_id'].value_counts(normalize=True).sort_index())

    eval_path = Path(cfg['eval_csv']) if cfg.get('eval_csv') else None
    if eval_path is None or not eval_path.exists():
        print('eval_csv missing — skipping target eval distribution.')
        return

    eval_df = pd.read_csv(eval_path)
    eval_text_col = _resolve_text_column(eval_df)
    eval_label_col = _resolve_label_column(eval_df)
    eval_df_clean = clean_dataframe(eval_df, eval_text_col, eval_label_col)
    eval_df_clean['label_id'] = eval_df_clean[eval_label_col].map(LABEL2ID)

    # Filter target only
    if 'origin' in eval_df_clean.columns:
        eval_df_clean = eval_df_clean[eval_df_clean['origin'].astype(str).str.lower() == 'target']

    print("\n=== TARGET EVAL CLASS DISTRIBUTION ===")
    print(eval_df_clean['label_id'].value_counts().sort_index())
    print(eval_df_clean['label_id'].value_counts(normalize=True).sort_index())

    if not show_loader_check:
        return

    # Create target eval loader and check imbalance through the actual pipeline
    tokenizer_eda = AutoTokenizer.from_pretrained(cfg['model_name'])
    target_eval_loader_eda, target_eval_samples = make_target_eval_loader(
        tokenizer=tokenizer_eda,
        eval_path=eval_path,
        batch_size=cfg['batch_size'],
        max_length=cfg['max_length'],
        filter_origin='target',
    )

    print(f"\n✓ Target eval loader created: {target_eval_samples} samples")
    if target_eval_loader_eda is not None:
        print(f"  Batches: {len(target_eval_loader_eda)}")

        all_eval_labels = []
        for batch in target_eval_loader_eda:
            all_eval_labels.extend(batch['labels'].cpu().tolist())

        eval_df_full = pd.DataFrame({'label_id': all_eval_labels})

        print("\n⚠️  TARGET EVAL CLASS IMBALANCE")
        print("Class distribution (counts):")
        print(eval_df_full['label_id'].value_counts().sort_index())
        print("\nClass distribution (%):")
        dist = eval_df_full['label_id'].value_counts(normalize=True).sort_index()
        print(dist * 100)

        if 0 in dist.index and dist[0] > 0:
            imbalance_ratio = (dist[1] / dist[0])
            print(f"\nImbalance ratio (pos/neg): {imbalance_ratio:.2f}:1")
            print("→ Monitor recall/F1 for negative class carefully!")


def main(argv: Optional[List[str]] = None):
    global CFG

    args = _get_parser().parse_args(argv)
    CFG = apply_cli_overrides(CFG, args)

    # Verbosity first: everything below is silenced unless --use-verbose.
    set_verbose(args.use_verbose)

    set_seed(CFG['seed'])
    print_header(CFG)
    print_config(CFG, args)

    model_dir = Path(args.model_dir) if args.model_dir else Path(CFG['output_dir']) / 'best'
    show = not args.no_show

    try:
        if not args.skip_train:
            best_model_dir = train_dann(CFG)
            print('Best checkpoint dir:', best_model_dir)
            model_dir = best_model_dir

        if not args.skip_inference:
            run_inference(CFG, model_dir=model_dir, texts=args.text)

        if not args.skip_visualize:
            run_visualization(
                CFG,
                model_dir=model_dir,
                method=args.vis_method or 'tsne',
                stage=args.vis_stage or 'encoder+domain+task',
                max_per_domain=args.max_per_domain or 2000,
                show=show,
            )

        if not args.skip_eda:
            run_eda(CFG)
    finally:
        # Result summary survives quiet mode (and prints even on a crash).
        if not args.skip_train:
            print_result_summary(CFG)


if __name__ == '__main__':
    main()
