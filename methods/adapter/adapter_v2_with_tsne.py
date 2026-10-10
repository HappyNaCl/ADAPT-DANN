#!/usr/bin/env python3
"""
Adapter V2 (domain adapter + task adapter, with t-SNE visualization).

Converted from `Adapter_V2_with_tsne.ipynb` into a standalone script.

Every hyper-parameter can be overridden on the command line; anything you do
NOT pass keeps the notebook's original default value.

    python adapter_v2_with_tsne.py                       # run with defaults
    python adapter_v2_with_tsne.py --batch-size 16       # change one thing
    python adapter_v2_with_tsne.py --domain-epochs 5 --task-lr 5e-4
    python adapter_v2_with_tsne.py --use_verbose false   # only print the summary
    python adapter_v2_with_tsne.py --help                # list everything
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import adapters
from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer
from adapters.composition import Stack

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

# ── Fixed label mapping (binary sentiment) ───────────────────────────────────
LABEL2ID = {'negative': 0, 'positive': 1}
ID2LABEL = {0: 'negative', 1: 'positive'}
CLASS_NAMES = ['negative', 'positive']

_HERE = Path(__file__).resolve().parent


def _default(*parts: str) -> str:
    """Default path relative to this script (matches the notebook layout)."""
    return os.path.normpath(str(_HERE.joinpath(*parts)))


# ═══════════════════════════════════════════════════════════════════════════
# Verbose control — set from --use-verbose / --use_verbose
# ═══════════════════════════════════════════════════════════════════════════
VERBOSE = True


def vprint(*args, **kwargs) -> None:
    """print() that is silenced when --use_verbose false."""
    if VERBOSE:
        print(*args, **kwargs)


def str2bool(value) -> bool:
    """Accept true/false, 1/0, yes/no (case-insensitive)."""
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ('true', 't', 'yes', 'y', '1', 'on'):
        return True
    if v in ('false', 'f', 'no', 'n', '0', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean, got {value!r}')


# ═══════════════════════════════════════════════════════════════════════════
# CLI parameters — defaults are exactly the notebook's values
# ═══════════════════════════════════════════════════════════════════════════
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Adapter V2 training + t-SNE visualization (converted from notebook).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ────────────────────────────────────────────────────────────
    g = p.add_argument_group('model')
    g.add_argument('--base-model', default='indolem/indobert-base-uncased',
                   help='HuggingFace base model id')
    g.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                   help='device to train on')
    g.add_argument('--fp16', default='auto', choices=['auto', 'on', 'off'],
                   help='mixed-precision training for the task adapter')

    # ── Data ─────────────────────────────────────────────────────────────
    g = p.add_argument_group('data')
    g.add_argument('--source-train-csv', default=_default('..', '..', 'datasets', 'telemedicine_preprocessed', 'train.csv'),
                   help='source train (labeled)')
    g.add_argument('--source-test-csv', default=_default('..', '..', 'datasets', 'telemedicine_preprocessed', 'test.csv'),
                   help='source test (labeled)')
    g.add_argument('--source-unlabeled-csv', default=_default('..', '..', 'datasets', 'telemedicine_preprocessed', 'train.csv'),
                   help='source unlabeled, for domain MMD')
    g.add_argument('--target-unlabeled-csv', default=_default('..', '..', 'datasets', 'lazada_train_preprocessed.csv'),
                   help='target unlabeled (domain MMD)')
    g.add_argument('--target-eval-csv', default=_default('..', '..', 'datasets', 'lazada_test_preprocessed.csv'),
                   help='target labeled (evaluation)')
    g.add_argument('--max-length', type=int, default=128, help='tokenizer max length')
    g.add_argument('--batch-size', type=int, default=32, help='batch size (all stages)')
    g.add_argument('--seed', type=int, default=42, help='random seed')
    g.add_argument('--val-split', type=float, default=0.1,
                   help='stratified fraction of source train held out for checkpoint '
                        'selection (the source test set is never used for selection)')

    # ── Domain adapter (MMD) ─────────────────────────────────────────────
    g = p.add_argument_group('domain adapter (MMD)')
    g.add_argument('--domain-adapter-name', default='domain_adapter')
    g.add_argument('--domain-adapter-reduction', type=int, default=16,
                   help='pfeiffer reduction factor')
    g.add_argument('--domain-epochs', type=int, default=3)
    g.add_argument('--domain-lr', type=float, default=1e-5)
    g.add_argument('--mmd-kernel-mul', type=float, default=2.0)
    g.add_argument('--mmd-kernel-num', type=int, default=5)

    # ── Task adapter ─────────────────────────────────────────────────────
    g = p.add_argument_group('task adapter')
    g.add_argument('--task-adapter-name', default='task_adapter')
    g.add_argument('--task-adapter-reduction', type=int, default=16)
    g.add_argument('--task-epochs', type=int, default=3)
    g.add_argument('--task-lr', type=float, default=1e-3)
    g.add_argument('--label-smoothing', type=float, default=0.05)
    g.add_argument('--dropout', type=float, default=0.2,
                   help='reserved (unused by the notebook pipeline)')

    # ── Optimizer / schedule ─────────────────────────────────────────────
    g = p.add_argument_group('optimizer / schedule')
    g.add_argument('--warmup-ratio', type=float, default=0.1)
    g.add_argument('--weight-decay', type=float, default=0.02)
    g.add_argument('--max-grad-norm', type=float, default=1.0)
    g.add_argument('--logging-steps', type=int, default=50)

    # ── Output ───────────────────────────────────────────────────────────
    g = p.add_argument_group('output')
    g.add_argument('--output-dir', default='./indobert_adapter_only')

    # ── Verbosity / summary ──────────────────────────────────────────────
    g = p.add_argument_group('verbosity / summary')
    g.add_argument('--use-verbose', '--use_verbose', dest='use_verbose',
                   type=str2bool, nargs='?', const=True, default=True,
                   help='true: show the full training process. '
                        'false: only print the final summary of the results')
    g.add_argument('--summary-style', '--summary_style', dest='summary_style',
                   choices=['text', 'table'], default='text',
                   help='summary format: compact text lines or a table')

    # ── Optional stages ──────────────────────────────────────────────────
    g = p.add_argument_group('optional stages')
    g.add_argument('--skip-inference', action='store_true',
                   help='skip the quick text inference demo')
    g.add_argument('--skip-tsne', action='store_true',
                   help='skip the t-SNE embedding visualization')
    g.add_argument('--max-vis-samples', type=int, default=2000,
                   help='max samples per domain for t-SNE')
    g.add_argument('--tsne-perplexity', type=int, default=30)
    g.add_argument('--tsne-title', default='Adapter', help='t-SNE plot title')
    g.add_argument('--show-plot', action='store_true',
                   help='also open an interactive plot window')

    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def print_config(args: argparse.Namespace) -> None:
    vprint('Configuration loaded.')
    vprint(f'  Base model               : {args.base_model}')
    vprint(f'  Source train CSV         : {args.source_train_csv}')
    vprint(f'  Source test  CSV         : {args.source_test_csv}')
    vprint(f'  Target unlabeled CSV     : {args.target_unlabeled_csv}')
    vprint(f'  Target eval  CSV         : {args.target_eval_csv}')
    vprint(f'  Batch size               : {args.batch_size}')
    vprint(f'  Source val split         : {args.val_split}')
    vprint(f'  Domain LR / Task LR      : {args.domain_lr} / {args.task_lr}')
    vprint(f'  Warmup ratio             : {args.warmup_ratio}')
    vprint(f'  Label smoothing          : {args.label_smoothing}')


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_text_column(df):
    for col in ['content', 'reviewContent', 'text', 'review']:
        if col in df.columns:
            return col
    raise ValueError(f'No text column found. Got: {df.columns.tolist()}')


def _resolve_label_column(df):
    for col in ['label', 'sentiment', 'target']:
        if col in df.columns:
            return col
    raise ValueError(f'No label column found. Got: {df.columns.tolist()}')


def _normalize_labels(series):
    return series.astype(str).str.strip().str.lower().replace({
        'pos':     'positive', 'neg':     'negative',
        '1':       'positive', '0':       'negative',
        'positif': 'positive', 'negatif': 'negative',
        'true':    'positive', 'false':   'negative',
    })


def clean_dataframe(df, text_col, label_col=None):
    cols = [text_col] + ([label_col] if label_col else [])
    out = df[cols].copy().dropna(subset=[text_col])
    out[text_col] = out[text_col].astype(str).str.strip()
    out = out[out[text_col] != '']
    if label_col:
        out = out.dropna(subset=[label_col])
        out[label_col] = _normalize_labels(out[label_col])
        out = out[out[label_col].isin(LABEL2ID)]
    return out.reset_index(drop=True)


def load_data(args: argparse.Namespace):
    # ── source TRAIN (domain MMD + task training) ───────────────────────────
    source_train_df = pd.read_csv(args.source_train_csv)
    text_col_source = _resolve_text_column(source_train_df)
    label_col = _resolve_label_column(source_train_df)
    source_train_df = clean_dataframe(source_train_df, text_col_source, label_col)
    source_train_df['label_id'] = source_train_df[label_col].map(LABEL2ID)

    # ── source VAL (checkpoint selection) — held out from source train ─────
    source_train_df, source_val_df = train_test_split(
        source_train_df,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=source_train_df['label_id'],
    )

    # ── source TEST (source evaluation) ────────────────────────────────────
    source_test_df = pd.read_csv(args.source_test_csv)
    source_test_df = clean_dataframe(source_test_df, text_col_source, label_col)
    source_test_df['label_id'] = source_test_df[label_col].map(LABEL2ID)

    # ── target UNLABELED (domain MMD) ──────────────────────────────────────
    target_df = pd.read_csv(args.target_unlabeled_csv)
    text_col_target = _resolve_text_column(target_df)
    target_df = clean_dataframe(target_df, text_col_target)
    target_texts = target_df[text_col_target].tolist()

    # ── target LABELED TEST (target evaluation) ────────────────────────────
    target_test_df = pd.read_csv(args.target_eval_csv)
    text_col_target_test = _resolve_text_column(target_test_df)
    label_col_target = _resolve_label_column(target_test_df)
    target_test_df = clean_dataframe(target_test_df, text_col_target_test, label_col_target)
    target_test_df['label_id'] = target_test_df[label_col_target].map(LABEL2ID)

    vprint(f'Source TRAIN : {source_train_df.shape}  | '
          f'{dict(source_train_df[label_col].value_counts())}')
    vprint(f'Source VAL   : {source_val_df.shape}  | '
          f'{dict(source_val_df[label_col].value_counts())}')
    vprint(f'Source TEST  : {source_test_df.shape}  | '
          f'{dict(source_test_df[label_col].value_counts())}')
    vprint(f'Target unlabeled  : {len(target_texts):,} samples')
    vprint(f'Target labeled test: {target_test_df.shape} | '
          f'{dict(target_test_df[label_col_target].value_counts())}')
    vprint(f'Sample source : {source_train_df[text_col_source].iloc[0]}')
    vprint(f'Sample target : {target_texts[0]}')

    return {
        'source_train_df': source_train_df,
        'source_val_df': source_val_df,
        'source_test_df': source_test_df,
        'target_texts': target_texts,
        'target_test_df': target_test_df,
        'text_col_source': text_col_source,
        'text_col_target_test': text_col_target_test,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MMD loss
# ═══════════════════════════════════════════════════════════════════════════
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.detach()) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size(0)
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)

    XX = kernels[:batch_size, :batch_size].mean()
    YY = kernels[batch_size:, batch_size:].mean()
    XY = kernels[:batch_size, batch_size:].mean()

    return XX + YY - 2 * XY


def mean_pool(hidden_states, attention_mask):
    """Mean pooling over the last hidden state (mask-aware)."""
    last_h = hidden_states[-1]                     # (B, L, H)
    mask = attention_mask.unsqueeze(-1).float()    # (B, L, 1)
    return (last_h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# Tokenization helpers
# ═══════════════════════════════════════════════════════════════════════════
def tokenize_texts(tokenizer, texts, max_length=128):
    ds = Dataset.from_dict({'text': texts})
    ds = ds.map(
        lambda ex: tokenizer(ex['text'], truncation=True,
                             max_length=max_length, padding=False),
        batched=True,
        remove_columns=['text'],
        desc='Tokenizing',
    )
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return ds


def make_labeled_dataset(tokenizer, df, text_col, max_length=128):
    ds = Dataset.from_dict({
        'text': df[text_col].tolist(),
        'labels': df['label_id'].tolist(),
    })
    ds = ds.map(
        lambda ex: tokenizer(ex['text'], truncation=True,
                             max_length=max_length, padding=False),
        batched=True,
        remove_columns=['text'],
        desc='Tokenizing',
    )
    return ds


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — domain adapter (MMD alignment)
# ═══════════════════════════════════════════════════════════════════════════
def train_domain_adapter(model, source_loader, target_loader, args, device):
    optimizer_domain = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.domain_lr,
        weight_decay=args.weight_decay,
    )

    model.train()

    steps_per_epoch = max(len(source_loader), len(target_loader))
    total_steps = args.domain_epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer_domain,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    vprint(f'Domain training: {args.domain_epochs} epochs x {steps_per_epoch} steps '
          f'= {total_steps} total')
    vprint(f'Warmup steps: {warmup_steps}')

    domain_losses = []

    epoch_pbar = tqdm(range(args.domain_epochs), desc='Epochs', position=0,
                      disable=not VERBOSE)
    for epoch in epoch_pbar:
        epoch_loss = 0.0

        # max() + cycle(): every sample of BOTH domains is seen each epoch
        steps_this_epoch = max(len(source_loader), len(target_loader))
        src_iter = itertools.cycle(source_loader)
        tgt_iter = itertools.cycle(target_loader)

        step_pbar = tqdm(range(steps_this_epoch),
                         desc=f'  Epoch {epoch + 1}/{args.domain_epochs}',
                         position=1, leave=False, disable=not VERBOSE)

        for step in step_pbar:
            src_batch = {k: v.to(device) for k, v in next(src_iter).items()}
            tgt_batch = {k: v.to(device) for k, v in next(tgt_iter).items()}

            src_outputs = model(input_ids=src_batch['input_ids'],
                                attention_mask=src_batch['attention_mask'],
                                output_hidden_states=True)
            tgt_outputs = model(input_ids=tgt_batch['input_ids'],
                                attention_mask=tgt_batch['attention_mask'],
                                output_hidden_states=True)

            src_feat = mean_pool(src_outputs.hidden_states, src_batch['attention_mask'])
            tgt_feat = mean_pool(tgt_outputs.hidden_states, tgt_batch['attention_mask'])

            loss = mmd_loss(src_feat, tgt_feat,
                            args.mmd_kernel_mul, args.mmd_kernel_num)

            optimizer_domain.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                args.max_grad_norm,
            )
            optimizer_domain.step()
            scheduler.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)

            step_pbar.set_postfix({
                'MMD Loss': f'{avg_loss:.6f}',
                'step': f'{step + 1}/{steps_this_epoch}',
            })

        step_pbar.close()

        avg_epoch_loss = epoch_loss / steps_this_epoch
        domain_losses.append(avg_epoch_loss)

        epoch_pbar.set_postfix({
            'avg MMD Loss': f'{avg_epoch_loss:.6f}',
            'epoch': f'{epoch + 1}/{args.domain_epochs}',
        })

        if VERBOSE:
            tqdm.write(f'Epoch {epoch + 1}/{args.domain_epochs} done — '
                       f'avg MMD loss: {avg_epoch_loss:.6f}')
            tqdm.write('-' * 60)

    vprint(f'MMD loss per epoch: {[f"{l:.6f}" for l in domain_losses]}')
    return domain_losses


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — task adapter (supervised, weighted loss)
# ═══════════════════════════════════════════════════════════════════════════
class WeightedAdapterTrainer(AdapterTrainer):
    def __init__(self, class_weights, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            label_smoothing=self.label_smoothing,
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1': f1}


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════
def print_eval(name, metrics):
    vprint('=' * 60)
    vprint(f'{name} EVALUATION')
    vprint('=' * 60)
    vprint(f'  Accuracy  : {metrics["eval_accuracy"]:.4f}')
    vprint(f'  Precision : {metrics["eval_precision"]:.4f}')
    vprint(f'  Recall    : {metrics["eval_recall"]:.4f}')
    vprint(f'  F1 Score  : {metrics["eval_f1"]:.4f}')
    vprint(f'  Loss      : {metrics["eval_loss"]:.4f}')


def print_report(title, true_labels, pred_labels):
    vprint(f'\n{title}:')
    vprint('=' * 60)
    vprint(classification_report(true_labels, pred_labels,
                                target_names=CLASS_NAMES, digits=4))


SUMMARY_METRICS = [
    ('accuracy',  'eval_accuracy'),
    ('precision', 'eval_precision'),
    ('recall',    'eval_recall'),
    ('f1',        'eval_f1'),
]


def build_summary_rows(src_metrics, tgt_metrics):
    """[(name, source value, target value, gap = source - target), ...]"""
    rows = []
    for name, key in SUMMARY_METRICS:
        src = float(src_metrics[key])
        tgt = float(tgt_metrics[key])
        rows.append((name, src, tgt, src - tgt))
    return rows


def print_summary(src_metrics, tgt_metrics, style='text'):
    """Final summary — always printed, even when --use_verbose false."""
    rows = build_summary_rows(src_metrics, tgt_metrics)

    print()
    print('=' * 60)
    print('TRAINING RESULT SUMMARY')
    print('=' * 60)

    if style == 'table':
        header = f'{"metric":<12}{"source":>10}{"target":>10}{"gap":>10}'
        print(header)
        print('-' * len(header))
        for name, src, tgt, gap in rows:
            print(f'{name:<12}{src:>10.4f}{tgt:>10.4f}{gap:>+10.4f}')
        print('-' * len(header))
        print('gap = source - target')
    else:
        for name, src, tgt, gap in rows:
            print(f'{name}: source={src:.4f}  target={tgt:.4f}  gap={gap:+.4f}')

    print('=' * 60)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Quick inference demo
# ═══════════════════════════════════════════════════════════════════════════
TEST_SAMPLES = [
    # App / Tech
    'Aplikasinya sangat membantu dan mudah digunakan!',
    'Aplikasi sering error dan lambat. Tolong diperbaiki.',
    # E-commerce
    'Pengiriman cepat, barang sesuai deskripsi. Sangat puas!',
    'Barang datang rusak dan tidak sesuai ekspektasi.',
    # Food / Restaurant
    'Makanannya enak banget, bumbunya pas!',
    'Pelayanan lama dan makanan datang dingin.',
    # Transport
    'Driver sangat ramah dan tepat waktu.',
    'Driver tidak sopan dan rutenya muter-muter.',
    # Hotel / Travel
    'Kamarnya bersih dan nyaman, sangat rekomendasi!',
    'AC tidak dingin dan kamar kurang terawat.',
    # Education
    'Materinya mudah dipahami dan sangat membantu belajar.',
    'Penjelasan kurang jelas dan membingungkan.',
    # Neutral / mixed
    'Pelayanannya lumayan, masih bisa diimprove',
    'Cukup oke, tapi ada beberapa kekurangan kecil',
]


def run_inference_demo(model, tokenizer, device, max_length):
    model.eval()
    vprint('\n' + '=' * 60)
    vprint('QUICK INFERENCE TEST')
    vprint('=' * 60)
    for text in TEST_SAMPLES:
        inputs = tokenizer(text, return_tensors='pt',
                           truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = probs.argmax().item()
        confidence = probs[0][pred_id].item()

        vprint(f'Text       : {text}')
        vprint(f'Prediction : {ID2LABEL[pred_id].upper()} '
              f'(confidence: {confidence:.4f})')
        vprint('-' * 60)


# ═══════════════════════════════════════════════════════════════════════════
# t-SNE visualization
# ═══════════════════════════════════════════════════════════════════════════
def extract_embeddings(model, dataloader, device, max_samples=2000):
    """Mean-pooled last-hidden-state embeddings from AutoAdapterModel."""
    model.eval()
    Xs, Ys = [], []
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if total >= max_samples:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch.get('labels', None)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)

            pooled = mean_pool(outputs.hidden_states, attention_mask)
            Xs.append(pooled.cpu().numpy())

            if labels is not None:
                Ys.append(labels.cpu().numpy())
            else:
                Ys.append(np.full((pooled.size(0),), -1, dtype=int))

            total += pooled.size(0)

    X = np.concatenate(Xs, axis=0)[:max_samples]
    Y = np.concatenate(Ys, axis=0)[:max_samples]
    return X, Y


def project_embeddings(X, method='tsne', n_components=2, random_state=42,
                       perplexity=30):
    if X.shape[0] == 0:
        return X
    if method == 'tsne':
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        if X.shape[1] > 50:   # PCA to 50 dims first for speed
            X = PCA(n_components=50, random_state=random_state).fit_transform(X)
        tsne = TSNE(n_components=n_components, random_state=random_state,
                    init='pca', perplexity=perplexity)
        return tsne.fit_transform(X)

    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=random_state).fit_transform(X)


def plot_single_panel(proj, labels, domains, title, outpath=None,
                      figsize=(7, 6), show=False):
    import matplotlib
    if not show:
        matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt

    palette = {
        ('source', 0): '#d62728',   # source negative — red
        ('source', 1): '#1f77b4',   # source positive — blue
        ('target', 0): '#2ca02c',   # target negative — green
        ('target', 1): '#9467bd',   # target positive — purple
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
    xs, ys = proj[:, 0], proj[:, 1]

    for dom, lab in plot_order:
        mask = (domains_np == dom) & (labels == lab)
        if mask.sum() == 0:
            continue
        ax.scatter(xs[mask], ys[mask], s=10, c=palette[(dom, lab)],
                   label=display_names[(dom, lab)], alpha=0.8)

    ax.set_title(title, fontsize=24)
    ax.grid(False)
    ax.legend(loc='best', fontsize=18, frameon=True, markerscale=2)
    fig.tight_layout()

    if outpath:
        out_parent = os.path.dirname(str(outpath))
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
        fig.savefig(str(outpath), dpi=200, bbox_inches='tight')
        vprint(f'Saved plot to: {outpath}')

    if show:
        plt.show()
    plt.close(fig)


def run_tsne(model, tokenizer, args, data, device):
    import seaborn as sns
    sns.set(style='whitegrid')

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    model.eval()
    model.to(device)

    g_src = torch.Generator(); g_src.manual_seed(args.seed)
    g_tgt = torch.Generator(); g_tgt.manual_seed(args.seed + 1)

    source_loader = DataLoader(data['source_test_dataset'], batch_size=args.batch_size,
                               shuffle=False, collate_fn=collator, generator=g_src)
    target_loader = DataLoader(data['target_test_dataset'], batch_size=args.batch_size,
                               shuffle=False, collate_fn=collator, generator=g_tgt)

    max_samples = args.max_vis_samples
    vprint(f'Extracting embeddings from source '
          f'({len(data["source_test_dataset"]):,} samples, capped at {max_samples})...')
    Xs_src, Ys_src = extract_embeddings(model, source_loader, device, max_samples)

    vprint(f'Extracting embeddings from target '
          f'({len(data["target_test_dataset"]):,} samples, capped at {max_samples})...')
    Xs_tgt, Ys_tgt = extract_embeddings(model, target_loader, device, max_samples)

    vprint(f'Source embeddings: {Xs_src.shape} | Target embeddings: {Xs_tgt.shape}')

    X_all = np.concatenate([Xs_src, Xs_tgt], axis=0)
    Y_all = np.concatenate([Ys_src, Ys_tgt], axis=0)
    domains = ['source'] * Xs_src.shape[0] + ['target'] * Xs_tgt.shape[0]

    vprint('Running t-SNE projection (this may take ~1-2 minutes)...')
    proj = project_embeddings(X_all, method='tsne', random_state=args.seed,
                              perplexity=args.tsne_perplexity)

    outpath = os.path.join(args.output_dir, 'tsne_dapt_adapter.png')
    plot_single_panel(proj=proj, labels=Y_all, domains=domains,
                      title=args.tsne_title, outpath=outpath,
                      figsize=(7, 6), show=args.show_plot)
    vprint(f'Done. Plot saved to: {outpath}')


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════
def main(argv=None) -> None:
    global VERBOSE

    args = parse_args(argv)
    VERBOSE = args.use_verbose

    if not VERBOSE:
        # Silent mode: hide library progress bars / info logs / warnings too.
        import warnings
        warnings.filterwarnings('ignore')
        os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        try:
            import logging as _logging
            _logging.getLogger('adapters').setLevel(_logging.ERROR)
        except Exception:
            pass
        try:
            import datasets as hf_datasets
            hf_datasets.disable_progress_bar()
        except Exception:
            pass
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
            hf_logging.set_verbosity_error()
        except Exception:
            pass
        try:
            from huggingface_hub import disable_progress_bars
            disable_progress_bars()
        except Exception:
            pass

    set_seed(args.seed)
    device = pick_device(args.device)
    vprint(f'Using device: {device}')
    if device.type == 'cuda':
        vprint(f'GPU: {torch.cuda.get_device_name(0)}')

    print_config(args)

    # ── Data ──────────────────────────────────────────────────────────────
    data = load_data(args)
    source_train_df = data['source_train_df']
    source_val_df = data['source_val_df']
    source_test_df = data['source_test_df']
    target_test_df = data['target_test_df']
    text_col_source = data['text_col_source']
    text_col_target_test = data['text_col_target_test']

    # ── Model + tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoAdapterModel.from_pretrained(args.base_model)
    adapters.init(model)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    vprint(f'Model loaded  : {model.__class__.__name__}')
    vprint(f'Total params  : {total_params:,}')

    vprint(f'MMD loss defined. Kernel multiplier: {args.mmd_kernel_mul} | '
          f'kernels: {args.mmd_kernel_num}')

    # ── Domain data loaders ───────────────────────────────────────────────
    source_domain_ds = tokenize_texts(tokenizer, source_train_df[text_col_source].tolist(),
                                      args.max_length)
    target_domain_ds = tokenize_texts(tokenizer, data['target_texts'], args.max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    g_src = torch.Generator(); g_src.manual_seed(args.seed)
    g_tgt = torch.Generator(); g_tgt.manual_seed(args.seed + 1)

    source_domain_loader = DataLoader(source_domain_ds, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=collator, generator=g_src)
    target_domain_loader = DataLoader(target_domain_ds, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=collator, generator=g_tgt)

    vprint(f'Source domain batches : {len(source_domain_loader)}')
    vprint(f'Target domain batches : {len(target_domain_loader)}')

    # ── Domain adapter ────────────────────────────────────────────────────
    domain_adapter_config = AdapterConfig.load('pfeiffer',
                                               reduction_factor=args.domain_adapter_reduction)
    model.add_adapter(args.domain_adapter_name, config=domain_adapter_config)
    model.train_adapter(args.domain_adapter_name)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    vprint(f'Domain adapter added : {args.domain_adapter_name}')
    vprint(f'Trainable params     : {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)')
    vprint(f'Model devices        : {set(p.device for p in model.parameters())}')

    # ── Stage 1: MMD alignment ────────────────────────────────────────────
    train_domain_adapter(model, source_domain_loader, target_domain_loader,
                         args, device)

    domain_adapter_dir = os.path.join(args.output_dir, args.domain_adapter_name)
    os.makedirs(domain_adapter_dir, exist_ok=True)
    model.save_adapter(domain_adapter_dir, args.domain_adapter_name)
    vprint(f'Domain adapter saved to: {domain_adapter_dir}')
    for f in os.listdir(domain_adapter_dir):
        fpath = os.path.join(domain_adapter_dir, f)
        vprint(f'  {f:30s} ({os.path.getsize(fpath) / 1024:.1f} KB)')

    # ── Class weights from TRAIN distribution ─────────────────────────────
    counts = source_train_df['label_id'].value_counts().sort_index()
    total_n = len(source_train_df)
    class_weights = torch.tensor(
        [total_n / (2 * counts[i]) for i in range(2)], dtype=torch.float).to(device)
    vprint(f'Train  : {len(source_train_df):,} samples')
    vprint(f'Val    : {len(source_val_df):,} samples  (checkpoint selection)')
    vprint(f'Test   : {len(source_test_df):,} samples  (source test set)')
    vprint(f'Class weights: neg={class_weights[0]:.3f}  pos={class_weights[1]:.3f}')

    # ── Labeled datasets ──────────────────────────────────────────────────
    train_dataset = make_labeled_dataset(tokenizer, source_train_df,
                                         text_col_source, args.max_length)
    source_val_dataset = make_labeled_dataset(tokenizer, source_val_df,
                                              text_col_source, args.max_length)
    source_test_dataset = make_labeled_dataset(tokenizer, source_test_df,
                                               text_col_source, args.max_length)
    target_test_dataset = make_labeled_dataset(tokenizer, target_test_df,
                                               text_col_target_test, args.max_length)
    vprint(f'Train dataset features : {train_dataset.features}')
    vprint(f'Source test features   : {source_test_dataset.features}')
    vprint(f'Target test features   : {target_test_dataset.features}')

    # ── Task adapter stacked on the (frozen) domain adapter ───────────────
    task_adapter_config = AdapterConfig.load('pfeiffer',
                                             reduction_factor=args.task_adapter_reduction)
    model.add_adapter(args.task_adapter_name, config=task_adapter_config)
    model.add_classification_head(args.task_adapter_name, num_labels=2, id2label=ID2LABEL)

    model.active_adapters = Stack(args.domain_adapter_name, args.task_adapter_name)
    model.train_adapter(args.task_adapter_name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    vprint(f'Task adapter added : {args.task_adapter_name}')
    vprint(f'Active stack       : {args.domain_adapter_name} → {args.task_adapter_name}')
    vprint(f'Trainable params   : {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)')

    # ── Training arguments ────────────────────────────────────────────────
    if args.fp16 == 'auto':
        use_fp16 = torch.cuda.is_available()
    else:
        use_fp16 = args.fp16 == 'on'

    task_training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, 'task_adapter_checkpoints'),
        num_train_epochs=args.task_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.task_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
        fp16=use_fp16,
        push_to_hub=False,
        report_to='none',
        disable_tqdm=not args.use_verbose,
        seed=args.seed,
    )

    task_trainer = WeightedAdapterTrainer(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        model=model,
        args=task_training_args,
        train_dataset=train_dataset,
        eval_dataset=source_val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    if not args.use_verbose:
        # disable_tqdm=True makes transformers swap in PrinterCallback,
        # which prints every log dict — drop it for summary-only output.
        try:
            from transformers import PrinterCallback
            task_trainer.pop_callback(PrinterCallback)
        except Exception:
            pass

    # ── Stage 2: task adapter training ────────────────────────────────────
    task_result = task_trainer.train()
    vprint(f'Training loss : {task_result.training_loss:.4f}')
    vprint(f'Training time : {task_result.metrics["train_runtime"]:.1f} s')

    # ── Evaluation ────────────────────────────────────────────────────────
    src_eval = task_trainer.evaluate(eval_dataset=source_test_dataset)
    print_eval(f'SOURCE — {os.path.basename(args.source_test_csv)}', src_eval)

    src_preds = task_trainer.predict(source_test_dataset)
    src_pred_labels = np.argmax(src_preds.predictions, axis=1)
    src_true_labels = src_preds.label_ids
    print_report('Source Classification Report', src_true_labels, src_pred_labels)

    tgt_eval = task_trainer.evaluate(eval_dataset=target_test_dataset)
    print_eval(f'TARGET — {os.path.basename(args.target_eval_csv)}', tgt_eval)

    tgt_preds = task_trainer.predict(target_test_dataset)
    tgt_pred_labels = np.argmax(tgt_preds.predictions, axis=1)
    tgt_true_labels = tgt_preds.label_ids
    print_report('Target Classification Report', tgt_true_labels, tgt_pred_labels)

    # ── Summary of the training results (always printed) ──────────────────
    print_summary(src_eval, tgt_eval, style=args.summary_style)

    # ── Save ──────────────────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    model.save_adapter(os.path.join(final_dir, args.domain_adapter_name), args.domain_adapter_name)
    model.save_adapter(os.path.join(final_dir, args.task_adapter_name), args.task_adapter_name)
    tokenizer.save_pretrained(final_dir)

    vprint(f'Saved to: {final_dir}')
    vprint('\nTo load later:')
    vprint(f'  model = AutoAdapterModel.from_pretrained("{args.base_model}")')
    vprint(f'  model.load_adapter("{final_dir}/{args.domain_adapter_name}")')
    vprint(f'  model.load_adapter("{final_dir}/{args.task_adapter_name}")')
    vprint(f'  model.active_adapters = Stack("{args.domain_adapter_name}", '
          f'"{args.task_adapter_name}")')

    # ── Optional: inference demo ──────────────────────────────────────────
    if not args.skip_inference:
        run_inference_demo(model, tokenizer, device, args.max_length)

    # ── Optional: t-SNE ───────────────────────────────────────────────────
    if not args.skip_tsne:
        data['source_test_dataset'] = source_test_dataset
        data['target_test_dataset'] = target_test_dataset
        run_tsne(model, tokenizer, args, data, device)


if __name__ == '__main__':
    main()
