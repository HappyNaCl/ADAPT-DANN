#!/usr/bin/env python
"""
ADAPT-DANN v2 — DAPT encoder + domain adapter (GRL) + task adapter, trained jointly.

Changes vs. adapt_dann.py (v1), all shared hyperparameters unchanged:
  1. One joint stage: task loss on source + adversarial domain loss on source/target
     every step, instead of a separate domain stage followed by a task stage.
  2. The reversed domain gradient reaches the unfrozen top encoder layers (v1 kept the
     encoder frozen during the domain stage, so alignment could not happen there).
  3. Domain classifier trains at the head LR (1e-3) like every other head (v1: 1e-5).
  4. LP-FT: top encoder layers stay frozen during the LR warmup, then unfreeze, so the
     randomly initialised heads don't distort the DAPT features early on.
  5. The best epoch restores the whole trainable state (encoder top layers + adapters +
     heads) together.
Optional: --target_entropy_weight > 0 adds entropy minimisation on target predictions.

Usage (from methods/ADAPT-DANN):
    python3 adapt-dann-v2.py --seed 42
    python3 adapt-dann-v2.py --seed 42 --use-verbose
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.autograd import Function
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

_HERE = Path(__file__).resolve().parent
_DATA = _HERE / '..' / '..' / 'datasets'

CFG = {
    'dapt_model_dir':        str(_HERE / '..' / '..' / 'dapt_lazada'),
    'source_labeled_csv':    str(_DATA / 'telemedicine_preprocessed' / 'train.csv'),
    'source_test_csv':       str(_DATA / 'telemedicine_preprocessed' / 'test.csv'),
    'target_unlabeled_csv':  str(_DATA / 'lazada_train_preprocessed.csv'),
    'target_eval_csv':       str(_DATA / 'lazada_test_preprocessed.csv'),
    'output_dir':            None,   # default: outputs/adapt_dann_v2/seed_<seed>

    # Shared hyperparameters (identical to the other methods)
    'batch_size':            32,
    'max_length':            128,
    'epochs':                3,
    'learning_rate':         1e-5,   # unfrozen encoder layers
    'head_lr':               1e-3,   # adapters + task head + domain head
    'weight_decay':          0.02,
    'warmup_ratio':          0.1,
    'dropout':               0.2,
    'domain_adapter_dim':    16,
    'task_adapter_dim':      16,
    'max_grad_norm':         1.0,
    'unfreeze_top_layers':   4,
    'val_split':             0.15,
    'early_stopping_patience': 3,
    'early_stopping_min_delta': 1e-4,
    'seed':                  42,

    # Adaptation
    'domain_loss_weight':    0.1,    # same as DANN / DANN+Adapter
    'target_entropy_weight': 0.0,    # optional, off by default
    'probe_samples':         1500,
}

LABEL2ID = {'negative': 0, 'positive': 1}
ID2LABEL = {0: 'negative', 1: 'positive'}


# ---------- Verbose control ---------------------------------------------------
_ORIG_STDOUT = sys.stdout
USE_VERBOSE = False


def set_verbose(verbose: bool):
    """Silence stdout + library logs/progress bars unless verbose is set."""
    global USE_VERBOSE
    USE_VERBOSE = bool(verbose)
    if USE_VERBOSE:
        return
    sys.stdout = open(os.devnull, 'w')
    try:
        from transformers.utils.logging import set_verbosity_error
        set_verbosity_error()
    except Exception:
        pass


def always_print(*args):
    print(*args, file=_ORIG_STDOUT, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_summary(model_line: str, hparams_line: str, src: dict, tgt: dict):
    """Compact summary — always printed, same format as the other methods."""
    always_print(model_line)
    always_print(hparams_line)
    for key in ('accuracy', 'precision', 'recall', 'f1'):
        s, t = src[key], tgt[key]
        always_print(f'{key}: source={s:.4f}  target={t:.4f}  gap={s - t:+.4f}')


# ---------- Data --------------------------------------------------------------
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
        'pos': 'positive', 'neg': 'negative',
        '1': 'positive', '0': 'negative',
        'positif': 'positive', 'negatif': 'negative',
        'true': 'positive', 'false': 'negative',
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


def load_labeled(path):
    """-> (texts, label_ids); fails loudly if label normalisation left < 2 classes."""
    df = pd.read_csv(path)
    tc, lc = _resolve_text_column(df), _resolve_label_column(df)
    df = clean_dataframe(df, tc, lc)
    if df[lc].nunique() < 2:
        raise ValueError(f'Only {df[lc].nunique()} class after cleaning {path}')
    print(f'  [{Path(path).name}] {len(df):,} rows | classes: {dict(df[lc].value_counts())}')
    return df[tc].tolist(), df[lc].map(LABEL2ID).tolist()


def load_unlabeled(path):
    df = pd.read_csv(path)
    tc = _resolve_text_column(df)
    df = clean_dataframe(df, tc)
    assert len(df) > 0, f'Empty dataframe after cleaning: {path}'
    print(f'  [{Path(path).name}] {len(df):,} unlabeled rows')
    return df[tc].tolist()


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128, labels=None):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True,
                             max_length=self.max_length, padding='max_length',
                             return_tensors='pt')
        item = {k: enc[k].squeeze(0) for k in ['input_ids', 'attention_mask']}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_loader(texts, tokenizer, cfg, labels=None, shuffle=False, seed=0):
    ds = TextDataset(texts, tokenizer, cfg['max_length'], labels)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle, generator=g)


def infinite(loader):
    """Re-iterate the loader forever (reshuffles each pass, unlike itertools.cycle)."""
    while True:
        for batch in loader:
            yield batch


# ---------- Model -------------------------------------------------------------
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
    """Pfeiffer bottleneck adapter: x + dropout(LayerNorm(up(gelu(down(x)))))"""
    def __init__(self, hidden_size, adapter_dim, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.layer_norm(self.up(self.act(self.down(x)))))


def _head(H, dropout):
    return nn.Sequential(nn.Linear(H, H // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(H // 2, 2))


class AdaptDANNv2(nn.Module):
    """encoder -> mean pool -> domain adapter -> z
         z -> task adapter -> classifier            (sentiment)
         z -> GRL -> domain classifier              (source vs target)
    The reversed domain gradient flows into the domain adapter and the unfrozen encoder layers."""
    def __init__(self, base_model_dir, domain_adapter_dim, task_adapter_dim, dropout):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_dir)
        H = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.domain_adapter = PfeifferAdapter(H, domain_adapter_dim, dropout=dropout)
        self.task_adapter = PfeifferAdapter(H, task_adapter_dim, dropout=dropout)
        self.classifier = _head(H, dropout)
        self.grl = GradientReversalLayer()
        self.domain_classifier = _head(H, dropout)

    def pooled(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, alpha=0.0):
        z = self.domain_adapter(self.dropout(self.pooled(input_ids, attention_mask)))
        class_logits = self.classifier(self.task_adapter(z))
        domain_logits = self.domain_classifier(self.grl(z, alpha=alpha))
        return class_logits, domain_logits


def top_layer_params(encoder, num_layers):
    if num_layers <= 0:
        return []
    if not (hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer')):
        raise RuntimeError('Cannot find encoder layers to unfreeze')
    return [p for layer in encoder.encoder.layer[-num_layers:] for p in layer.parameters()]


def snapshot(model, names):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in names}


def restore(model, state):
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f'Unexpected keys when restoring: {result.unexpected_keys}')
    not_loaded = set(state) & set(result.missing_keys)
    if not_loaded:
        raise RuntimeError(f'Keys not restored: {sorted(not_loaded)}')


# ---------- Evaluation --------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    preds, labels, total = [], [], 0.0
    for b in loader:
        logits, _ = model(b['input_ids'].to(device), b['attention_mask'].to(device))
        y = b['labels'].to(device)
        total += loss_fn(logits, y).item()
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(y.cpu().tolist())
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return {'loss': total / max(1, len(loader)), 'accuracy': accuracy_score(labels, preds),
            'precision': p, 'recall': r, 'f1': f1}


@torch.no_grad()
def domain_probe(model, tokenizer, src_texts, tgt_texts, cfg, device):
    """5-fold linear-probe accuracy for source vs target (0.5 = indistinguishable),
    on pooled encoder features and after the domain adapter."""
    model.eval()
    rng = np.random.RandomState(0)
    n = cfg['probe_samples']
    src = [src_texts[i] for i in rng.choice(len(src_texts), min(n, len(src_texts)), replace=False)]
    tgt = [tgt_texts[i] for i in rng.choice(len(tgt_texts), min(n, len(tgt_texts)), replace=False)]

    def feats(texts):
        pooled, adapted = [], []
        for i in range(0, len(texts), 64):
            e = tokenizer(texts[i:i + 64], truncation=True, max_length=cfg['max_length'],
                          padding=True, return_tensors='pt').to(device)
            x = model.pooled(e['input_ids'], e['attention_mask'])
            pooled.append(x.cpu())
            adapted.append(model.domain_adapter(x).cpu())
        return torch.cat(pooled).numpy(), torch.cat(adapted).numpy()

    ps, as_ = feats(src)
    pt, at = feats(tgt)
    y = np.r_[np.zeros(len(ps)), np.ones(len(pt))]
    probe = lambda X: float(cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=5).mean())
    return {'pooled': probe(np.r_[ps, pt]), 'after_domain_adapter': probe(np.r_[as_, at])}


# ---------- Training ----------------------------------------------------------
def train(cfg, device):
    set_seed(cfg['seed'])
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg['dapt_model_dir'])

    print('Loading data...')
    src_texts, src_labels = load_labeled(cfg['source_labeled_csv'])
    tr_x, va_x, tr_y, va_y = train_test_split(src_texts, src_labels, test_size=cfg['val_split'],
                                              random_state=cfg['seed'], stratify=src_labels)
    tgt_texts = load_unlabeled(cfg['target_unlabeled_csv'])
    stest_x, stest_y = load_labeled(cfg['source_test_csv'])
    ttest_x, ttest_y = load_labeled(cfg['target_eval_csv'])
    print(f'  Source train {len(tr_x):,} | val {len(va_x):,} | target unlabeled {len(tgt_texts):,}')

    train_loader = make_loader(tr_x, tokenizer, cfg, tr_y, shuffle=True, seed=cfg['seed'])
    val_loader = make_loader(va_x, tokenizer, cfg, va_y)
    target_iter = infinite(make_loader(tgt_texts, tokenizer, cfg, shuffle=True, seed=cfg['seed'] + 1))

    model = AdaptDANNv2(cfg['dapt_model_dir'], cfg['domain_adapter_dim'],
                        cfg['task_adapter_dim'], cfg['dropout']).to(device)

    # Encoder frozen except the top layers; those also start frozen (LP-FT) and are
    # unfrozen once the warmup ends.
    for p in model.encoder.parameters():
        p.requires_grad = False
    enc_params = top_layer_params(model.encoder, cfg['unfreeze_top_layers'])
    head_params = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    enc_ids = {id(p) for p in enc_params}
    trainable_names = {n for n, p in model.named_parameters()
                       if not n.startswith('encoder.') or id(p) in enc_ids}

    groups = [{'params': head_params, 'lr': cfg['head_lr']}]
    if enc_params:
        groups.append({'params': enc_params, 'lr': cfg['learning_rate']})
    optimizer = AdamW(groups, weight_decay=cfg['weight_decay'])

    steps_per_epoch = len(train_loader)
    total_steps = cfg['epochs'] * steps_per_epoch
    warmup_steps = int(cfg['warmup_ratio'] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f'  {steps_per_epoch} steps/epoch, {total_steps} total, warmup {warmup_steps} '
          f'(encoder top-{cfg["unfreeze_top_layers"]} layers unfreeze after warmup)')

    ce = nn.CrossEntropyLoss()
    best_val, best_epoch, best_state, es_counter = float('inf'), 0, None, 0
    global_step = 0
    history = []

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        sums = {'total': 0.0, 'cls': 0.0, 'dom': 0.0, 'dom_acc': 0.0, 'ent': 0.0}
        progress = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg["epochs"]}', disable=not USE_VERBOSE)
        for step_idx, sb in enumerate(progress, 1):
            if global_step == warmup_steps and enc_params:
                for p in enc_params:
                    p.requires_grad = True
                print(f'  step {global_step}: unfroze top-{cfg["unfreeze_top_layers"]} encoder layers')

            tb = next(target_iter)
            p_prog = global_step / max(1, total_steps - 1)
            alpha = 2.0 / (1.0 + math.exp(-10 * p_prog)) - 1.0

            n_src = sb['input_ids'].size(0)
            ids = torch.cat([sb['input_ids'], tb['input_ids']]).to(device)
            mask = torch.cat([sb['attention_mask'], tb['attention_mask']]).to(device)
            class_logits, domain_logits = model(ids, mask, alpha=alpha)

            cls_loss = ce(class_logits[:n_src], sb['labels'].to(device))
            dom_labels = torch.cat([torch.zeros(n_src), torch.ones(ids.size(0) - n_src)]).long().to(device)
            dom_loss = 0.5 * (ce(domain_logits[:n_src], dom_labels[:n_src])
                              + ce(domain_logits[n_src:], dom_labels[n_src:]))
            loss = cls_loss + cfg['domain_loss_weight'] * dom_loss

            ent = torch.tensor(0.0, device=device)
            if cfg['target_entropy_weight'] > 0:
                pt = F.softmax(class_logits[n_src:], dim=1)
                ent = -(pt * torch.log(pt.clamp(min=1e-8))).sum(1).mean()
                loss = loss + cfg['target_entropy_weight'] * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], cfg['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            global_step += 1

            sums['total'] += loss.item()
            sums['cls'] += cls_loss.item()
            sums['dom'] += dom_loss.item()
            sums['dom_acc'] += (domain_logits.argmax(1) == dom_labels).float().mean().item()
            sums['ent'] += ent.item()
            progress.set_postfix(cls=f'{sums["cls"]/step_idx:.4f}', dom=f'{sums["dom"]/step_idx:.4f}',
                                 dom_acc=f'{sums["dom_acc"]/step_idx:.3f}', alpha=f'{alpha:.3f}')

        avg = {k: v / steps_per_epoch for k, v in sums.items()}
        val = evaluate(model, val_loader, device)
        history.append({'epoch': epoch, **{f'train_{k}': round(v, 4) for k, v in avg.items()},
                        'val_loss': round(val['loss'], 4), 'val_f1': round(val['f1'], 4)})
        print(f'Epoch {epoch} | cls={avg["cls"]:.4f} dom={avg["dom"]:.4f} dom_acc={avg["dom_acc"]:.3f} '
              f'| val_loss={val["loss"]:.4f} val_f1={val["f1"]:.4f}')

        if val['loss'] < best_val - cfg['early_stopping_min_delta']:
            best_val, best_epoch, es_counter = val['loss'], epoch, 0
            best_state = snapshot(model, trainable_names)
            print(f'  -> new best (val_loss={val["loss"]:.4f})')
        else:
            es_counter += 1
            print(f'  -> no improvement ({es_counter}/{cfg["early_stopping_patience"]})')
            if es_counter >= cfg['early_stopping_patience']:
                print(f'  Early stopping at epoch {epoch}; best epoch {best_epoch}')
                break

    # Restore the whole trainable state of the best epoch (encoder top layers + adapters + heads)
    restore(model, best_state)
    torch.save(best_state, out_dir / 'best_state.pt')
    print(f'Restored best epoch {best_epoch}; saved trainable state to {out_dir / "best_state.pt"}')

    src_metrics = evaluate(model, make_loader(stest_x, tokenizer, cfg, stest_y), device)
    tgt_metrics = evaluate(model, make_loader(ttest_x, tokenizer, cfg, ttest_y), device)
    print('Source test :', src_metrics)
    print('Target test :', tgt_metrics)

    probe = None
    if not cfg['skip_domain_probe']:
        print('Running domain probe...')
        probe = domain_probe(model, tokenizer, tr_x, tgt_texts, cfg, device)

    metrics = {'method': 'ADAPT-DANN v2', 'seed': cfg['seed'], 'best_epoch': best_epoch,
               'source_test': src_metrics, 'target_eval': tgt_metrics,
               'domain_probe': probe, 'history': history,
               'config': {k: v for k, v in cfg.items()}}
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print_summary(
        f"Model: ADAPT-DANN v2 ({Path(cfg['dapt_model_dir']).name})",
        f"batch={cfg['batch_size']}  max_len={cfg['max_length']}  epochs={cfg['epochs']}  "
        f"lr_enc={cfg['learning_rate']:g}  lr_heads={cfg['head_lr']:g}  wd={cfg['weight_decay']}  "
        f"dom_w={cfg['domain_loss_weight']:g}  ent_w={cfg['target_entropy_weight']:g}  "
        f"val_split={cfg['val_split']}  seed={cfg['seed']}",
        src_metrics, tgt_metrics,
    )
    if probe is not None:
        always_print(f"domain_probe: pooled={probe['pooled']:.4f}  "
                     f"after_domain_adapter={probe['after_domain_adapter']:.4f}  (0.5 = aligned)")
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description='ADAPT-DANN v2 (joint adversarial training + LP-FT)',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--use-verbose', '--use_verbose', dest='use_verbose', action='store_true',
                   help='Show all training logs (default: summary only)')
    p.add_argument('--skip_domain_probe', action='store_true', help='Skip the source/target linear probe')
    for key, val in CFG.items():
        typ = str if val is None else type(val)
        p.add_argument(f'--{key}', type=typ, default=None, help=f'default: {val}')
    return p.parse_args()


def main():
    args = parse_args()
    set_verbose(args.use_verbose)
    cfg = dict(CFG)
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    if cfg['output_dir'] is None:
        cfg['output_dir'] = str(_HERE / 'outputs' / 'adapt_dann_v2' / f'seed_{cfg["seed"]}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('Config:', json.dumps(cfg, indent=2))
    train(cfg, device)


if __name__ == '__main__':
    main()
