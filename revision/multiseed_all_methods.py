"""
Multi-seed evaluation for all 7 UDA methods.

Run from project root:
    python scripts/multiseed_all_methods.py

Outputs:
    outputs/multiseed/all_methods_raw.csv      — per-seed per-method results
    outputs/multiseed/all_methods_summary.csv  — mean ± std per method

Methods:
    IndoBERT       base IndoBERT, full fine-tune on source, no adaptation
    DAPT           DAPT backbone, full fine-tune on source
    Adapter        UDApter TS-DT (MMD), base IndoBERT, adapters library
    DANN           base IndoBERT, full fine-tune, GRL adversarial joint loss
    DAPT+Adapter   DAPT backbone, frozen encoder, adapter + top-4 unfreeze
    DANN+Adapter   base IndoBERT, joint cls + GRL domain loss, label/domain adapters
    ADAPT-DANN     DAPT backbone, 3-stage sequential pipeline

NOTE: DAPT model must already exist at <project_root>/dapt_lazada/
      Run pretraining/bert_mlm_target.ipynb first if missing.
"""

import os, sys, math, random, itertools, shutil
from pathlib import Path

# ---------- Verbose control ---------------------------------------------------
# Default: only the compact per-seed summary + final tables are printed.
# Pass --use-verbose to see the full training logs.
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
    try:
        from datasets.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass


def print_summary(model_line: str, hparams_line: str, src: dict, tgt: dict):
    """Compact summary — always printed, even in quiet mode."""
    out = _ORIG_STDOUT
    print(model_line, file=out)
    print(hparams_line, file=out)
    for key in ('accuracy', 'precision', 'recall', 'f1'):
        s, t = src.get(key), tgt.get(key)
        if s is None:
            s, t = src.get('acc'), tgt.get('acc')   # eval_metrics() short alias
        if s is None or t is None:
            continue
        print(f'{key}: source={s:.4f}  target={t:.4f}  gap={s - t:+.4f}', file=out)
    out.flush()


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import (
    AutoModel, AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset as HFDataset

# ── Paths (relative to project root) ─────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
DAPT_MODEL_DIR   = PROJECT_ROOT / 'dapt_lazada'
SOURCE_TRAIN_CSV = PROJECT_ROOT / 'datasets/telemedicine_preprocessed/train.csv'
SOURCE_TEST_CSV  = PROJECT_ROOT / 'datasets/telemedicine_preprocessed/test.csv'
TARGET_TRAIN_CSV = PROJECT_ROOT / 'datasets/lazada_train_preprocessed.csv'
TARGET_TEST_CSV  = PROJECT_ROOT / 'datasets/lazada_test_preprocessed.csv'
BASE_MODEL_NAME  = 'indolem/indobert-base-uncased'
OUTPUT_DIR       = PROJECT_ROOT / 'outputs/multiseed'

# ── Hyperparameters (match paper Table I) ────────────────────────────────────
BATCH_SIZE   = 32
MAX_LENGTH   = 128
EPOCHS       = 3
LR           = 1e-5
TASK_LR      = 1e-3
WD           = 0.02
DROPOUT      = 0.2
ADAPTER_DIM  = 16
MAX_GRAD     = 1.0
UNFREEZE_TOP = 4
VAL_SPLIT    = 0.10  # Table I "All: Val split 10%"
DOMAIN_LW    = 0.1   # DANN / DANN+Adapter domain loss weight
PATIENCE     = 3

# Per-method warmup / label smoothing (Table I; FT and DAPT+Adapter not listed -> original code)
FT_WARMUP          = 0.0
DANN_WARMUP        = 0.05
MMD_WARMUP         = 0.02
MMD_LABEL_SMOOTH   = 0.05
DA_WARMUP          = 0.1
DA_VAL_SPLIT       = 0.20
DA_ENCODER_LR      = 5e-6  # unfrozen top-4 layers; from methods/DAPT + Adapter/fine_tune_dapt_adapter_source.py
DANN_AD_WARMUP     = 0.02
ADAPT_DANN_WARMUP  = 0.02
ADAPT_DANN_LS      = 0.05

SEEDS   = [42, 84, 123, 256, 512]
METHODS = ['IndoBERT', 'DAPT', 'Adapter', 'DANN', 'DAPT+Adapter', 'DANN+Adapter', 'ADAPT-DANN']

LABEL2ID = {'negative': 0, 'positive': 1}
ID2LABEL = {0: 'negative', 1: 'positive'}

# ═════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _resolve_text_col(df):
    for c in ['content', 'reviewContent', 'text', 'review']:
        if c in df.columns: return c
    raise ValueError(f'No text column in {df.columns.tolist()}')

def _resolve_label_col(df):
    for c in ['label', 'sentiment', 'target']:
        if c in df.columns: return c
    raise ValueError(f'No label column in {df.columns.tolist()}')

def _normalize_labels(series):
    return series.astype(str).str.strip().str.lower().replace({
        'pos': 'positive', 'neg': 'negative',
        '1':   'positive', '0':   'negative',
        'positif': 'positive', 'negatif': 'negative',
    })

def load_labeled_csv(path):
    df = pd.read_csv(path)
    tc = _resolve_text_col(df)
    lc = _resolve_label_col(df)
    df = df.dropna(subset=[tc, lc])
    df[tc] = df[tc].astype(str).str.strip()
    df = df[df[tc] != '']
    df[lc] = _normalize_labels(df[lc])
    df = df[df[lc].isin(LABEL2ID)].copy()
    df['label_id'] = df[lc].map(LABEL2ID)
    return df.reset_index(drop=True), tc

def load_unlabeled_csv(path):
    df = pd.read_csv(path)
    tc = _resolve_text_col(df)
    df = df.dropna(subset=[tc])
    df[tc] = df[tc].astype(str).str.strip()
    df = df[df[tc] != ''].copy()
    return df.reset_index(drop=True), tc

def eval_metrics(all_preds, all_labels):
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    return {'acc': acc, 'precision': float(p), 'recall': float(r), 'f1': float(f1)}

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH, labels=None):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]), truncation=True,
            max_length=self.max_length, padding='max_length', return_tensors='pt'
        )
        item = {k: enc[k].squeeze(0) for k in ['input_ids', 'attention_mask']}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def make_labeled_loader(df, tc, tokenizer, seed, batch_size=BATCH_SIZE):
    g = torch.Generator(); g.manual_seed(seed)
    ds = TextDataset(df[tc].tolist(), tokenizer, labels=df['label_id'].tolist())
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

def make_unlabeled_loader(df, tc, tokenizer, seed, batch_size=BATCH_SIZE):
    g = torch.Generator(); g.manual_seed(seed)
    ds = TextDataset(df[tc].tolist(), tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

@torch.no_grad()
def evaluate(model_fn, loader, device):
    all_preds, all_labels = [], []
    for batch in loader:
        preds = model_fn(batch, device)
        all_preds.extend(preds)
        all_labels.extend(batch['labels'].tolist())
    return eval_metrics(all_preds, all_labels)

def class_weights(df, device):
    counts = df['label_id'].value_counts().sort_index()
    total  = len(df)
    return torch.tensor([total / (2 * counts[i]) for i in range(2)], dtype=torch.float).to(device)

def freeze(module):
    for p in module.parameters(): p.requires_grad = False

def unfreeze(module):
    for p in module.parameters(): p.requires_grad = True

def unfreeze_top_layers(encoder, n):
    if n <= 0: return
    if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
        for layer in encoder.encoder.layer[-n:]:
            unfreeze(layer)

# ═════════════════════════════════════════════════════════════════════════════
# Shared model components
# ═════════════════════════════════════════════════════════════════════════════

class PfeifferAdapter(nn.Module):
    def __init__(self, hidden_size, adapter_dim, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim)
        self.act  = nn.GELU()
        self.up   = nn.Linear(adapter_dim, hidden_size)
        self.ln   = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.ln(self.up(self.act(self.down(x)))))

class GRLFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return -ctx.alpha * grad, None

class GRL(nn.Module):
    def forward(self, x, alpha=1.0): return GRLFunction.apply(x, alpha)

def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

# ═════════════════════════════════════════════════════════════════════════════
# Method 1 & 2: IndoBERT / DAPT (HF Trainer, full fine-tune)
# ═════════════════════════════════════════════════════════════════════════════

def run_hf_finetune(seed, backbone_path, method_name, work_dir):
    """Full fine-tune with HF Trainer. Used for IndoBERT and DAPT."""
    set_seed(seed)
    device = get_device()

    train_df, tc = load_labeled_csv(SOURCE_TRAIN_CSV)
    src_test_df, _ = load_labeled_csv(SOURCE_TEST_CSV)
    tgt_test_df, _ = load_labeled_csv(TARGET_TEST_CSV)

    tokenizer = AutoTokenizer.from_pretrained(str(backbone_path))
    model     = AutoModelForSequenceClassification.from_pretrained(
        str(backbone_path), num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    def tok_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=MAX_LENGTH, padding=False)

    def make_hf_ds(df):
        ds = HFDataset.from_dict({'text': df[tc].tolist(), 'label': df['label_id'].tolist()})
        return ds.map(tok_fn, batched=True, remove_columns=['text'])

    from sklearn.model_selection import train_test_split as tts
    tr_df, val_df = tts(train_df, test_size=VAL_SPLIT, stratify=train_df['label_id'], random_state=seed)
    train_ds    = make_hf_ds(tr_df)
    val_ds      = make_hf_ds(val_df)
    src_test_ds = make_hf_ds(src_test_df)
    tgt_test_ds = make_hf_ds(tgt_test_df)

    def compute_metrics(ep):
        preds = np.argmax(ep.predictions, axis=1)
        acc   = accuracy_score(ep.label_ids, preds)
        p, r, f1, _ = precision_recall_fscore_support(ep.label_ids, preds, average='weighted')
        return {'accuracy': acc, 'f1': f1, 'precision': p, 'recall': r}

    out = work_dir / f'seed_{seed}'
    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=WD,
        warmup_ratio=FT_WARMUP,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_only_model=True,   # skip optimizer state (~1.3GB per checkpoint); disk is tight
        save_total_limit=1,     # keeps best + latest only
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_steps=100,
        disable_tqdm=not USE_VERBOSE,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to='none',
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    try:
        trainer.train()
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise

    def get_metrics(ds):
        out = trainer.predict(ds)
        preds = np.argmax(out.predictions, axis=1)
        p, r, f1, _ = precision_recall_fscore_support(out.label_ids, preds, average='weighted')
        return {'acc': accuracy_score(out.label_ids, preds), 'precision': float(p),
                'recall': float(r), 'f1': float(f1)}

    src = get_metrics(src_test_ds)
    tgt = get_metrics(tgt_test_ds)
    shutil.rmtree(out, ignore_errors=True)
    return src, tgt


# ═════════════════════════════════════════════════════════════════════════════
# Method 3: Adapter (UDApter TS-DT with MMD)
# ═════════════════════════════════════════════════════════════════════════════

def run_adapter_mmd(seed, work_dir):
    """UDApter TS-DT: domain adapter (MMD) stacked with task adapter."""
    try:
        import adapters
        from adapters import AutoAdapterModel, AdapterConfig
        from adapters.composition import Stack
    except ImportError:
        print('[Adapter] adapters library not installed. pip install adapters')
        return None, None

    set_seed(seed)
    device = get_device()

    train_df, tc = load_labeled_csv(SOURCE_TRAIN_CSV)
    src_test_df, _ = load_labeled_csv(SOURCE_TEST_CSV)
    tgt_test_df, _ = load_labeled_csv(TARGET_TEST_CSV)
    tgt_df, ttc   = load_unlabeled_csv(TARGET_TRAIN_CSV)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model     = AutoAdapterModel.from_pretrained(BASE_MODEL_NAME)
    adapters.init(model); model.to(device)

    def tok_texts(texts):
        ds = HFDataset.from_dict({'text': texts})
        ds = ds.map(
            lambda ex: tokenizer(ex['text'], truncation=True, max_length=MAX_LENGTH, padding=False),
            batched=True, remove_columns=['text']
        )
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        return ds

    DOMAIN_ADAPTER = 'domain_adapter'
    TASK_ADAPTER   = 'task_adapter'

    # Stage 1: domain adapter via MMD
    domain_cfg = AdapterConfig.load('pfeiffer', reduction_factor=ADAPTER_DIM)
    model.add_adapter(DOMAIN_ADAPTER, config=domain_cfg)
    model.train_adapter(DOMAIN_ADAPTER)
    model.to(device)  # new adapter weights are created on CPU

    collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')
    g1 = torch.Generator(); g1.manual_seed(seed)
    g2 = torch.Generator(); g2.manual_seed(seed + 1)
    src_dom_loader = DataLoader(tok_texts(train_df[tc].tolist()), batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collator, generator=g1)
    tgt_dom_loader = DataLoader(tok_texts(tgt_df[ttc].tolist()), batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collator, generator=g2)

    opt_d = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
    steps = max(len(src_dom_loader), len(tgt_dom_loader))
    sch_d = get_cosine_schedule_with_warmup(opt_d, int(MMD_WARMUP * steps * EPOCHS), steps * EPOCHS)

    def mmd_loss(s, t, mul=2.0, num=5):
        # as in methods/adapter/Adapter_V2_with_tsne.ipynb; source/target batch sizes may differ
        bs  = s.size(0)
        n   = s.size(0) + t.size(0)
        tot = torch.cat([s, t], 0)
        t0  = tot.unsqueeze(0).expand(n, n, -1)
        t1  = tot.unsqueeze(1).expand(n, n, -1)
        L2  = ((t0 - t1) ** 2).sum(2)
        bw  = torch.sum(L2.detach()) / (n * n - n)
        bws = [bw / (mul ** (num // 2)) * (mul ** i) for i in range(num)]
        K   = sum(torch.exp(-L2 / b) for b in bws)
        return K[:bs, :bs].mean() + K[bs:, bs:].mean() - 2 * K[:bs, bs:].mean()

    model.train()
    for _ in range(EPOCHS):
        si = itertools.cycle(src_dom_loader)
        ti = itertools.cycle(tgt_dom_loader)
        for _ in range(steps):
            sb = {k: v.to(device) for k, v in next(si).items()}
            tb = {k: v.to(device) for k, v in next(ti).items()}
            so = model(**sb, output_hidden_states=True)
            to_ = model(**tb, output_hidden_states=True)
            sf  = mean_pool(so.hidden_states[-1], sb['attention_mask'])
            tf  = mean_pool(to_.hidden_states[-1], tb['attention_mask'])
            loss = mmd_loss(sf, tf)
            opt_d.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), MAX_GRAD)
            opt_d.step(); sch_d.step()

    # Stage 2: task adapter stacked on frozen domain adapter
    task_cfg = AdapterConfig.load('pfeiffer', reduction_factor=ADAPTER_DIM)
    model.add_adapter(TASK_ADAPTER, config=task_cfg)
    model.add_classification_head(TASK_ADAPTER, num_labels=2, id2label=ID2LABEL)
    model.train_adapter(TASK_ADAPTER)  # note: resets active adapters, so set Stack after
    model.active_adapters = Stack(DOMAIN_ADAPTER, TASK_ADAPTER)
    model.to(device)  # move task adapter + head to device

    def tok_labeled(df):
        ds = HFDataset.from_dict({'text': df[tc].tolist(), 'labels': df['label_id'].tolist()})
        return ds.map(
            lambda ex: tokenizer(ex['text'], truncation=True, max_length=MAX_LENGTH, padding=False),
            batched=True, remove_columns=['text']
        )

    cw = class_weights(train_df, device)

    class WAdapterTrainer:
        pass  # use manual loop instead of AdapterTrainer to avoid HF Trainer seed issues

    g3 = torch.Generator(); g3.manual_seed(seed)
    train_task_ds = tok_labeled(train_df); train_task_ds.set_format(type='torch')
    task_loader = DataLoader(train_task_ds, batch_size=BATCH_SIZE, shuffle=True,
                             collate_fn=collator, generator=g3)

    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=MMD_LABEL_SMOOTH)
    opt_t   = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=TASK_LR, weight_decay=WD)
    sch_t   = get_cosine_schedule_with_warmup(opt_t, int(MMD_WARMUP * len(task_loader) * EPOCHS),
                                               len(task_loader) * EPOCHS)

    for _ in range(EPOCHS):
        model.train()
        for batch in task_loader:
            labels = batch.pop('labels').to(device)
            batch  = {k: v.to(device) for k, v in batch.items()}
            out    = model(**batch)
            loss   = loss_fn(out.logits, labels)
            opt_t.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), MAX_GRAD)
            opt_t.step(); sch_t.step()

    # Evaluate
    model.eval()
    def eval_hf(df_ev):
        ev_ds = tok_labeled(df_ev); ev_ds.set_format(type='torch')
        loader = DataLoader(ev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop('labels').tolist()
                batch  = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_labels.extend(labels)
        return eval_metrics(all_preds, all_labels)

    src = eval_hf(src_test_df)
    tgt = eval_hf(tgt_test_df)
    del model
    return src, tgt


# ═════════════════════════════════════════════════════════════════════════════
# Method 4: DANN (full fine-tune, base IndoBERT, GRL joint loss)
# ═════════════════════════════════════════════════════════════════════════════

class DANNModel(nn.Module):
    def __init__(self, model_name, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.label_cls = nn.Sequential(
            nn.Linear(H, H // 2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(H // 2, 2)
        )
        self.grl = GRL()
        self.domain_cls = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(), nn.Dropout(dropout), nn.Linear(H, 2)
        )

    def _cls(self, ids, mask):
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]

    def forward(self, ids, mask, alpha=1.0):
        feat  = self.drop(self._cls(ids, mask))
        cls   = self.label_cls(feat)
        dom   = self.domain_cls(self.grl(feat, alpha))
        return cls, dom


def run_dann(seed, backbone_path, work_dir):
    """Full fine-tune DANN with GRL. backbone_path = base IndoBERT or DAPT."""
    set_seed(seed)
    device = get_device()

    train_df, tc = load_labeled_csv(SOURCE_TRAIN_CSV)
    src_test_df, _ = load_labeled_csv(SOURCE_TEST_CSV)
    tgt_test_df, _ = load_labeled_csv(TARGET_TEST_CSV)
    tgt_df, ttc   = load_unlabeled_csv(TARGET_TRAIN_CSV)

    tokenizer = AutoTokenizer.from_pretrained(str(backbone_path))
    model     = DANNModel(str(backbone_path)).to(device)

    from sklearn.model_selection import train_test_split as tts
    tr_df, val_df = tts(train_df, test_size=VAL_SPLIT, stratify=train_df['label_id'], random_state=seed)

    g1 = torch.Generator(); g1.manual_seed(seed)
    g2 = torch.Generator(); g2.manual_seed(seed + 1)
    src_loader = DataLoader(TextDataset(tr_df[tc].tolist(), tokenizer, labels=tr_df['label_id'].tolist()),
                            batch_size=BATCH_SIZE, shuffle=True, generator=g1)
    tgt_loader = DataLoader(TextDataset(tgt_df[ttc].tolist(), tokenizer),
                            batch_size=BATCH_SIZE, shuffle=True, generator=g2)

    cls_fn = nn.CrossEntropyLoss()
    dom_fn = nn.CrossEntropyLoss()
    opt    = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps  = min(len(src_loader), len(tgt_loader))
    total  = steps * EPOCHS
    sch    = get_linear_schedule_with_warmup(opt, int(DANN_WARMUP * total), total)

    best_f1, best_state, gs = -1.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        si = iter(src_loader); ti = itertools.cycle(tgt_loader)
        for _ in range(steps):
            p     = float(gs) / max(1, total - 1)
            alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
            sb = next(si); tb = next(ti)
            s_ids, s_mask = sb['input_ids'].to(device), sb['attention_mask'].to(device)
            t_ids, t_mask = tb['input_ids'].to(device), tb['attention_mask'].to(device)
            s_lab = sb['labels'].to(device)

            cls_logits, s_dom = model(s_ids, s_mask, alpha)
            _, t_dom          = model(t_ids, t_mask, alpha)

            loss = cls_fn(cls_logits, s_lab) + DOMAIN_LW * 0.5 * (
                dom_fn(s_dom, torch.zeros(s_dom.size(0), dtype=torch.long, device=device)) +
                dom_fn(t_dom, torch.ones(t_dom.size(0), dtype=torch.long, device=device))
            )
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
            opt.step(); sch.step(); gs += 1

        # Validation
        model.eval()
        vl = DataLoader(TextDataset(val_df[tc].tolist(), tokenizer, labels=val_df['label_id'].tolist()),
                        batch_size=BATCH_SIZE)
        preds, labels = [], []
        with torch.no_grad():
            for b in vl:
                logits, _ = model(b['input_ids'].to(device), b['attention_mask'].to(device), alpha=0.0)
                preds.extend(logits.argmax(1).cpu().tolist())
                labels.extend(b['labels'].tolist())
        f1 = precision_recall_fscore_support(labels, preds, average='weighted')[2]
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    def eval_dann(df_ev, tc_ev):
        loader = DataLoader(TextDataset(df_ev[tc_ev].tolist(), tokenizer,
                                        labels=df_ev['label_id'].tolist()), batch_size=BATCH_SIZE)
        preds, labels = [], []
        with torch.no_grad():
            for b in loader:
                logits, _ = model(b['input_ids'].to(device), b['attention_mask'].to(device), alpha=0.0)
                preds.extend(logits.argmax(1).cpu().tolist())
                labels.extend(b['labels'].tolist())
        return eval_metrics(preds, labels)

    src = eval_dann(src_test_df, tc)
    tgt = eval_dann(tgt_test_df, tc)
    del model
    return src, tgt


# ═════════════════════════════════════════════════════════════════════════════
# Method 5: DAPT+Adapter (frozen encoder, adapter, top-4 unfreeze)
# ═════════════════════════════════════════════════════════════════════════════

class DAPTAdapterClassifier(nn.Module):
    def __init__(self, model_dir, adapter_dim=ADAPTER_DIM, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(model_dir))
        H = self.encoder.config.hidden_size
        self.drop    = nn.Dropout(dropout)
        self.adapter = PfeifferAdapter(H, adapter_dim, dropout)
        self.cls     = nn.Sequential(
            nn.Linear(H, H // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(H // 2, 2)
        )

    def forward(self, ids, mask):
        h    = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        feat = self.drop(mean_pool(h, mask))
        return self.cls(self.adapter(feat))


def run_dapt_adapter(seed, work_dir):
    set_seed(seed)
    device = get_device()

    train_df, tc = load_labeled_csv(SOURCE_TRAIN_CSV)
    src_test_df, _ = load_labeled_csv(SOURCE_TEST_CSV)
    tgt_test_df, _ = load_labeled_csv(TARGET_TEST_CSV)

    tokenizer = AutoTokenizer.from_pretrained(str(DAPT_MODEL_DIR))
    model     = DAPTAdapterClassifier(DAPT_MODEL_DIR).to(device)

    freeze(model.encoder)
    unfreeze(model.adapter); unfreeze(model.cls)
    unfreeze_top_layers(model.encoder, UNFREEZE_TOP)

    from sklearn.model_selection import train_test_split as tts
    tr_df, val_df = tts(train_df, test_size=DA_VAL_SPLIT, stratify=train_df['label_id'], random_state=seed)

    cw = class_weights(train_df, device)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    frozen_enc = [p for p in model.encoder.parameters() if p.requires_grad]
    opt = AdamW([
        {'params': model.adapter.parameters(), 'lr': LR},
        {'params': model.cls.parameters(),     'lr': TASK_LR},
        {'params': frozen_enc,                 'lr': DA_ENCODER_LR},
    ], weight_decay=WD)

    g = torch.Generator(); g.manual_seed(seed)
    tr_loader  = DataLoader(TextDataset(tr_df[tc].tolist(), tokenizer, labels=tr_df['label_id'].tolist()),
                            batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(TextDataset(val_df[tc].tolist(), tokenizer, labels=val_df['label_id'].tolist()),
                            batch_size=BATCH_SIZE)

    total = EPOCHS * len(tr_loader)
    sch   = get_cosine_schedule_with_warmup(opt, int(DA_WARMUP * total), total)

    best_loss, best_state, patience, wait = float('inf'), None, PATIENCE, 0
    for epoch in range(EPOCHS):
        model.train()
        for b in tr_loader:
            ids, mask, lab = b['input_ids'].to(device), b['attention_mask'].to(device), b['labels'].to(device)
            loss = loss_fn(model(ids, mask), lab)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
            opt.step(); sch.step()

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in val_loader:
                ids, mask, lab = b['input_ids'].to(device), b['attention_mask'].to(device), b['labels'].to(device)
                val_loss += loss_fn(model(ids, mask), lab).item()
        val_loss /= len(val_loader)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss; wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    def eval_da(df_ev, tc_ev):
        loader = DataLoader(TextDataset(df_ev[tc_ev].tolist(), tokenizer,
                                        labels=df_ev['label_id'].tolist()), batch_size=BATCH_SIZE)
        preds, labels = [], []
        with torch.no_grad():
            for b in loader:
                logits = model(b['input_ids'].to(device), b['attention_mask'].to(device))
                preds.extend(logits.argmax(1).cpu().tolist())
                labels.extend(b['labels'].tolist())
        return eval_metrics(preds, labels)

    src = eval_da(src_test_df, tc)
    tgt = eval_da(tgt_test_df, tc)
    del model
    return src, tgt


# ═════════════════════════════════════════════════════════════════════════════
# Method 6: DANN+Adapter (base IndoBERT, frozen encoder, domain adapter GRL, task adapter)
# ═════════════════════════════════════════════════════════════════════════════

class DomainAdapterDANN(nn.Module):
    def __init__(self, model_dir, adapter_dim=ADAPTER_DIM, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(model_dir))
        H = self.encoder.config.hidden_size
        self.drop       = nn.Dropout(dropout)
        self.adapter    = PfeifferAdapter(H, adapter_dim, dropout)
        self.grl        = GRL()
        self.domain_cls = nn.Sequential(
            nn.Linear(H, H // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(H // 2, 2)
        )

    def forward(self, ids, mask, alpha=1.0):
        h    = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        feat = self.drop(mean_pool(h, mask))
        adap = self.adapter(feat)
        return self.domain_cls(self.grl(adap, alpha))


class TaskAdapterClassifier(nn.Module):
    def __init__(self, model_dir, adapter_dim=ADAPTER_DIM, dropout=DROPOUT):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(str(model_dir))
        H = self.encoder.config.hidden_size
        self.drop       = nn.Dropout(dropout)
        self.dom_adapt  = PfeifferAdapter(H, adapter_dim, dropout)
        self.task_adapt = PfeifferAdapter(H, adapter_dim, dropout)
        self.cls        = nn.Sequential(
            nn.Linear(H, H // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(H // 2, 2)
        )

    def forward(self, ids, mask):
        h    = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        feat = self.drop(mean_pool(h, mask))
        feat = self.dom_adapt(feat)
        feat = self.task_adapt(feat)
        return self.cls(feat)


def run_two_stage_adapter(seed, backbone_path, work_dir):
    """ADAPT-DANN stages 2-3: frozen encoder, domain adapter GRL, then task adapter."""
    set_seed(seed)
    device = get_device()

    train_df, tc = load_labeled_csv(SOURCE_TRAIN_CSV)
    src_test_df, _ = load_labeled_csv(SOURCE_TEST_CSV)
    tgt_test_df, _ = load_labeled_csv(TARGET_TEST_CSV)
    tgt_df, ttc   = load_unlabeled_csv(TARGET_TRAIN_CSV)

    tokenizer = AutoTokenizer.from_pretrained(str(backbone_path))
    out_path  = work_dir / f'seed_{seed}'
    out_path.mkdir(parents=True, exist_ok=True)
    dom_path  = out_path / 'domain_adapter.pt'

    # Stage 2: domain adapter with GRL
    dann_model = DomainAdapterDANN(backbone_path).to(device)
    freeze(dann_model.encoder)
    unfreeze(dann_model.adapter); unfreeze(dann_model.domain_cls)

    g1 = torch.Generator(); g1.manual_seed(seed)
    g2 = torch.Generator(); g2.manual_seed(seed + 1)
    steps = max(len(DataLoader(TextDataset(train_df[tc].tolist(), tokenizer), batch_size=BATCH_SIZE)),
                len(DataLoader(TextDataset(tgt_df[ttc].tolist(), tokenizer), batch_size=BATCH_SIZE)))
    src_loader = DataLoader(TextDataset(train_df[tc].tolist(), tokenizer),
                            batch_size=BATCH_SIZE, shuffle=True, generator=g1)
    tgt_loader = DataLoader(TextDataset(tgt_df[ttc].tolist(), tokenizer),
                            batch_size=BATCH_SIZE, shuffle=True, generator=g2)

    dom_fn = nn.CrossEntropyLoss()
    opt_d  = AdamW(list(dann_model.adapter.parameters()) + list(dann_model.domain_cls.parameters()),
                   lr=LR, weight_decay=WD)
    total_d = EPOCHS * steps
    sch_d   = get_cosine_schedule_with_warmup(opt_d, int(ADAPT_DANN_WARMUP * total_d), total_d)

    gs = 0
    for _ in range(EPOCHS):
        dann_model.train()
        si = itertools.cycle(src_loader); ti = itertools.cycle(tgt_loader)
        for _ in range(steps):
            p     = float(gs) / max(1, total_d - 1)
            alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
            sb = next(si); tb = next(ti)
            s_ids, s_mask = sb['input_ids'].to(device), sb['attention_mask'].to(device)
            t_ids, t_mask = tb['input_ids'].to(device), tb['attention_mask'].to(device)
            sl = dann_model(s_ids, s_mask, alpha)
            tl = dann_model(t_ids, t_mask, alpha)
            loss = 0.5 * (
                dom_fn(sl, torch.zeros(sl.size(0), dtype=torch.long, device=device)) +
                dom_fn(tl, torch.ones(tl.size(0), dtype=torch.long, device=device))
            )
            opt_d.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(dann_model.parameters(), MAX_GRAD); opt_d.step(); sch_d.step(); gs += 1

    torch.save(dann_model.adapter.state_dict(), dom_path)
    del dann_model

    # Stage 3: task adapter
    task_model = TaskAdapterClassifier(backbone_path).to(device)
    task_model.dom_adapt.load_state_dict(torch.load(dom_path, map_location=device))
    freeze(task_model.encoder); freeze(task_model.dom_adapt)
    unfreeze(task_model.task_adapt); unfreeze(task_model.cls)
    unfreeze_top_layers(task_model.encoder, UNFREEZE_TOP)

    from sklearn.model_selection import train_test_split as tts
    tr_df, val_df = tts(train_df, test_size=VAL_SPLIT, stratify=train_df['label_id'], random_state=seed)

    cw      = class_weights(train_df, device)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=ADAPT_DANN_LS)

    g3 = torch.Generator(); g3.manual_seed(seed)
    tr_loader  = DataLoader(TextDataset(tr_df[tc].tolist(), tokenizer, labels=tr_df['label_id'].tolist()),
                            batch_size=BATCH_SIZE, shuffle=True, generator=g3)
    val_loader = DataLoader(TextDataset(val_df[tc].tolist(), tokenizer, labels=val_df['label_id'].tolist()),
                            batch_size=BATCH_SIZE)

    frozen_enc_params = [p for p in task_model.encoder.parameters() if p.requires_grad]
    opt_t = AdamW([
        {'params': task_model.task_adapt.parameters(), 'lr': TASK_LR},
        {'params': task_model.cls.parameters(),        'lr': TASK_LR},
        {'params': frozen_enc_params,                  'lr': LR},
    ], weight_decay=WD)
    total_t = EPOCHS * len(tr_loader)
    sch_t   = get_cosine_schedule_with_warmup(opt_t, int(ADAPT_DANN_WARMUP * total_t), total_t)

    best_loss, best_state, patience, wait = float('inf'), None, PATIENCE, 0
    for _ in range(EPOCHS):
        task_model.train()
        for b in tr_loader:
            ids, mask, lab = b['input_ids'].to(device), b['attention_mask'].to(device), b['labels'].to(device)
            loss = loss_fn(task_model(ids, mask), lab)
            opt_t.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(task_model.parameters(), MAX_GRAD); opt_t.step(); sch_t.step()

        task_model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in val_loader:
                ids, mask, lab = b['input_ids'].to(device), b['attention_mask'].to(device), b['labels'].to(device)
                vl += loss_fn(task_model(ids, mask), lab).item()
        vl /= len(val_loader)
        if vl < best_loss - 1e-4:
            best_loss = vl; wait = 0
            best_state = {k: v.cpu().clone() for k, v in task_model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break

    task_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    task_model.eval()

    def ev(df_ev, tc_ev):
        loader = DataLoader(TextDataset(df_ev[tc_ev].tolist(), tokenizer,
                                        labels=df_ev['label_id'].tolist()), batch_size=BATCH_SIZE)
        preds, labels = [], []
        with torch.no_grad():
            for b in loader:
                logits = task_model(b['input_ids'].to(device), b['attention_mask'].to(device))
                preds.extend(logits.argmax(1).cpu().tolist())
                labels.extend(b['labels'].tolist())
        return eval_metrics(preds, labels)

    src = ev(src_test_df, tc)
    tgt = ev(tgt_test_df, tc)
    shutil.rmtree(out_path, ignore_errors=True)
    del task_model
    return src, tgt


class DANNAdapterModel(nn.Module):
    """DANN with a label adapter before the sentiment head and a domain adapter before the GRL."""
    def __init__(self, model_name, adapter_dim=ADAPTER_DIM, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(model_name))
        H = self.encoder.config.hidden_size
        self.drop          = nn.Dropout(dropout)
        self.label_adapter  = PfeifferAdapter(H, adapter_dim, dropout)
        self.domain_adapter = PfeifferAdapter(H, adapter_dim, dropout)
        self.label_cls = nn.Sequential(
            nn.Linear(H, H // 2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(H // 2, 2)
        )
        self.grl = GRL()
        self.domain_cls = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(), nn.Dropout(dropout), nn.Linear(H, 2)
        )

    def forward(self, ids, mask, alpha=1.0):
        feat = self.drop(self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
        cls  = self.label_cls(self.label_adapter(feat))
        dom  = self.domain_cls(self.grl(self.domain_adapter(feat), alpha))
        return cls, dom


def run_dann_adapter(seed, backbone_path, work_dir):
    """DANN+Adapter: joint sentiment + adversarial domain loss with adapters (no DAPT)."""
    set_seed(seed)
    device = get_device()

    train_df, tc = load_labeled_csv(SOURCE_TRAIN_CSV)
    src_test_df, _ = load_labeled_csv(SOURCE_TEST_CSV)
    tgt_test_df, _ = load_labeled_csv(TARGET_TEST_CSV)
    tgt_df, ttc   = load_unlabeled_csv(TARGET_TRAIN_CSV)

    tokenizer = AutoTokenizer.from_pretrained(str(backbone_path))
    model     = DANNAdapterModel(backbone_path).to(device)

    from sklearn.model_selection import train_test_split as tts
    tr_df, val_df = tts(train_df, test_size=VAL_SPLIT, stratify=train_df['label_id'], random_state=seed)

    g1 = torch.Generator(); g1.manual_seed(seed)
    g2 = torch.Generator(); g2.manual_seed(seed + 1)
    src_loader = DataLoader(TextDataset(tr_df[tc].tolist(), tokenizer, labels=tr_df['label_id'].tolist()),
                            batch_size=BATCH_SIZE, shuffle=True, generator=g1)
    tgt_loader = DataLoader(TextDataset(tgt_df[ttc].tolist(), tokenizer),
                            batch_size=BATCH_SIZE, shuffle=True, generator=g2)
    val_loader = DataLoader(TextDataset(val_df[tc].tolist(), tokenizer, labels=val_df['label_id'].tolist()),
                            batch_size=BATCH_SIZE)

    cls_fn = nn.CrossEntropyLoss()
    dom_fn = nn.CrossEntropyLoss()
    opt    = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps  = min(len(src_loader), len(tgt_loader))
    total  = steps * EPOCHS
    sch    = get_linear_schedule_with_warmup(opt, int(DANN_AD_WARMUP * total), total)

    best_f1, best_state, gs = -1.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        si = iter(src_loader); ti = iter(tgt_loader)
        for _ in range(steps):
            p     = float(gs) / max(1, total - 1)
            alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
            sb = next(si); tb = next(ti)
            s_ids, s_mask = sb['input_ids'].to(device), sb['attention_mask'].to(device)
            t_ids, t_mask = tb['input_ids'].to(device), tb['attention_mask'].to(device)
            s_lab = sb['labels'].to(device)

            cls_logits, s_dom = model(s_ids, s_mask, alpha)
            _, t_dom          = model(t_ids, t_mask, alpha)

            loss = cls_fn(cls_logits, s_lab) + DOMAIN_LW * 0.5 * (
                dom_fn(s_dom, torch.zeros(s_dom.size(0), dtype=torch.long, device=device)) +
                dom_fn(t_dom, torch.ones(t_dom.size(0), dtype=torch.long, device=device))
            )
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
            opt.step(); sch.step(); gs += 1

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for b in val_loader:
                logits, _ = model(b['input_ids'].to(device), b['attention_mask'].to(device), alpha=0.0)
                preds.extend(logits.argmax(1).cpu().tolist())
                labels.extend(b['labels'].tolist())
        f1 = precision_recall_fscore_support(labels, preds, average='weighted')[2]
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    def ev(df_ev, tc_ev):
        loader = DataLoader(TextDataset(df_ev[tc_ev].tolist(), tokenizer,
                                        labels=df_ev['label_id'].tolist()), batch_size=BATCH_SIZE)
        preds, labels = [], []
        with torch.no_grad():
            for b in loader:
                logits, _ = model(b['input_ids'].to(device), b['attention_mask'].to(device), alpha=0.0)
                preds.extend(logits.argmax(1).cpu().tolist())
                labels.extend(b['labels'].tolist())
        return eval_metrics(preds, labels)

    src = ev(src_test_df, tc)
    tgt = ev(tgt_test_df, tc)
    del model
    return src, tgt



# ═════════════════════════════════════════════════════════════════════════════
# Method 7: ADAPT-DANN (DAPT backbone, 3-stage pipeline)
# ═════════════════════════════════════════════════════════════════════════════

def run_adapt_dann(seed, work_dir):
    """ADAPT-DANN: DAPT backbone → domain adapter GRL → task adapter."""
    # Same as DANN+Adapter but uses DAPT model as backbone (Stage 1 already done)
    return run_two_stage_adapter(seed, DAPT_MODEL_DIR, work_dir)


# ═════════════════════════════════════════════════════════════════════════════
# Multi-seed runner
# ═════════════════════════════════════════════════════════════════════════════

def run_all(seeds=SEEDS, methods=METHODS):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DAPT_MODEL_DIR.exists():
        print(f'WARNING: DAPT model not found at {DAPT_MODEL_DIR}')
        print('Methods requiring DAPT (DAPT, DAPT+Adapter, ADAPT-DANN) will be skipped.')

    raw_path = OUTPUT_DIR / 'all_methods_raw.csv'
    records  = []
    if raw_path.exists():  # keep earlier results of methods not being rerun
        prev    = pd.read_csv(raw_path)
        records = prev[~prev['method'].isin(methods)].to_dict('records')

    for method in methods:
        print(f'\n{"="*60}')
        print(f'METHOD: {method}')
        print(f'{"="*60}')

        need_dapt = method in ('DAPT', 'DAPT+Adapter', 'ADAPT-DANN')
        if need_dapt and not DAPT_MODEL_DIR.exists():
            print(f'  Skipping {method} — DAPT model missing')
            continue

        work = OUTPUT_DIR / method.replace('+', '_plus_').replace(' ', '_')
        work.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            print(f'\n  Seed {seed}...')
            try:
                if method == 'IndoBERT':
                    src, tgt = run_hf_finetune(seed, BASE_MODEL_NAME, method, work)
                elif method == 'DAPT':
                    src, tgt = run_hf_finetune(seed, DAPT_MODEL_DIR, method, work)
                elif method == 'Adapter':
                    src, tgt = run_adapter_mmd(seed, work)
                elif method == 'DANN':
                    src, tgt = run_dann(seed, BASE_MODEL_NAME, work)
                elif method == 'DAPT+Adapter':
                    src, tgt = run_dapt_adapter(seed, work)
                elif method == 'DANN+Adapter':
                    src, tgt = run_dann_adapter(seed, BASE_MODEL_NAME, work)
                elif method == 'ADAPT-DANN':
                    src, tgt = run_adapt_dann(seed, work)
                else:
                    print(f'  Unknown method: {method}')
                    continue

                if src is None or tgt is None:
                    continue

                records.append({
                    'method': method, 'seed': seed,
                    'src_acc': src['acc'], 'src_precision': src['precision'],
                    'src_recall': src['recall'], 'src_f1': src['f1'],
                    'tgt_acc': tgt['acc'], 'tgt_precision': tgt['precision'],
                    'tgt_recall': tgt['recall'], 'tgt_f1': tgt['f1'],
                })
                print(f'  src_f1={src["f1"]:.4f}  tgt_f1={tgt["f1"]:.4f}')

                # Compact per-seed summary (always printed, even in quiet mode)
                _backbone = str(DAPT_MODEL_DIR) if method in ('DAPT', 'DAPT+Adapter', 'ADAPT-DANN') \
                    else BASE_MODEL_NAME
                print_summary(
                    f'Model: {method} ({_backbone})',
                    f'batch={BATCH_SIZE}  max_len={MAX_LENGTH}  epochs={EPOCHS}  '
                    f'lr={LR:g}  wd={WD}  seed={seed}',
                    src, tgt,
                )

                # Save incrementally after every run
                pd.DataFrame(records).to_csv(OUTPUT_DIR / 'all_methods_raw.csv', index=False)

            except Exception as e:
                print(f'  ERROR seed={seed} method={method}: {e}')
                import traceback; traceback.print_exc()

    # Restore stdout so the final tables are always visible in quiet mode
    sys.stdout = _ORIG_STDOUT

    raw_df = pd.DataFrame(records)
    raw_df.to_csv(OUTPUT_DIR / 'all_methods_raw.csv', index=False)
    print(f'\nSaved raw results: {OUTPUT_DIR / "all_methods_raw.csv"}')

    # Summary
    summary = []
    for method, grp in raw_df.groupby('method'):
        for col in ['src_f1', 'tgt_f1', 'src_acc', 'tgt_acc',
                    'src_precision', 'tgt_precision', 'src_recall', 'tgt_recall']:
            summary.append({
                'method': method, 'metric': col,
                'mean': grp[col].mean(), 'std': grp[col].std(),
                'n_seeds': len(grp),
                'formatted': f"{grp[col].mean():.4f} ± {grp[col].std():.4f}",
            })
    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(OUTPUT_DIR / 'all_methods_summary.csv', index=False)
    print(f'Saved summary:     {OUTPUT_DIR / "all_methods_summary.csv"}')

    # Print key tgt_f1 table
    print('\n── Target F1 (mean ± std) ──────────────────────────────────')
    tgt_f1 = sum_df[sum_df['metric'] == 'tgt_f1'][['method', 'formatted', 'n_seeds']]
    print(tgt_f1.to_string(index=False))

    return raw_df, sum_df


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)  # progress lines show up in the log immediately
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=METHODS,
                        help='Methods to run (default: all 7)')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='Seeds to use (default: 42 84 123 256 512)')
    parser.add_argument('--use-verbose', dest='use_verbose', action='store_true',
                        help='Show all training logs (default: summary only)')
    args = parser.parse_args()
    set_verbose(args.use_verbose)

    print(f'Project root : {PROJECT_ROOT}')
    print(f'DAPT model   : {DAPT_MODEL_DIR} (exists={DAPT_MODEL_DIR.exists()})')
    print(f'Methods      : {args.methods}')
    print(f'Seeds        : {args.seeds}')
    print(f'Output       : {OUTPUT_DIR}')

    run_all(seeds=args.seeds, methods=args.methods)
