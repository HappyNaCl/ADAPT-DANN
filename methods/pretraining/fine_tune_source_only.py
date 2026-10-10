"""
Fine-tune IndoBERT on coastSent (source) only and evaluate.

This script fine-tunes the domain-adapted IndoBERT (DAPT) on the coastSent
labeled dataset (source only), evaluates on both coastSent (source) and
Lazada (target) test sets, and produces a t-SNE visualization of encoder
embeddings (source vs target). Adjust paths/parameters as needed.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

sns.set(style='whitegrid')

# Default configuration - override via command-line arguments
DEFAULT_MODEL_DIR = '../../dapt_lazada/'  # domain-adapted IndoBERT (MLM)
DEFAULT_OUTPUT_DIR = './models/indobert_telemedicine_finetuned'
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 128
DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-5
DEFAULT_SEED = 42
DEFAULT_TRAIN_CSV = '../../datasets/telemedicine_preprocessed/train.csv'
DEFAULT_COAST_TEST_CSV = '../../datasets/telemedicine_preprocessed/test.csv'
DEFAULT_LAZADA_TEST_CSV = '../../datasets/lazada_test_preprocessed.csv'
DEFAULT_TSNE_OUT = 'outputs/tsne_coastsent_vs_lazada.png'

LABEL2ID = {'negative': 0, 'positive': 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ---------- Verbose control ---------------------------------------------------
# Default: only the compact 2-line header + 4-line summary is printed.
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
        if s is None or t is None:
            continue
        print(f'{key}: source={s:.4f}  target={t:.4f}  gap={s - t:+.4f}', file=out)
    out.flush()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_text_column(df):
    for c in ['reviewContent', 'content', 'text', 'review']:
        if c in df.columns:
            return c
    raise ValueError('No text-like column found: ' + ','.join(df.columns))


def _normalize_labels(series):
    return series.astype(str).str.strip().str.lower().replace({
        'pos': 'positive', 'neg': 'negative',
        '1': 'positive', '0': 'negative',
        'positif': 'positive', 'negatif': 'negative',
        'true': 'positive', 'false': 'negative',
    })


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def eval_on_df(trainer, tokenize_fn, df, name='eval'):
    tc = resolve_text_column(df)
    d = df.copy()
    d['label_id'] = d['label'].map(LABEL2ID)
    ds = Dataset.from_dict({'text': d[tc].astype(str).tolist(), 'label': d['label_id'].tolist()})
    tok = ds.map(tokenize_fn, batched=True, remove_columns=['text'])
    pred = trainer.predict(tok)
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    print(f'\nEvaluation on {name} - samples: {len(labels)}')
    print(classification_report(labels, preds, target_names=['negative', 'positive'], digits=4))
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def extract_embeddings_from_dataloader(model, dataloader, device):
    """Extract mean-pooled embeddings from a dataloader."""
    model.eval()
    Xs = []
    Ys = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Extract mean-pooled features from base model
            base = getattr(model, 'base_model', model)
            o = base(input_ids=input_ids, attention_mask=attention_mask)
            h = o.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            feats = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

            Xs.append(feats.cpu().numpy())

            # Extract labels - handle both tensor and missing cases
            if 'label_id' in batch:
                label_id = batch['label_id']
                if isinstance(label_id, torch.Tensor):
                    Ys.append(label_id.cpu().numpy())
                else:
                    Ys.append(np.array(label_id))
            else:
                Ys.append(np.full((feats.size(0),), -1, dtype=int))

    if len(Xs) == 0:
        return np.zeros((0, model.config.hidden_size)), np.array([])

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def project_embeddings(X, method='tsne', n_components=2, random_state=42):
    """Project embeddings to 2D using t-SNE or PCA."""
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


def plot_single_panel(proj, labels, domains, title, outpath=None, figsize=(7, 6)):
    """Plot a single projection panel with legend for domain+label combinations."""
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
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=200)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune IndoBERT (DAPT) on coastSent (source) only and evaluate.'
    )
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR,
                        help='Path to the domain-adapted (MLM) IndoBERT checkpoint')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Where to save the fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--max_length', type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--train_csv', type=str, default=DEFAULT_TRAIN_CSV)
    parser.add_argument('--coast_test_csv', type=str, default=DEFAULT_COAST_TEST_CSV)
    parser.add_argument('--lazada_test_csv', type=str, default=DEFAULT_LAZADA_TEST_CSV)
    parser.add_argument('--tsne_out', type=str, default=DEFAULT_TSNE_OUT,
                        help='Output path for the t-SNE plot')
    parser.add_argument('--use-verbose', dest='use_verbose', action='store_true',
                        help='Show all training logs (default: summary only)')
    return parser.parse_args()


def main():
    args = parse_args()
    set_verbose(args.use_verbose)

    MODEL_DIR = args.model_dir
    OUTPUT_DIR = args.output_dir
    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length
    EPOCHS = args.epochs
    LR = args.lr

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    os.makedirs('outputs', exist_ok=True)
    print('Config set')

    # Load datasets
    train_df = pd.read_csv(args.train_csv)
    coast_test_df = pd.read_csv(args.coast_test_csv)
    lazada_test_df = pd.read_csv(args.lazada_test_csv)
    print('Loaded shapes:', train_df.shape, coast_test_df.shape, lazada_test_df.shape)

    # Clean and prepare labels/text
    text_col = resolve_text_column(train_df)
    for df in (train_df, coast_test_df, lazada_test_df):
        tc = resolve_text_column(df)
        df[tc] = df[tc].astype(str).str.strip()
        if 'label' in df.columns:
            df['label'] = _normalize_labels(df['label'])

    train_df = train_df.dropna(subset=[text_col])
    train_df = train_df[train_df[text_col] != '']
    train_df['label_id'] = train_df['label'].map(LABEL2ID)
    train_df = train_df.dropna(subset=['label_id'])

    coast_test_df['label_id'] = coast_test_df['label'].map(LABEL2ID)
    lazada_test_df['label_id'] = lazada_test_df['label'].map(LABEL2ID)

    print('Train label counts:', train_df['label_id'].value_counts().to_dict())

    # Load tokenizer and classification model (adds head automatically)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    print('Model and tokenizer loaded')

    # Prepare HF datasets and tokenization
    hf_train = Dataset.from_dict({
        'text': train_df[text_col].astype(str).tolist(),
        'label': train_df['label_id'].tolist(),
    })
    split = hf_train.train_test_split(test_size=0.1, seed=args.seed)
    train_ds = split['train']
    val_ds = split['test']

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=MAX_LENGTH)

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=['text'])
    val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=['text'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print('Datasets tokenized:', len(train_tok), len(val_tok))

    # Training setup
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.02,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_steps=50,
        disable_tqdm=not USE_VERBOSE,
        fp16=torch.cuda.is_available(),
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print('Trainer ready')

    # Train (can take time) - reduce EPOCHS for a quick smoke test
    train_result = trainer.train()
    print(train_result.metrics)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print('Saved model to', OUTPUT_DIR)

    # Evaluation
    src_res = eval_on_df(trainer, tokenize_fn, coast_test_df, name='coast_test (source)')
    tgt_res = eval_on_df(trainer, tokenize_fn, lazada_test_df, name='lazada_test (target)')

    # Compact summary (always printed; silenced logs otherwise)
    print_summary(
        f"Model: DAPT fine-tune source-only ({MODEL_DIR.rstrip('/')})",
        f"batch={BATCH_SIZE}  max_len={MAX_LENGTH}  epochs={EPOCHS}  "
        f"lr={LR:g}  wd=0.02  seed={args.seed}",
        src_res,
        tgt_res,
    )

    # Extract encoder embeddings (mean-pooled) and plot t-SNE
    hf_coast_test = Dataset.from_dict({
        'text': coast_test_df[resolve_text_column(coast_test_df)].astype(str).tolist(),
        'label_id': coast_test_df['label_id'].astype(int).tolist(),
    })
    hf_lazada_test = Dataset.from_dict({
        'text': lazada_test_df[resolve_text_column(lazada_test_df)].astype(str).tolist(),
        'label_id': lazada_test_df['label_id'].astype(int).tolist(),
    })

    def tokenize_fn_simple(examples):
        enc = tokenizer(examples['text'], truncation=True, max_length=MAX_LENGTH, padding=False)
        enc['label_id'] = examples['label_id']
        return enc

    coast_test_tok = hf_coast_test.map(tokenize_fn_simple, batched=True, remove_columns=['text'])
    lazada_test_tok = hf_lazada_test.map(tokenize_fn_simple, batched=True, remove_columns=['text'])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    src_loader = DataLoader(coast_test_tok, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    tgt_loader = DataLoader(lazada_test_tok, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    Xs_src, Ys_src = extract_embeddings_from_dataloader(trainer.model, src_loader, device)
    Xs_tgt, Ys_tgt = extract_embeddings_from_dataloader(trainer.model, tgt_loader, device)

    X = np.concatenate([Xs_src, Xs_tgt], axis=0)
    Y = np.concatenate([Ys_src, Ys_tgt], axis=0)
    domains = np.array(['source'] * Xs_src.shape[0] + ['target'] * Xs_tgt.shape[0])

    proj = project_embeddings(X, method='tsne', n_components=2, random_state=42)

    outpath = args.tsne_out
    plot_single_panel(proj, Y, domains, title='DAPT (Fine tune Source)', outpath=outpath, figsize=(7, 6))
    print(f'Saved t-SNE visualization to: {outpath}')


if __name__ == '__main__':
    main()
