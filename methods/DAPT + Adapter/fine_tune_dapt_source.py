"""
DAPT-only baseline (no adapter).

Loads the DAPT backbone (IndoBERT further pretrained on Lazada reviews), freezes it,
and trains only a small classifier head on the labelled Telemedicine reviews (source).
The trained model is then tested on Telemedicine (source) and Lazada (target).

Comparing this with DAPT + Adapter shows how much the adapter adds.

Usage (from methods/DAPT + Adapter):
    python3 fine_tune_dapt_source.py --seed 42
    python3 fine_tune_dapt_source.py --seed 42 --head_lr 5e-4 --epochs 5
    python3 fine_tune_dapt_source.py --help      # every CFG key can be set this way
"""

import argparse
import json
import os
import sys
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    set_seed,
)

_HERE = Path(__file__).resolve().parent
_DATA = _HERE / '..' / '..' / 'datasets'

CFG = {
    # Paths
    'dapt_model_dir':     str(_HERE / '..' / '..' / 'dapt_lazada'),
    'source_labeled_csv': str(_DATA / 'telemedicine_preprocessed' / 'train_split.csv'),
    'source_val_csv':     str(_DATA / 'telemedicine_preprocessed' / 'val.csv'),  # shared by all methods
    'source_test_csv':    str(_DATA / 'telemedicine_preprocessed' / 'test.csv'),
    'target_eval_csv':    str(_DATA / 'lazada_test_preprocessed.csv'),
    'output_dir':         str(_HERE / 'models' / 'dapt_lazada_source_finetuned'),  # checkpoints
    'final_model_dir':    str(_HERE / 'models' / 'dapt_lazada_source_final'),
    'tsne_path':          str(_HERE / 'outputs' / 'tsne_dapt_source_vs_lazada.png'),

    # Training (same values as the other methods)
    'batch_size':    32,
    'max_length':    128,
    'epochs':        3,
    'head_lr':       1e-3,   # only the classifier head trains; the encoder is frozen
    'weight_decay':  0.02,
    'warmup_ratio':  0.1,
    'dropout':       0.2,
    'logging_steps': 50,
    'seed':          42,
}

LABELS = ['negative', 'positive']
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

# In quiet mode stdout is muted, so the summary is written here instead.
REAL_STDOUT = sys.stdout


# ---------- Setup -------------------------------------------------------------
def load_config():
    """Return CFG with any command-line overrides, plus the verbose flag."""
    parser = argparse.ArgumentParser(
        description='DAPT-only baseline: train a classifier head on source data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--use-verbose', '--use_verbose', dest='use_verbose', action='store_true',
                        help='show the full training log instead of just the summary')
    for key, default in CFG.items():
        parser.add_argument(f'--{key}', type=type(default), default=default)
    args = parser.parse_args()
    return {key: getattr(args, key) for key in CFG}, args.use_verbose


def mute_output():
    """Hide prints, library logs and progress bars (used when not verbose)."""
    sys.stdout = open(os.devnull, 'w')
    transformers.logging.set_verbosity_error()
    datasets.disable_progress_bars()


# ---------- Data --------------------------------------------------------------
def read_labeled_csv(path):
    df = pd.read_csv(path)
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    return df


def to_dataset(df, tokenizer, max_length):
    """Turn a dataframe with 'text' and 'label' columns into a tokenized Dataset."""
    dataset = Dataset.from_dict({
        'text': df['text'].tolist(),
        'label': df['label'].map(LABEL2ID).tolist(),
    })
    return dataset.map(
        lambda batch: tokenizer(batch['text'], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=['text'],
    )


# ---------- Model -------------------------------------------------------------
class DAPTClassifier(nn.Module):
    """Frozen DAPT encoder -> average of token vectors -> small classifier head."""

    def __init__(self, model_dir, dropout):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir)
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, len(LABELS)),
        )

    def embed(self, input_ids, attention_mask):
        """One vector per review: the mean of its token vectors, ignoring padding."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        logits = self.classifier(self.dropout(self.embed(input_ids, attention_mask)))
        if labels is None:
            return {'logits': logits}
        return {'loss': nn.functional.cross_entropy(logits, labels), 'logits': logits}


# ---------- Evaluation --------------------------------------------------------
def scores(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {'accuracy': accuracy_score(labels, preds), 'precision': precision, 'recall': recall, 'f1': f1}


def compute_metrics(eval_pred):
    """Used by the Trainer on the validation split after each epoch."""
    logits, labels = eval_pred
    return scores(labels, np.argmax(logits, axis=1))


def evaluate(trainer, dataset, name):
    """Predict on a test set, print a per-class report and return the scores."""
    output = trainer.predict(dataset)
    preds = np.argmax(output.predictions, axis=1)
    print(f'\n{name} test set ({len(dataset)} reviews)')
    print(classification_report(output.label_ids, preds, target_names=LABELS, digits=4))
    return scores(output.label_ids, preds)


def print_summary(cfg, source, target):
    """The short result block, shown even in quiet mode.
    run_dapt_source_seeds.sh greps these lines, so keep the format."""
    lines = [
        f"Model: DAPT-only, no adapter ({Path(cfg['dapt_model_dir']).name})",
        f"batch={cfg['batch_size']}  max_len={cfg['max_length']}  epochs={cfg['epochs']}  "
        f"lr_heads={cfg['head_lr']:g}  wd={cfg['weight_decay']}  "
        f"val={Path(cfg['source_val_csv']).name}  seed={cfg['seed']}",
    ]
    for key in ['accuracy', 'precision', 'recall', 'f1']:
        s, t = source[key], target[key]
        lines.append(f'{key}: source={s:.4f}  target={t:.4f}  gap={s - t:+.4f}')
    print('\n'.join(lines), file=REAL_STDOUT, flush=True)


def plot_tsne(model, tokenizer, source_test, target_test, path, device, title='DAPT'):
    """Squash the source and target test embeddings to 2D and save a scatter plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    @torch.no_grad()
    def embed_all(dataset):
        loader = DataLoader(dataset, batch_size=32, collate_fn=DataCollatorWithPadding(tokenizer))
        vectors, labels = [], []
        for batch in loader:
            vectors.append(model.embed(batch['input_ids'].to(device),
                                       batch['attention_mask'].to(device)).cpu().numpy())
            labels.append(batch['labels'].numpy())
        return np.concatenate(vectors), np.concatenate(labels)

    model.eval()
    src_x, src_y = embed_all(source_test)
    tgt_x, tgt_y = embed_all(target_test)
    x = np.concatenate([src_x, tgt_x])
    y = np.concatenate([src_y, tgt_y])
    domain = np.array(['source'] * len(src_x) + ['target'] * len(tgt_x))

    if x.shape[1] > 50:  # PCA first makes t-SNE faster and less noisy
        x = PCA(n_components=50, random_state=42).fit_transform(x)
    points = TSNE(n_components=2, random_state=42, init='pca', perplexity=30).fit_transform(x)

    colors = {
        ('source', 0): '#d62728', ('source', 1): '#1f77b4',
        ('target', 0): '#2ca02c', ('target', 1): '#9467bd',
    }
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(7, 6))
    for (dom, label), color in colors.items():
        mask = (domain == dom) & (y == label)
        if mask.any():
            ax.scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.8, c=color,
                       label=f'{dom} {LABELS[label]}')
    ax.set_title(title, fontsize=24)
    ax.grid(False)
    ax.legend(loc='best', fontsize=18, frameon=True, markerscale=2)
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------- Main --------------------------------------------------------------
def main():
    cfg, verbose = load_config()
    if not verbose:
        mute_output()
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('Config:', json.dumps(cfg, indent=2))

    # Data
    train_df = read_labeled_csv(cfg['source_labeled_csv'])
    val_df = read_labeled_csv(cfg['source_val_csv'])
    source_test_df = read_labeled_csv(cfg['source_test_csv'])
    target_test_df = read_labeled_csv(cfg['target_eval_csv'])
    print('Source training labels:', train_df['label'].value_counts().to_dict())

    # Model
    tokenizer = AutoTokenizer.from_pretrained(cfg['dapt_model_dir'])
    model = DAPTClassifier(cfg['dapt_model_dir'], cfg['dropout']).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable params: {trainable:,} of {total:,}')

    # Tokenize (the validation set is the fixed val.csv shared by all methods)
    train_set = to_dataset(train_df, tokenizer, cfg['max_length'])
    val_set = to_dataset(val_df, tokenizer, cfg['max_length'])
    source_test = to_dataset(source_test_df, tokenizer, cfg['max_length'])
    target_test = to_dataset(target_test_df, tokenizer, cfg['max_length'])
    print(f'Train {len(train_set)} | val {len(val_set)} | '
          f'source test {len(source_test)} | target test {len(target_test)}')

    # We give the Trainer our own optimizer and schedule, so learning-rate
    # settings inside TrainingArguments would be ignored.
    optimizer = AdamW(model.classifier.parameters(), lr=cfg['head_lr'], weight_decay=cfg['weight_decay'])
    total_steps = len(train_set) // cfg['batch_size'] * cfg['epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg['warmup_ratio'] * total_steps), num_training_steps=total_steps)

    training_args = TrainingArguments(
        output_dir=cfg['output_dir'],
        num_train_epochs=cfg['epochs'],
        per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['batch_size'],
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,             # each checkpoint is ~440 MB
        load_best_model_at_end=True,    # keep the epoch with the best validation F1
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_steps=cfg['logging_steps'],
        disable_tqdm=not verbose,
        fp16=torch.cuda.is_available(),
        report_to='none',
        seed=cfg['seed'],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    # Train
    trainer.train()
    print('Validation:', trainer.evaluate())
    trainer.save_model(cfg['final_model_dir'])
    tokenizer.save_pretrained(cfg['final_model_dir'])
    print('Saved model to', cfg['final_model_dir'])

    # Test
    source_scores = evaluate(trainer, source_test, 'Source (Telemedicine)')
    target_scores = evaluate(trainer, target_test, 'Target (Lazada)')

    # The plot is a nice extra; don't lose the results if it fails
    try:
        plot_tsne(model, tokenizer, source_test, target_test, cfg['tsne_path'], device)
        print('Saved t-SNE plot to', cfg['tsne_path'])
    except Exception as e:
        print(f'Warning: t-SNE plot failed: {e}', file=sys.stderr)

    print_summary(cfg, source_scores, target_scores)


if __name__ == '__main__':
    main()
