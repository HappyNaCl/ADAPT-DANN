"""
IndoBERT Baseline: t-SNE Visualization

Script version of indobert_baseline.ipynb.

Extracts [CLS] embeddings from the IndoBERT encoder only (no DANN, no adapters)
for the source and target domains, projects them with t-SNE and saves the plot,
coordinates and metadata. Compare the figure with the DANN / ADAPT-DANN plots.

Usage:
  python3 indobert_baseline.py
  python3 indobert_baseline.py --seed 84 --output_dir outputs/indobert_baseline_seed84
  MPLBACKEND=Agg python3 indobert_baseline.py      # headless: save plot, no window
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

import matplotlib.pyplot as plt
import seaborn as sns

# Relative paths below resolve against this file's folder, so the script works
# no matter which directory it is launched from.
SCRIPT_DIR = Path(__file__).resolve().parent


# =========================
# 1) Configuration
# =========================
CFG = {
    'model_name': 'indolem/indobert-base-uncased',
    'source_csv': '../../datasets/telemedicine_preprocessed/train.csv',
    'target_csv': '../../datasets/lazada_train_preprocessed.csv',
    'eval_csv': '../../datasets/lazada_test_preprocessed.csv',
    'output_dir': 'outputs/indobert_baseline',

    'batch_size': 32,
    'max_length': 128,
    'seed': 42,
    'max_samples_per_domain': 2000,
}


def parse_args():
    p = argparse.ArgumentParser(
        description='IndoBERT baseline t-SNE (encoder only, no domain adaptation)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model_name', type=str, default=CFG['model_name'])
    p.add_argument('--source_csv', type=str, default=CFG['source_csv'])
    p.add_argument('--eval_csv', type=str, default=CFG['eval_csv'])
    p.add_argument('--output_dir', type=str, default=CFG['output_dir'])
    p.add_argument('--batch_size', type=int, default=CFG['batch_size'])
    p.add_argument('--max_length', type=int, default=CFG['max_length'])
    p.add_argument('--seed', type=int, default=CFG['seed'])
    p.add_argument('--max_samples_per_domain', type=int, default=CFG['max_samples_per_domain'])
    p.add_argument('--title', type=str, default='IndoBERT Baseline')
    return p.parse_args()


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else SCRIPT_DIR / p


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 2) Utility Functions
# =========================
LABEL2ID = {'negative': 0, 'positive': 1}
ID2LABEL = {0: 'negative', 1: 'positive'}


def _resolve_text_column(df: pd.DataFrame) -> str:
    for col in ['content', 'reviewContent', 'text', 'review']:
        if col in df.columns:
            return col
    raise ValueError(f'Cannot find text column. Available: {df.columns.tolist()}')


def _resolve_label_column(df: pd.DataFrame) -> str:
    for col in ['label', 'sentiment', 'target']:
        if col in df.columns:
            return col
    raise ValueError(f'Cannot find label column. Available: {df.columns.tolist()}')


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


def load_labeled_csv(path: Path, filter_target_origin: bool = False):
    df = pd.read_csv(path)
    text_col = _resolve_text_column(df)
    label_col = _resolve_label_column(df)

    # Filter to target origin if column exists
    if filter_target_origin and 'origin' in df.columns:
        origin = df['origin'].astype(str).str.lower()
        df = df[origin == 'target']
        print(f"  Filtered to 'target' origin: {len(df)} samples")

    df = clean_dataframe(df, text_col, label_col)
    df['label_id'] = df[label_col].map(LABEL2ID)
    print(f"  Samples: {len(df)}")
    print(f"  Class distribution:\n{df['label_id'].value_counts().sort_index()}")
    return df, text_col


# =========================
# 3) Extract IndoBERT Embeddings
# =========================
def extract_embeddings_indobert(encoder, dataloader, device, max_samples=None):
    """Extract [CLS] token embeddings from IndoBERT encoder."""
    encoder.eval()
    embeddings = []
    labels = []
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if max_samples is not None and sample_count >= max_samples:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].cpu().numpy()

            # Get encoder output and extract [CLS] token (first token)
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

            embeddings.append(cls_embeddings.cpu().numpy())
            labels.extend(batch_labels.tolist())
            sample_count += len(batch_labels)

    if not embeddings:
        return np.zeros((0, encoder.config.hidden_size)), np.array([])

    return np.concatenate(embeddings, axis=0), np.array(labels)


# =========================
# 4) t-SNE Projection
# =========================
def project_tsne(X, seed):
    # Apply PCA first if necessary (speeds up t-SNE)
    if X.shape[1] > 50:
        print("  Applying PCA to reduce dimensionality...")
        X = PCA(n_components=50, random_state=seed).fit_transform(X)
        print(f"  PCA output shape: {X.shape}")

    tsne = TSNE(n_components=2, random_state=seed, init='pca', perplexity=30)
    return tsne.fit_transform(X)


# =========================
# 5) Plot t-SNE Combined (Source + Target)
# =========================
sns.set(style="whitegrid")


def plot_tsne_combined(proj, labels, domains, title, figsize=(7, 6), outpath=None):
    """Plot t-SNE using the same visual style as the ADAPT-DANN script."""
    palette = {
        ('source', 0): '#d62728',  # red
        ('source', 1): '#1f77b4',  # blue
        ('target', 0): '#2ca02c',  # green
        ('target', 1): '#9467bd',  # purple
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
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=200)
        print(f"  Saved: {outpath}")
    plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Device:', device)
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))

    source_csv = _resolve_path(args.source_csv)
    eval_csv = _resolve_path(args.eval_csv)
    output_dir = _resolve_path(args.output_dir)

    for key, path in [('source_csv', source_csv), ('eval_csv', eval_csv)]:
        print(f"{key}: {path} -> {'✓' if path.exists() else '✗'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput dir: {output_dir}")

    # Load IndoBERT
    print("\nLoading IndoBERT tokenizer and encoder...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name).to(device)
    encoder.eval()
    hidden_size = encoder.config.hidden_size
    print(f"✓ IndoBERT loaded | hidden_size={hidden_size} | device={device}")

    # Load and clean datasets
    print("\nLoading source dataset...")
    source_df, source_text_col = load_labeled_csv(source_csv)
    print("\nLoading target (eval) dataset...")
    target_df, target_text_col = load_labeled_csv(eval_csv, filter_target_origin=True)

    # Build DataLoaders
    source_ds = TextDataset(
        texts=source_df[source_text_col].tolist(),
        labels=source_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    target_ds = TextDataset(
        texts=target_df[target_text_col].tolist(),
        labels=target_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    source_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=False)
    target_loader = DataLoader(target_ds, batch_size=args.batch_size, shuffle=False)

    # Extract embeddings
    print("\nExtracting source embeddings...")
    X_source, y_source = extract_embeddings_indobert(
        encoder, source_loader, device, max_samples=args.max_samples_per_domain)
    print(f"  Source embeddings: shape={X_source.shape}, labels={len(y_source)}")

    print("\nExtracting target embeddings...")
    X_target, y_target = extract_embeddings_indobert(
        encoder, target_loader, device, max_samples=args.max_samples_per_domain)
    print(f"  Target embeddings: shape={X_target.shape}, labels={len(y_target)}")

    # Combine for unified t-SNE projection
    X_combined = np.concatenate([X_source, X_target], axis=0)
    y_combined = np.concatenate([y_source, y_target], axis=0)
    domains = ['source'] * len(y_source) + ['target'] * len(y_target)
    print(f"\n✓ Combined embeddings: shape={X_combined.shape}")

    print("\nComputing t-SNE projection (this may take a minute)...")
    X_tsne = project_tsne(X_combined, args.seed)
    print(f"✓ t-SNE projection complete: shape={X_tsne.shape}")

    plot_tsne_combined(
        X_tsne,
        y_combined,
        domains,
        title=args.title,
        figsize=(7, 6),
        outpath=output_dir / 'tsne_indobert_combined.png',
    )

    # Export results and metadata
    results_df = pd.DataFrame({
        'tsne_dim1': X_tsne[:, 0],
        'tsne_dim2': X_tsne[:, 1],
        'label_id': y_combined,
        'label_name': [ID2LABEL[lab] for lab in y_combined],
        'domain': domains,
    })
    results_csv = output_dir / 'tsne_indobert_coordinates.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Saved coordinates: {results_csv}")

    metadata = {
        'model': args.model_name,
        'framework': 'indobert_baseline',
        'projection_method': 'tsne',
        'seed': args.seed,
        'n_samples_total': len(X_tsne),
        'n_samples_source': len(y_source),
        'n_samples_target': len(y_target),
        'hidden_size': hidden_size,
        'output_files': {
            'combined': 'tsne_indobert_combined.png',
            'coordinates': 'tsne_indobert_coordinates.csv',
        },
    }
    metadata_json = output_dir / 'tsne_indobert_metadata.json'
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_json}")

    print(f"\n✓ All outputs saved to: {output_dir}")
    print("\nSummary Statistics:")
    print(f"  • Total samples: {len(X_tsne)}")
    print(f"  • Source: {len(y_source)} ({100*len(y_source)/len(X_tsne):.1f}%)")
    print(f"  • Target: {len(y_target)} ({100*len(y_target)/len(X_tsne):.1f}%)")
    print(f"\n  Source class dist:\n{pd.Series(y_source).value_counts().sort_index()}")
    print(f"\n  Target class dist:\n{pd.Series(y_target).value_counts().sort_index()}")


if __name__ == '__main__':
    main()
