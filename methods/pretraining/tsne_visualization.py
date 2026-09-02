"""
t-SNE Visualization for IndoBERT Models
========================================
Mirrors the plot style from fine_tune_coastsent.ipynb:
  - "Fine tune ID-CoastSent" style scatter plots
  - Mean-pooled encoder embeddings
  - Source (coastSent) vs Target (Lazada) coloured by sentiment label

Usage
-----
  python tsne_visualization.py --mode source_finetune
  python tsne_visualization.py --mode mlm_target
  python tsne_visualization.py --mode both          (default)

Paths (edit the CONFIG block below if needed)
-----
  SOURCE_TEST  : coastsent_test.csv   (text column: 'text')
  TARGET_TEST  : lazada_test.csv      (text column: 'reviewContent')
  SOURCE_MODEL : ./models/indobert_source_final          (fine_tune_source.py output)
  MLM_MODEL    : ./models/indobert_mlm_target_final      (bert_mlm_target.py output)
  OUTPUT_DIR   : ./outputs/
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")          # headless — change to "TkAgg" if you want a window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    DataCollatorWithPadding,
)

sns.set(style="whitegrid")


# ============================================================================
# CONFIG  –  edit paths here if your layout differs
# ============================================================================
SOURCE_TEST_CSV   = "../../datasets/telemedicine_preprocessed/train.csv"
TARGET_TEST_CSV   = "../../datasets/lazada_test_preprocessed.csv"
SOURCE_TEXT_COL   = "text"         # column in coastsent_test.csv
TARGET_TEXT_COL   = "text"  # column in lazada_test.csv
LABEL_COL         = "label"

SOURCE_FINETUNE_MODEL = "./models/indobert_source_final"   # from fine_tune_source.py
MLM_TARGET_MODEL      = "./models/indobert_mlm_target_final"  # from bert_mlm_target.py

OUTPUT_DIR = "./outputs"
BATCH_SIZE = 32
MAX_LENGTH = 128
SEED       = 42

LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL  = {0: "negative", 1: "positive"}

# Colour palette — exactly as in the notebook
PALETTE = {
    ("source", 0): "#d62728",   # source negative  – red
    ("source", 1): "#1f77b4",   # source positive  – blue
    ("target", 0): "#2ca02c",   # target negative  – green
    ("target", 1): "#9467bd",   # target positive  – purple
}
DISPLAY_NAMES = {
    ("source", 0): "source negative",
    ("source", 1): "source positive",
    ("target", 0): "target negative",
    ("target", 1): "target positive",
}
PLOT_ORDER = [("source", 0), ("source", 1), ("target", 0), ("target", 1)]
# ============================================================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_labels(series: pd.Series) -> pd.Series:
    """Map various label spellings to 'negative' / 'positive'."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "pos": "positive", "neg": "negative",
            "1": "positive",   "0": "negative",
            "positif": "positive", "negatif": "negative",
            "true": "positive", "false": "negative",
        })
    )


# ── Data loading ─────────────────────────────────────────────────────────────

def load_test_sets():
    print("\n" + "=" * 60)
    print("Loading test sets …")
    print("=" * 60)

    src_df = pd.read_csv(SOURCE_TEST_CSV)
    tgt_df = pd.read_csv(TARGET_TEST_CSV)

    for df, tcol in [(src_df, SOURCE_TEXT_COL), (tgt_df, TARGET_TEXT_COL)]:
        df[tcol]    = df[tcol].astype(str).str.strip()
        df["label"] = normalize_labels(df[LABEL_COL])
        df["label_id"] = df["label"].map(LABEL2ID)

    src_df = src_df.dropna(subset=["label_id"])
    tgt_df = tgt_df.dropna(subset=["label_id"])

    print(f"  Source (coastsent_test) : {len(src_df)} rows")
    print(f"  Target (lazada_test)    : {len(tgt_df)} rows")
    return src_df, tgt_df


# ── Tokenisation helpers ──────────────────────────────────────────────────────

def make_hf_dataset(df: pd.DataFrame, text_col: str) -> Dataset:
    return Dataset.from_dict({
        "text":     df[text_col].astype(str).tolist(),
        "label_id": df["label_id"].astype(int).tolist(),
    })


def tokenize_dataset(hf_ds: Dataset, tokenizer, max_length: int = MAX_LENGTH) -> Dataset:
    def _tok(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["label_id"] = examples["label_id"]
        return enc

    return hf_ds.map(_tok, batched=True, remove_columns=["text"])


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_embeddings(model, dataloader, device) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean-pool the last hidden state from the base encoder.
    Works for both AutoModelForSequenceClassification and AutoModelForMaskedLM
    because both share the same bert/roberta base under the hood.
    """
    model.eval()
    Xs, Ys = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get the encoder (strip any classification / MLM head)
            base = getattr(model, "bert",
                   getattr(model, "roberta",
                   getattr(model, "base_model", model)))

            out  = base(input_ids=input_ids, attention_mask=attention_mask)
            h    = out.last_hidden_state                        # (B, L, H)
            mask = attention_mask.unsqueeze(-1).float()
            feats = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

            Xs.append(feats.cpu().numpy())

            if "label_id" in batch:
                lbl = batch["label_id"]
                Ys.append(lbl.cpu().numpy() if isinstance(lbl, torch.Tensor)
                          else np.array(lbl))
            else:
                Ys.append(np.full(feats.shape[0], -1, dtype=int))

    return np.concatenate(Xs, 0), np.concatenate(Ys, 0)


# ── Dimensionality reduction ──────────────────────────────────────────────────

def project_tsne(X: np.ndarray, random_state: int = SEED) -> np.ndarray:
    if X.shape[0] == 0:
        return X
    # PCA pre-reduction if high-dimensional
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=random_state).fit_transform(X)
    perplexity = min(30, max(5, X.shape[0] // 10))
    tsne = TSNE(n_components=2, random_state=random_state,
                init="pca", perplexity=perplexity)
    return tsne.fit_transform(X)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_tsne(proj: np.ndarray, labels: np.ndarray, domains: np.ndarray,
              title: str, outpath: str, figsize=(7, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    xs, ys = proj[:, 0], proj[:, 1]
    for dom, lab in PLOT_ORDER:
        mask = (domains == dom) & (labels == lab)
        if mask.sum() == 0:
            continue
        ax.scatter(
            xs[mask], ys[mask],
            s=10,
            c=PALETTE[(dom, lab)],
            label=DISPLAY_NAMES[(dom, lab)],
            alpha=0.8,
        )

    ax.set_title(title, fontsize=18)
    ax.grid(False)
    ax.legend(loc="lower left", fontsize=14, frameon=True, markerscale=2)
    fig.tight_layout()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=200)
    print(f"  Saved → {outpath}")
    plt.close(fig)


# ── Per-model pipeline ────────────────────────────────────────────────────────

def run_tsne_for_model(
    model_path: str,
    model_type: str,   # "classification" or "mlm"
    title: str,
    output_filename: str,
    src_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    device,
):
    print("\n" + "=" * 60)
    print(f"Model : {model_path}")
    print(f"Type  : {model_type}")
    print("=" * 60)

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if model_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    else:  # mlm
        model = AutoModelForMaskedLM.from_pretrained(model_path)

    model.to(device)
    model.eval()
    print(f"  Parameters: {model.num_parameters():,}")

    # Build dataloaders
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    src_tok = tokenize_dataset(make_hf_dataset(src_df, SOURCE_TEXT_COL), tokenizer)
    tgt_tok = tokenize_dataset(make_hf_dataset(tgt_df, TARGET_TEXT_COL), tokenizer)

    src_loader = DataLoader(src_tok, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collator)
    tgt_loader = DataLoader(tgt_tok, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collator)

    # Extract embeddings
    print("  Extracting source embeddings …")
    Xs_src, Ys_src = extract_embeddings(model, src_loader, device)
    print(f"    source  → {Xs_src.shape}")

    print("  Extracting target embeddings …")
    Xs_tgt, Ys_tgt = extract_embeddings(model, tgt_loader, device)
    print(f"    target  → {Xs_tgt.shape}")

    # Combine and run t-SNE
    X = np.concatenate([Xs_src, Xs_tgt], axis=0)
    Y = np.concatenate([Ys_src, Ys_tgt], axis=0)
    domains = np.array(
        ["source"] * len(Xs_src) + ["target"] * len(Xs_tgt)
    )

    print("  Running t-SNE (PCA→50 dims first if needed) …")
    proj = project_tsne(X)

    # Plot
    outpath = os.path.join(OUTPUT_DIR, output_filename)
    print(f"  Plotting → {outpath}")
    plot_tsne(proj, Y, domains, title=title, outpath=outpath)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="t-SNE visualisation for IndoBERT experiments"
    )
    parser.add_argument(
        "--mode",
        choices=["source_finetune", "mlm_target", "both"],
        default="both",
        help=(
            "source_finetune  →  plot for ./models/indobert_source_final  \n"
            "mlm_target       →  plot for ./models/indobert_mlm_target_final  \n"
            "both             →  run both (default)"
        ),
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    src_df, tgt_df = load_test_sets()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.mode in ("source_finetune", "both"):
        run_tsne_for_model(
            model_path=SOURCE_FINETUNE_MODEL,
            model_type="classification",
            title="Fine tune Source (Lazada → CoastSent)",
            output_filename="tsne_source_finetune.png",
            src_df=src_df,
            tgt_df=tgt_df,
            device=device,
        )

    if args.mode in ("mlm_target", "both"):
        run_tsne_for_model(
            model_path=MLM_TARGET_MODEL,
            model_type="mlm",
            title="DAPT (Telecomunication)",
            output_filename="dapt.png",
            src_df=src_df,
            tgt_df=tgt_df,
            device=device,
        )

    print("\n" + "=" * 60)
    print("Done!  Plots saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
