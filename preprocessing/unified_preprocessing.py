"""
Unified preprocessing pipeline for Hospital and Lazada datasets.
Applies the SAME pipeline used for ID-CoastSent source domain:
  1. basic_clean (HTML unescape, newline removal, quote removal)
  2. safe_replace_slang (indoNLP slang dictionary)
  3. emoji_to_words (convert emojis to text descriptions)
  4. replace_word_elongation (e.g., "baguuuus" → "bagus")
  5. strip_leftover_emojis (remove remaining Unicode emojis)
  6. Drop duplicates and empty texts
"""

import html
import os
import re

import pandas as pd
from indoNLP.preprocessing import (
    SLANG_DATA,
    SLANG_PATTERN,
    emoji_to_words,
    pipeline,
    replace_word_elongation,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def safe_replace_slang(text):
    """Replace slang words, keep original if not found."""
    return re.sub(
        SLANG_PATTERN,
        lambda mo: SLANG_DATA.get(mo.group(0).lower(), mo.group(0)),
        text,
    )


LEFTOVER_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF"
    "\uFE0F\u200D\u2049\u203C\u2139\u2194-\u21AA\u231A-\u23FA]"
)

QUOTES_RE = re.compile(r'[\'"\u2018\u2019\u201a\u201b\u201c\u201d\u201e\xab\xbb`]')


def basic_clean(text):
    text = html.unescape(str(text))
    text = re.sub(r"[\r\n]+", " ", text)
    text = QUOTES_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_leftover_emojis(text):
    return LEFTOVER_EMOJI_RE.sub(" ", text)


# Build the exact same pipeline used for ID-CoastSent
nlp_pipe = pipeline([safe_replace_slang, emoji_to_words, replace_word_elongation])


def preprocess_text(text):
    """Full preprocessing pipeline matching ID-CoastSent."""
    text = basic_clean(text)
    text = nlp_pipe(text)
    text = strip_leftover_emojis(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Hospital preprocessing
# ---------------------------------------------------------------------------

def preprocess_hospital(input_path, output_dir, test_size=0.2, seed=42):
    """
    Preprocess hospital dataset with the same pipeline as ID-CoastSent.
    Produces train.csv and test.csv in output_dir.
    """
    print("=" * 60)
    print(" HOSPITAL PREPROCESSING")
    print("=" * 60)

    df = pd.read_excel(input_path)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    print(f"  Loaded: {len(df):,} rows")
    print(f"  Columns: {df.columns.tolist()}")

    # Drop NaN
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    print(f"  After dropna: {len(df):,} rows (dropped {before - len(df):,})")

    # Remove duplicate texts
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"  After dedup:  {len(df):,} rows (dropped {before - len(df):,})")

    # Apply preprocessing pipeline
    print("  Applying preprocessing pipeline (slang + emoji + elongation)...")
    df["text"] = df["text"].apply(preprocess_text)

    # Post-pipeline cleanup
    before = len(df)
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"  After cleanup: {len(df):,} rows (dropped {before - len(df):,})")

    # Label distribution
    print(f"\n  Label distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"    {label}: {count:,}")

    # Stratified train/test split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n  Saved: {train_path} ({len(train_df):,} rows)")
    print(f"  Saved: {test_path} ({len(test_df):,} rows)")

    # Sample
    print("\n  Sample preprocessed texts:")
    for _, row in train_df.head(3).iterrows():
        print(f"    [{row['label']}] {row['text'][:100]}")

    return train_df, test_df


# ---------------------------------------------------------------------------
# Lazada preprocessing
# ---------------------------------------------------------------------------

def preprocess_lazada(input_path, output_dir, text_col="reviewContent", 
                      label_from="rating", threshold=3, test_size=0.2, seed=42):
    """
    Preprocess Lazada dataset with the same pipeline as ID-CoastSent.
    Produces train.csv and test.csv in output_dir.
    """
    print("\n" + "=" * 60)
    print(" LAZADA PREPROCESSING")
    print("=" * 60)

    df = pd.read_csv(input_path)
    print(f"  Loaded: {len(df):,} rows")
    print(f"  Columns: {df.columns.tolist()}")

    # Create binary labels from rating
    df["label"] = df[label_from].apply(lambda x: "positive" if x > threshold else "negative")
    df = df.rename(columns={text_col: "text"})
    df = df[["text", "label"]].copy()

    # Drop NaN
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    print(f"  After dropna: {len(df):,} rows (dropped {before - len(df):,})")

    # Remove duplicate texts
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"  After dedup:  {len(df):,} rows (dropped {before - len(df):,})")

    # Apply preprocessing pipeline
    print("  Applying preprocessing pipeline (slang + emoji + elongation)...")
    df["text"] = df["text"].apply(preprocess_text)

    # Post-pipeline cleanup
    before = len(df)
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"  After cleanup: {len(df):,} rows (dropped {before - len(df):,})")

    # Label distribution
    print(f"\n  Label distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"    {label}: {count:,}")

    # Stratified train/test split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n  Saved: {train_path} ({len(train_df):,} rows)")
    print(f"  Saved: {test_path} ({len(test_df):,} rows)")

    # Sample
    print("\n  Sample preprocessed texts:")
    for _, row in train_df.head(3).iterrows():
        print(f"    [{row['label']}] {row['text'][:100]}")

    return train_df, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASETS = os.path.join(BASE, "datasets")

    # Preprocess hospital
    hospital_train, hospital_test = preprocess_hospital(
        input_path=os.path.join(DATASETS, "hospital_dataset.xlsx"),
        output_dir=os.path.join(DATASETS, "hospital_preprocessed"),
    )

    # Preprocess Lazada
    lazada_train, lazada_test = preprocess_lazada(
        input_path=os.path.join(DATASETS, "original", "lazada", "lazada.csv"),
        output_dir=os.path.join(DATASETS, "lazada_preprocessed"),
    )

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  Hospital train: {len(hospital_train):,} rows")
    print(f"  Hospital test:  {len(hospital_test):,} rows")
    print(f"  Lazada train:   {len(lazada_train):,} rows")
    print(f"  Lazada test:    {len(lazada_test):,} rows")
    print(f"\n  Output dirs:")
    print(f"    {os.path.join(DATASETS, 'hospital_preprocessed')}")
    print(f"    {os.path.join(DATASETS, 'lazada_preprocessed')}")
