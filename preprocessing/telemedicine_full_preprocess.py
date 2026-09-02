"""
Full Telemedicine preprocessing pipeline:
1. basic_clean (HTML unescape, newline, quotes)
2. indoNLP (slang replacement, emoji_to_words, elongation)
3. Case folding (lowercase)
4. URL removal
5. Mention removal (@user)
6. Punctuation removal
7. Emoji removal (leftover Unicode)
8. Strip extra whitespace
9. Split 80/20 train/test
10. Downsample to 1:1 balanced ratio
11. Split train 70/30 for domain/task
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

SEED = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(BASE, "datasets")
OUTPUT_DIR = os.path.join(DATASETS, "telemedicine_preprocessed")

# --- Preprocessing helpers ---

def safe_replace_slang(text):
    return re.sub(
        SLANG_PATTERN,
        lambda mo: SLANG_DATA.get(mo.group(0).lower(), mo.group(0)),
        text,
    )

LEFTOVER_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⬀-⯿"
    "️‍⁉‼ℹ↔-↪⌚-⏺]"
)
QUOTES_RE = re.compile(r'[\'"‘’‚‛“”„\xab\xbb`]')
URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
# Keep basic punctuation that might carry sentiment: !?
PUNCTUATION_RE = re.compile(r'[^\w\s!?]')

def basic_clean(text):
    text = html.unescape(str(text))
    text = re.sub(r"[\r\n]+", " ", text)
    text = QUOTES_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_urls(text):
    return URL_RE.sub("", text)

def remove_mentions(text):
    return MENTION_RE.sub("", text)

def remove_punctuation(text):
    return PUNCTUATION_RE.sub(" ", text)

def remove_emojis(text):
    return LEFTOVER_EMOJI_RE.sub(" ", text)

def case_fold(text):
    return text.lower()

def sastrawi_stem(text):
    return sastrawi_stemmer.stem(text)

# indoNLP pipeline (slang + emoji_to_words + elongation)
nlp_pipe = pipeline([safe_replace_slang, emoji_to_words, replace_word_elongation])

def preprocess_text(text):
    # Step 1: Basic clean
    text = basic_clean(text)
    # Step 2: indoNLP (slang, emoji words, elongation)
    text = nlp_pipe(text)
    # Step 3: URL removal
    text = remove_urls(text)
    # Step 4: Mention removal
    text = remove_mentions(text)
    # Step 5: Punctuation removal
    text = remove_punctuation(text)
    # Step 6: Emoji removal (leftover Unicode)
    text = remove_emojis(text)
    # Step 7: Case folding
    text = case_fold(text)
    # Step 8: Strip extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def downsample(df, seed=SEED):
    pos = df[df['label'].str.upper() == 'POSITIVE']
    neg = df[df['label'].str.upper() == 'NEGATIVE']
    n = min(len(pos), len(neg))
    pos_down = pos.sample(n=n, random_state=seed)
    neg_down = neg.sample(n=n, random_state=seed)
    return pd.concat([pos_down, neg_down]).sample(frac=1, random_state=seed).reset_index(drop=True)

def print_stats(df, name):
    pos = (df['label'].str.upper() == 'POSITIVE').sum()
    neg = (df['label'].str.upper() == 'NEGATIVE').sum()
    print(f"  {name:<20}: {len(df):>6} rows | pos={pos:,} neg={neg:,} ratio={pos/neg:.1f}:1")

# --- Main ---

print("=" * 60)
print(" TELEMEDICINE FULL PREPROCESSING")
print("=" * 60)

# Load raw
df = pd.read_csv(os.path.join("../datasets/telemedicine.csv"))
# We only need review and sentiment
df = df[['review', 'sentiment']].rename(columns={"review": "text", "sentiment": "label"})
print(f"\n1. Loaded: {len(df):,} rows")

# Drop NaN/empty
before = len(df)
df = df.dropna(subset=["text", "label"])
df = df[df["text"].astype(str).str.strip() != ""]
print(f"2. After dropna: {len(df):,} rows (dropped {before - len(df):,})")

# Dedup
before = len(df)
df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
print(f"3. After dedup:  {len(df):,} rows (dropped {before - len(df):,})")

# For binary classification, we keep only POSITIVE and NEGATIVE, discard NEUTRAL
before = len(df)
df = df[df['label'].isin(['POSITIVE', 'NEGATIVE'])].reset_index(drop=True)
print(f"4. After keeping only POSITIVE/NEGATIVE: {len(df):,} rows (dropped {before - len(df):,})")

# Full preprocessing
print("5. Applying full preprocessing pipeline...")
print("   (slang -> emoji_words -> elongation -> urls -> mentions -> punct -> emoji -> casefold)")
df["text"] = df["text"].apply(preprocess_text)

# Post-cleanup
before = len(df)
df = df[df["text"].str.strip() != ""]
df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
print(f"6. After cleanup: {len(df):,} rows (dropped {before - len(df):,})")

print(f"\n   Label distribution:")
for label, count in df["label"].value_counts().items():
    print(f"     {label}: {count:,}")

# Sample
print(f"\n   Sample preprocessed texts:")
for _, row in df.head(3).iterrows():
    print(f"     [{row['label']}] {row['text'][:100]}")

# Split 80/20
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"\n7. After 80/20 split:")
print_stats(train_df, "Train")
print_stats(test_df, "Test")

# Downsample to 1:1
train_bal = downsample(train_df)
test_bal = downsample(test_df)

print(f"\n8. After downsampling to 1:1:")
print_stats(train_bal, "Train")
print_stats(test_bal, "Test")

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_bal.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
test_bal.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print(f"\n9. Saved to {OUTPUT_DIR}/")
print(f"     train.csv ({len(train_bal):,} rows)")
print(f"     test.csv  ({len(test_bal):,} rows)")

# Optional: Comparison with original distribution
print(f"\n{'=' * 60}")
print(" COMPARISON WITH ORIGINAL TELEMEDICINE (after removing NEUTRAL)")
print(f"{'=' * 60}")
original_df = pd.read_csv(os.path.join("../datasets/telemedicine.csv"))
original_df = original_df[['review', 'sentiment']].rename(columns={"review": "text", "sentiment": "label"})
original_df = original_df.dropna(subset=["text", "label"])
original_df = original_df[original_df["text"].astype(str).str.strip() != ""]
original_df = original_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
original_df = original_df[original_df['label'].isin(['POSITIVE', 'NEGATIVE'])].reset_index(drop=True)

pos_orig = (original_df['label'] == 'POSITIVE').sum()
neg_orig = (original_df['label'] == 'NEGATIVE').sum()
ratio_orig = pos_orig / neg_orig if neg_orig > 0 else 0

print(f"\nOriginal (after removing NEUTRAL): {len(original_df):,} rows | pos={pos_orig:,} neg={neg_orig:,} ratio={ratio_orig:.1f}:1")
print(f"Preprocessed train: {len(train_bal):,} rows | pos={(train_bal['label'].str.upper() == 'POSITIVE').sum():,} neg={(train_bal['label'].str.upper() == 'NEGATIVE').sum():,} ratio={len(train_bal)/2/((train_bal['label'].str.upper() == 'POSITIVE').sum()):.1f}:1 (balanced)")
print(f"Preprocessed test:  {len(test_bal):,} rows | pos={(test_bal['label'].str.upper() == 'POSITIVE').sum():,} neg={(test_bal['label'].str.upper() == 'NEGATIVE').sum():,} ratio={len(test_bal)/2/((test_bal['label'].str.upper() == 'POSITIVE').sum()):.1f}:1 (balanced)")