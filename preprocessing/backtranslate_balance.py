"""
Telemedicine preprocessing + backtranslation balancing (no downsampling).

Run on GPU:
    pip install -r requirements.txt
    python backtranslate_balance.py
"""

import html
import os
import re
import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from transformers import MarianMTModel, MarianTokenizer

try:
    from indoNLP.preprocessing import (
        SLANG_DATA,
        SLANG_PATTERN,
        emoji_to_words,
        pipeline,
        replace_word_elongation,
    )
    INDONLP_AVAILABLE = True
except ImportError:
    INDONLP_AVAILABLE = False
    print("Warning: indoNLP not available, skipping slang/emoji/elongation")

SEED = 42
MAX_TOKENS = 128
BATCH_SIZE = 32

BASE = Path(__file__).parent
OUTPUT_DIR = BASE / "datasets" / "telemedicine_preprocessed"

# --- Sastrawi Stemmer ---
factory = StemmerFactory()
sastrawi_stemmer = factory.create_stemmer()

# --- Regex patterns ---
QUOTES_RE = re.compile(r'[\'"\u2018\u2019\u201a\u201b\u201c\u201d\u201e\xab\xbb`]')
URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
PUNCTUATION_RE = re.compile(r'[^\w\s!?]')
LEFTOVER_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF"
    "\uFE0F\u203C\u2049\u2139\u2190-\u21AA\u231A-\u23FA]"
)

if INDONLP_AVAILABLE:
    def safe_replace_slang(text):
        return re.sub(
            SLANG_PATTERN,
            lambda mo: SLANG_DATA.get(mo.group(0).lower(), mo.group(0)),
            text,
        )
    nlp_pipe = pipeline([safe_replace_slang, emoji_to_words, replace_word_elongation])

# --- Preprocessing functions ---
def basic_clean(text):
    text = html.unescape(str(text))
    text = re.sub(r"[\r\n]+", " ", text)
    text = QUOTES_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text):
    text = basic_clean(text)
    if INDONLP_AVAILABLE:
        text = nlp_pipe(text)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = PUNCTUATION_RE.sub(" ", text)
    text = LEFTOVER_EMOJI_RE.sub(" ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = text.split()
    tokens = [sastrawi_stemmer.stem(token) for token in tokens]
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
    else:
        tokens = tokens + ["[PAD]"] * (MAX_TOKENS - len(tokens))
    
    return " ".join(tokens)

def print_stats(df, name):
    pos = (df['label'] == 'POSITIVE').sum()
    neg = (df['label'] == 'NEGATIVE').sum()
    print(f"  {name:<20}: {len(df):>6} rows | pos={pos:,} neg={neg:,} ratio={pos/neg if neg>0 else 'inf':.1f}:1")

def load_and_preprocess():
    print("=" * 60)
    print(" LOADING & PREPROCESSING")
    print("=" * 60)
    
    df = pd.read_csv(BASE / "Telemedicine.csv")
    df = df[['review', 'sentiment']].rename(columns={"review": "text", "sentiment": "label"})
    print(f"Loaded: {len(df):,} rows")
    
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    print(f"After dropna: {len(df):,} rows (dropped {before - len(df):,})")
    
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"After dedup: {len(df):,} rows (dropped {before - len(df):,})")
    
    before = len(df)
    df = df[df['label'].isin(['POSITIVE', 'NEGATIVE'])].reset_index(drop=True)
    print(f"After filter POSITIVE/NEGATIVE: {len(df):,} rows (dropped {before - len(df):,})")
    
    print("Preprocessing...")
    start = time.time()
    df["text"] = df["text"].apply(preprocess_text)
    print(f"Preprocessing done in {time.time() - start:.1f}s")
    
    before = len(df)
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"After cleanup: {len(df):,} rows (dropped {before - len(df):,})")
    
    for label, count in df["label"].value_counts().items():
        print(f"  {label}: {count:,}")
    
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"\nSplit: train={len(train_df):,}, test={len(test_df):,}")
    print_stats(train_df, "Train")
    print_stats(test_df, "Test")
    
    return train_df, test_df

def load_models(device):
    print("\nLoading translation models...")
    model_id_en = "Helsinki-NLP/opus-mt-id-en"
    model_id_id = "Helsinki-NLP/opus-mt-en-id"
    
    tokenizer_en = MarianTokenizer.from_pretrained(model_id_en)
    model_en = MarianMTModel.from_pretrained(model_id_en).to(device)
    tokenizer_id = MarianTokenizer.from_pretrained(model_id_id)
    model_id = MarianMTModel.from_pretrained(model_id_id).to(device)
    
    return tokenizer_en, model_en, tokenizer_id, model_id

def translate(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
    translated = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(decoded)
    return translated

def backtranslate_id_en_id(texts, tokenizer_en, model_en, tokenizer_id, model_id, device):
    en_texts = translate(texts, tokenizer_en, model_en, device)
    id_texts = translate(en_texts, tokenizer_id, model_id, device)
    return id_texts

def run_backtranslation(train_df, test_df):
    print("\n" + "=" * 60)
    print(" BACKTRANSLATION BALANCING")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    train_pos = train_df[train_df['label'] == 'POSITIVE']
    train_neg = train_df[train_df['label'] == 'NEGATIVE']
    
    print(f"Train POSITIVE: {len(train_pos):,}")
    print(f"Train NEGATIVE: {len(train_neg):,}")
    
    if len(train_pos) == len(train_neg):
        print("Already balanced!")
        return train_df, test_df
    
    minority_class = 'POSITIVE' if len(train_pos) < len(train_neg) else 'NEGATIVE'
    majority_class = 'NEGATIVE' if minority_class == 'POSITIVE' else 'POSITIVE'
    minority_df = train_pos if minority_class == 'POSITIVE' else train_neg
    majority_count = len(train_neg) if minority_class == 'POSITIVE' else len(train_pos)
    minority_count = len(minority_df)
    
    needed = majority_count - minority_count
    print(f"\nImbalance: {minority_class}={minority_count:,} vs {majority_class}={majority_count:,}")
    print(f"Need {needed:,} more {minority_class} samples")
    
    tokenizer_en, model_en, tokenizer_id, model_id = load_models(device)
    
    texts_to_translate = minority_df['text'].tolist()
    print(f"\nBacktranslating {len(texts_to_translate)} samples (2 rounds)...")
    
    print("Round 1:")
    start = time.time()
    backtranslated_1 = backtranslate_id_en_id(texts_to_translate, tokenizer_en, model_en, tokenizer_id, model_id, device)
    print(f"  Done in {time.time() - start:.1f}s")
    
    print("Round 2:")
    start = time.time()
    backtranslated_2 = backtranslate_id_en_id(texts_to_translate, tokenizer_en, model_en, tokenizer_id, model_id, device)
    print(f"  Done in {time.time() - start:.1f}s")
    
    all_backtranslated = backtranslated_1 + backtranslated_2
    selected = all_backtranslated[:needed]
    
    print(f"Generated {len(selected)} backtranslated samples")
    
    new_rows = pd.DataFrame({
        'text': selected,
        'label': [minority_class] * len(selected)
    })
    
    train_balanced = pd.concat([train_df, new_rows], ignore_index=True)
    train_balanced = train_balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"\nFinal train distribution:")
    print_stats(train_balanced, "Train")
    print_stats(test_df, "Test")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_balanced.to_csv(OUTPUT_DIR / "train_balanced.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_balanced.csv", index=False)
    
    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  train_balanced.csv ({len(train_balanced):,} rows)")
    print(f"  test_balanced.csv ({len(test_df):,} rows)")
    
    return train_balanced, test_df

def main():
    train_df, test_df = load_and_preprocess()
    run_backtranslation(train_df, test_df)

if __name__ == "__main__":
    main()