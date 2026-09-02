import json
import os
import random
import re

import pandas as pd
import argostranslate.translate
from sklearn.model_selection import train_test_split

DATASET_PATH = "../datasets/hospital_dataset.xlsx"
OUTPUT_DIR = "../datasets/hospital_backtranslation"
CACHE_PATH = os.path.join(OUTPUT_DIR, "back_translation_cache.json")
TRAIN_FINAL_PATH = os.path.join(OUTPUT_DIR, "train_final.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")

SEED = 42
TEST_SIZE = 0.2
EMOJI_TOKEN_RE = re.compile(r"!\w+(?:_\w+)*!")


def remove_emoji_tokens(text):
    return EMOJI_TOKEN_RE.sub("", text)


def back_translate(text):
    en = argostranslate.translate.translate(text, "id", "en")
    return argostranslate.translate.translate(en, "en", "id")


def vary_text(text):
    """Create a slight variation of the text for diverse back-translation."""
    variations = [
        text.lower(),
        text.upper(),
        re.sub(r"[.,!?;:]+", " ", text).strip(),
        re.sub(r"\s+", " ", text).strip(),
    ]
    return random.choice(variations)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)

    # --- Load and split ---
    df = pd.read_excel(DATASET_PATH)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df["id"] = range(1, len(train_df) + 1)
    test_df["id"] = range(1, len(test_df) + 1)

    # --- Clean ---
    train_df = train_df[train_df["text"].str.len() >= 5].reset_index(drop=True)
    test_df = test_df[test_df["text"].str.len() >= 5].reset_index(drop=True)
    train_df["id"] = range(1, len(train_df) + 1)
    train_df["text"] = train_df["text"].str.replace("\u200b", "", regex=False)
    test_df["text"] = test_df["text"].str.replace("\u200b", "", regex=False)
    train_df["text"] = train_df["text"].apply(remove_emoji_tokens)
    test_df["text"] = test_df["text"].apply(remove_emoji_tokens)
    train_df["text"] = train_df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    test_df["text"] = test_df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    train_df = train_df[train_df["text"].str.len() >= 5].reset_index(drop=True)
    test_df = test_df[test_df["text"].str.len() >= 5].reset_index(drop=True)
    train_df["id"] = range(1, len(train_df) + 1)

    pos_count = len(train_df[train_df["label"] == "positive"])
    neg_count = len(train_df[train_df["label"] == "negative"])
    target_neg = pos_count  # 50/50 balance
    need_more = target_neg - neg_count

    print(f"Positive: {pos_count}")
    print(f"Negative (original): {neg_count}")
    print(f"Target negative: {target_neg}")
    print(f"Need to generate: {need_more}")

    # --- Load existing cache ---
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    # Build text->translation map
    text_to_bt = {}
    for _, row in df.iterrows():
        cache_key = str(row["id"])
        if cache_key in cache:
            text_to_bt[row["text"]] = cache[cache_key]

    # --- Generate additional back-translations ---
    neg_train = train_df[train_df["label"] == "negative"]

    # Count existing augmentations per original text
    existing_aug = {}
    for _, row in neg_train.iterrows():
        if row["text"] in text_to_bt:
            existing_aug[row["text"]] = existing_aug.get(row["text"], 0) + 1

    augmented_rows = []
    generated = 0
    skipped_existing = 0

    for _, row in neg_train.iterrows():
        if generated >= need_more:
            break

        orig_text = row["text"]
        orig_id = row["id"]

        # Generate first augmentation (standard back-translation)
        if orig_text in text_to_bt:
            bt1 = text_to_bt[orig_text]
            if not (bt1.endswith("..") or bt1.endswith("...")):
                bt1_clean = remove_emoji_tokens(bt1)
                bt1_clean = re.sub(r"\s+", " ", bt1_clean).strip()
                if len(bt1_clean) >= 5:
                    augmented_rows.append({
                        "id": f"{orig_id}_bt1",
                        "original_id": orig_id,
                        "text": bt1_clean,
                        "label": "negative",
                        "is_augmented": True,
                    })
                    generated += 1
                    if generated >= need_more:
                        break

        # Generate second augmentation (varied input)
        varied = vary_text(orig_text)
        cache_key_v = f"{orig_id}_v"
        if cache_key_v in cache:
            bt2 = cache[cache_key_v]
        else:
            try:
                bt2 = back_translate(varied)
                cache[cache_key_v] = bt2
            except Exception:
                continue

        if not (bt2.endswith("..") or bt2.endswith("...")):
            bt2_clean = remove_emoji_tokens(bt2)
            bt2_clean = re.sub(r"\s+", " ", bt2_clean).strip()
            if len(bt2_clean) >= 5:
                augmented_rows.append({
                    "id": f"{orig_id}_bt2",
                    "original_id": orig_id,
                    "text": bt2_clean,
                    "label": "negative",
                    "is_augmented": True,
                })
                generated += 1

        # Generate third augmentation (another variation)
        if generated < need_more:
            varied2 = re.sub(r"[^\w\s]", "", orig_text).strip()
            cache_key_v2 = f"{orig_id}_v2"
            if cache_key_v2 in cache:
                bt3 = cache[cache_key_v2]
            else:
                try:
                    bt3 = back_translate(varied2)
                    cache[cache_key_v2] = bt3
                except Exception:
                    bt3 = None

            if bt3 and not (bt3.endswith("..") or bt3.endswith("...")):
                bt3_clean = remove_emoji_tokens(bt3)
                bt3_clean = re.sub(r"\s+", " ", bt3_clean).strip()
                if len(bt3_clean) >= 5:
                    augmented_rows.append({
                        "id": f"{orig_id}_bt3",
                        "original_id": orig_id,
                        "text": bt3_clean,
                        "label": "negative",
                        "is_augmented": True,
                    })
                    generated += 1

        # Save cache periodically
        if generated % 50 == 0 and generated > 0:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            print(f"  ... {generated}/{need_more} generated")

    # Final cache save
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated: {generated} new augmented samples")

    # --- Build final datasets ---
    aug_df = pd.DataFrame(augmented_rows)

    train_df["is_augmented"] = False
    train_df["original_id"] = train_df["id"]
    test_df["is_augmented"] = False
    test_df["original_id"] = test_df["id"]

    cols = ["id", "original_id", "text", "label", "is_augmented"]
    train_final = pd.concat([train_df[cols], aug_df[cols]], ignore_index=True)

    # --- Leakage check ---
    overlap = set(train_final["text"].str.lower()) & set(test_df["text"].str.lower())
    print(f"Leakage: {len(overlap)} overlaps")

    # --- Save ---
    train_final.to_csv(TRAIN_FINAL_PATH, index=False)
    test_df[cols].to_csv(TEST_PATH, index=False)

    # --- Report ---
    pos = len(train_final[train_final["label"] == "positive"])
    neg = len(train_final[train_final["label"] == "negative"])
    print(f"\nTRAIN: {pos} pos / {neg} neg = {pos+neg} total (ratio {pos/neg:.2f}:1)")
    print(f"TEST:  {len(test_df[test_df['label']=='positive'])} pos / {len(test_df[test_df['label']=='negative'])} neg")


if __name__ == "__main__":
    main()
