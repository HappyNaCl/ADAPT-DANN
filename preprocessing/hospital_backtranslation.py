import json
import os
import random
import time

import pandas as pd
import argostranslate.translate
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = "../datasets/hospital_dataset.xlsx"
DATA_DIR = "../datasets"
OUTPUT_DIR = os.path.join(DATA_DIR, "hospital_backtranslation")

TRAIN_ORIGINAL_PATH = os.path.join(OUTPUT_DIR, "train_original.csv")
TRAIN_AUGMENTED_PATH = os.path.join(OUTPUT_DIR, "train_augmented.csv")
TRAIN_FINAL_PATH = os.path.join(OUTPUT_DIR, "train_final.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")
CACHE_PATH = os.path.join(OUTPUT_DIR, "back_translation_cache.json")

SEED = 42
TEST_SIZE = 0.2
MAX_TRANSLATE_RETRIES = 3
RETRY_DELAY = 2  # seconds
REQUEST_DELAY = 0.1  # seconds between requests

# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _translate_with_retry(text, src, tgt, retries=MAX_TRANSLATE_RETRIES):
    for attempt in range(retries):
        try:
            result = argostranslate.translate.translate(text, src, tgt)
            return result
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise e


def back_translate(text):
    """Indonesian -> English -> Indonesian"""
    en = _translate_with_retry(text, "id", "en")
    id_back = _translate_with_retry(en, "en", "id")
    return id_back


# ---------------------------------------------------------------------------
# Data lineage
# ---------------------------------------------------------------------------

def assign_ids(df):
    df = df.copy()
    df["id"] = range(1, len(df) + 1)
    return df


def build_augmented_row(original_id, original_text, new_text, label):
    return {
        "id": f"{original_id}_bt",
        "original_id": original_id,
        "text": new_text,
        "label": label,
        "is_augmented": True,
    }


# ---------------------------------------------------------------------------
# Data leakage checks
# ---------------------------------------------------------------------------

def check_leakage(train_df, test_df):
    print("\n[Leakage check] verifying no overlap between TRAIN and TEST ...")
    train_texts = set(train_df["text"].str.lower())
    test_texts = set(test_df["text"].str.lower())

    overlap = train_texts & test_texts
    if overlap:
        print(f"  WARNING: {len(overlap)} duplicate texts found across TRAIN and TEST!")
        return False

    print("  OK: no text overlap detected.")
    return True


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def print_section(title):
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")


def print_distribution(df, label_col="label"):
    counts = df[label_col].value_counts()
    total = len(df)
    for lbl, cnt in counts.items():
        print(f"  {lbl:<10}: {cnt:>6}")
    print(f"  {'Total':<10}: {total:>6}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load dataset ---------------------------------------------------
    print_section("ORIGINAL DATASET")
    df = pd.read_excel(DATASET_PATH)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    print_distribution(df)

    # --- Assign IDs -----------------------------------------------------
    df = assign_ids(df)

    # --- Stratified train/test split ------------------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Re-assign IDs after split so they are contiguous within each set
    train_df = assign_ids(train_df)
    test_df = assign_ids(test_df)

    print_section("TRAIN / TEST SPLIT")
    print("\n  TRAIN:")
    print_distribution(train_df)
    print("\n  TEST:")
    print_distribution(test_df)

    # --- Back-translate negative TRAIN samples only ----------------------
    print_section("BACK TRANSLATION")
    neg_train = train_df[train_df["label"] == "negative"].copy()
    neg_train["text"] = neg_train["text"].astype(str)
    total_neg = len(neg_train)
    print(f"  Negative TRAIN samples: {total_neg}")

    cache = load_cache(CACHE_PATH)
    augmented_rows = []
    success = 0
    failed = 0

    for idx, row in neg_train.iterrows():
        orig_id = row["id"]
        orig_text = row["text"]

        # Check cache first
        cache_key = str(orig_id)
        if cache_key in cache:
            bt_text = cache[cache_key]
            augmented_rows.append(
                build_augmented_row(orig_id, orig_text, bt_text, row["label"])
            )
            success += 1
            continue

        try:
            bt_text = back_translate(orig_text)
            cache[cache_key] = bt_text
            augmented_rows.append(
                build_augmented_row(orig_id, orig_text, bt_text, row["label"])
            )
            success += 1

            # Save cache every 50 translations
            if success % 50 == 0:
                save_cache(cache, CACHE_PATH)
                print(f"  ... {success}/{total_neg} translated (cache saved)")

        except Exception as e:
            failed += 1
            print(f"  FAILED id={orig_id}: {e}")

        # Delay between requests to avoid rate limits
        time.sleep(REQUEST_DELAY)

    # Final cache save
    save_cache(cache, CACHE_PATH)

    print(f"\n  Successfully augmented: {success}")
    print(f"  Failed:                 {failed}")
    if total_neg > 0:
        print(f"  Success rate:           {success / total_neg:.2%}")

    # --- Build augmented dataframe --------------------------------------
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
    else:
        aug_df = pd.DataFrame(columns=["id", "original_id", "text", "label", "is_augmented"])

    # --- Add provenance columns to original train/test -------------------
    train_df["is_augmented"] = False
    train_df["original_id"] = train_df["id"]
    test_df["is_augmented"] = False
    test_df["original_id"] = test_df["id"]

    # --- Final train = original train + augmented -----------------------
    train_final = pd.concat(
        [train_df[["id", "original_id", "text", "label", "is_augmented"]],
         aug_df[["id", "original_id", "text", "label", "is_augmented"]]],
        ignore_index=True,
    )

    # --- Leakage check --------------------------------------------------
    leakage_ok = check_leakage(train_final, test_df)
    if not leakage_ok:
        print("\n  ABORT: data leakage detected. Check the data.")
        return

    # --- Save output files -----------------------------------------------
    train_df[["id", "original_id", "text", "label", "is_augmented"]].to_csv(
        TRAIN_ORIGINAL_PATH, index=False
    )
    aug_df[["id", "original_id", "text", "label", "is_augmented"]].to_csv(
        TRAIN_AUGMENTED_PATH, index=False
    )
    train_final.to_csv(TRAIN_FINAL_PATH, index=False)
    test_df[["id", "original_id", "text", "label", "is_augmented"]].to_csv(
        TEST_PATH, index=False
    )

    # --- Final reports --------------------------------------------------
    print_section("FINAL TRAIN")
    print_distribution(train_final)
    pos = len(train_final[train_final["label"] == "positive"])
    neg = len(train_final[train_final["label"] == "negative"])
    if neg > 0 and pos % neg == 0:
        ratio = pos // neg
        print(f"  Ratio:      {ratio}:1")

    print_section("TEST (untouched)")
    print_distribution(test_df)

    print_section("SAVED FILES")
    for name, path in [
        ("train_original", TRAIN_ORIGINAL_PATH),
        ("train_augmented", TRAIN_AUGMENTED_PATH),
        ("train_final", TRAIN_FINAL_PATH),
        ("test", TEST_PATH),
        ("cache", CACHE_PATH),
    ]:
        n = "rows"
        if name != "cache":
            n = pd.read_csv(path).shape[0]
        print(f"  {name:<20}: {path}  ({n})")

    print(f"\n  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
