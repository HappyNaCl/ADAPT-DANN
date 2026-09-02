import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load dataset ---------------------------------------------------
    print_section("ORIGINAL DATASET")
    df = pd.read_excel(DATASET_PATH)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"  Removed {before - len(df)} duplicate texts")
    print_distribution(df)

    # --- Assign IDs -----------------------------------------------------
    df["id"] = range(1, len(df) + 1)

    # --- Stratified train/test split ------------------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df["id"] = range(1, len(train_df) + 1)
    test_df["id"] = range(1, len(test_df) + 1)

    print_section("TRAIN / TEST SPLIT")
    print("\n  TRAIN:")
    print_distribution(train_df)
    print("\n  TEST:")
    print_distribution(test_df)

    # --- Load cache and build augmented rows ----------------------------
    print_section("BUILDING FROM CACHE")
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # Build text->translated_text lookup from old cache
    # Old cache maps old_id -> translated_text
    # We need to map new train negative texts to their translations
    # Load original data again to get old text->id mapping
    orig_df = pd.read_excel(DATASET_PATH)
    orig_df = orig_df.rename(columns={"review": "text", "sentiment": "label"})
    orig_df = orig_df.dropna(subset=["text", "label"]).reset_index(drop=True)
    orig_df = orig_df[orig_df["text"].str.strip() != ""].reset_index(drop=True)
    orig_df = orig_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    orig_df["id"] = range(1, len(orig_df) + 1)

    # Create text->translation mapping from cache using original data
    text_to_bt = {}
    for _, row in orig_df.iterrows():
        cache_key = str(row["id"])
        if cache_key in cache:
            text_to_bt[row["text"]] = cache[cache_key]

    neg_train = train_df[train_df["label"] == "negative"]
    total_neg = len(neg_train)
    print(f"  Negative TRAIN samples: {total_neg}")
    print(f"  Cache entries:          {len(cache)}")
    print(f"  Text->translation map:  {len(text_to_bt)}")

    augmented_rows = []
    success = 0
    failed = 0

    for _, row in neg_train.iterrows():
        orig_text = row["text"]
        if orig_text in text_to_bt:
            bt_text = text_to_bt[orig_text]
            augmented_rows.append({
                "id": f"{row['id']}_bt",
                "original_id": row["id"],
                "text": bt_text,
                "label": row["label"],
                "is_augmented": True,
            })
            success += 1
        else:
            failed += 1

    print(f"  Matched from cache:     {success}")
    print(f"  Missing from cache:     {failed}")
    if total_neg > 0:
        print(f"  Success rate:           {success / total_neg:.2%}")

    # --- Build augmented dataframe --------------------------------------
    aug_df = pd.DataFrame(augmented_rows)

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
    print("\n[Leakage check] verifying no overlap between TRAIN and TEST ...")
    train_texts = set(train_final["text"].str.lower())
    test_texts = set(test_df["text"].str.lower())
    overlap = train_texts & test_texts
    if overlap:
        print(f"  WARNING: {len(overlap)} duplicate texts found!")
    else:
        print("  OK: no text overlap detected.")

    # --- Save output files -----------------------------------------------
    cols = ["id", "original_id", "text", "label", "is_augmented"]
    train_df[cols].to_csv(TRAIN_ORIGINAL_PATH, index=False)
    aug_df[cols].to_csv(TRAIN_AUGMENTED_PATH, index=False)
    train_final.to_csv(TRAIN_FINAL_PATH, index=False)
    test_df[cols].to_csv(TEST_PATH, index=False)

    # --- Final reports --------------------------------------------------
    print_section("FINAL TRAIN")
    print_distribution(train_final)
    pos = len(train_final[train_final["label"] == "positive"])
    neg = len(train_final[train_final["label"] == "negative"])
    if neg > 0 and pos % neg == 0:
        print(f"  Ratio:      {pos // neg}:1")

    print_section("TEST (untouched)")
    print_distribution(test_df)

    print_section("SAVED FILES")
    for name, path in [
        ("train_original", TRAIN_ORIGINAL_PATH),
        ("train_augmented", TRAIN_AUGMENTED_PATH),
        ("train_final", TRAIN_FINAL_PATH),
        ("test", TEST_PATH),
    ]:
        n = pd.read_csv(path).shape[0]
        print(f"  {name:<20}: {path}  ({n} rows)")


if __name__ == "__main__":
    main()
