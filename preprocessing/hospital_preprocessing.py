import html
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

INPUT_PATH = "../datasets/hospital_dataset.xlsx"
TRAIN_OUTPUT_PATH = "../datasets/hospital_train.csv"
# VAL_OUTPUT_PATH = "../datasets/hospital_val.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

POSITIVE_WORDS = [
    "bagus", "ramah", "baik", "cepat", "memuaskan", "mantap", "recommended",
    "puas", "bersih", "nyaman", "terima kasih", "terimakasih", "best",
    "luar biasa", "sigap",
]
NEGATIVE_WORDS = [
    "kecewa", "buruk", "jelek", "payah", "jutek", "lelet", "mahal", "kotor",
    "parah", "mengecewakan", "bad service", "tidak profesional",
]


def safe_replace_slang(text):
    return re.sub(
        SLANG_PATTERN,
        lambda mo: SLANG_DATA.get(mo.group(0).lower(), mo.group(0)),
        text,
    )


pipe = pipeline([safe_replace_slang, emoji_to_words, replace_word_elongation])

LEFTOVER_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF"
    "\uFE0F\u200D\u2049\u203C\u2139\u2194-\u21AA\u231A-\u23FA]"
)


def strip_leftover_emojis(text):
    return LEFTOVER_EMOJI_RE.sub(" ", text)


QUOTES_RE = re.compile(r'[\'"\u2018\u2019\u201a\u201b\u201c\u201d\u201e«»`]')


def basic_clean(text):
    text = html.unescape(str(text))
    text = re.sub(r"[\r\n]+", " ", text)
    text = QUOTES_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def report_label_conflicts(df):
    t = df["text"].str.lower()
    has_pos = t.apply(lambda x: any(w in x for w in POSITIVE_WORDS))
    has_neg = t.apply(lambda x: any(w in x for w in NEGATIVE_WORDS))
    pos_text_neg_label = ((has_pos & ~has_neg) & (df["label"] == "negative")).sum()
    neg_text_pos_label = ((has_neg & ~has_pos) & (df["label"] == "positive")).sum()
    total = len(df)
    print("\n[Label conflict report - review recommended, nothing removed]")
    print(f"  positive-text labeled negative: {pos_text_neg_label} "
          f"({pos_text_neg_label / total:.1%})")
    print(f"  negative-text labeled positive: {neg_text_pos_label} "
          f"({neg_text_pos_label / total:.1%})")


def print_stats(df, title):
    print(f"\n[{title}]")
    print(f"  rows: {len(df)}")
    print(f"  label distribution:\n{df['label'].value_counts().to_string()}")


def main():
    df = pd.read_excel(INPUT_PATH)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    print_stats(df, "Raw data")

    df["text"] = df["text"].apply(basic_clean)

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"\n[Cleaning] dropped {before - len(df)} rows "
          f"(empty text/label or duplicate text)")

    print("\n[Text normalization] applying indoNLP pipeline "
          "(safe_replace_slang, emoji_to_words, replace_word_elongation)...")
    df["text"] = df["text"].apply(pipe)
    df["text"] = df["text"].map(strip_leftover_emojis)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    before = len(df)
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"[Post-pipeline cleanup] dropped {before - len(df)} additional rows")

    print_stats(df, "Preprocessed data")
    report_label_conflicts(df)

    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    # val_df.to_csv(VAL_OUTPUT_PATH, index=False)

    print_stats(train_df, f"Saved {TRAIN_OUTPUT_PATH}")
    # print_stats(val_df, f"Saved {VAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
