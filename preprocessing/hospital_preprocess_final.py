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

TRAIN_PATH = "../datasets/hospital_backtranslation/train_final.csv"
TEST_PATH = "../datasets/hospital_backtranslation/test.csv"

QUOTES_RE = re.compile(r'[\'"\u2018\u2019\u201a\u201b\u201c\u201d\u201e«»`]')

LEFTOVER_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF"
    "\uFE0F\u200D\u2049\u203C\u2139\u2194-\u21AA\u231A-\u23FA]"
)


def safe_replace_slang(text):
    return re.sub(
        SLANG_PATTERN,
        lambda mo: SLANG_DATA.get(mo.group(0).lower(), mo.group(0)),
        text,
    )


def strip_leftover_emojis(text):
    return LEFTOVER_EMOJI_RE.sub(" ", text)


def basic_clean(text):
    text = html.unescape(str(text))
    text = re.sub(r"[\r\n]+", " ", text)
    text = QUOTES_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df, title):
    print(f"\n[{title}]")
    print(f"  rows before: {len(df)}")

    df["text"] = df["text"].astype(str).str.replace('"', '', regex=False)
    df["text"] = df["text"].str.replace("'", '', regex=False)
    df["text"] = df["text"].apply(basic_clean)

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"  basic_clean: dropped {before - len(df)} rows")

    pipe = pipeline([safe_replace_slang, emoji_to_words, replace_word_elongation])
    df["text"] = df["text"].apply(pipe)
    df["text"] = df["text"].map(strip_leftover_emojis)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    before = len(df)
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"  post-pipeline: dropped {before - len(df)} rows")

    print(f"  rows after: {len(df)}")
    print(f"  label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = preprocess(train_df, "TRAIN")
    test_df = preprocess(test_df, "TEST")

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"\nSaved: {TRAIN_PATH} ({len(train_df)} rows)")
    print(f"Saved: {TEST_PATH} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
