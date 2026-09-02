# Indonesian Hospital Review — Train/Test Split + Back-Translation Augmentation

## Objective

Implement a reproducible data preprocessing pipeline for the Indonesian hospital review sentiment dataset.

The current dataset has approximately:

- Positive: 20,000
- Negative: 5,000
- Total: 25,000

The goal is to create:

```text
TRAIN
Positive: 16,000
Negative:  4,000
                 ↓
       Back-translate negative TRAIN only
                 ↓
Final TRAIN
Positive: 16,000
Negative:  8,000

TEST
Positive: 4,000
Negative: 1,000
```

Therefore:

- Original TRAIN ratio = 4:1
- Final TRAIN ratio = 2:1
- TEST remains completely untouched.

---

## Critical Rules

### 1. Split BEFORE augmentation

The original dataset must first be split into TRAIN and TEST.

Do NOT perform back translation before the split.

Correct:

```text
Original Dataset
       │
       ▼
Train/Test Split
       │
       ├── TRAIN ──► Back Translation ──► Final TRAIN
       │
       └── TEST ───────────────────────► TEST
```

Incorrect:

```text
Original Dataset
       │
       ▼
Back Translation
       │
       ▼
Train/Test Split
```

The second approach can cause data leakage because an augmented version of a review could end up in TEST.

---

## 2. Do NOT create a validation set

Only create:

```text
TRAIN
TEST
```

There should be no validation split in this preprocessing pipeline.

---

## 3. Use a stratified split

The split must preserve the original class distribution.

Target distribution:

```text
Original:
Positive = 20,000
Negative =  5,000

TRAIN:
Positive = 16,000
Negative =  4,000
Total    = 20,000

TEST:
Positive =  4,000
Negative =  1,000
Total    =  5,000
```

This is an 80/20 train/test split.

Use a fixed random seed, preferably:

```python
random_state=42
```

Use stratification based on the sentiment label.

---

# Back-Translation Augmentation

## 4. Augment ONLY the negative TRAIN samples

After the train/test split:

```text
TRAIN
Positive = 16,000
Negative =  4,000
```

Take all 4,000 negative training samples.

Generate exactly one back-translated version for each negative training sample.

Translation pipeline:

```text
Indonesian
    ↓
English
    ↓
Indonesian
```

Example:

```text
Original:
"Pelayanannya sangat buruk dan dokternya tidak ramah."

       ↓

Indonesian → English

"The service was very bad and the doctor was not friendly."

       ↓

English → Indonesian

"Pelayanannya sangat buruk dan dokternya tidak bersikap ramah."
```

The generated sample keeps the same label:

```text
label = negative
```

---

## 5. Expected final training distribution

Before augmentation:

```text
Positive = 16,000
Negative =  4,000
Total    = 20,000
```

Generated samples:

```text
Positive =     0
Negative =  4,000
Total    =  4,000
```

Final TRAIN:

```text
Positive = 16,000
Negative =  8,000
Total    = 24,000
```

Final ratio:

```text
Positive : Negative
16,000 : 8,000
2 : 1
```

Do NOT generate synthetic positive samples.

Do NOT remove positive samples.

Do NOT oversample the negative class beyond one generated sample per original negative training sample.

---

# TEST DATA

## 6. Never augment TEST

The TEST dataset must remain exactly as created by the original stratified split.

Expected:

```text
TEST
Positive = 4,000
Negative = 1,000
Total    = 5,000
```

No back translation.

No SMOTE.

No oversampling.

No undersampling.

No synthetic samples.

The test set must represent unseen, original data.

---

# Data Lineage

## 7. Preserve sample provenance

The pipeline should preserve the identity of every original sample.

Add/use the following columns where appropriate:

```text
id
text
label
is_augmented
original_id
```

For original samples:

```text
is_augmented = false
original_id = original sample ID
```

For back-translated samples:

```text
is_augmented = true
original_id = ID of the original negative review
```

Example:

```text
id        original_id    text                              label   is_augmented
1001      1001           pelayanan sangat buruk            0       false
1001_bt   1001           pelayanan sangat mengecewakan     0       true
```

This is important for reproducibility and later analysis.

---

# Data Leakage Prevention

## 8. Check for leakage

Before saving the final datasets, verify that:

1. No original review exists in both TRAIN and TEST.
2. No TEST review is used as input to back translation.
3. No augmented sample was generated from a TEST review.
4. Back-translated samples originate exclusively from TRAIN.
5. Exact duplicate text does not exist across TRAIN and TEST.

If an `id` column exists, use it for the primary leakage check.

Also check normalized text where appropriate.

The pipeline should fail loudly or report an error if leakage is detected.

---

# Back-Translation Implementation

## 9. Make the translation backend modular

Implement back translation as a separate function/module, for example:

```python
back_translate(text)
```

The rest of the preprocessing pipeline should not depend directly on a specific translation provider.

The implementation should allow the translation backend to be replaced later.

For example:

```text
preprocessing/
├── split.py
├── augmentation.py
└── pipeline.py
```

or an equivalent clean structure appropriate for the existing project.

Do NOT introduce unnecessary architectural changes if the project already has an established structure.

---

## 10. Translation failures

Translation can fail due to:

- API errors
- rate limits
- network errors
- invalid text
- timeout
- malformed response

Do NOT silently use the original text as the augmented text when translation fails.

Instead:

```text
Original sample
      ↓
Back translation
      ↓
FAIL
      ↓
Log failure
      ↓
Do not add augmented sample
```

Report:

```text
Total negative TRAIN samples: 4000
Successful augmentations: XXXX
Failed augmentations: XXXX
Success rate: XX.XX%
```

If the final number of augmented samples is less than 4,000, clearly report this instead of pretending the dataset is balanced.

---

# Recommended Output Files

Save the datasets separately:

```text
data/
├── train_original.csv
├── train_augmented.csv
├── train_final.csv
└── test.csv
```

### train_original.csv

Contains only the original TRAIN samples:

```text
Positive = 16,000
Negative =  4,000
Total    = 20,000
```

### train_augmented.csv

Contains only successfully generated back-translated negative samples:

```text
Positive = 0
Negative = up to 4,000
```

### train_final.csv

Contains:

```text
Original TRAIN
+
Back-translated negative TRAIN
```

Expected:

```text
Positive = 16,000
Negative =  8,000
Total    = 24,000
```

assuming all 4,000 translations succeed.

### test.csv

Contains only original test samples:

```text
Positive = 4,000
Negative = 1,000
Total    = 5,000
```

---

# Logging and Verification

## 11. Print class distributions

The script should print class distributions at every important stage.

Expected output:

```text
========================================
ORIGINAL DATASET
========================================
Positive: 20000
Negative:  5000
Total:    25000

========================================
TRAIN / TEST SPLIT
========================================
TRAIN
Positive: 16000
Negative:  4000
Total:    20000

TEST
Positive: 4000
Negative: 1000
Total:    5000

========================================
BACK TRANSLATION
========================================
Negative TRAIN samples: 4000
Successfully augmented: XXXX
Failed:                 XXXX

========================================
FINAL TRAIN
========================================
Positive: 16000
Negative: XXXX
Total:    XXXX

========================================
TEST
========================================
Positive: 4000
Negative: 1000
Total:    5000
```

If all augmentations succeed:

```text
========================================
FINAL TRAIN
========================================
Positive: 16000
Negative:  8000
Total:    24000
Ratio:       2:1
```

---

# Reproducibility

## 12. Fixed random seed

Use:

```python
SEED = 42
```

Set the relevant random seeds for the libraries used by the project.

The same input dataset and configuration should produce the same TRAIN/TEST split.

Note that external translation APIs/models may not always produce byte-for-byte identical translations. Therefore, cache the generated translations.

---

# Translation Cache

## 13. Cache translations

Do not repeatedly translate the same review every time the pipeline runs.

Use a cache such as:

```text
data/
└── back_translation_cache.json
```

or an appropriate database/CSV/JSON format.

The cache should map:

```text
original_id
    ↓
translated result
```

This allows the pipeline to resume after a failed API request and avoids unnecessary translation calls.

---

# Important Text Handling Rules

## 14. Preserve Indonesian review characteristics

This dataset contains user-generated Indonesian hospital reviews.

Do not aggressively normalize the text before back translation.

Avoid unnecessarily:

- removing stopwords
- stemming
- lemmatizing
- removing sentiment-bearing punctuation
- removing emojis
- replacing slang with formal Indonesian
- manually rewriting reviews

For Transformer-based sentiment classification, preserve as much meaningful linguistic information as possible.

Only apply preprocessing already required by the existing project.

---

# Implementation Requirements

## 15. Before coding

First inspect the existing project and determine:

- Dataset file location
- Text column name
- Sentiment/label column name
- Label encoding
- Existing preprocessing code
- Existing train/test split code
- Existing dependencies
- Existing Python environment
- Whether a translation library/API is already used

Do not assume column names such as `text` or `label` if the project uses different names.

Reuse existing project conventions where possible.

---

## 16. Do not change the model training code unnecessarily

This task is primarily a **data preprocessing and augmentation pipeline**.

Do not modify:

- IndoBERT architecture
- tokenizer configuration
- training hyperparameters
- optimizer
- loss function
- evaluation code

unless required to integrate the generated `train_final.csv`.

The goal is to create clean datasets first.

---

# Final Acceptance Criteria

The implementation is complete only if all of the following are true:

- [ ] Original dataset is loaded correctly.
- [ ] Data is split into TRAIN and TEST before augmentation.
- [ ] Split is stratified by sentiment label.
- [ ] Random seed is fixed.
- [ ] TRAIN is approximately/exactly 80% of the original data.
- [ ] TEST is approximately/exactly 20% of the original data.
- [ ] Target TRAIN distribution is 16,000 positive / 4,000 negative for the stated 25K dataset.
- [ ] Only negative TRAIN samples are back-translated.
- [ ] Each negative TRAIN sample receives at most one augmented version.
- [ ] No TEST sample is back-translated.
- [ ] No validation dataset is created.
- [ ] Augmented samples retain the negative label.
- [ ] Original IDs/provenance are preserved.
- [ ] Translation failures are logged.
- [ ] Failed translations are not silently duplicated as originals.
- [ ] Back translations are cached.
- [ ] TRAIN and TEST leakage checks are implemented.
- [ ] `train_original.csv` is saved.
- [ ] `train_augmented.csv` is saved.
- [ ] `train_final.csv` is saved.
- [ ] `test.csv` is saved.
- [ ] Class distributions are printed.
- [ ] If all translations succeed, final TRAIN is exactly:
  - Positive = 16,000
  - Negative = 8,000
  - Total = 24,000
- [ ] TEST remains exactly:
  - Positive = 4,000
  - Negative = 1,000
  - Total = 5,000

## Expected Final Pipeline

```text
                 ORIGINAL DATASET
                 20K Positive
                  5K Negative
                       │
                       │
                Stratified 80/20
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
           TRAIN                TEST
          16K Pos              4K Pos
           4K Neg              1K Neg
             │                   │
             │                   │
             ▼                   │
      Back Translate              │
      NEGATIVE ONLY               │
             │                   │
             ▼                   │
       4K BT Negative             │
             │                   │
             ▼                   │
        ┌────────────┐            │
        │ Final Train│            │
        │ 16K Pos    │            │
        │  8K Neg    │            │
        │ 24K Total  │            │
        └────────────┘            │
                                  │
                                  ▼
                           UNTOUCHED TEST
                            4K Pos / 1K Neg
```

## Important

Do not implement SMOTE in this task.

This pipeline is specifically for evaluating **back-translation augmentation of the minority class**.

SMOTE can be implemented later as a separate experiment so that the effect of back translation can be compared fairly against other imbalance-handling techniques.
