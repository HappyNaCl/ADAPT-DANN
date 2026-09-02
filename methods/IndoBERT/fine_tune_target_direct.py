"""
IndoBERT Baseline: Source Train -> Target Evaluation

Fine-tunes base IndoBERT on source domain (labeled) data, then evaluates
on both source and target test sets. This provides a fair baseline comparison
against domain adaptation methods (DANN, DANN+Adapters) which also train
on source labeled data.

Setup matches DANN v2:
  - Source (labeled): telemedicine
  - Target (unlabeled during DANN, but used for eval here): Lazada
"""

import os
import pandas as pd
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================================
# Config
# ============================================================================
BASE_MODEL       = "indolem/indobert-base-uncased"
SOURCE_TRAIN_CSV = "../../datasets/telemedicine_preprocessed/train.csv"
SOURCE_TEST_CSV  = "../../datasets/telemedicine_preprocessed/test.csv"
TARGET_TEST_CSV  = "../../datasets/lazada_test_preprocessed.csv"
OUTPUT_DIR       = "./indobert_baseline_source"
TEXT_COL         = "text"
LABEL_COL        = "label"

LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "negative", 1: "positive"}

def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize labels to lowercase to handle mixed case formats (POSITIVE/positive)."""
    return series.astype(str).str.strip().str.lower()

MAX_LENGTH      = 128
NUM_EPOCHS      = 3
TRAIN_BATCH     = 32
EVAL_BATCH      = 32
LEARNING_RATE   = 1e-5
WEIGHT_DECAY    = 0.02
WARMUP_RATIO    = 0.1
VAL_SIZE        = 0.15
SEED            = 42


# ============================================================================
# Metrics
# ============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ---- Load Data ----
    print("\n" + "=" * 60)
    print("Loading source train and test datasets ...")
    print("=" * 60)

    # --- Source train set ---
    print(f"\nSource train file: {SOURCE_TRAIN_CSV}")
    train_df = pd.read_csv(SOURCE_TRAIN_CSV)
    print(f"  Shape  : {train_df.shape}")
    print(f"  Columns: {train_df.columns.tolist()}")
    print(f"  Label distribution:\n{train_df[LABEL_COL].value_counts().to_string()}")

    before = len(train_df)
    train_df = train_df.dropna(subset=[TEXT_COL, LABEL_COL])
    train_df[TEXT_COL] = train_df[TEXT_COL].astype(str).str.strip()
    train_df = train_df[train_df[TEXT_COL] != ""]
    print(f"  Rows after cleaning: {len(train_df)}  (dropped {before - len(train_df)})")

    train_df[LABEL_COL] = normalize_labels(train_df[LABEL_COL])
    train_df["label_id"] = train_df[LABEL_COL].map(LABEL2ID)

    # --- Source test set ---
    print(f"\nSource test file: {SOURCE_TEST_CSV}")
    source_test_df = pd.read_csv(SOURCE_TEST_CSV)
    print(f"  Shape  : {source_test_df.shape}")
    print(f"  Columns: {source_test_df.columns.tolist()}")
    print(f"  Label distribution:\n{source_test_df[LABEL_COL].value_counts().to_string()}")

    before = len(source_test_df)
    source_test_df = source_test_df.dropna(subset=[TEXT_COL, LABEL_COL])
    source_test_df[TEXT_COL] = source_test_df[TEXT_COL].astype(str).str.strip()
    source_test_df = source_test_df[source_test_df[TEXT_COL] != ""]
    print(f"  Rows after cleaning: {len(source_test_df)}  (dropped {before - len(source_test_df)})")

    source_test_df[LABEL_COL] = normalize_labels(source_test_df[LABEL_COL])
    source_test_df["label_id"] = source_test_df[LABEL_COL].map(LABEL2ID)

    # --- Target test set ---
    print(f"\nTarget test file: {TARGET_TEST_CSV}")
    target_test_df = pd.read_csv(TARGET_TEST_CSV)
    print(f"  Shape  : {target_test_df.shape}")
    print(f"  Columns: {target_test_df.columns.tolist()}")
    print(f"  Label distribution:\n{target_test_df[LABEL_COL].value_counts().to_string()}")

    before = len(target_test_df)
    target_test_df = target_test_df.dropna(subset=[TEXT_COL, LABEL_COL])
    target_test_df[TEXT_COL] = target_test_df[TEXT_COL].astype(str).str.strip()
    target_test_df = target_test_df[target_test_df[TEXT_COL] != ""]
    print(f"  Rows after cleaning: {len(target_test_df)}  (dropped {before - len(target_test_df)})")

    target_test_df[LABEL_COL] = normalize_labels(target_test_df[LABEL_COL])
    target_test_df["label_id"] = target_test_df[LABEL_COL].map(LABEL2ID)

    # --- Split source train into train / val ---
    train_df, val_df = train_test_split(
        train_df, test_size=VAL_SIZE, stratify=train_df["label_id"], random_state=SEED
    )

    print(f"\nSplit  : train={len(train_df)}  val={len(val_df)}  source_test={len(source_test_df)}  target_test={len(target_test_df)}")

    # ---- Tokenizer & Model ----
    print("\n" + "=" * 60)
    print(f"Loading model: {BASE_MODEL}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ---- Tokenize ----
    def make_hf_dataset(dataframe):
        ds = Dataset.from_dict({
            "text":  dataframe[TEXT_COL].tolist(),
            "label": dataframe["label_id"].tolist(),
        })
        return ds.map(
            lambda ex: tokenizer(
                ex["text"],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
            ),
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
        )

    print("\nTokenizing splits ...")
    train_dataset        = make_hf_dataset(train_df)
    val_dataset          = make_hf_dataset(val_df)
    source_test_dataset  = make_hf_dataset(source_test_df)
    target_test_dataset  = make_hf_dataset(target_test_df)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ---- Training Arguments ----
    print("\n" + "=" * 60)
    print("Training configuration")
    print("=" * 60)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none",
        seed=SEED,
    )

    for key in ["num_train_epochs", "per_device_train_batch_size",
                "learning_rate", "weight_decay", "warmup_ratio", "fp16"]:
        print(f"  {key}: {getattr(training_args, key)}")

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ---- Train ----
    print("\n" + "=" * 60)
    print("Training started ...")
    print("=" * 60)

    train_result = trainer.train()

    print(f"\nTraining loss : {train_result.training_loss:.4f}")
    print(f"Training time : {train_result.metrics['train_runtime']:.1f} s")

    # ---- Validation Evaluation (per epoch already printed above) ----
    print("\n" + "=" * 60)
    print("Validation set evaluation (best checkpoint)")
    print("=" * 60)

    val_results = trainer.evaluate(val_dataset)
    print(f"  Loss      : {val_results['eval_loss']:.4f}")
    print(f"  Accuracy  : {val_results['eval_accuracy']:.4f}")
    print(f"  Precision : {val_results['eval_precision']:.4f}")
    print(f"  Recall    : {val_results['eval_recall']:.4f}")
    print(f"  F1        : {val_results['eval_f1']:.4f}")

    # ---- Target Test Evaluation ----
    print("\n" + "=" * 60)
    print("Target test set evaluation (primary metric)")
    print("=" * 60)

    target_output  = trainer.predict(target_test_dataset)
    target_preds   = np.argmax(target_output.predictions, axis=1)
    target_labels  = target_output.label_ids

    target_accuracy  = accuracy_score(target_labels, target_preds)
    target_precision, target_recall, target_f1, _ = precision_recall_fscore_support(
        target_labels, target_preds, average="weighted"
    )
    target_loss = target_output.metrics["test_loss"]

    print(f"  Loss      : {target_loss:.4f}")
    print(f"  Accuracy  : {target_accuracy:.4f}  ({target_accuracy * 100:.2f}%)")
    print(f"  Precision : {target_precision:.4f}")
    print(f"  Recall    : {target_recall:.4f}")
    print(f"  F1        : {target_f1:.4f}")

    # Per-class classification report
    print("\nClassification Report (target):")
    print("-" * 60)
    print(classification_report(
        target_labels,
        target_preds,
        target_names=["negative", "positive"],
        digits=4,
    ))

    # Confusion matrix
    cm = confusion_matrix(target_labels, target_preds)
    print("Confusion Matrix (target)  (rows=true, cols=predicted):")
    print(f"               negative  positive")
    print(f"  negative      {cm[0][0]:6d}    {cm[0][1]:6d}")
    print(f"  positive      {cm[1][0]:6d}    {cm[1][1]:6d}")

    # Per-label accuracy breakdown
    print("\nPer-class accuracy (target):")
    for idx, name in ID2LABEL.items():
        mask = target_labels == idx
        if mask.sum() > 0:
            cls_acc = accuracy_score(target_labels[mask], target_preds[mask])
            print(f"  {name:10s}: {cls_acc:.4f}  ({mask.sum()} samples)")

    # ---- Source Test Evaluation ----
    print("\n" + "=" * 60)
    print("Source test set evaluation")
    print("=" * 60)

    source_output   = trainer.predict(source_test_dataset)
    source_preds    = np.argmax(source_output.predictions, axis=1)
    source_labels   = source_output.label_ids

    source_accuracy  = accuracy_score(source_labels, source_preds)
    source_precision, source_recall, source_f1, _ = precision_recall_fscore_support(
        source_labels, source_preds, average="weighted"
    )
    source_loss = source_output.metrics["test_loss"]

    print(f"  Loss      : {source_loss:.4f}")
    print(f"  Accuracy  : {source_accuracy:.4f}  ({source_accuracy * 100:.2f}%)")
    print(f"  Precision : {source_precision:.4f}")
    print(f"  Recall    : {source_recall:.4f}")
    print(f"  F1        : {source_f1:.4f}")

    print("\nClassification Report (source):")
    print("-" * 60)
    print(classification_report(
        source_labels,
        source_preds,
        target_names=["negative", "positive"],
        digits=4,
    ))

    cm_src = confusion_matrix(source_labels, source_preds)
    print("Confusion Matrix (source)  (rows=true, cols=predicted):")
    print(f"               negative  positive")
    print(f"  negative      {cm_src[0][0]:6d}    {cm_src[0][1]:6d}")
    print(f"  positive      {cm_src[1][0]:6d}    {cm_src[1][1]:6d}")

    # ---- Save ----
    print("\n" + "=" * 60)
    print(f"Saving model to {OUTPUT_DIR}_final ...")
    print("=" * 60)

    final_dir = OUTPUT_DIR + "_final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    if os.path.exists(final_dir):
        files = os.listdir(final_dir)
        for f in sorted(files):
            fpath = os.path.join(final_dir, f)
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"  {f:40s} {size_mb:.2f} MB")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model        : {BASE_MODEL}")
    print(f"  Source train : {SOURCE_TRAIN_CSV}  ({len(train_df)} train / {len(val_df)} val)")
    print(f"  Source test  : {SOURCE_TEST_CSV}  ({len(source_test_df)} samples)")
    print(f"  Target test  : {TARGET_TEST_CSV}  ({len(target_test_df)} samples)")
    print(f"  Epochs       : {NUM_EPOCHS}")
    print(f"  Saved to     : {final_dir}")

    # -- Source vs Target Comparison Table --
    print(f"\n{'Metric':<12} {'Source':<15} {'Target':<15} {'Gap':<10}")
    print("-" * 52)
    for metric_name, src_val, tgt_val in [
        ("Loss",      source_loss,      target_loss),
        ("Accuracy",  source_accuracy,  target_accuracy),
        ("Precision", source_precision, target_precision),
        ("Recall",    source_recall,    target_recall),
        ("F1",        source_f1,        target_f1),
    ]:
        gap = src_val - tgt_val
        print(f"{metric_name:<12} {src_val:<15.4f} {tgt_val:<15.4f} {gap:<+10.4f}")
    print("=" * 60)

    print(f"\nTo load the saved model:")
    print(f"  model = AutoModelForSequenceClassification.from_pretrained('{final_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{final_dir}')")


if __name__ == "__main__":
    main()
