"""
Fine-tune Domain-Adapted IndoBERT on Source Dataset

This script fine-tunes the domain-adapted IndoBERT (from MLM training) for sentiment 
classification using source_balanced.csv
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
    pipeline
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def compute_metrics(eval_pred):
    """Calculate evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_on_test_set(trainer, tokenizer, test_df, text_column, label2id, id2label, 
                         domain_name, tokenize_function):
    """Evaluate the trained model on a given test set and print detailed metrics."""
    print("\n" + "="*60)
    print(f"Evaluating on {domain_name} test set...")
    print("="*60)
    
    print(f"\nTest set shape: {test_df.shape}")
    print(f"Label distribution:")
    print(test_df['label'].value_counts())
    
    # Convert labels to numeric
    test_df = test_df.copy()
    test_df['label_id'] = test_df['label'].map(label2id)
    
    # Create dataset
    test_dict = {
        "text": test_df[text_column].tolist(),
        "label": test_df['label_id'].tolist()
    }
    test_dataset = Dataset.from_dict(test_dict)
    
    # Tokenize
    print(f"\nTokenizing {domain_name} test data...")
    test_tokenized = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing {domain_name} test"
    )
    
    # Make predictions
    print(f"\nMaking predictions on {domain_name} test set...")
    predictions = trainer.predict(test_tokenized)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted'
    )
    
    print(f"\n{domain_name.upper()} TEST SET RESULTS:")
    print(f"  Samples:   {len(test_df)}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report ({domain_name}):")
    print("-"*60)
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=['negative', 'positive'],
        digits=4
    ))
    
    return {
        'domain': domain_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'samples': len(test_df)
    }


def main():
    # ========================================================================
    # Check GPU availability
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========================================================================
    # Load Source Dataset
    # ========================================================================
    print("\n" + "="*60)
    print("Loading source dataset...")
    print("="*60)
    
    # Load .csv
    df = pd.read_csv('./datasets/coastsent_train.csv')
    
    # Load target domain test .csv
    target_test_df = pd.read_csv('./datasets/lazada_test_preprocessed.csv')

    # Load source domain test .csv
    source_test_df = pd.read_csv('./datasets/coastsent_test.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Create label mapping
    label2id = {'negative': 0, 'positive': 1}
    id2label = {0: 'negative', 1: 'positive'}
    
    # Convert labels to numeric
    df['label_id'] = df['label'].map(label2id)
    
    print("\nLabel mapping:")
    print(f"  negative → 0")
    print(f"  positive → 1")
    print(f"\nLabel distribution (numeric):")
    print(df['label_id'].value_counts().sort_index())
    
    # ========================================================================
    # Load Domain-Adapted Model
    # ========================================================================
    print("\n" + "="*60)
    print("Loading domain-adapted model...")
    print("="*60)
    
    model_path = "./models/indobert_mlm_target_final"
    
    print(f"Loading domain-adapted model from: {model_path}")
    
    # Load tokenizer (same as before)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model for sequence classification (not MLM)
    # This will add a classification head on top of the MLM-adapted base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # Ignore the MLM head, use classification head instead
    )
    
    print(f"Model loaded successfully!")
    print(f"Model type: {model.__class__.__name__}")
    print(f"Number of labels: {model.config.num_labels}")
    
    # ========================================================================
    # Prepare Data for Training
    # ========================================================================
    print("\n" + "="*60)
    print("Preparing data for training...")
    print("="*60)
    
    # Create dataset from DataFrame
    dataset_dict = {
        "text": df['text'].tolist(),
        "label": df['label_id'].tolist()
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")
    print(f"\nFirst example:")
    print(dataset[0])
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False  # Dynamic padding will be handled by data collator
        )
    
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    print(f"Tokenization complete!")
    print(f"Tokenized dataset features: {tokenized_dataset.features}")
    
    # Split dataset into train and test (80/20 split)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    print("\nData collator created for dynamic padding")
    
    # ========================================================================
    # Configure Training Parameters
    # ========================================================================
    print("\n" + "="*60)
    print("Configuring training parameters...")
    print("="*60)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./models/indobert_source_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none"
    )
    
    print("Training configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  FP16: {training_args.fp16}")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("\nTrainer initialized successfully!")
    
    # ========================================================================
    # Train the Model
    # ========================================================================
    print("\n" + "="*60)
    print("Starting fine-tuning on source domain data...")
    print("="*60)
    
    train_result = trainer.train()
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    
    # ========================================================================
    # Evaluate the Model on internal validation split
    # ========================================================================
    print("\n" + "="*60)
    print("Evaluating model on internal validation split...")
    print("="*60)
    
    eval_results = trainer.evaluate()
    
    print("\nInternal Validation Results:")
    print(f"  Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall:    {eval_results['eval_recall']:.4f}")
    print(f"  F1 Score:  {eval_results['eval_f1']:.4f}")
    print(f"  Loss:      {eval_results['eval_loss']:.4f}")
    
    # Get detailed predictions for classification report
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Print detailed classification report
    print("\nDetailed Classification Report (internal val):")
    print("="*60)
    print(classification_report(
        true_labels, 
        pred_labels, 
        target_names=['negative', 'positive'],
        digits=4
    ))
    
    # ========================================================================
    # Save the Fine-tuned Model
    # ========================================================================
    print("\n" + "="*60)
    print("Saving the fine-tuned model...")
    print("="*60)
    
    output_dir = "./models/indobert_source_final"
    
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Verify saved files
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"\nSaved files in {output_dir}:")
        for file in files:
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file:30s} ({size_mb:.2f} MB)")
    
    print(f"\n✓ Model and tokenizer saved successfully to: {output_dir}")
    print(f"\nTo load the model later, use:")
    print(f"  from transformers import AutoModelForSequenceClassification, AutoTokenizer")
    print(f"  model = AutoModelForSequenceClassification.from_pretrained('{output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    
    # ========================================================================
    # Evaluate on SOURCE test set (lazada_test.csv)
    # ========================================================================
    # NOTE: adjust the text column name below if your source test csv uses a
    # different column than 'reviewContent'
    source_results = evaluate_on_test_set(
        trainer=trainer,
        tokenizer=tokenizer,
        test_df=source_test_df,
        text_column='text',
        label2id=label2id,
        id2label=id2label,
        domain_name='source',
        tokenize_function=tokenize_function
    )
    
    # ========================================================================
    # Evaluate on TARGET test set (coastsent_test.csv)
    # ========================================================================
    # NOTE: adjust the text column name below if your target test csv uses a
    # different column than 'content'
    target_results = evaluate_on_test_set(
        trainer=trainer,
        tokenizer=tokenizer,
        test_df=target_test_df,
        text_column='text',
        label2id=label2id,
        id2label=id2label,
        domain_name='target',
        tokenize_function=tokenize_function
    )
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*60)
    print("FINAL SUMMARY - Source vs Target Performance")
    print("="*60)
    print(f"\n{'Metric':<12} {'Source':<15} {'Target':<15} {'Gap':<10}")
    print("-"*52)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        src_val = source_results[metric]
        tgt_val = target_results[metric]
        gap = src_val - tgt_val
        print(f"{metric.capitalize():<12} {src_val:<15.4f} {tgt_val:<15.4f} {gap:<+10.4f}")
    
    print(f"\n{'Samples':<12} {source_results['samples']:<15} {target_results['samples']:<15}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()