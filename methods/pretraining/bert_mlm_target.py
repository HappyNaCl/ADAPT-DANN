"""
IndoBERT Masked Language Modeling - Target Domain Adaptation

This script trains IndoBERT using Masked Language Modeling (MLM) on the target domain data 
(reviewContent from target_final.csv)
"""

import os
import pandas as pd
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline
)
from datasets import Dataset


def main():
    # ========================================================================
    # Check GPU availability
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========================================================================
    # Load Target Data
    # ========================================================================
    print("\n" + "="*60)
    print("Loading target data...")
    print("="*60)
    
    df = pd.read_csv('./datasets/lazada_train_preprocessed.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Extract text data (use 'content' column based on preprocessing notebook)
    texts = df['text'].dropna().tolist()
    
    print(f"\nTotal texts for MLM training: {len(texts)}")
    print(f"\nSample text:")
    print(texts[0])
    
    # ========================================================================
    # Load IndoBERT Model and Tokenizer
    # ========================================================================
    print("\n" + "="*60)
    print("Loading IndoBERT model and tokenizer...")
    print("="*60)
    
    model_name = "indolem/indobert-base-uncased"
    
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {model.num_parameters():,}")
    
    # ========================================================================
    # Prepare Data for MLM Training
    # ========================================================================
    print("\n" + "="*60)
    print("Preparing data for MLM training...")
    print("="*60)
    
    # Create dataset from texts
    dataset_dict = {"text": texts}
    dataset = Dataset.from_dict(dataset_dict)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"\nDataset features: {dataset.features}")
    print(f"\nFirst example:")
    print(dataset[0])
    
    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize with truncation and padding
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_special_tokens_mask=True
        )
    
    # Apply tokenization
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    print(f"Tokenization complete!")
    print(f"Tokenized dataset features: {tokenized_dataset.features}")
    
    # Split dataset into train and validation (90/10 split)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    # Create data collator for MLM
    # mlm_probability=0.15 means 15% of tokens will be masked
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    print("\nData collator created for MLM with 15% masking probability")
    
    # ========================================================================
    # Configure Training Parameters
    # ========================================================================
    print("\n" + "="*60)
    print("Configuring training parameters...")
    print("="*60)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./models/indobert_mlm_target",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none"  # Disable wandb/tensorboard
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
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("\nTrainer initialized successfully!")
    
    # ========================================================================
    # Train the Model
    # ========================================================================
    print("\n" + "="*60)
    print("Starting MLM training on target domain data...")
    print("="*60)
    
    train_result = trainer.train()
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    
    # ========================================================================
    # Evaluate the Model
    # ========================================================================
    print("\n" + "="*60)
    print("Evaluating model on validation set...")
    print("="*60)
    
    eval_results = trainer.evaluate()
    
    print("\nValidation Results:")
    print(f"  Validation Loss: {eval_results['eval_loss']:.4f}")
    print(f"  Perplexity: {np.exp(eval_results['eval_loss']):.4f}")
    
    # ========================================================================
    # Save the Fine-tuned Model
    # ========================================================================
    print("\n" + "="*60)
    print("Saving the fine-tuned model...")
    print("="*60)
    
    output_dir = "./models/indobert_mlm_target_final"
    
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Also save model state dict (PyTorch weights only)
    weights_path = f"{output_dir}/pytorch_model.bin"
    print(f"\nModel weights saved at: {weights_path}")
    
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
    print("\nYou can now use this model for downstream tasks like sentiment analysis!")
    print(f"\nTo load the model later, use:")
    print(f"  model = AutoModelForMaskedLM.from_pretrained('{output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    
    # ========================================================================
    # Check Training Checkpoints
    # ========================================================================
    print("\n" + "="*60)
    print("Checking training checkpoints...")
    print("="*60)
    
    checkpoint_dir = "./indobert_mlm_target"
    
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
        checkpoints.sort()
        
        if checkpoints:
            print(f"Available checkpoints in {checkpoint_dir}:")
            for cp in checkpoints:
                cp_path = os.path.join(checkpoint_dir, cp)
                print(f"  - {cp}")
            
            print(f"\nTo load a specific checkpoint:")
            print(f"  model = AutoModelForMaskedLM.from_pretrained('{checkpoint_dir}/checkpoint-XXX')")
        else:
            print("No checkpoints found (training may not have started yet)")
    else:
        print(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # ========================================================================
    # Test the Fine-tuned Model (Optional)
    # ========================================================================
    print("\n" + "="*60)
    print("Testing the fine-tuned model...")
    print("="*60)
    
    # Load the fine-tuned model for testing
    fill_mask = pipeline(
        "fill-mask",
        model=output_dir,
        tokenizer=output_dir
    )
    
    # Test with a sample sentence with [MASK]
    test_sentence = "Produk ini sangat [MASK] dan berkualitas."
    print(f"Test sentence: {test_sentence}\n")
    
    # Get predictions
    predictions = fill_mask(test_sentence, top_k=5)
    
    print("Top 5 predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['token_str']:15s} - Score: {pred['score']:.4f}")
    
    print("\n" + "="*60)
    print("All done!")
    print("="*60)


if __name__ == "__main__":
    main()
