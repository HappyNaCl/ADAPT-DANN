"""
Fine-tune DAPT Model on Source Dataset (No Adapter - DAPT Only Baseline)

This script loads the DAPT (Domain Adaptive Pretrained) model and fine-tunes
ONLY the classifier head for sentiment classification on SOURCE domain
(Telemedicine) data, then evaluates on both source and target test sets.

Purpose:
  Baseline to measure the contribution of the adapter layer.
  Comparing DAPT+Adapter vs DAPT-only shows how much the adapter helps.

Pipeline:
  1. bert_mlm_target.py  -> MLM pretraining on target -> DAPT model
  2. This script          -> Fine-tune DAPT on SOURCE labeled data (classifier-only)
  3. Evaluate on source + target test sets
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def compute_metrics(eval_pred):
    """Calculate evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
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
    
    test_df = test_df.copy()
    test_df['label_id'] = test_df['label'].map(label2id)
    
    test_dict = {
        "text": test_df[text_column].tolist(),
        "label": test_df['label_id'].tolist()
    }
    test_dataset = Dataset.from_dict(test_dict)
    
    print(f"\nTokenizing {domain_name} test data...")
    test_tokenized = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing {domain_name} test"
    )
    
    print(f"\nMaking predictions on {domain_name} test set...")
    predictions = trainer.predict(test_tokenized)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
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


def run_tsne_visualization(model, tokenizer, source_test_df, target_test_df, 
                           text_column, label2id, tokenize_function, 
                           output_path, device, title):
    """Extract embeddings and run t-SNE visualization"""
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from torch.utils.data import DataLoader
        
        sns.set(style="whitegrid")
        
        PALETTE = {
            ('source', 0): '#d62728',
            ('source', 1): '#1f77b4',
            ('target', 0): '#2ca02c',
            ('target', 1): '#9467bd',
        }
        DISPLAY_NAMES = {
            ('source', 0): 'source negative',
            ('source', 1): 'source positive',
            ('target', 0): 'target negative',
            ('target', 1): 'target positive',
        }
        PLOT_ORDER = [('source', 0), ('source', 1), ('target', 0), ('target', 1)]
        
        def extract_embeddings(model, dataloader, device):
            model.eval()
            Xs, Ys = [], []
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    feat = model.dropout(model.mean_pool(input_ids, attention_mask))
                    Xs.append(feat.cpu().numpy())
                    
                    labels = batch.get('labels')
                    if isinstance(labels, torch.Tensor):
                        Ys.append(labels.cpu().numpy())
                    elif labels is not None:
                        Ys.append(np.array(labels))
                    else:
                        Ys.append(np.full((feat.size(0),), -1, dtype=int))
            return np.concatenate(Xs, 0), np.concatenate(Ys, 0)
        
        # Tokenize test sets
        source_dict = {
            "text": source_test_df[text_column].tolist(),
            "label": source_test_df['label'].map(label2id).tolist()
        }
        target_dict = {
            "text": target_test_df[text_column].tolist(),
            "label": target_test_df['label'].map(label2id).tolist()
        }
        
        source_dataset = Dataset.from_dict(source_dict)
        target_dataset = Dataset.from_dict(target_dict)
        
        source_tokenized = source_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        target_tokenized = target_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        source_loader = DataLoader(source_tokenized, batch_size=32, shuffle=False, collate_fn=data_collator)
        target_loader = DataLoader(target_tokenized, batch_size=32, shuffle=False, collate_fn=data_collator)
        
        print("  Extracting source embeddings...")
        Xs_src, Ys_src = extract_embeddings(model, source_loader, device)
        print("  Extracting target embeddings...")
        Xs_tgt, Ys_tgt = extract_embeddings(model, target_loader, device)
        
        X = np.concatenate([Xs_src, Xs_tgt], axis=0)
        Y = np.concatenate([Ys_src, Ys_tgt], axis=0)
        domains = np.array(['source'] * Xs_src.shape[0] + ['target'] * Xs_tgt.shape[0])
        
        print("  Running t-SNE...")
        if X.shape[1] > 50:
            X = PCA(n_components=50, random_state=42).fit_transform(X)
        tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=30)
        proj = tsne.fit_transform(X)
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        xs, ys = proj[:, 0], proj[:, 1]
        
        for dom, lab in PLOT_ORDER:
            mask = (domains == dom) & (Y == lab)
            if mask.sum() == 0:
                continue
            ax.scatter(xs[mask], ys[mask], s=10, alpha=0.8, 
                      c=PALETTE[(dom, lab)], label=DISPLAY_NAMES[(dom, lab)])
        
        ax.set_title(title, fontsize=18)
        ax.grid(False)
        ax.legend(loc='lower left', fontsize=14, frameon=True, markerscale=2)
        fig.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"  Saved t-SNE visualization to: {output_path}")
        
    except Exception as e:
        print(f"  t-SNE visualization failed: {e}")


class DAPTClassifier(nn.Module):
    """DAPT model with classifier head only (NO adapter)"""
    def __init__(self, base_model_dir, dropout=0.2, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_dir)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )
        # FREEZE encoder - only classifier trainable
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.config = self.encoder.config
        self.config.problem_type = "single_label_classification"
        self.config.num_labels = num_labels

    def mean_pool(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        feat = self.dropout(self.mean_pool(input_ids, attention_mask))
        logits = self.classifier(feat)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}


def main():
    # ========================================================================
    # Check GPU availability
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========================================================================
    # Load Source Dataset (for fine-tuning)
    # ========================================================================
    print("\n" + "="*60)
    print("Loading SOURCE dataset for fine-tuning...")
    print("="*60)
    
    # Source training data (Telemedicine)
    df = pd.read_csv('../../datasets/telemedicine_preprocessed/train.csv')
    # Target test data (Lazada)
    target_test_df = pd.read_csv('../../datasets/lazada_test_preprocessed.csv')
    # Source test data (Telemedicine)
    source_test_df = pd.read_csv('../../datasets/telemedicine_preprocessed/test.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    label2id = {'negative': 0, 'positive': 1}
    id2label = {0: 'negative', 1: 'positive'}
    
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    target_test_df['label'] = target_test_df['label'].astype(str).str.strip().str.lower()
    source_test_df['label'] = source_test_df['label'].astype(str).str.strip().str.lower()
    
    df['label_id'] = df['label'].map(label2id)
    
    print("\nLabel mapping:")
    print(f"  negative -> 0")
    print(f"  positive -> 1")
    print(f"\nLabel distribution (numeric):")
    print(df['label_id'].value_counts().sort_index())
    
    # ========================================================================
    # Load DAPT Model (No Adapter)
    # ========================================================================
    print("\n" + "="*60)
    print("Loading DAPT model (no adapter)...")
    print("="*60)
    
    model_path = "../../dapt_lazada"
    print(f"Loading domain-adapted model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = DAPTClassifier(
        base_model_dir=model_path,
        dropout=0.2,
        num_labels=2
    ).to(device)
    
    print(f"Model loaded successfully!")
    print(f"Model type: {model.__class__.__name__}")
    print(f"Number of labels: {model.config.num_labels}")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # ========================================================================
    # Prepare Data for Training
    # ========================================================================
    print("\n" + "="*60)
    print("Preparing data for training...")
    print("="*60)
    
    dataset_dict = {
        "text": df['text'].tolist(),
        "label": df['label_id'].tolist()
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    print(f"Dataset size: {len(dataset)}")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding=False
        )
    
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # ========================================================================
    # Configure Training Parameters
    # ========================================================================
    print("\n" + "="*60)
    print("Configuring training parameters...")
    print("="*60)
    
    CLASSIFIER_LR = 1e-3
    
    training_args = TrainingArguments(
        output_dir="./models/dapt_lazada_source_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-5,
        weight_decay=0.02,
        warmup_ratio=0.1,
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
    
    # Custom optimizer with classifier LR
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad], 'lr': CLASSIFIER_LR},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.02)
    
    total_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    print("Training configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Classifier LR: {CLASSIFIER_LR}")
    print(f"  FP16: {training_args.fp16}")
    print(f"  No adapter - classifier head only")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )
    
    print("\nTrainer initialized successfully!")
    
    # ========================================================================
    # Train the Model
    # ========================================================================
    print("\n" + "="*60)
    print("Starting fine-tuning on SOURCE domain data (classifier-only)...")
    print("="*60)
    
    train_result = trainer.train()
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    
    # ========================================================================
    # Evaluate on internal validation split
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
    
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
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
    
    output_dir = "./models/dapt_lazada_source_final"
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"\nSaved files in {output_dir}:")
        for file in files:
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file:30s} ({size_mb:.2f} MB)")
    
    print(f"\n[OK] Model and tokenizer saved successfully to: {output_dir}")
    
    # ========================================================================
    # Evaluate on SOURCE test set
    # ========================================================================
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
    # Evaluate on TARGET test set
    # ========================================================================
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
    print("FINAL SUMMARY - DAPT Only (No Adapter)")
    print("="*60)
    print(f"\nPipeline: MLM on target -> Fine-tune DAPT on SOURCE (frozen encoder, no adapter) -> Evaluate")
    print(f"\n{'Metric':<12} {'Source':<15} {'Target':<15} {'Gap':<10}")
    print("-"*52)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        src_val = source_results[metric]
        tgt_val = target_results[metric]
        gap = src_val - tgt_val
        print(f"{metric.capitalize():<12} {src_val:<15.4f} {tgt_val:<15.4f} {gap:<+10.4f}")
    
    print(f"\n{'Samples':<12} {source_results['samples']:<15} {target_results['samples']:<15}")
    
    print("\n" + "="*60)
    print("Comparison with DAPT + Adapter:")
    print("="*60)
    print("DAPT + Adapter:  Source ~88.35%  Target ~79.24%  Gap ~9.12%")
    print("DAPT Only:       Source  ?.??%   Target  ?.??%   Gap  ?.??%")
    
    # ========================================================================
    # t-SNE Visualization
    # ========================================================================
    print("\n" + "="*60)
    print("Generating t-SNE visualization...")
    print("="*60)
    
    run_tsne_visualization(
        model=model,
        tokenizer=tokenizer,
        source_test_df=source_test_df,
        target_test_df=target_test_df,
        text_column='text',
        label2id=label2id,
        tokenize_function=tokenize_function,
        output_path="./outputs/tsne_dapt_source_vs_lazada.png",
        device=device,
        title="DAPT"
    )
    
    print("\n" + "="*60)
    print("All done!")
    print("="*60)


if __name__ == "__main__":
    main()
