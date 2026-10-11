"""
Fine-tune DAPT + Adapter Model on Source Dataset (Adapter-only)

This script loads the DAPT (Domain Adaptive Pretrained) model from
indobert_mlm_target_final and fine-tunes ONLY the adapter + classifier 
for sentiment classification on SOURCE domain (Telemedicine) data,
then evaluates on both source and target test sets.

Pipeline:
  1. bert_mlm_target.py  -> MLM pretraining on target -> DAPT model
  2. This script          -> Fine-tune DAPT + Adapter on SOURCE labeled data (adapter-only)
  3. Evaluate on source + target test sets

Key difference from fine_tune_dapt_adapter_target.py:
  - Trains on SOURCE domain (Telemedicine) not target
  - Only adapter + classifier trainable (encoder frozen)
  - Matches the dapt_adapter.ipynb approach
"""

import argparse
import math
import os
import sys

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
    set_seed,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


# ---------- Verbose control ---------------------------------------------------
# Default: only the compact 2-line header + 4-line summary is printed.
# Pass --use-verbose to see the full training logs.
_ORIG_STDOUT = sys.stdout
USE_VERBOSE = False


def set_verbose(verbose: bool):
    """Silence stdout + library logs/progress bars unless verbose is set."""
    global USE_VERBOSE
    USE_VERBOSE = bool(verbose)
    if USE_VERBOSE:
        return
    sys.stdout = open(os.devnull, 'w')
    try:
        from transformers.utils.logging import set_verbosity_error
        set_verbosity_error()
    except Exception:
        pass
    try:
        from datasets.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass


def print_summary(model_line: str, hparams_line: str, src: dict, tgt: dict):
    """Compact summary — always printed, even in quiet mode."""
    out = _ORIG_STDOUT
    print(model_line, file=out)
    print(hparams_line, file=out)
    for key in ('accuracy', 'precision', 'recall', 'f1'):
        s, t = src.get(key), tgt.get(key)
        if s is None or t is None:
            continue
        print(f'{key}: source={s:.4f}  target={t:.4f}  gap={s - t:+.4f}', file=out)
    out.flush()


# ---------- Training configuration (same key names as ADAPT-DANN's PIPE_CFG) ----------
# Note: this baseline has NO domain adapter. DAPT here = MLM-pretrained backbone
# (dapt_model_dir); only a single task adapter + classifier head is trained.
CFG = {
'dapt_model_dir':           '../../dapt_lazada',
'source_labeled_csv':       '../../datasets/telemedicine_preprocessed/train_split.csv',
'source_val_csv':           '../../datasets/telemedicine_preprocessed/val.csv',  # shared by all methods
'source_test_csv':          '../../datasets/telemedicine_preprocessed/test.csv',
'target_eval_csv':          '../../datasets/lazada_test_preprocessed.csv',
'output_dir':               './models/dapt_lazada_adapter_source_finetuned',
'final_model_dir':          './models/dapt_lazada_adapter_source_final',

'batch_size':               32,
'max_length':               128,
'epochs_task':              3,
'learning_rate':            1e-5,   # encoder LR (unfrozen top layers)
'task_adapter_lr':          1e-3,   # task adapter LR
'classifier_lr':            1e-3,   # classifier head LR
'weight_decay':             0.02,
'warmup_ratio':             0.1,
'dropout':                  0.2,
'task_adapter_dim':         16,
'max_grad_norm':            1.0,
'unfreeze_top_layers':      4,
'logging_steps':            50,
'seed':                     42,
}


def parse_args():
    """CLI overrides: python fine_tune_dapt_adapter_source.py --seed 84 --task_adapter_lr 5e-4 ..."""
    parser = argparse.ArgumentParser(
        description='Fine-tune DAPT + Adapter on SOURCE data (adapter-only).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--use-verbose', dest='use_verbose', action='store_true',
                        help='Show all training logs (default: summary only)')
    for k, v in CFG.items():
        parser.add_argument(f'--{k}', type=type(v), default=None)
    args = parser.parse_args()
    for k in CFG:
        v = getattr(args, k)
        if v is not None:
            CFG[k] = v
    return args


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


class ResidualBottleneckAdapter(nn.Module):
    """Residual bottleneck adapter (Pfeiffer-style with LayerNorm)"""
    def __init__(self, hidden_size, adapter_dim=16, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.layer_norm(self.up(self.act(self.down(x)))))


class DAPTAdapterClassifier(nn.Module):
    """DAPT model with adapter layer for sequence classification"""
    def __init__(self, base_model_dir, adapter_dim=16, dropout=0.2, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_dir)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.adapter = ResidualBottleneckAdapter(hidden_size, adapter_dim=adapter_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )
        # FREEZE encoder - only adapter + classifier trainable
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

    def extract_features(self, input_ids, attention_mask):
        feat = self.dropout(self.mean_pool(input_ids, attention_mask))
        return self.adapter(feat)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        feat = self.extract_features(input_ids, attention_mask)
        logits = self.classifier(feat)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}


def unfreeze_top_layers(model, num_layers):
    """Unfreeze the top N transformer layers of the encoder"""
    if num_layers <= 0:
        return
    encoder = model.encoder
    if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
        layers = encoder.encoder.layer
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"  Unfroze top {num_layers} transformer layers")
    else:
        print("  Warning: cannot find encoder layers to unfreeze")


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
                    
                    feats = model.extract_features(input_ids=input_ids, attention_mask=attention_mask)
                    Xs.append(feats.cpu().numpy())
                    
                    labels = batch.get('labels')
                    if isinstance(labels, torch.Tensor):
                        Ys.append(labels.cpu().numpy())
                    elif labels is not None:
                        Ys.append(np.array(labels))
                    else:
                        Ys.append(np.full((feats.size(0),), -1, dtype=int))
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
        
        ax.set_title(title, fontsize=24)
        ax.grid(False)
        ax.legend(loc='best', fontsize=18, frameon=True, markerscale=2)
        fig.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"  Saved t-SNE visualization to: {output_path}")
        
    except Exception as e:
        print(f"  t-SNE visualization failed: {e}")


def main():
    args = parse_args()
    set_verbose(args.use_verbose)
    # Seed before building the model so adapter/classifier init is reproducible
    # (same as ADAPT-DANN's set_seed(cfg['seed']) before model construction).
    set_seed(CFG['seed'])

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
    df = pd.read_csv(CFG['source_labeled_csv'])
    # Target test data (Lazada)
    target_test_df = pd.read_csv(CFG['target_eval_csv'])
    # Source test data (Telemedicine)
    source_test_df = pd.read_csv(CFG['source_test_csv'])
    
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
    # Load Domain-Adapted Model with Adapter
    # ========================================================================
    print("\n" + "="*60)
    print("Loading domain-adapted model with adapter...")
    print("="*60)
    
    model_path = CFG['dapt_model_dir']
    print(f"Loading domain-adapted model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = DAPTAdapterClassifier(
        base_model_dir=model_path,
        adapter_dim=CFG['task_adapter_dim'],
        dropout=CFG['dropout'],
        num_labels=2
    )
    
    # Unfreeze top N encoder layers for task-adaptive fine-tuning
    unfreeze_top_layers(model, CFG['unfreeze_top_layers'])
    
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
    
    # Fixed validation set (val.csv), shared by all methods; df is already train_split.csv
    train_df = df
    val_df = pd.read_csv(CFG['source_val_csv'])
    val_df['label_id'] = val_df['label'].astype(str).str.strip().str.lower().map(label2id)
    train_dataset = Dataset.from_dict({"text": train_df['text'].tolist(), "label": train_df['label_id'].tolist()})
    test_dataset = Dataset.from_dict({"text": val_df['text'].tolist(), "label": val_df['label_id'].tolist()})
    
    print(f"Dataset size: {len(df)}")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=CFG['max_length'],
            padding=False
        )
    
    print("\nTokenizing dataset...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing train")
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing val")
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # ========================================================================
    # Configure Training Parameters
    # ========================================================================
    print("\n" + "="*60)
    print("Configuring training parameters...")
    print("="*60)
    
    # LRs come from CFG: learning_rate (encoder), task_adapter_lr, classifier_lr
    training_args = TrainingArguments(
        output_dir=CFG['output_dir'],
        num_train_epochs=CFG['epochs_task'],
        per_device_train_batch_size=CFG['batch_size'],
        per_device_eval_batch_size=CFG['batch_size'],
        learning_rate=CFG['learning_rate'],  # Default LR (overridden by custom optimizer)
        weight_decay=CFG['weight_decay'],
        warmup_ratio=CFG['warmup_ratio'],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # ADAPT-DANN keeps the epoch with the lowest validation loss
        metric_for_best_model="loss",
        greater_is_better=False,
        max_grad_norm=CFG['max_grad_norm'],
        logging_steps=CFG['logging_steps'],
        disable_tqdm=not USE_VERBOSE,
        fp16=False,  # ADAPT-DANN trains in fp32
        push_to_hub=False,
        report_to="none",
        seed=CFG['seed'],
    )
    
    # Separate learning rates: encoder < adapter = classifier
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad], 'lr': CFG['learning_rate']},
        {'params': [p for n, p in model.named_parameters() if 'adapter' in n and p.requires_grad], 'lr': CFG['task_adapter_lr']},
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad], 'lr': CFG['classifier_lr']},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=CFG['weight_decay'])
    
    total_steps = math.ceil(len(train_dataset) / training_args.per_device_train_batch_size) * int(training_args.num_train_epochs)
    warmup_steps = int(CFG['warmup_ratio'] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    print("Training configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Encoder LR: {CFG['learning_rate']} (unfrozen top {CFG['unfreeze_top_layers']} layers)")
    print(f"  Task adapter LR: {CFG['task_adapter_lr']}")
    print(f"  Classifier LR: {CFG['classifier_lr']}")
    print(f"  FP16: {training_args.fp16}")
    
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
    print("Starting fine-tuning on SOURCE domain data (adapter-only)...")
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
    
    output_dir = CFG['final_model_dir']
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
    # Final Summary — printed at the very end of main()
    # ========================================================================
    
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
        output_path="./outputs/tsne_dapt_adapter_source_adapter_only.png",
        device=device,
        title="DAPT + Adapter"
    )
    
    print("\n" + "="*60)
    print("All done!")
    print("="*60)

    # Compact summary — always printed (this is the only output in quiet mode)
    print_summary(
        f"Model: DAPT + Adapter on source, adapter-only ({model_path})",
        f"batch={CFG['batch_size']}  max_len={CFG['max_length']}  epochs={CFG['epochs_task']}  "
        f"lr_enc={CFG['learning_rate']:g}  lr_task_adapter={CFG['task_adapter_lr']:g}  lr_head={CFG['classifier_lr']:g}  "
        f"wd={CFG['weight_decay']}  val={os.path.basename(CFG['source_val_csv'])}  "
        f"seed={CFG['seed']}",
        source_results,
        target_results,
    )


if __name__ == "__main__":
    main()