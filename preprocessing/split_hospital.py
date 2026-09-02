"""
Split hospital preprocessed data into:
  - domain_train.csv: unlabeled source for Phase 1 (domain adapter)
  - task_train.csv:   labeled source for Phase 2 (task adapter)
  - test.csv:         held-out for evaluation (unchanged)

This prevents the domain adapter and task adapter from seeing the same examples.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DOMAIN_RATIO = 0.7  # 70% for domain adapter, 30% for task adapter

# Load preprocessed hospital train
df = pd.read_csv(r"D:\UDA\datasets\hospital_preprocessed\train.csv")
print(f"Total hospital train: {len(df):,} rows")
print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

# Split into domain adapter data (unlabeled) and task adapter data (labeled)
domain_df, task_df = train_test_split(
    df, test_size=1 - DOMAIN_RATIO, random_state=SEED, stratify=df["label"]
)
domain_df = domain_df.reset_index(drop=True)
task_df = task_df.reset_index(drop=True)

print(f"\nDomain adapter split: {len(domain_df):,} rows")
print(f"  {domain_df['label'].value_counts().to_string()}")
print(f"\nTask adapter split:   {len(task_df):,} rows")
print(f"  {task_df['label'].value_counts().to_string()}")

# Save
domain_df.to_csv(r"D:\UDA\datasets\hospital_preprocessed\domain_train.csv", index=False)
task_df.to_csv(r"D:\UDA\datasets\hospital_preprocessed\task_train.csv", index=False)

print(f"\nSaved: hospital_preprocessed/domain_train.csv ({len(domain_df):,} rows)")
print(f"Saved: hospital_preprocessed/task_train.csv   ({len(task_df):,} rows)")
print(f"Saved: hospital_preprocessed/test.csv         (unchanged)")
