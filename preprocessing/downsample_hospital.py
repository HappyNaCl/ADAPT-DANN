"""
Downsample Hospital positive class to match negative class (1:1 ratio).
Creates balanced versions of domain_train, task_train, and test.
"""
import pandas as pd

SEED = 42

def downsample(df):
    """Downsample majority class to match minority class."""
    pos = df[df['label'] == 'positive']
    neg = df[df['label'] == 'negative']
    n = min(len(pos), len(neg))
    pos_down = pos.sample(n=n, random_state=SEED)
    neg_down = neg.sample(n=n, random_state=SEED)
    return pd.concat([pos_down, neg_down]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# Load current data
domain = pd.read_csv(r"D:\UDA\datasets\hospital_preprocessed\domain_train.csv")
task = pd.read_csv(r"D:\UDA\datasets\hospital_preprocessed\task_train.csv")
test = pd.read_csv(r"D:\UDA\datasets\hospital_preprocessed\test.csv")

print("=== BEFORE downsampling ===")
for name, df in [("Domain train", domain), ("Task train", task), ("Test", test)]:
    pos = (df['label'] == 'positive').sum()
    neg = (df['label'] == 'negative').sum()
    print(f"  {name:<15}: {len(df):>6} rows | pos={pos:,} neg={neg:,} ratio={pos/neg:.1f}:1")

# Downsample
domain_bal = downsample(domain)
task_bal = downsample(task)
test_bal = downsample(test)

print("\n=== AFTER downsampling ===")
for name, df in [("Domain train", domain_bal), ("Task train", task_bal), ("Test", test_bal)]:
    pos = (df['label'] == 'positive').sum()
    neg = (df['label'] == 'negative').sum()
    print(f"  {name:<15}: {len(df):>6} rows | pos={pos:,} neg={neg:,} ratio={pos/neg:.1f}:1")

# Save
domain_bal.to_csv(r"D:\UDA\datasets\hospital_preprocessed\domain_train.csv", index=False)
task_bal.to_csv(r"D:\UDA\datasets\hospital_preprocessed\task_train.csv", index=False)
test_bal.to_csv(r"D:\UDA\datasets\hospital_preprocessed\test.csv", index=False)

print("\n=== CoastSent comparison ===")
cs_train = pd.read_csv(r"D:\UDA\datasets\coastsent_train.csv")
cs_test = pd.read_csv(r"D:\UDA\datasets\coastsent_test.csv")
for name, df in [("CoastSent Train", cs_train), ("CoastSent Test", cs_test)]:
    pos = (df['label'] == 'positive').sum()
    neg = (df['label'] == 'negative').sum()
    print(f"  {name:<15}: {len(df):>6} rows | pos={pos:,} neg={neg:,} ratio={pos/neg:.1f}:1")

print("\nSaved balanced files to hospital_preprocessed/")
