#Dataset exploration script for the SMS Spam Collection dataset.

import pandas as pd
import os
import sys
import csv


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")


def inspect_raw_lines(path: str, n: int = 5) -> None:
    
    print("=" * 55)
    print("  RAW FILE INSPECTION (first 5 lines)")
    print("=" * 55)

    with open(path, encoding="latin-1") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            visible = line.rstrip("\n").replace("\t", "[TAB]")
            tab_count   = line.count("\t")
            comma_count = line.count(",")
            print(f"\n  Line {i+1}:")
            print(f"    tabs={tab_count}  commas={comma_count}")
            print(f"    {visible[:120]}{'...' if len(visible) > 120 else ''}")

    print()


def load_dataset(path: str) -> pd.DataFrame:
    
    if not os.path.exists(path):
        print(f"  [ERROR]  File not found: {path}")
        sys.exit(1)

    records = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip().strip('"')
            parts = line.split("\t", maxsplit=1)
            if len(parts) == 2:
                records.append({"label": parts[0], "message": parts[1]})

    df = pd.DataFrame(records)
    print(f"  [OK]  Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df

def explore(df: pd.DataFrame) -> None:

    print("=" * 55)
    print("  SECTION 1 - Raw Column Names")
    print("=" * 55)
    print(f"  Columns : {list(df.columns)}")
    print(f"  Total   : {len(df.columns)}")

    print("\n" + "=" * 55)
    print("  SECTION 2 - Dataset Shape")
    print("=" * 55)
    print(f"  Rows    : {df.shape[0]}")
    print(f"  Columns : {df.shape[1]}")

    if df.shape[1] < 2:
        print("  [ERROR]  Still only 1 column. See raw inspection above.")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  SECTION 3 - Column Data Types")
    print("=" * 55)
    print(df.dtypes.to_string())

    print("\n" + "=" * 55)
    print("  SECTION 4 - Missing Values Per Column")
    print("=" * 55)
    null_counts = df.isnull().sum()
    print(null_counts.to_string())
    total_nulls = null_counts.sum()
    if total_nulls == 0:
        print("\n  [OK]  No missing values found.")
    else:
        print(f"\n  [WARN]  Total missing values: {total_nulls}")
        print(  "          Rows with nulls will be dropped in preprocessing.")

    print("\n" + "=" * 55)
    print("  SECTION 5 - First 5 Rows (Raw)")
    print("=" * 55)
    print(df.head().to_string())

    print("\n" + "=" * 55)
    print("  SECTION 6 - Label Distribution (column: 'label')")
    print("=" * 55)
    label_counts = df["label"].value_counts()
    total = len(df)
    for label, count in label_counts.items():
        pct   = (count / total) * 100
        bar   = "#" * int(pct / 2)
        print(f"  {label:>6}  {count:>5}  ({pct:5.1f}%)  {bar}")

    print("\n" + "=" * 55)
    print("  SECTION 7 - Class Imbalance Check")
    print("=" * 55)
    counts = label_counts.values
    if len(counts) >= 2:
        ratio = counts[0] / counts[1]
        print(f"  Majority : Minority ratio = {ratio:.1f} : 1")
        if ratio > 3:
            print("  [WARN]  Imbalanced dataset.")
            print("          Will use stratified split during training.")
        else:
            print("  [OK]  Reasonably balanced.")

    print("\n" + "=" * 55)
    print("  SECTION 8 - Sample Messages")
    print("=" * 55)
    for label in df["label"].unique():
        sample  = df[df["label"] == label]["message"].iloc[0]
        display = (str(sample)[:100] + "...") if len(str(sample)) > 100 else str(sample)
        print(f"\n  [{label.upper()}]\n  {display}")

    print("\n" + "=" * 55)
    print("  [OK]  Exploration complete.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    print("\n  Loading dataset...\n")
    inspect_raw_lines(DATA_PATH)
    df = load_dataset(DATA_PATH)
    explore(df)