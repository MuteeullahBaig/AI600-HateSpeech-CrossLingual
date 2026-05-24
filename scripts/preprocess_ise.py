"""
ISE-Hate preprocessing script.

Reads ISE_Level_1_Dataset.xlsx, standardises columns to match the
existing pipeline format (id, text, hs), creates stratified
train / dev / test TSV splits, and generates few-shot JSON samples.

Usage:
    python scripts/preprocess_ise.py \
        --input     data/raw/ISE_Level_1_Dataset.xlsx \
        --test_size 0.20 \
        --dev_size  0.10
"""

import os
import sys
import json
import argparse
import random

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed

PROCESSED_DIR = 'data/processed'
FEW_SHOT_DIR  = 'data/processed/few_shot_samples'
SHOT_COUNTS   = [8, 16, 32, 64]
NUM_SEEDS     = 3


def save_tsv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False, encoding='utf-8')
    n_hate = int(df['hs'].sum())
    n_non  = int((df['hs'] == 0).sum())
    print(f"  Saved {len(df)} rows -> {path}  [hate={n_hate} | non-hate={n_non}]")


def save_json(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(records)} records -> {path}")


def create_few_shot_splits(
    df: pd.DataFrame,
    shot_counts: list,
    num_seeds: int,
    lang: str = 'ise',
) -> None:
    """Create stratified K-shot splits from the ISE train set."""
    labels = df['hs'].tolist()
    for k in shot_counts:
        if k > len(df):
            print(f"  Skipping {k}-shot (train only has {len(df)} samples)")
            continue
        for seed in range(num_seeds):
            random.seed(seed)
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=k, random_state=seed
            )
            for idx, _ in splitter.split(df, labels):
                sample = df.iloc[sorted(idx)]
            records = sample[['id', 'text', 'hs']].to_dict(orient='records')
            path = os.path.join(FEW_SHOT_DIR, f'{lang}_{k}shot_seed{seed}.json')
            save_json(records, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',     default='data/raw/ISE_Level_1_Dataset.xlsx',
                   help='Path to ISE_Level_1_Dataset.xlsx')
    p.add_argument('--test_size', type=float, default=0.20,
                   help='Fraction of data for test set (default 0.20)')
    p.add_argument('--dev_size',  type=float, default=0.10,
                   help='Fraction of total data for dev set (default 0.10)')
    args = p.parse_args()

    set_seed(42)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FEW_SHOT_DIR,  exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading {args.input} ...")
    df = pd.read_excel(args.input)

    # Standardise to pipeline format: id | text | hs
    df = df[['Tweet_ID', 'Tweet_Text', 'Final_decision']].copy()
    df.columns = ['id', 'text', 'hs']
    df['text'] = df['text'].astype(str).str.strip()
    df['hs']   = df['hs'].astype(int)
    df = df.dropna(subset=['text']).reset_index(drop=True)

    total  = len(df)
    n_hate = int(df['hs'].sum())
    n_non  = total - n_hate
    print(f"  Total: {total} | hate={n_hate} ({100*n_hate/total:.1f}%) "
          f"| non-hate={n_non} ({100*n_non/total:.1f}%)")

    # ── Stratified splits ────────────────────────────────────────────────────
    # Step 1: carve out test set
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=42
    )
    for traindev_idx, test_idx in sss1.split(df, df['hs']):
        df_traindev = df.iloc[traindev_idx].reset_index(drop=True)
        df_test     = df.iloc[test_idx].reset_index(drop=True)

    # Step 2: carve out dev set from remaining
    dev_ratio = args.dev_size / (1.0 - args.test_size)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=dev_ratio, random_state=42
    )
    for train_idx, dev_idx in sss2.split(df_traindev, df_traindev['hs']):
        df_train = df_traindev.iloc[train_idx].reset_index(drop=True)
        df_dev   = df_traindev.iloc[dev_idx].reset_index(drop=True)

    print("\nSplit sizes:")
    for name, split in [('train', df_train), ('dev', df_dev), ('test', df_test)]:
        save_tsv(split, os.path.join(PROCESSED_DIR, f'ise_{name}.tsv'))

    # ── Few-shot splits from train ───────────────────────────────────────────
    print("\nCreating few-shot splits from ISE train set ...")
    create_few_shot_splits(df_train, SHOT_COUNTS, NUM_SEEDS, lang='ise')
    print(f"  Created {len(SHOT_COUNTS) * NUM_SEEDS} split files in {FEW_SHOT_DIR}/")

    print("\nISE preprocessing complete.")


if __name__ == '__main__':
    main()
