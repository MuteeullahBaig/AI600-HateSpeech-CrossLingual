"""
Preprocessing script: validates data files, prints statistics, and creates
stratified few-shot splits for Urdu (Roman Urdu) transfer experiments.

Usage:
    python scripts/preprocess.py \
        --en_train data/raw/hateval2019_en_train.tsv \
        --en_dev   data/raw/hateval2019_en_dev.tsv \
        --ur_train data/raw/hateval2019_ur_train.tsv \
        --ur_dev   data/raw/hateval2019_ur_dev.tsv \
        --ur_test  data/raw/hateval2019_ur_test.tsv
"""

import os
import sys
import json
import argparse
import random

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_hateval_tsv, df_to_hf_dataset, print_dataset_stats
from src.utils import set_seed

PROCESSED_DIR = 'data/processed'
FEW_SHOT_DIR  = 'data/processed/few_shot_samples'
SHOT_COUNTS   = [8, 16, 32, 64]
NUM_SEEDS     = 3


def save_json(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(records)} records → {path}")


def df_to_records(df: pd.DataFrame) -> list[dict]:
    cols = ['id', 'text', 'hs']
    for c in ['tr', 'ag']:
        if c in df.columns:
            cols.append(c)
    return df[cols].to_dict(orient='records')


def create_few_shot_splits(df: pd.DataFrame, shot_counts: list[int], num_seeds: int, lang: str = 'ur') -> None:
    """Create stratified K-shot splits from the Urdu train set."""
    labels = df['hs'].tolist()
    for k in shot_counts:
        if k > len(df):
            print(f"  Skipping {k}-shot (dataset only has {len(df)} samples)")
            continue
        for seed in range(num_seeds):
            random.seed(seed)
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=k, random_state=seed)
            for idx, _ in splitter.split(df, labels):
                sample = df.iloc[sorted(idx)]
            records = df_to_records(sample)
            path = os.path.join(FEW_SHOT_DIR, f'{lang}_{k}shot_seed{seed}.json')
            save_json(records, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--en_train', default=None)
    p.add_argument('--en_dev',   default=None)
    p.add_argument('--ur_train', default=None)
    p.add_argument('--ur_dev',   default=None)
    p.add_argument('--ur_test',  default=None)
    args = p.parse_args()

    set_seed(42)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FEW_SHOT_DIR,  exist_ok=True)

    # ---- English ----
    print("\n=== English (Davidson et al. 2017) ===")
    for tag, path in [('en_train', args.en_train), ('en_dev', args.en_dev)]:
        if path is None or not os.path.exists(path):
            print(f"  {tag}: not found, skipping")
            continue
        df = load_hateval_tsv(path)
        print_dataset_stats({tag: df_to_hf_dataset(df)})
        save_json(df_to_records(df), os.path.join(PROCESSED_DIR, f'{tag}.json'))

    # ---- Urdu ----
    print("\n=== Urdu (Roman Urdu Hate Speech) ===")
    ur_train_df = None
    for tag, path in [('ur_train', args.ur_train), ('ur_dev', args.ur_dev), ('ur_test', args.ur_test)]:
        if path is None or not os.path.exists(path):
            print(f"  {tag}: not found, skipping")
            continue
        df = load_hateval_tsv(path)
        has_labels = 'hs' in df.columns
        print_dataset_stats({tag: df_to_hf_dataset(df, has_labels=has_labels)})
        save_json(
            df_to_records(df) if has_labels else df[['id', 'text']].to_dict('records'),
            os.path.join(PROCESSED_DIR, f'{tag}.json'),
        )
        if tag == 'ur_train':
            ur_train_df = df

    # ---- Few-shot splits from Urdu train ----
    if ur_train_df is not None and 'hs' in ur_train_df.columns:
        print("\n=== Creating Urdu few-shot splits ===")
        create_few_shot_splits(ur_train_df, SHOT_COUNTS, NUM_SEEDS, lang='ur')
        print(f"  Created {len(SHOT_COUNTS) * NUM_SEEDS} few-shot split files in {FEW_SHOT_DIR}/")

    print("\nPreprocessing complete.")


if __name__ == '__main__':
    main()
