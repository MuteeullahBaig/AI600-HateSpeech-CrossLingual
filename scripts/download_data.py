"""
Download and standardise hate speech datasets for EN → Urdu cross-lingual transfer.

English training data  : tdavidson/hate_speech_offensive  (Davidson et al. 2017, ICWSM)
Urdu evaluation data   : community-datasets/roman_urdu_hate_speech  (Roman Urdu, RUHSOLD)

Both are freely accessible on HuggingFace — no registration required.

Usage:
    python scripts/download_data.py
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DIR = 'data/raw'


# ─────────────────────────────────────────────────────────────────────────────
# English: Davidson et al. 2017
#   class 0 = hate speech  → label 1
#   class 1 = offensive    → label 0  (kept as non-hate for larger training set)
#   class 2 = neither      → label 0
# ─────────────────────────────────────────────────────────────────────────────

def download_english():
    from datasets import load_dataset
    print("\n[English] Loading tdavidson/hate_speech_offensive …")
    ds = load_dataset("tdavidson/hate_speech_offensive", split="train", trust_remote_code=False)
    df = ds.to_pandas()

    # Standardise columns
    df = df.rename(columns={'tweet': 'text'})
    df['id']    = range(len(df))
    df['hs']    = (df['class'] == 0).astype(int)   # hate=1, offensive+neither=0
    df['tr']    = 0   # Davidson does not label target type
    df['ag']    = 0   # Davidson does not label aggression separately
    df = df[['id', 'text', 'hs', 'tr', 'ag']]

    # 80 / 10 / 10 stratified split
    train_df, tmp_df = train_test_split(df, test_size=0.20, stratify=df['hs'], random_state=42)
    dev_df,   test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df['hs'], random_state=42)

    _save(train_df, 'hateval2019_en_train.tsv')
    _save(dev_df,   'hateval2019_en_dev.tsv')
    _save(test_df,  'hateval2019_en_test.tsv')
    _stats(train_df, 'EN train')
    _stats(dev_df,   'EN dev')
    _stats(test_df,  'EN test')


# ─────────────────────────────────────────────────────────────────────────────
# Urdu: Roman Urdu Hate Speech (RUHSOLD)
# ─────────────────────────────────────────────────────────────────────────────

def download_urdu():
    from datasets import load_dataset

    # Try primary source first, fall back to alternative
    sources = [
        ("community-datasets/roman_urdu_hate_speech", None),
        ("devzohaib/roman-urdu-HateSpeech",           None),
    ]

    df = None
    for hf_id, config in sources:
        try:
            print(f"\n[Urdu] Trying {hf_id} …")
            kwargs = dict(trust_remote_code=False)
            if config:
                kwargs['name'] = config
            ds_dict = load_dataset(hf_id, **kwargs)
            print(f"  Splits found: {list(ds_dict.keys())}")
            df = _merge_splits(ds_dict)
            print(f"  Loaded {len(df)} total rows from {hf_id}")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if df is None:
        print("ERROR: Could not load any Urdu dataset. Check your internet connection.")
        sys.exit(1)

    df = _normalise_urdu(df)

    # Use existing splits if available, else create 70/15/15
    split_keys = list(load_dataset(sources[0][0], trust_remote_code=False).keys()) if df is not None else []

    # Re-load to check for pre-existing splits
    ds_dict = load_dataset(sources[0][0], trust_remote_code=False)
    split_map = _map_urdu_splits(ds_dict)

    if split_map:
        for part, key in split_map.items():
            part_df = _normalise_urdu(ds_dict[key].to_pandas())
            _save(part_df, f'hateval2019_ur_{part}.tsv')
            _stats(part_df, f'UR {part}')
    else:
        # No predefined splits — create them
        train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df['hs'], random_state=42)
        dev_df,   test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df['hs'], random_state=42)
        _save(train_df, 'hateval2019_ur_train.tsv')
        _save(dev_df,   'hateval2019_ur_dev.tsv')
        _save(test_df,  'hateval2019_ur_test.tsv')
        _stats(train_df, 'UR train')
        _stats(dev_df,   'UR dev')
        _stats(test_df,  'UR test')


def _merge_splits(ds_dict) -> pd.DataFrame:
    frames = []
    for key in ds_dict:
        frames.append(ds_dict[key].to_pandas())
    return pd.concat(frames, ignore_index=True)


def _map_urdu_splits(ds_dict) -> dict:
    """Map dataset split keys to our part names (train/dev/test)."""
    keys = list(ds_dict.keys())
    mapping = {}
    for part, aliases in [('train', ['train']), ('dev', ['validation', 'dev', 'valid']), ('test', ['test'])]:
        for a in aliases:
            if a in keys:
                mapping[part] = a
                break
    return mapping


def _normalise_urdu(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Column rename variants
    for col in list(df.columns):
        if col in ('sentence', 'tweet', 'comment', 'message', 'content'):
            df.rename(columns={col: 'text'}, inplace=True)
        if col in ('label', 'labels', 'class', 'category', 'hate_speech'):
            df.rename(columns={col: 'raw_label'}, inplace=True)

    if 'text' not in df.columns:
        raise ValueError(f"Cannot find text column. Columns: {df.columns.tolist()}")

    if 'id' not in df.columns:
        df.insert(0, 'id', range(len(df)))

    df['tr'] = 0
    df['ag'] = 0

    # Binary conversion
    if 'raw_label' in df.columns:
        # If all labels are None the split is unlabeled — skip hs column
        non_null = df['raw_label'].dropna()
        if len(non_null) == 0:
            # Unlabeled test set — return without hs column
            return df[['id', 'text', 'tr', 'ag']]
        raw = df['raw_label'].astype(str).str.lower()
        df['hs'] = raw.apply(lambda x: 1 if any(h in x for h in ['hate', '1', 'true', 'yes', 'abusive']) else 0)
    elif 'hs' in df.columns:
        df['hs'] = df['hs'].astype(int)
    else:
        raise ValueError(f"Cannot find label column. Columns: {df.columns.tolist()}")

    return df[['id', 'text', 'hs', 'tr', 'ag']]


def _save(df: pd.DataFrame, filename: str):
    path = os.path.join(RAW_DIR, filename)
    df.to_csv(path, sep='\t', index=False)
    print(f"  Saved -> {path}")


def _stats(df: pd.DataFrame, tag: str):
    n = len(df)
    h = int(df['hs'].sum())
    print(f"  {tag}: {n} samples | hate={h} ({100*h/n:.1f}%) | non-hate={n-h} ({100*(n-h)/n:.1f}%)")


def check_existing() -> tuple[list, list]:
    needed = (
        ['hateval2019_en_train.tsv', 'hateval2019_en_dev.tsv', 'hateval2019_en_test.tsv'] +
        ['hateval2019_ur_train.tsv', 'hateval2019_ur_dev.tsv', 'hateval2019_ur_test.tsv']
    )
    found   = [f for f in needed if os.path.exists(os.path.join(RAW_DIR, f))]
    missing = [f for f in needed if f not in found]
    return found, missing


if __name__ == '__main__':
    os.makedirs(RAW_DIR, exist_ok=True)
    found, missing = check_existing()

    if found:
        print("Already present:")
        for f in found: print(f"  data/raw/{f}")

    en_missing = any('en' in f for f in missing)
    ur_missing = any('ur' in f for f in missing)

    if en_missing:
        download_english()
    if ur_missing:
        download_urdu()

    if not missing:
        print("All data files already present.")
    else:
        print("\nDownload complete. Files saved in data/raw/")
