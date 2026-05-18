"""
HatEval SemEval-2019 data loading utilities.

TSV format: id  text  HS  TR  AG
  HS = 1 if hate speech, 0 otherwise
  TR = 1 if target is an individual, 0 if generic group
  AG = 1 if tweet is aggressive
Test files from some releases omit HS/TR/AG columns.
"""

import pandas as pd
from datasets import Dataset


def load_hateval_tsv(filepath: str) -> pd.DataFrame:
    """Load a HatEval TSV file into a DataFrame.

    Handles files with or without a header row and with 2 or 5 columns.
    """
    try:
        df = pd.read_csv(filepath, sep='\t', header=0, dtype=str, quoting=3)
        df.columns = [c.strip().lower() for c in df.columns]
    except Exception:
        df = pd.read_csv(filepath, sep='\t', header=None, dtype=str, quoting=3)

    # If columns are unnamed integers, assign names based on column count
    if not isinstance(df.columns[0], str) or df.columns[0].isdigit():
        if df.shape[1] == 5:
            df.columns = ['id', 'text', 'hs', 'tr', 'ag']
        elif df.shape[1] == 2:
            df.columns = ['id', 'text']
        else:
            raise ValueError(f"Unexpected number of columns ({df.shape[1]}) in {filepath}")

    # Remap alternative header spellings
    rename_map = {'HS': 'hs', 'TR': 'tr', 'AG': 'ag', 'Text': 'text', 'ID': 'id'}
    df.rename(columns=rename_map, inplace=True)

    # Convert label columns to int
    for col in ['hs', 'tr', 'ag']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    df['text'] = df['text'].astype(str).str.strip()
    return df


def df_to_hf_dataset(df: pd.DataFrame, has_labels: bool = True) -> Dataset:
    """Convert a HatEval DataFrame to a HuggingFace Dataset."""
    data: dict = {'text': df['text'].tolist()}
    if has_labels and 'hs' in df.columns:
        data['label'] = df['hs'].tolist()
    if 'tr' in df.columns:
        data['tr'] = df['tr'].tolist()
    if 'ag' in df.columns:
        data['ag'] = df['ag'].tolist()
    return Dataset.from_dict(data)


def load_splits(
    train_path: str | None = None,
    dev_path: str | None = None,
    test_path: str | None = None,
) -> dict[str, Dataset]:
    """Load HatEval dataset splits.

    Returns a dict with keys 'train', 'dev', 'test' for each provided path.
    """
    splits = {}
    for name, path in [('train', train_path), ('dev', dev_path), ('test', test_path)]:
        if path is None:
            continue
        df = load_hateval_tsv(path)
        has_labels = 'hs' in df.columns
        splits[name] = df_to_hf_dataset(df, has_labels=has_labels)
    return splits


def get_class_counts(dataset: Dataset) -> tuple[int, int]:
    """Return (count_non_hate, count_hate) from a labeled dataset."""
    labels = dataset['label']
    n_non_hate = sum(1 for l in labels if l == 0)
    n_hate = sum(1 for l in labels if l == 1)
    return n_non_hate, n_hate


def print_dataset_stats(splits: dict[str, Dataset]) -> None:
    """Print class distribution for each split."""
    for name, ds in splits.items():
        if 'label' not in ds.column_names:
            print(f"  {name}: {len(ds)} samples (no labels)")
            continue
        n_non_hate, n_hate = get_class_counts(ds)
        total = len(ds)
        print(
            f"  {name}: {total} samples | "
            f"non-hate={n_non_hate} ({100*n_non_hate/total:.1f}%) | "
            f"hate={n_hate} ({100*n_hate/total:.1f}%)"
        )
