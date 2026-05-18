"""
Zero-shot cross-lingual evaluation.

Loads an EN-trained checkpoint and evaluates it directly on a target-language
test set without any target-language fine-tuning.

Usage:
    python scripts/zero_shot_eval.py \
        --checkpoint outputs/checkpoints/xlmr_large_en/best_model \
        --test_file  data/raw/hateval2019_es_test.tsv \
        --lang es \
        --output_tag xlmr_large_zero_shot_es
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from datasets import Dataset

from src.data_loader import load_hateval_tsv, df_to_hf_dataset
from src.evaluator import compute_metrics, save_predictions, full_evaluation
from src.utils import setup_logging, get_device
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = setup_logging()


def tokenize_for_inference(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tok(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)
    cols_to_remove = [c for c in dataset.column_names if c not in ('label', 'tr', 'ag')]
    tok = dataset.map(_tok, batched=True, remove_columns=['text'])
    tok.set_format('torch')
    return tok


@torch.no_grad()
def run_inference(model, dataset: Dataset, batch_size: int = 32, device=None) -> np.ndarray:
    """Run model inference and return logits array of shape (N, 2)."""
    if device is None:
        device = get_device()
    model.eval().to(device)

    # Build a DataLoader over only the tokenizer columns
    input_cols = ['input_ids', 'attention_mask']
    if 'token_type_ids' in dataset.column_names:
        input_cols.append('token_type_ids')

    loader = DataLoader(
        dataset.select_columns(input_cols),
        batch_size=batch_size,
        shuffle=False,
    )

    all_logits = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        all_logits.append(out.logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True,  help='Path to fine-tuned model directory')
    p.add_argument('--test_file',   required=True,  help='Path to target-language TSV')
    p.add_argument('--lang',        default='es',   help='Language tag for output filenames')
    p.add_argument('--output_tag',  default=None,   help='Identifier used in output filenames')
    p.add_argument('--max_length',  type=int, default=128)
    p.add_argument('--batch_size',  type=int, default=32)
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    os.makedirs('outputs/predictions', exist_ok=True)

    tag = args.output_tag or f'zeroshot_{args.lang}'

    logger.info(f"Checkpoint : {args.checkpoint}")
    logger.info(f"Test file  : {args.test_file}")

    # ---- Load data ----
    df        = load_hateval_tsv(args.test_file)
    has_labels = 'hs' in df.columns
    dataset   = df_to_hf_dataset(df, has_labels=has_labels)
    texts     = df['text'].tolist()

    # ---- Load model ----
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model     = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)

    # ---- Tokenise ----
    tok_ds = tokenize_for_inference(dataset, tokenizer, args.max_length)

    # ---- Inference ----
    logger.info("Running inference …")
    logits = run_inference(model, tok_ds, batch_size=args.batch_size, device=device)
    preds  = np.argmax(logits, axis=-1)

    # ---- Metrics (if labels available) ----
    if has_labels:
        true_labels = dataset['label']
        metrics = full_evaluation(true_labels, preds, logits, verbose=True)
        metrics['checkpoint'] = args.checkpoint
        metrics['test_file']  = args.test_file
        metrics['lang']       = args.lang

        metrics_path = f'outputs/predictions/{tag}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics → {metrics_path}")

        pred_path = f'outputs/predictions/{tag}_predictions.json'
        save_predictions(texts, true_labels, preds, logits, pred_path)
    else:
        logger.info("No ground-truth labels found — saving predictions only.")
        true_labels = [-1] * len(preds)
        pred_path = f'outputs/predictions/{tag}_predictions.json'
        save_predictions(texts, true_labels, preds, logits, pred_path)

    logger.info("Done.")


if __name__ == '__main__':
    main()
