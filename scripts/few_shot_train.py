"""
Few-shot cross-lingual fine-tuning (EN → Urdu).

Loads an EN-trained checkpoint, fine-tunes briefly on K Urdu examples,
then evaluates on the Urdu test set.

Repeats across multiple seeds and K values; saves a summary CSV.

Usage:
    python scripts/few_shot_train.py \
        --checkpoint outputs/checkpoints/xlmr_large_en/best_model \
        --ur_test    data/raw/hateval2019_ur_test.tsv \
        --shot_dir   data/processed/few_shot_samples \
        --config     configs/few_shot.yaml
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_hateval_tsv, df_to_hf_dataset
from src.evaluator import compute_metrics, save_predictions, full_evaluation
from src.utils import set_seed, load_config, setup_logging, get_device, model_shortname
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

logger = setup_logging()


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        weights = torch.tensor(self.class_weights, dtype=torch.float32, device=outputs.logits.device)
        loss = torch.nn.CrossEntropyLoss(weight=weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def tokenize(dataset, tokenizer, max_length):
    def _tok(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)
    tok = dataset.map(_tok, batched=True, remove_columns=['text'])
    tok.set_format('torch')
    return tok


def load_few_shot_split(json_path: str):
    """Load a K-shot JSON split and return a HuggingFace Dataset."""
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    from datasets import Dataset
    return Dataset.from_dict({
        'text':  [r['text'] for r in records],
        'label': [int(r['hs']) for r in records],
    })


def run_few_shot(
    checkpoint: str,
    few_shot_path: str,
    test_dataset,
    test_texts: list[str],
    test_labels: list[int],
    cfg: dict,
    output_dir: str,
    tag: str,
) -> dict:
    """Fine-tune on K-shot data and evaluate on test set. Returns metrics dict."""
    set_seed(cfg.get('seed', 42))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model     = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    if cfg.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()

    max_length    = cfg.get('max_length', 128)
    class_weights = cfg.get('class_weights', [1.0, 2.5])

    train_tok = tokenize(few_shot_path, tokenizer, max_length)
    train_tok = train_tok.rename_column('label', 'labels')
    test_tok  = tokenize(test_dataset,  tokenizer, max_length)
    test_tok  = test_tok.rename_column('label', 'labels')

    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get('num_train_epochs', 20),
        per_device_train_batch_size=cfg.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=cfg.get('per_device_eval_batch_size', 16),
        gradient_accumulation_steps=cfg.get('gradient_accumulation_steps', 1),
        learning_rate=float(cfg.get('learning_rate', 5e-6)),
        warmup_ratio=cfg.get('warmup_ratio', 0.1),
        weight_decay=cfg.get('weight_decay', 0.01),
        fp16=cfg.get('fp16', True),
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_hate',
        greater_is_better=True,
        save_total_limit=1,
        report_to='none',
        logging_steps=10,
        dataloader_num_workers=0,
        seed=cfg.get('seed', 42),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    preds_out   = trainer.predict(test_tok)
    logits      = preds_out.predictions
    preds       = np.argmax(logits, axis=-1)

    metrics = full_evaluation(test_labels, preds, logits, verbose=False)

    pred_path = os.path.join('outputs/predictions', f'{tag}_predictions.json')
    save_predictions(test_texts, test_labels, preds, logits, pred_path)

    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ur_test',    required=True)
    p.add_argument('--shot_dir',   default='data/processed/few_shot_samples')
    p.add_argument('--config',     default='configs/few_shot.yaml')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    get_device()

    shot_counts = cfg.get('shot_counts', [8, 16, 32, 64])
    num_seeds   = cfg.get('num_seeds', 3)

    # Load Urdu test set once (reused across all experiments)
    test_df     = load_hateval_tsv(args.ur_test)
    test_ds     = df_to_hf_dataset(test_df)
    test_texts  = test_df['text'].tolist()
    test_labels = test_df['hs'].tolist()

    summary_rows = []

    for k in shot_counts:
        seed_metrics = []
        for seed in range(num_seeds):
            split_path = os.path.join(args.shot_dir, f'ur_{k}shot_seed{seed}.json')
            if not os.path.exists(split_path):
                logger.warning(f"Few-shot split not found: {split_path} — run preprocess.py first")
                continue

            tag        = f'fewshot_ur_{k}shot_seed{seed}'
            output_dir = f'outputs/checkpoints/fewshot_ur_{k}shot_seed{seed}'

            logger.info(f"K={k}, seed={seed} — {split_path}")
            train_ds = load_few_shot_split(split_path)

            m = run_few_shot(
                checkpoint=args.checkpoint,
                few_shot_path=train_ds,
                test_dataset=test_ds,
                test_texts=test_texts,
                test_labels=test_labels,
                cfg={**cfg, 'seed': seed},
                output_dir=output_dir,
                tag=tag,
            )
            seed_metrics.append(m)
            logger.info(f"  K={k} seed={seed} -> Hate F1={m['f1_hate']:.4f} | Macro F1={m['f1_macro']:.4f}")

        if seed_metrics:
            f1_hate_vals  = [m['f1_hate']  for m in seed_metrics]
            f1_macro_vals = [m['f1_macro'] for m in seed_metrics]
            fpr_vals      = [m['fpr']      for m in seed_metrics]
            row = {
                'k': k,
                'f1_hate_mean':  np.mean(f1_hate_vals),
                'f1_hate_std':   np.std(f1_hate_vals),
                'f1_macro_mean': np.mean(f1_macro_vals),
                'f1_macro_std':  np.std(f1_macro_vals),
                'fpr_mean':      np.mean(fpr_vals),
            }
            summary_rows.append(row)
            logger.info(f"K={k} summary -> Hate F1={row['f1_hate_mean']:.4f} +/- {row['f1_hate_std']:.4f}")

    # Save summary CSV
    summary_path = 'outputs/predictions/few_shot_summary.csv'
    if summary_rows:
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Summary -> {summary_path}")

    logger.info("Few-shot experiments complete.")


if __name__ == '__main__':
    main()
