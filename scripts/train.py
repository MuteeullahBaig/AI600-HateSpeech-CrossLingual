"""
Main fine-tuning script for hate speech detection.

Usage:
    python scripts/train.py --config configs/xlmr_large.yaml \
                            --train_en data/raw/hateval2019_en_train.tsv \
                            --dev_en   data/raw/hateval2019_en_dev.tsv \
                            --lang en  \
                            --output_dir outputs/checkpoints/xlmr_large_en

For Spanish supervised upper-bound:
    python scripts/train.py --config configs/xlmr_large.yaml \
                            --train_en data/raw/hateval2019_es_train.tsv \
                            --dev_en   data/raw/hateval2019_es_dev.tsv \
                            --lang es  \
                            --output_dir outputs/checkpoints/xlmr_large_es_full
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

# Make src importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_hateval_tsv, df_to_hf_dataset, print_dataset_stats
from src.evaluator import compute_metrics, save_predictions, full_evaluation
from src.utils import set_seed, load_config, setup_logging, model_shortname
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

logger = setup_logging()

# ---------------------------------------------------------------------------
# Weighted Trainer (handles class imbalance)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize_dataset(dataset, tokenizer, max_length: int):
    def _tokenize(batch):
        return tokenizer(
            batch['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )
    return dataset.map(_tokenize, batched=True, remove_columns=['text'])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',      required=True,  help='Path to YAML config file')
    p.add_argument('--train_en',    required=True,  help='Training TSV file')
    p.add_argument('--dev_en',      required=True,  help='Dev/validation TSV file')
    p.add_argument('--output_dir',  required=True,  help='Directory for checkpoints and predictions')
    p.add_argument('--lang',        default='en',   help='Language tag (en/es), used in prediction filenames')
    p.add_argument('--model_name_or_path', default=None,
                   help='Override model_name in config (use to load a fine-tuned checkpoint)')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    seed = cfg.get('seed', 42)
    set_seed(seed)

    model_name = args.model_name_or_path or cfg['model_name']
    max_length  = cfg.get('max_length', 128)
    class_weights = cfg.get('class_weights', [1.0, 2.5])
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('outputs/predictions', exist_ok=True)

    logger.info(f"Model : {model_name}")
    logger.info(f"Output: {args.output_dir}")

    # ---- Load data ----
    logger.info("Loading data …")
    train_df = load_hateval_tsv(args.train_en)
    dev_df   = load_hateval_tsv(args.dev_en)

    train_ds = df_to_hf_dataset(train_df)
    dev_ds   = df_to_hf_dataset(dev_df)

    print("Training data:")
    print_dataset_stats({'train': train_ds, 'dev': dev_ds})

    # ---- Tokenise ----
    logger.info(f"Tokenising (max_length={max_length}) …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_tok = tokenize_dataset(train_ds, tokenizer, max_length)
    dev_tok   = tokenize_dataset(dev_ds,   tokenizer, max_length)
    train_tok = train_tok.rename_column('label', 'labels')
    dev_tok   = dev_tok.rename_column('label', 'labels')
    train_tok.set_format('torch')
    dev_tok.set_format('torch')

    # ---- Model ----
    logger.info("Loading model …")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if cfg.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=cfg.get('num_train_epochs', 5),
        per_device_train_batch_size=cfg.get('per_device_train_batch_size', 16),
        per_device_eval_batch_size=cfg.get('per_device_eval_batch_size', 32),
        gradient_accumulation_steps=cfg.get('gradient_accumulation_steps', 1),
        learning_rate=float(cfg.get('learning_rate', 2e-5)),
        warmup_ratio=cfg.get('warmup_ratio', 0.1),
        weight_decay=cfg.get('weight_decay', 0.01),
        fp16=cfg.get('fp16', True),
        eval_strategy=cfg.get('eval_strategy', cfg.get('evaluation_strategy', 'epoch')),
        save_strategy=cfg.get('save_strategy', 'epoch'),
        load_best_model_at_end=cfg.get('load_best_model_at_end', True),
        metric_for_best_model=cfg.get('metric_for_best_model', 'f1_hate'),
        greater_is_better=cfg.get('greater_is_better', True),
        save_total_limit=cfg.get('save_total_limit', 2),
        report_to=cfg.get('report_to', 'none'),
        seed=seed,
        logging_steps=50,
        dataloader_num_workers=0,   # avoid issues on Windows
    )

    # ---- Train ----
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Training …")
    trainer.train()

    # ---- Evaluate on dev ----
    logger.info("Evaluating on dev set …")
    preds_output = trainer.predict(dev_tok)
    logits       = preds_output.predictions
    preds        = np.argmax(logits, axis=-1)
    true_labels  = preds_output.label_ids

    metrics = full_evaluation(true_labels, preds, logits, verbose=True)
    metrics_path = os.path.join(args.output_dir, f'metrics_{args.lang}_dev.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    # Save predictions for fairness analysis
    pred_path = f'outputs/predictions/{model_shortname(model_name)}_{args.lang}_dev.json'
    save_predictions(dev_df['text'].tolist(), true_labels, preds, logits, pred_path)

    # Save the best model and tokenizer to a clean location
    best_dir = os.path.join(args.output_dir, 'best_model')
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info(f"Best model saved → {best_dir}")
    logger.info("Done.")


if __name__ == '__main__':
    main()
