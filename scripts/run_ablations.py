"""
Orchestrates all backbone ablation experiments (A1):
trains each model on English, then zero-shot evaluates on Urdu.

Usage:
    python scripts/run_ablations.py
"""

import os
import subprocess
import sys
import json

PYTHON = sys.executable

BACKBONE_ABLATIONS = [
    ('mbert',      'configs/mbert_baseline.yaml', 'outputs/checkpoints/mbert_en'),
    ('xlmr_base',  'configs/xlmr_base.yaml',      'outputs/checkpoints/xlmr_base_en'),
    ('xlmr_large', 'configs/xlmr_large.yaml',     'outputs/checkpoints/xlmr_large_en'),
]

EN_TRAIN = 'data/raw/hateval2019_en_train.tsv'
EN_DEV   = 'data/raw/hateval2019_en_dev.tsv'
UR_TEST  = 'data/raw/hateval2019_ur_dev.tsv'   # dev split has labels; test split is unlabeled


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    all_metrics = {}

    for label, config, output_dir in BACKBONE_ABLATIONS:
        print(f"\n{'='*60}\nTRAINING: {label}\n{'='*60}")

        run([PYTHON, 'scripts/train.py',
             '--config', config,
             '--train_en', EN_TRAIN,
             '--dev_en',   EN_DEV,
             '--output_dir', output_dir,
             '--lang', 'en'])

        checkpoint = os.path.join(output_dir, 'best_model')
        tag = f'{label}_zero_shot_ur'

        run([PYTHON, 'scripts/zero_shot_eval.py',
             '--checkpoint', checkpoint,
             '--test_file',  UR_TEST,
             '--lang', 'ur',
             '--output_tag', tag])

        metrics_path = f'outputs/predictions/{tag}_metrics.json'
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                all_metrics[label] = json.load(f)

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'Hate F1':>10} {'Macro F1':>10} {'FPR':>10}")
    print("-" * 70)
    for label, m in all_metrics.items():
        print(f"{label:<20} {m['f1_hate']:>10.4f} {m['f1_macro']:>10.4f} {m['fpr']:>10.4f}")
    print("=" * 70)

    with open('outputs/predictions/ablation_backbone_summary.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("\nSummary → outputs/predictions/ablation_backbone_summary.json")


if __name__ == '__main__':
    main()
