"""
Full supervised Urdu fine-tuning — upper bound experiment.

Starts from the English-trained XLM-R-large checkpoint and fine-tunes
on the full Urdu training set (7,208 samples). Evaluates on Urdu dev.

Usage:
    python scripts/run_ur_supervised.py
"""

import os
import sys
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKPOINT  = "outputs/checkpoints/xlmr_large_en/best_model"
TRAIN_UR    = "data/raw/hateval2019_ur_train.tsv"
DEV_UR      = "data/raw/hateval2019_ur_dev.tsv"
CONFIG      = "configs/xlmr_large_ur_supervised.yaml"
OUTPUT_DIR  = "outputs/checkpoints/xlmr_large_ur_supervised"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

cmd = [
    sys.executable, "scripts/train.py",
    "--config",           CONFIG,
    "--train_en",         TRAIN_UR,
    "--dev_en",           DEV_UR,
    "--output_dir",       OUTPUT_DIR,
    "--model_name_or_path", CHECKPOINT,
    "--lang",             "ur",
]

print("=" * 60)
print("Full supervised Urdu fine-tuning")
print(f"  Checkpoint : {CHECKPOINT}")
print(f"  Train data : {TRAIN_UR}")
print(f"  Dev data   : {DEV_UR}")
print(f"  Output     : {OUTPUT_DIR}")
print("=" * 60)

subprocess.run(cmd, check=True)
