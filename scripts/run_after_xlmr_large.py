"""
Run after XLM-R-large training completes.
Chains: zero-shot Urdu eval → few-shot experiments → figure generation.

Usage:
    python scripts/run_after_xlmr_large.py
"""

import os, sys, subprocess

PYTHON = r"C:\Users\Matee\miniconda3\envs\gpu_env\python.exe"
ENV    = {**os.environ, "PYTHONIOENCODING": "utf-8",
          "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}

def run(cmd, log_path=None):
    print(f"\n>>> {' '.join(cmd)}")
    if log_path:
        with open(log_path, 'w', encoding='utf-8') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV,
                                    cwd=r"D:\Deep Learning Project")
    else:
        result = subprocess.run(cmd, env=ENV, cwd=r"D:\Deep Learning Project")
    if result.returncode != 0:
        print(f"ERROR: exit code {result.returncode}")
    return result.returncode == 0


if __name__ == '__main__':
    checkpoint = r"outputs\checkpoints\xlmr_large_en\best_model"

    # 1. Zero-shot eval on Urdu
    ok = run([PYTHON, "-u", "scripts/zero_shot_eval.py",
              "--checkpoint", checkpoint,
              "--test_file", "data/raw/hateval2019_ur_dev.tsv",
              "--lang", "ur",
              "--output_tag", "xlmr_large_zero_shot_ur"],
             r"C:\Users\Matee\AppData\Local\Temp\xlmr_large_zs.txt")
    if not ok:
        print("Zero-shot eval failed — check log"); sys.exit(1)

    print("\n=== Zero-shot complete ===")
    try:
        with open(r"C:\Users\Matee\AppData\Local\Temp\xlmr_large_zs.txt",
                  encoding='utf-8', errors='replace') as f:
            print(f.read().encode('ascii', errors='replace').decode('ascii'))
    except Exception as e:
        print(f"(Could not print log: {e})")

    # 2. Few-shot experiments
    ok = run([PYTHON, "-u", "scripts/few_shot_train.py",
              "--checkpoint", checkpoint,
              "--ur_test",    "data/raw/hateval2019_ur_dev.tsv",
              "--shot_dir",   "data/processed/few_shot_samples",
              "--config",     "configs/few_shot.yaml"],
             r"C:\Users\Matee\AppData\Local\Temp\few_shot.txt")
    if not ok:
        print("Few-shot training failed — check log")

    # 3. Regenerate figures with complete results
    run([PYTHON, "-u", "scripts/generate_figures.py"])

    # 4. Patch mid-report with final numbers
    run([PYTHON, "-u", "scripts/update_report.py"])

    print("\nAll post-training steps complete. mid_report.md is fully updated.")
