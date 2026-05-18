# Cross-Lingual Hate Speech Detection: English → Roman Urdu

**AI-600 Deep Learning · LUMS Spring 2026 · Project 6**

**Team:** Muteeullah Baig (25280036) · Rana Muhammad Hamza (25280103)

---

## Overview

This project investigates cross-lingual transfer learning for hate speech detection,
using English as the source language and Roman Urdu as the target. We fine-tune three
multilingual transformers (mBERT, XLM-R-base, XLM-R-large) on English hate speech data
and evaluate zero-shot and few-shot transfer to Roman Urdu, a severely under-resourced
informal script used on Pakistani social media.

### Key Results

| Method | K | Hate F1 | Macro F1 | FPR |
|---|---|---|---|---|
| XLM-R-large (EN dev) | — | 0.509 | 0.740 | 0.030 |
| XLM-R-large (zero-shot UR) | 0 | 0.005 | 0.319 | 0.003 |
| Standard fine-tuning | 64 | 0.715 | 0.528 | 0.777 |
| **Prompt-based fine-tuning** | **64** | **0.787** | **0.745** | **0.355** |
| TF-IDF + FuzzyWuzzy + LR | full | 0.857 | 0.841 | 0.207 |
| XLM-R-large (supervised UR) | full | **0.895** | **0.885** | **0.140** |

---

## Project Structure

```
configs/          Training hyperparameter YAML files
notebooks/        EDA, results, and fairness analysis notebooks
outputs/
  figures/        Plots (class distribution, model comparison, fairness)
  predictions/    Per-model metrics JSON + few-shot summary CSVs
report/
  mid_report/     LaTeX source for mid-report and final report
scripts/          Training and evaluation scripts
src/              Reusable Python modules (data_loader, evaluator, utils)
requirements.txt
```

---

## Setup

```bash
# Create and activate environment (Python 3.10+, CUDA 12.x)
conda create -n hateval python=3.10
conda activate hateval

# Install dependencies
pip install -r requirements.txt

# Prevent CUDA OOM fragmentation (Windows)
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## Data

### Download

**English source data** (Davidson et al. 2017 — binarised hate vs. non-hate):
```python
from datasets import load_dataset
ds = load_dataset("tdavidson/hate_speech_offensive")
```

**Roman Urdu target data:**
```python
from datasets import load_dataset
ds = load_dataset("community-datasets/roman_urdu_hate_speech")
```

Place the exported TSV files in `data/raw/` with the following names:
```
data/raw/hateval2019_en_train.tsv    # 21,159 samples
data/raw/hateval2019_en_dev.tsv      #  2,612 samples
data/raw/hateval2019_ur_train.tsv    #  7,208 samples
data/raw/hateval2019_ur_dev.tsv      #    800 samples
```

TSV format: `id \t text \t HS \t TR \t AG` (HS=hate label, 0/1).

### Preprocess

```bash
python scripts/preprocess.py
```

Outputs JSON files and few-shot stratified splits to `data/processed/`.

---

## Running Experiments

### 1. Train on English (backbone ablation)

```bash
# mBERT baseline
python scripts/train.py --config configs/mbert_baseline.yaml

# XLM-R-base
python scripts/train.py --config configs/xlmr_base.yaml

# XLM-R-large (primary model)
python scripts/train.py --config configs/xlmr_large.yaml
```

### 2. Zero-shot evaluation on Roman Urdu

```bash
python scripts/zero_shot_eval.py \
  --checkpoint outputs/checkpoints/xlmr_large_en/best_model \
  --test_file  data/raw/hateval2019_ur_dev.tsv \
  --output_tag xlmr_large_zero_shot_ur
```

### 3. Standard few-shot fine-tuning (K = 8 / 16 / 32 / 64)

```bash
python scripts/few_shot_train.py \
  --checkpoint outputs/checkpoints/xlmr_large_en/best_model \
  --config     configs/few_shot.yaml
```

Results saved to `outputs/predictions/few_shot_summary.csv`.

### 4. Prompt-based few-shot fine-tuning

```bash
python scripts/prompt_finetune.py
```

Uses prefix-prompt template `"{tweet} This text is [MASK]."` with an MLM
verbalizer (Ullah et al. 2025). Results saved to
`outputs/predictions/prompt_finetune_summary.csv`.

### 5. Classical ML baselines (TF-IDF + FuzzyWuzzy)

```bash
python scripts/classical_baseline.py
```

Results saved to `outputs/predictions/classical_baseline_results.json`.

### 6. Supervised upper bound (full Roman Urdu fine-tuning)

```bash
python scripts/run_ur_supervised.py
```

### 7. Generate figures

```bash
python scripts/generate_figures.py
```

Or run notebooks in order:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preliminary_results.ipynb`
3. `notebooks/03_fairness_analysis.ipynb`

---

## Ablation Studies

| ID | Variable | How to run |
|---|---|---|
| A1 | Model backbone (mBERT / XLM-R-base / XLM-R-large) | `scripts/train.py` with different configs |
| A2 | K-shot data ($K \in \{8,16,32,64,\text{full}\}$) | `scripts/few_shot_train.py` |
| A3 | Training method (standard FT vs prompt-based vs classical ML) | separate scripts above |
| A4 | Class weighting | `class_weights` field in YAML configs |

---

## Hardware

All experiments run on an NVIDIA RTX 4070 GPU (8.6 GB VRAM) with CUDA 12.1.
XLM-R-large training uses `gradient_checkpointing=True` and fp16.
Estimated runtimes: mBERT ~10 min, XLM-R-base ~12 min, XLM-R-large ~35 min (5 epochs).

---

## Report

LaTeX source for the final report is in `report/mid_report/mid_report.tex`.
Compile with `pdflatex` using the ICML 2025 style files (download from
https://icml.cc/Conferences/2025/StyleAuthorInstructions).

---

## References

- Davidson et al. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. ICWSM.
- Conneau et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL.
- Rizwan et al. (2020). Hate-Speech and Offensive Language Detection in Roman Urdu. EMNLP.
- Ullah et al. (2024). Towards Unveiling the Potential of Fuzzy Values as Features. ICONIP.
- Ullah et al. (2025). Prompt-based fine-tuning with multilingual transformers. Scientific Reports.

---

## AI Tools Statement

Claude Sonnet 4.6 (Anthropic) assisted with code generation, debugging, and report
drafting. All experimental results are produced by our code running on our hardware.
All results were verified by the authors.
