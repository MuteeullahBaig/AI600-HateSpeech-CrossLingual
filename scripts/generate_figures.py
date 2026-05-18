"""
Generate all figures for the mid-report.
Run after training is complete. Figures saved to outputs/figures/.

Usage:
    python scripts/generate_figures.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_hateval_tsv

os.makedirs('outputs/figures', exist_ok=True)

COLORS = {'mbert': '#4C72B0', 'xlmr_base': '#DD8452', 'xlmr_large': '#55A868'}
GRAY = '#888888'

# ---------------------------------------------------------------------------
# Figure 1: Dataset class distribution (EN + UR)
# ---------------------------------------------------------------------------

def fig_class_distribution():
    en_train = load_hateval_tsv('data/raw/hateval2019_en_train.tsv')
    en_dev   = load_hateval_tsv('data/raw/hateval2019_en_dev.tsv')
    ur_train = load_hateval_tsv('data/raw/hateval2019_ur_train.tsv')
    ur_dev   = load_hateval_tsv('data/raw/hateval2019_ur_dev.tsv')

    splits = {
        'EN Train\n(n=21,159)': en_train,
        'EN Dev\n(n=2,612)':   en_dev,
        'UR Train\n(n=7,208)': ur_train,
        'UR Dev\n(n=800)':     ur_dev,
    }

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(splits))
    w = 0.35

    hate_pcts    = [100 * (df['hs'].sum() / len(df)) for df in splits.values()]
    nonhate_pcts = [100 - p for p in hate_pcts]

    bars_nh = ax.bar(x - w/2, nonhate_pcts, w, label='Non-hate', color='#4C72B0', alpha=0.85)
    bars_h  = ax.bar(x + w/2, hate_pcts,    w, label='Hate',     color='#C44E52', alpha=0.85)

    for bar, pct in zip(bars_nh, nonhate_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, pct in zip(bars_h, hate_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(splits.keys(), fontsize=10)
    ax.set_ylabel('Percentage of samples (%)', fontsize=11)
    ax.set_title('Class Distribution Across Dataset Splits\n(English and Roman Urdu)', fontsize=12, pad=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = 'outputs/figures/fig1_class_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# Figure 2: EN dev performance comparison (backbone ablation)
# ---------------------------------------------------------------------------

def fig_en_dev_comparison(results):
    """results: list of (model_label, metrics_dict)"""
    models   = [r[0] for r in results]
    f1_hate  = [r[1]['f1_hate']  for r in results]
    f1_macro = [r[1]['f1_macro'] for r in results]
    fpr      = [r[1]['fpr']      for r in results]

    x = np.arange(len(models))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w,   f1_hate,  w, label='Hate F1',  color='#C44E52', alpha=0.85)
    b2 = ax.bar(x,       f1_macro, w, label='Macro F1', color='#4C72B0', alpha=0.85)
    b3 = ax.bar(x + w,   fpr,      w, label='FPR',      color=GRAY,      alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('English Dev Set Performance — Backbone Comparison', fontsize=12, pad=10)
    ax.set_ylim(0, 0.95)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = 'outputs/figures/fig2_en_dev_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# Figure 3: Zero-shot Urdu performance comparison
# ---------------------------------------------------------------------------

def fig_zeroshot_comparison(zs_results):
    """zs_results: list of (model_label, metrics_dict)"""
    models   = [r[0] for r in zs_results]
    f1_hate  = [r[1].get('f1_hate',  0) for r in zs_results]
    f1_macro = [r[1].get('f1_macro', 0) for r in zs_results]
    fpr      = [r[1].get('fpr',      0) for r in zs_results]

    x = np.arange(len(models))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w, f1_hate,  w, label='Hate F1',  color='#C44E52', alpha=0.85)
    b2 = ax.bar(x,     f1_macro, w, label='Macro F1', color='#4C72B0', alpha=0.85)
    b3 = ax.bar(x + w, fpr,      w, label='FPR',      color=GRAY,      alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Zero-Shot Transfer to Roman Urdu — Backbone Comparison', fontsize=12, pad=10)
    ax.set_ylim(0, 0.95)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = 'outputs/figures/fig3_zeroshot_urdu.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# Figure 4: Fairness analysis
# ---------------------------------------------------------------------------

def fig_fairness(predictions_path: str):
    """Bar chart of group-conditional FPR and Hate F1 from a predictions file."""
    from src.evaluator import fairness_analysis
    results = fairness_analysis(predictions_path)

    groups   = [g for g in results if g != 'overall']
    fpr_vals = [results[g]['fpr']     if results[g]['fpr']     is not None else 0 for g in groups]
    f1_vals  = [results[g]['f1_hate'] if results[g]['f1_hate'] is not None else 0 for g in groups]
    counts   = [results[g]['count'] for g in groups]

    groups.append('overall')
    fpr_vals.append(results['overall']['fpr'])
    f1_vals.append(results['overall']['f1_hate'])
    counts.append(results['overall']['count'])

    labels = [f"{g.capitalize()}\n(n={c})" for g, c in zip(groups, counts)]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, fpr_vals, w, label='FPR (lower=better)',      color='#C44E52', alpha=0.85)
    b2 = ax.bar(x + w/2, f1_vals,  w, label='Hate F1 (higher=better)', color='#4C72B0', alpha=0.85)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(results['overall']['fpr'], color='#C44E52', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Fairness Analysis: Group-Conditional FPR and Hate F1\n(Roman Urdu zero-shot predictions)', fontsize=12, pad=10)
    ax.set_ylim(0, max(max(fpr_vals), max(f1_vals)) * 1.25 + 0.05)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = 'outputs/figures/fig4_fairness_analysis.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

if __name__ == '__main__':
    print('Generating figures...')

    # Fig 1: Always generate (uses raw data)
    fig_class_distribution()

    # Fig 2: EN dev comparison (use what's available)
    en_results = []
    m = load_json('outputs/checkpoints/mbert_en/metrics_en_dev.json')
    if m: en_results.append(('mBERT', m))
    m = load_json('outputs/checkpoints/xlmr_base_en/metrics_en_dev.json')
    if m: en_results.append(('XLM-R-base', m))
    m = load_json('outputs/checkpoints/xlmr_large_en/metrics_en_dev.json')
    if m: en_results.append(('XLM-R-large', m))

    if en_results:
        fig_en_dev_comparison(en_results)

    # Fig 3: Zero-shot Urdu comparison
    zs_results = []
    m = load_json('outputs/predictions/mbert_zero_shot_ur_metrics.json')
    if m: zs_results.append(('mBERT\n(zero-shot)', m))
    m = load_json('outputs/predictions/xlmr_base_zero_shot_ur_metrics.json')
    if m: zs_results.append(('XLM-R-base\n(zero-shot)', m))
    m = load_json('outputs/predictions/xlmr_large_zero_shot_ur_metrics.json')
    if m: zs_results.append(('XLM-R-large\n(zero-shot)', m))

    if zs_results:
        fig_zeroshot_comparison(zs_results)

    # Fig 4: Fairness analysis (group-conditional FPR)
    pred_file = 'outputs/predictions/xlmr_large_zero_shot_ur_predictions.json'
    if not os.path.exists(pred_file):
        pred_file = 'outputs/predictions/xlmr_base_zero_shot_ur_predictions.json'
    if os.path.exists(pred_file):
        fig_fairness(pred_file)

    print('Done.')
