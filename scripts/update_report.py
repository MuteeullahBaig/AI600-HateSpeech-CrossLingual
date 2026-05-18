"""
Patch mid_report.md with final XLM-R-large and few-shot results.
Run after run_after_xlmr_large.py completes.
"""

import os, sys, json, csv, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return None


def fmt(v, pct=False):
    if v is None:
        return '—'
    return f'{v:.3f}'


def main():
    report_path = 'mid_report.md'
    with open(report_path, encoding='utf-8') as f:
        text = f.read()

    # ---- XLM-R-large EN dev ----
    m = load_json('outputs/checkpoints/xlmr_large_en/metrics_en_dev.json')
    if m:
        row = f'| XLM-R-large | {fmt(m["f1_hate"])} | {fmt(m["f1_non_hate"])} | {fmt(m["f1_macro"])} | {fmt(m["fpr"])} |'
        text = text.replace(
            '| XLM-R-large | — (training) | — | — | — |',
            row
        )
        print(f'EN dev XLM-R-large: hate F1={m["f1_hate"]:.3f}')

    # ---- XLM-R-large zero-shot UR ----
    m = load_json('outputs/predictions/xlmr_large_zero_shot_ur_metrics.json')
    if m:
        row = f'| XLM-R-large (zero-shot) | {fmt(m["f1_hate"])} | {fmt(m["f1_non_hate"])} | {fmt(m["f1_macro"])} | {fmt(m["fpr"])} |'
        text = text.replace(
            '| XLM-R-large (zero-shot) | — (training) | — | — | — |',
            row
        )
        print(f'ZS Urdu XLM-R-large: hate F1={m["f1_hate"]:.3f}')

    # ---- Few-shot summary ----
    few_shot_csv = 'outputs/predictions/few_shot_summary.csv'
    if os.path.exists(few_shot_csv):
        with open(few_shot_csv, newline='') as f:
            rows = list(csv.DictReader(f))

        few_shot_table = '\n**Table 6: Few-Shot Transfer Results (XLM-R-large, mean ± std over 3 seeds)**\n\n'
        few_shot_table += '| K-shot | Hate F1 | Macro F1 | FPR |\n'
        few_shot_table += '|---|---|---|---|\n'
        few_shot_table += '| 0 (zero-shot) | see Table 4 | see Table 4 | see Table 4 |\n'
        for r in rows:
            k   = int(float(r['k']))
            f1h = f'{float(r["f1_hate_mean"]):.3f} ± {float(r["f1_hate_std"]):.3f}'
            f1m = f'{float(r["f1_macro_mean"]):.3f} ± {float(r["f1_macro_std"]):.3f}'
            fpr = f'{float(r["fpr_mean"]):.3f}'
            few_shot_table += f'| {k} | {f1h} | {f1m} | {fpr} |\n'

        # Insert before Remaining Work section
        text = text.replace(
            '## 6. Remaining Work',
            few_shot_table + '\n## 6. Remaining Work'
        )
        print('Few-shot table added.')

    # ---- Remove training note ----
    text = text.replace(
        '\n*[NOTE: Rows marked "— (training)" will be filled in once XLM-R-large training completes (~May 7, 11 PM). The report will be updated before final submission.]*',
        ''
    )

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Report updated: {report_path}')


if __name__ == '__main__':
    main()
