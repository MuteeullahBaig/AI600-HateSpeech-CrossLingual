"""
Evaluation utilities: metrics, prediction saving, fairness analysis.
"""

import json
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _fpr(labels, preds) -> float:
    """False Positive Rate = FP / (FP + TN)."""
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# HuggingFace Trainer compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """Compute metrics for HuggingFace Trainer.

    Returns a dict of scalar metrics. The Trainer expects this signature.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        'f1_hate':      float(f1_score(labels, preds, pos_label=1, average='binary',  zero_division=0)),
        'f1_non_hate':  float(f1_score(labels, preds, pos_label=0, average='binary',  zero_division=0)),
        'f1_macro':     float(f1_score(labels, preds, average='macro',    zero_division=0)),
        'f1_weighted':  float(f1_score(labels, preds, average='weighted', zero_division=0)),
        'accuracy':     float((preds == labels).mean()),
        'precision_hate': float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        'recall_hate':    float(recall_score(labels, preds,    pos_label=1, zero_division=0)),
        'fpr':          _fpr(labels, preds),
    }


# ---------------------------------------------------------------------------
# Full evaluation (used outside Trainer, e.g. zero-shot eval)
# ---------------------------------------------------------------------------

def full_evaluation(
    labels: list[int],
    preds: list[int],
    logits: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Compute comprehensive metrics and optionally print a report."""
    labels = list(labels)
    preds  = list(preds)

    metrics = {
        'f1_hate':      float(f1_score(labels, preds, pos_label=1, average='binary',  zero_division=0)),
        'f1_non_hate':  float(f1_score(labels, preds, pos_label=0, average='binary',  zero_division=0)),
        'f1_macro':     float(f1_score(labels, preds, average='macro',    zero_division=0)),
        'f1_weighted':  float(f1_score(labels, preds, average='weighted', zero_division=0)),
        'accuracy':     float((np.array(preds) == np.array(labels)).mean()),
        'precision_hate': float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        'recall_hate':    float(recall_score(labels, preds,    pos_label=1, zero_division=0)),
        'fpr':          _fpr(labels, preds),
    }

    if verbose:
        print("\n" + "=" * 60)
        print(classification_report(labels, preds, target_names=['non-hate', 'hate'], zero_division=0))
        print(f"  Macro F1      : {metrics['f1_macro']:.4f}")
        print(f"  Hate F1       : {metrics['f1_hate']:.4f}")
        print(f"  FPR           : {metrics['fpr']:.4f}")
        print("=" * 60 + "\n")

    return metrics


# ---------------------------------------------------------------------------
# Prediction saving
# ---------------------------------------------------------------------------

def save_predictions(
    texts: list[str],
    true_labels: list[int],
    predicted_labels: list[int],
    logits: np.ndarray,
    output_path: str,
) -> list[dict]:
    """Save predictions with confidence scores to a JSON file."""
    probs = _softmax(logits)
    records = []
    for text, true, pred, prob in zip(texts, true_labels, predicted_labels, probs):
        records.append({
            'text':             str(text),
            'true_label':       int(true),
            'predicted_label':  int(pred),
            'confidence_hate':  float(prob[1]),
            'confidence_non_hate': float(prob[0]),
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} predictions → {output_path}")
    return records


# ---------------------------------------------------------------------------
# Fairness analysis
# ---------------------------------------------------------------------------

GENDER_TERMS = {
    # English
    'woman', 'women', 'female', 'females', 'girl', 'girls',
    'she', 'her', 'hers', 'lady', 'ladies', 'feminist', 'feminism',
    # Roman Urdu transliterations
    'aurat', 'larki', 'larkiyan', 'khatoon', 'begum', 'bibi',
    'feminist', 'feminism', 'ladki', 'ladkiyan',
}

RELIGION_TERMS = {
    # Religious group mentions common in Urdu hate speech
    'muslim', 'muslims', 'islam', 'islamic', 'hindu', 'hindus',
    'christian', 'christians', 'kafir', 'kafirs', 'mushrik',
    'qadiani', 'ahmadi', 'shia', 'sunni',
}

ETHNIC_TERMS = {
    # Ethnic/regional group mentions
    'punjabi', 'sindhi', 'pashtun', 'balochi', 'mohajir',
    'indian', 'indians', 'bengali', 'afghan',
}

GROUP_LEXICONS = {
    'gender':   GENDER_TERMS,
    'religion': RELIGION_TERMS,
    'ethnicity': ETHNIC_TERMS,
}


def fairness_analysis(predictions_path: str) -> dict:
    """Compute group-conditional FPR and F1 from a saved predictions JSON file.

    Returns a dict keyed by group name with count, fpr, and f1_hate.
    """
    with open(predictions_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    results = {}
    for group_name, terms in GROUP_LEXICONS.items():
        group_records = [
            r for r in records
            if set(r['text'].lower().split()) & terms
        ]

        if not group_records:
            results[group_name] = {'count': 0, 'fpr': None, 'f1_hate': None}
            continue

        true = [r['true_label']       for r in group_records]
        pred = [r['predicted_label']  for r in group_records]

        results[group_name] = {
            'count':   len(group_records),
            'fpr':     _fpr(true, pred),
            'f1_hate': float(f1_score(true, pred, pos_label=1, average='binary', zero_division=0)),
        }

    # Overall
    all_true = [r['true_label']      for r in records]
    all_pred = [r['predicted_label'] for r in records]
    results['overall'] = {
        'count':   len(records),
        'fpr':     _fpr(all_true, all_pred),
        'f1_hate': float(f1_score(all_true, all_pred, pos_label=1, average='binary', zero_division=0)),
    }

    return results
