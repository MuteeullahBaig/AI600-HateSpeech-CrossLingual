"""
Classical ML baseline: TF-IDF + Logistic Regression on Roman Urdu hate speech.

Motivated by Ullah et al. (2024) who showed classical ML + fuzzy features
can outperform weak transformers on Roman Urdu datasets.

We run TF-IDF + LR/RF as a supervised upper-bound reference point:
  - Trained on full Urdu train set (7,208 samples)
  - Evaluated on Urdu dev set (800 samples)
  - Compared against our zero-shot and few-shot XLM-R results

Also computes FuzzyWuzzy features (if fuzzywuzzy is installed) and
evaluates TF-IDF + Fuzzy + RF as in Ullah et al. (2024).

Usage:
    python scripts/classical_baseline.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_hateval_tsv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report, confusion_matrix
)

TRAIN_PATH = "data/raw/hateval2019_ur_train.tsv"
DEV_PATH   = "data/raw/hateval2019_ur_dev.tsv"
OUT_PATH   = "outputs/predictions/classical_baseline_results.json"

os.makedirs("outputs/predictions", exist_ok=True)


def fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def evaluate(name, y_true, y_pred):
    f1h  = f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    f1nh = f1_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)
    f1m  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    fp_r = fpr(y_true, y_pred)
    acc  = float((np.array(y_pred) == np.array(y_true)).mean())
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred,
                                target_names=['non-hate', 'hate'],
                                zero_division=0))
    print(f"  Hate F1   : {f1h:.4f}")
    print(f"  Macro F1  : {f1m:.4f}")
    print(f"  FPR       : {fp_r:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    return {"model": name, "f1_hate": f1h, "f1_non_hate": f1nh,
            "f1_macro": f1m, "fpr": fp_r, "accuracy": acc}


def build_fuzzy_features(train_texts, train_labels, eval_texts, top_k=50):
    """
    Compute FuzzyWuzzy features as in Ullah et al. (2024).
    For each class, collect top-k frequent words, then compute 6 fuzzy
    similarity scores between each tweet and each class word list.
    Returns (train_features, eval_features) as numpy arrays.
    """
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        print("  [fuzzy] fuzzywuzzy not installed — skipping fuzzy features")
        print("  Install with: pip install fuzzywuzzy python-Levenshtein")
        return None, None

    from collections import Counter
    import re

    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    # Build per-class keyword lists
    classes = sorted(set(train_labels))
    class_keywords = {}
    for c in classes:
        words = []
        for txt, lbl in zip(train_texts, train_labels):
            if lbl == c:
                words.extend(tokenize(txt))
        # Remove stopwords-ish very common words
        freq = Counter(words)
        top_words = [w for w, _ in freq.most_common(top_k)]
        class_keywords[c] = " ".join(top_words)

    fuzzy_fns = [
        fuzz.ratio,
        fuzz.partial_ratio,
        fuzz.token_set_ratio,
        fuzz.token_sort_ratio,
        fuzz.partial_token_set_ratio,
        fuzz.partial_token_sort_ratio,
    ]

    def tweet_fuzzy_features(text):
        feats = []
        for c in classes:
            kw_str = class_keywords[c]
            for fn in fuzzy_fns:
                feats.append(fn(text.lower(), kw_str) / 100.0)
        return feats

    print(f"  [fuzzy] Computing fuzzy features for {len(train_texts)} train samples...")
    X_train = np.array([tweet_fuzzy_features(t) for t in train_texts])
    print(f"  [fuzzy] Computing fuzzy features for {len(eval_texts)} eval samples...")
    X_eval  = np.array([tweet_fuzzy_features(t) for t in eval_texts])
    print(f"  [fuzzy] Feature shape: {X_train.shape}")
    return X_train, X_eval


def main():
    print("Loading data...")
    train_df = load_hateval_tsv(TRAIN_PATH)
    dev_df   = load_hateval_tsv(DEV_PATH)

    X_train = train_df['text'].tolist()
    y_train = train_df['hs'].tolist()
    X_dev   = dev_df['text'].tolist()
    y_dev   = dev_df['hs'].tolist()

    print(f"Train: {len(X_train)} samples | Dev: {len(X_dev)} samples")
    print(f"Train hate rate: {sum(y_train)/len(y_train):.1%}")
    print(f"Dev   hate rate: {sum(y_dev)/len(y_dev):.1%}")

    results = []

    # ── Model 1: TF-IDF + Logistic Regression ──────────────────────────
    print("\nTraining TF-IDF + Logistic Regression...")
    pipe_lr = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                  sublinear_tf=True)),
        ('clf',   LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')),
    ])
    pipe_lr.fit(X_train, y_train)
    preds_lr = pipe_lr.predict(X_dev)
    results.append(evaluate("TF-IDF + Logistic Regression (balanced)", y_dev, preds_lr))

    # ── Model 2: TF-IDF + Random Forest ────────────────────────────────
    print("\nTraining TF-IDF + Random Forest...")
    pipe_rf = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                  sublinear_tf=True)),
        ('clf',   RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                         n_jobs=-1, random_state=42)),
    ])
    pipe_rf.fit(X_train, y_train)
    preds_rf = pipe_rf.predict(X_dev)
    results.append(evaluate("TF-IDF + Random Forest (balanced)", y_dev, preds_rf))

    # ── Model 3: TF-IDF + Fuzzy + LR (Ullah et al. 2024 style) ────────
    print("\nComputing fuzzy features (Ullah et al. 2024 approach)...")
    X_train_fuzzy, X_dev_fuzzy = build_fuzzy_features(X_train, y_train, X_dev)

    if X_train_fuzzy is not None:
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse import hstack, csr_matrix

        tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_dev_tfidf   = tfidf.transform(X_dev)

        X_train_fused = hstack([X_train_tfidf, csr_matrix(X_train_fuzzy)])
        X_dev_fused   = hstack([X_dev_tfidf,   csr_matrix(X_dev_fuzzy)])

        # LR
        lr_fused = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        lr_fused.fit(X_train_fused, y_train)
        preds_fused_lr = lr_fused.predict(X_dev_fused)
        results.append(evaluate("TF-IDF + Fuzzy + LR (Ullah et al. 2024)", y_dev, preds_fused_lr))

        # RF
        rf_fused = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                           n_jobs=-1, random_state=42)
        rf_fused.fit(X_train_fused, y_train)
        preds_fused_rf = rf_fused.predict(X_dev_fused)
        results.append(evaluate("TF-IDF + Fuzzy + RF (Ullah et al. 2024)", y_dev, preds_fused_rf))

    # ── Save results ────────────────────────────────────────────────────
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {OUT_PATH}")

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'Model':<45} {'Hate F1':>8} {'Macro F1':>9} {'FPR':>7}")
    print("-" * 72)
    for r in results:
        print(f"{r['model']:<45} {r['f1_hate']:>8.3f} {r['f1_macro']:>9.3f} {r['fpr']:>7.3f}")

    print("\nContext (from our transformer experiments):")
    print(f"{'XLM-R-large zero-shot (Urdu)':<45} {'0.005':>8} {'0.319':>9} {'0.003':>7}")
    print(f"{'XLM-R-large few-shot K=64 (balanced w)':<45} {'0.715':>8} {'0.528':>9} {'0.777':>7}")
    print(f"{'XLM-R-large supervised (running...)':<45} {'?':>8} {'?':>9} {'?':>7}")


if __name__ == '__main__':
    main()
