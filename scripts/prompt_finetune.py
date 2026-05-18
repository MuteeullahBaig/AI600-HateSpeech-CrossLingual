"""
Prompt-based few-shot fine-tuning for Roman Urdu hate speech detection.

Motivated by Ullah et al. (2025) who showed that prefix-prompt XLM-RoBERTa
matches standard fine-tuning on 80% of data using only 32 training examples
per class, across 8 languages in sentiment analysis.

We extend this to Roman Urdu hate speech (binary classification) to test
whether the MLM-based fine-tuning mitigates the all-hate collapse observed
in our standard few-shot experiments (K <= 32 all collapse to hate F1 ≈ 0.697).

Approach (prefix prompt):
  Template: "<tweet> This text is <mask>."
  Verbalizer: {hate=1: ["hateful","offensive","harmful","toxic"],
               non-hate=0: ["normal","fine","acceptable","okay"]}

Training: MLM loss restricted to verbalizer token logits at [MASK] position.
Inference: class with higher summed probability over its verbalizer tokens.

Usage:
    python scripts/prompt_finetune.py
"""

import os
import sys
import json
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT   = "outputs/checkpoints/xlmr_large_en/best_model"
DEV_PATH     = "data/raw/hateval2019_ur_dev.tsv"
TRAIN_PATH   = "data/raw/hateval2019_ur_train.tsv"
OUT_DIR      = "outputs/predictions"
SUMMARY_CSV  = "outputs/predictions/prompt_finetune_summary.csv"
PREDS_DIR    = "outputs/predictions/prompt_finetune"

os.makedirs(PREDS_DIR, exist_ok=True)

# ── Prompt config ─────────────────────────────────────────────────────────────
# Template: append after the tweet text. <mask> will be replaced by tokenizer's mask_token.
TEMPLATE = "{text} This text is {mask}."

# Verbalizer: words whose single-token ID maps to a class.
# Class 0 = non-hate, Class 1 = hate.
# These are common English adjectives that XLM-RoBERTa knows well.
VERBALIZERS = {
    # All words below tokenise as single SentencePiece tokens in XLM-RoBERTa
    0: ["normal", "fine", "okay", "good"],      # non-hate
    1: ["hate", "bad", "wrong", "vile", "terrible"],  # hate
}

# ── Training config ───────────────────────────────────────────────────────────
SHOT_COUNTS  = [8, 16, 32, 64]
NUM_SEEDS    = 3
LR           = 2e-5
NUM_EPOCHS   = 10           # prompt-based converges faster than standard
BATCH_SIZE   = 8
MAX_LENGTH   = 128
SEED_BASE    = 42

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def get_verbalizer_ids(tokenizer, verbalizers):
    """
    Map verbalizer words → single token IDs in the tokenizer vocabulary.
    XLM-RoBERTa uses SentencePiece; words get a leading '▁' when they follow
    a space.  We prepend a space so they tokenize as single tokens.
    Reports which words were successfully mapped.
    """
    verbalizer_ids = {}
    for cls_id, words in verbalizers.items():
        ids = []
        for word in words:
            # Prepend space so RoBERTa tokenises as a whole word token
            tokens = tokenizer.tokenize(" " + word)
            if len(tokens) == 1:
                tid = tokenizer.convert_tokens_to_ids(tokens[0])
                if tid != tokenizer.unk_token_id:
                    ids.append(tid)
                    print(f"    class {cls_id}: '{word}' -> token '{tokens[0]}' (id {tid})")
                else:
                    print(f"    class {cls_id}: '{word}' -> UNK (skipped)")
            else:
                # Multi-token word: use first subword token as fallback
                tid = tokenizer.convert_tokens_to_ids(tokens[0])
                ids.append(tid)
                print(f"    class {cls_id}: '{word}' -> multi-token {tokens}, using '{tokens[0]}' (id {tid})")
        verbalizer_ids[cls_id] = list(set(ids))
    return verbalizer_ids


def stratified_k_shot(texts, labels, k, seed):
    """Sample exactly k examples per class (stratified)."""
    rng = random.Random(seed)
    by_class = {0: [], 1: []}
    for t, l in zip(texts, labels):
        by_class[l].append((t, l))
    selected = []
    for cls_id in [0, 1]:
        pool = by_class[cls_id]
        chosen = rng.sample(pool, min(k, len(pool)))
        selected.extend(chosen)
    rng.shuffle(selected)
    return [x[0] for x in selected], [x[1] for x in selected]


# ── Dataset ───────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    """
    Wraps texts+labels in a cloze prompt and finds the [MASK] position.
    Each item: {input_ids, attention_mask, mask_pos, label}.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.examples = []
        mask_token = tokenizer.mask_token   # "<mask>" for XLM-R

        skipped = 0
        for text, label in zip(texts, labels):
            prompt = TEMPLATE.format(text=text, mask=mask_token)
            enc = tokenizer(
                prompt,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids     = enc["input_ids"].squeeze(0)       # (max_length,)
            attention_mask = enc["attention_mask"].squeeze(0)  # (max_length,)

            # Find the position of <mask> token
            mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                # Mask token was truncated away — skip this example
                skipped += 1
                continue
            mask_pos = mask_positions[0].item()

            self.examples.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "mask_pos":       mask_pos,
                "label":          label,
            })

        if skipped > 0:
            print(f"  [PromptDataset] Skipped {skipped} examples where [MASK] was truncated")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    mask_pos       = torch.tensor([b["mask_pos"]      for b in batch], dtype=torch.long)
    labels         = torch.tensor([b["label"]         for b in batch], dtype=torch.long)
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "mask_pos":       mask_pos,
        "labels":         labels,
    }


# ── Training & evaluation ─────────────────────────────────────────────────────

def prompt_loss(logits_at_mask, labels, verbalizer_ids):
    """
    Cross-entropy loss restricted to verbalizer token logits.

    logits_at_mask : (batch, vocab_size) — logits at the [MASK] position
    labels         : (batch,) — 0 or 1
    verbalizer_ids : {0: [token_ids], 1: [token_ids]}

    For each example, we:
      1. Extract logits for all verbalizer tokens across both classes.
      2. For each class c, aggregate by summing log-probs over its verbalizer tokens.
      3. Apply cross-entropy over the (2,) class-level scores.
    """
    batch_size = logits_at_mask.size(0)
    class_scores = []
    for cls_id in [0, 1]:
        ids = verbalizer_ids[cls_id]
        # Sum log-softmax scores over verbalizer tokens for this class
        cls_logits = logits_at_mask[:, ids]           # (batch, num_verbs)
        cls_score  = torch.logsumexp(cls_logits, dim=-1)  # (batch,)
        class_scores.append(cls_score)
    class_scores = torch.stack(class_scores, dim=1)   # (batch, 2)
    return F.cross_entropy(class_scores, labels)


def predict(logits_at_mask, verbalizer_ids):
    """
    Predict class by summing softmax probabilities over each class's verbalizer tokens.
    Returns predicted class ids (batch,).
    """
    probs = F.softmax(logits_at_mask, dim=-1)         # (batch, vocab_size)
    class_probs = []
    for cls_id in [0, 1]:
        ids = verbalizer_ids[cls_id]
        class_probs.append(probs[:, ids].sum(dim=-1))  # (batch,)
    class_probs = torch.stack(class_probs, dim=1)     # (batch, 2)
    return class_probs.argmax(dim=1)                  # (batch,)


def run_one_experiment(k, seed, tokenizer, dev_texts, dev_labels, all_train_texts, all_train_labels, verbalizer_ids):
    """Train prompt-based model on K examples per class, evaluate on dev set."""
    set_seed(seed)

    # Sample K-shot training data
    train_texts, train_labels = stratified_k_shot(all_train_texts, all_train_labels, k, seed)
    print(f"  K={k} seed={seed}: {len(train_texts)} train examples "
          f"(hate={sum(train_labels)}, non-hate={len(train_labels)-sum(train_labels)})")

    # Datasets
    train_ds = PromptDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    dev_ds   = PromptDataset(dev_texts,   dev_labels,   tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_ds,   batch_size=32,         shuffle=False, collate_fn=collate_fn)

    # Load fresh model for each run
    model = AutoModelForMaskedLM.from_pretrained(CHECKPOINT)
    model = model.to(device)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1   = -1.0
    best_preds = None

    for epoch in range(NUM_EPOCHS):
        # ── Train ───────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mask_pos       = batch["mask_pos"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # outputs.logits: (batch, seq_len, vocab_size)
            logits_at_mask = outputs.logits[
                torch.arange(input_ids.size(0), device=device), mask_pos
            ]  # (batch, vocab_size)

            loss = prompt_loss(logits_at_mask, labels, verbalizer_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # ── Evaluate ────────────────────────────────────────────────────
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                mask_pos       = batch["mask_pos"].to(device)
                true_labels    = batch["labels"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_at_mask = outputs.logits[
                    torch.arange(input_ids.size(0), device=device), mask_pos
                ]
                preds = predict(logits_at_mask, verbalizer_ids).cpu().tolist()
                all_preds.extend(preds)
                all_true.extend(true_labels.tolist())

        epoch_f1 = f1_score(all_true, all_preds, pos_label=1, average="binary", zero_division=0)
        if epoch_f1 > best_f1:
            best_f1    = epoch_f1
            best_preds = list(all_preds)

        print(f"    epoch {epoch+1}/{NUM_EPOCHS} | loss={total_loss/max(1,len(train_loader)):.4f} | hate_f1={epoch_f1:.4f}")

    del model
    torch.cuda.empty_cache()

    # Final metrics on best preds
    f1h  = f1_score(dev_labels, best_preds, pos_label=1, average="binary",   zero_division=0)
    f1nh = f1_score(dev_labels, best_preds, pos_label=0, average="binary",   zero_division=0)
    f1m  = f1_score(dev_labels, best_preds, average="macro",                 zero_division=0)
    fp_r = fpr(dev_labels, best_preds)
    acc  = float(np.mean(np.array(best_preds) == np.array(dev_labels)))

    return {
        "k": k, "seed": seed,
        "f1_hate": f1h, "f1_non_hate": f1nh, "f1_macro": f1m,
        "fpr": fp_r, "accuracy": acc,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Prompt-based few-shot fine-tuning (Ullah et al. 2025)")
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"  Device     : {device}")
    print(f"  Template   : {TEMPLATE!r}")
    print("=" * 60)

    # Load data (inline — avoids importing datasets lib alongside transformers)
    def load_tsv(path):
        df = pd.read_csv(path, sep="\t", header=None,
                         names=["id", "text", "hs", "tr", "ag"],
                         dtype=str,
                         on_bad_lines="skip")
        # Drop header row if the file includes it (hs == 'HS')
        df = df[df["hs"].str.strip() != "HS"].copy()
        df["hs"] = pd.to_numeric(df["hs"], errors="coerce").fillna(0).astype(int)
        df["tr"] = pd.to_numeric(df["tr"], errors="coerce").fillna(0).astype(int)
        df["ag"] = pd.to_numeric(df["ag"], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["text"]).reset_index(drop=True)
        return df

    print("\nLoading data...")
    train_df = load_tsv(TRAIN_PATH)
    dev_df   = load_tsv(DEV_PATH)
    all_train_texts  = train_df["text"].tolist()
    all_train_labels = train_df["hs"].tolist()
    dev_texts        = dev_df["text"].tolist()
    dev_labels       = dev_df["hs"].tolist()
    print(f"Train: {len(all_train_texts)} | Dev: {len(dev_texts)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    # Resolve verbalizer token IDs
    print("\nResolving verbalizer token IDs:")
    verbalizer_ids_cpu = get_verbalizer_ids(tokenizer, VERBALIZERS)
    # Convert to tensors on device
    verbalizer_ids = {
        cls_id: ids
        for cls_id, ids in verbalizer_ids_cpu.items()
    }
    for cls_id, ids in verbalizer_ids.items():
        if not ids:
            raise ValueError(f"No valid verbalizer tokens found for class {cls_id}!")
        print(f"  Class {cls_id} final IDs: {ids}")

    # Run experiments
    all_results = []
    for k in SHOT_COUNTS:
        print(f"\n{'─'*55}")
        print(f"  K = {k}")
        print(f"{'─'*55}")
        for seed_offset in range(NUM_SEEDS):
            seed = SEED_BASE + seed_offset * 100
            result = run_one_experiment(
                k, seed, tokenizer, dev_texts, dev_labels,
                all_train_texts, all_train_labels, verbalizer_ids,
            )
            all_results.append(result)
            print(f"  → K={k} seed={seed}: hate_F1={result['f1_hate']:.4f} "
                  f"macro_F1={result['f1_macro']:.4f} FPR={result['fpr']:.4f}")

    # Save per-run results
    per_run_path = os.path.join(PREDS_DIR, "per_run_results.json")
    with open(per_run_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Aggregate by K
    print(f"\n{'Model':<12} {'Hate F1':>9} {'±':>4} {'Macro F1':>9} {'±':>4} {'FPR':>7}")
    print("-" * 55)

    summary_rows = []
    for k in SHOT_COUNTS:
        runs = [r for r in all_results if r["k"] == k]
        f1h_vals  = [r["f1_hate"]  for r in runs]
        f1m_vals  = [r["f1_macro"] for r in runs]
        fpr_vals  = [r["fpr"]      for r in runs]
        row = {
            "k":            k,
            "f1_hate_mean": float(np.mean(f1h_vals)),
            "f1_hate_std":  float(np.std(f1h_vals)),
            "f1_macro_mean": float(np.mean(f1m_vals)),
            "f1_macro_std": float(np.std(f1m_vals)),
            "fpr_mean":     float(np.mean(fpr_vals)),
        }
        summary_rows.append(row)
        print(f"  K={k:<8} {row['f1_hate_mean']:>9.3f} {row['f1_hate_std']:>4.3f} "
              f"{row['f1_macro_mean']:>9.3f} {row['f1_macro_std']:>4.3f} "
              f"{row['fpr_mean']:>7.3f}")

    # Save summary CSV
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved → {SUMMARY_CSV}")

    # Context comparison
    print("\nContext (standard few-shot results):")
    std_results = [
        (8,  0.697, 0.000, 0.350, 0.998),
        (16, 0.697, 0.000, 0.351, 0.997),
        (32, 0.697, 0.000, 0.349, 1.000),
        (64, 0.715, 0.009, 0.528, 0.777),
    ]
    print(f"{'Method':<28} {'K':>4} {'Hate F1':>8} {'Macro F1':>9} {'FPR':>7}")
    print("-" * 60)
    for k, f1h, f1h_std, f1m, fp_r in std_results:
        print(f"  Standard fine-tuning       {k:>4} {f1h:>8.3f} {f1m:>9.3f} {fp_r:>7.3f}")
    for row in summary_rows:
        print(f"  Prompt-based (prefix)      {row['k']:>4} {row['f1_hate_mean']:>8.3f} "
              f"{row['f1_macro_mean']:>9.3f} {row['fpr_mean']:>7.3f}")


if __name__ == "__main__":
    main()
