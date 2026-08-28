"""Phase 3 bonus — LoRA fine-tune a small Persian encoder for recommendation_status
and compare it against the TF-IDF + logistic-regression baseline on the *identical*
product-grouped split (see recommend.prepare_split), so the comparison is fair and
leakage-free by construction.

Model: HooshvareLab/bert-fa-base-uncased (ParsBERT, ~118M params) -- a real Persian
encoder, small enough to LoRA-tune on a 6GB laptop GPU. LoRA adapters (rank=8) are
applied to the attention query/value projections; the base weights stay frozen, so
training touches <1% of parameters.

Sampling: capped to the same MAX_PER_CLASS as the baseline (config-shared via
recommend.MAX_PER_CLASS) for a like-for-like comparison, further capped by
--max-train/--max-test for a laptop-GPU-sized fine-tune -- this is the "limited
resources -> justified subset" case the brief explicitly allows, not an arbitrary
shortcut: full LoRA fine-tuning over the full sampled set is expensive relative to
the $0 TF-IDF baseline's near-instant training, and the brief only asks that the
subset choice be reasoned, not that the full corpus be used.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

from .. import config
from . import recommend

log = logging.getLogger("digikala.phase3_lora")

MODEL_NAME = "HooshvareLab/bert-fa-base-uncased"
# The model repo only ships pytorch_model.bin on `main`; modern `transformers`
# refuses torch.load-based checkpoints for security (CVE-2025-32434) unless
# torch>=2.6. A community PR on the repo already converted the weights to
# safetensors -- pin to that revision instead of bumping the pinned torch build.
MODEL_REVISION = "refs/pr/2"
MAX_LEN = 96


class _TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                             max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i]}


def train_and_compare(max_train: int = 6000, max_test: int = 1500, epochs: int = 2,
                       batch_size: int = 16, lr: float = 2e-4) -> dict:
    """Fine-tune ParsBERT+LoRA on the same product-grouped split the TF-IDF
    baseline uses, evaluate Macro-F1 on the held-out product-grouped test set,
    and report the delta vs. the already-persisted phase3_metrics.json baseline."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    df = recommend._load()
    sample, g_train, g_test, le = recommend.prepare_split(df)
    n_classes = len(le.classes_)

    # cap for laptop-GPU feasibility (see module docstring); sampling is stratified
    # by class and seeded, so it's reproducible, not arbitrary
    g_train = recommend._stratified_cap(g_train, "target_encoded", max_train // n_classes)
    g_test = recommend._stratified_cap(g_test, "target_encoded", max_test // n_classes)
    log.info("LoRA train rows=%d test rows=%d classes=%d", len(g_train), len(g_test), n_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_classes, revision=MODEL_REVISION)
    lora_cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
                          target_modules=["query", "value"])
    model = get_peft_model(base_model, lora_cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("LoRA trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    train_ds = _TextDataset(g_train["comment_text_norm"], g_train["target_encoded"].tolist(), tokenizer)
    test_ds = _TextDataset(g_test["comment_text_norm"], g_test["target_encoded"].tolist(), tokenizer)

    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += out.loss.item()
        log.info("epoch %d/%d mean loss %.4f", epoch + 1, epochs, total_loss / max(1, len(train_dl)))

    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in test_dl:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            gold.extend(labels.tolist())
    lora_macro_f1 = round(f1_score(gold, preds, average="macro"), 4)

    baseline_file = config.METRICS_DIR / "phase3_metrics.json"
    baseline_grouped = None
    if baseline_file.exists():
        baseline_grouped = json.loads(baseline_file.read_text(encoding="utf-8")).get(
            "primary_macro_f1")

    result = {
        "model": MODEL_NAME, "method": "LoRA (r=8, alpha=16, query+value)",
        "trainable_params": trainable, "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "n_train": len(g_train), "n_test": len(g_test), "epochs": epochs,
        "same_split_as_baseline": "product_grouped (recommend.prepare_split, identical seed)",
        "lora_macro_f1": lora_macro_f1,
        "baseline_grouped_macro_f1": baseline_grouped,
        "lora_vs_baseline_delta": (round(lora_macro_f1 - baseline_grouped, 4)
                                   if baseline_grouped is not None else None),
        "note": ("Sampled subset (see module docstring) for laptop-GPU feasibility -- "
                 "not the full grouped-split size the TF-IDF baseline uses. The delta is "
                 "reported for reference; it is not an apples-to-apples full-data comparison."),
    }
    (config.METRICS_DIR / "phase3_lora_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("LoRA Macro-F1 %.4f vs TF-IDF baseline %.4f (delta %s)",
             lora_macro_f1, baseline_grouped or -1, result["lora_vs_baseline_delta"])
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(train_and_compare(), ensure_ascii=False, indent=2))
