"""Shared demo pipeline — the single source of truth behind BOTH the standalone
notebook and `python run.py demo`.

Because the notebook's run-cells and the `demo` command call these exact functions
on the same deterministic sample (fixed `sample_size` + `seed`), their substantive
outputs (retrieval metrics, Macro-F1, predictions, citations) are identical — there
is no notebook-vs-application discrepancy. The full pipeline (`run.py all`) is the
same code at scale on the whole corpus.

The demo uses a deterministic head sample of the comments and indexes the products
those comments reference (Q&A/comparison/managerial only concern reviewed products),
so it runs in a couple of minutes instead of embedding all ~948k products.
"""
from __future__ import annotations

import logging

import pandas as pd

from . import config
from .core import dataio
from .phase1_data import clean
from .phase2_assistant import retrieval
from .phase2_assistant.assistant import ShoppingAssistant
from .phase2_assistant.router import Catalog
from .phase3_predict import recommend
from .phase4_eval import evaluate

log = logging.getLogger("digikala.demo")


def sample_frames(sample_size: int = 20_000, seed: int = config.RANDOM_SEED):
    """Deterministic, UNBIASED sample: `sample_size` comments drawn uniformly at
    random from the entire comments CSV (dataio.reservoir_sample, one-pass,
    vectorized) -- not just the first N rows, which would bias toward whatever
    order the raw file happens to be in. Plus the products those comments cite,
    both cleaned with the exact Phase-1 functions."""
    com_raw = dataio.reservoir_sample(config.COMMENTS_CSV, sample_size, seed)
    wanted = set(pd.to_numeric(com_raw["product_id"], errors="coerce").dropna().astype(int))
    keep = []
    for ch in pd.read_csv(config.PRODUCTS_CSV, chunksize=200_000, low_memory=False):
        ids = pd.to_numeric(ch["id"], errors="coerce")
        keep.append(ch[ids.isin(wanted)])
    prod_raw = pd.concat(keep, ignore_index=True)
    products, p_rep = clean.clean_products(prod_raw)
    valid = set(int(x) for x in products["product_id"].dropna())
    comments, c_rep = clean.clean_comments(com_raw, valid)
    return products, comments, p_rep, c_rep


def build_assistant(products: pd.DataFrame, comments: pd.DataFrame, llm=None) -> ShoppingAssistant:
    """In-memory assistant over the sample — same classes as the packaged app."""
    products = retrieval._prepare_products(products, comments)
    idx = retrieval.ProductIndex.build(products)
    with_text = comments[comments["has_text"].astype(bool)] if "has_text" in comments else comments
    by_product = retrieval.GroupedComments(with_text)
    catalog = Catalog.build(idx.products, by_product)
    return ShoppingAssistant(catalog, idx, retrieval.ReviewRetriever(by_product), llm=llm)


def run(sample_size: int = 20_000, seed: int = config.RANDOM_SEED, judge=None) -> dict:
    """Run the full demo end-to-end and return the substantive (deterministic) metrics."""
    from .core.llm import LLM
    products, comments, p_rep, c_rep = sample_frames(sample_size, seed)
    log.info("sample: %d products, %d comments", len(products), len(comments))

    assistant = build_assistant(products, comments, llm=LLM(mode=config.RUN_MODE))
    retr = evaluate.evaluate_retrieval(assistant, n_queries=20, k=10)

    _, p3 = recommend.train_from_frame(comments)

    top = assistant.c.products.sort_values("comment_count", ascending=False)["product_id"].head(2).astype(int).tolist()
    demos = {}
    for name, q in [("discovery", "یک کالای اقتصادی و باکیفیت زیر ۵۰۰ هزار تومان می‌خواهم"),
                    ("product_qa", f"آیا کاربران از کیفیت محصول {top[0]} راضی بودند؟"),
                    ("comparison", f"محصول {top[0]} و محصول {top[1]} را مقایسه کن")]:
        a = assistant.answer(q)
        demos[name] = {"intent": a.intent, "tier": a.tier,
                       "citations": sorted(a.citations), "review_citations": sorted(a.review_citations)}
    return {"sample_size": sample_size, "seed": seed,
            "n_products": int(len(products)), "n_comments": int(len(comments)),
            "retrieval_quality": retr,
            "prediction_test_macro_f1": p3["test_macro_f1"],
            "prediction_grouped_macro_f1": p3["grouped_macro_f1"],
            "prediction_primary_macro_f1": p3["primary_macro_f1"],
            "prediction_naive_split_product_overlap_pct": p3["naive_split_product_overlap_pct"],
            "prediction_leakage_ablation": p3["leakage_ablation"],
            "demos": demos}


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(run(), ensure_ascii=False, indent=2))
