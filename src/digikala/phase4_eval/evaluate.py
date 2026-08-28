"""Phase 4 — evaluate the whole system across the six axes the brief requires.

    1. Response quality  — LLM-as-judge relevance (0–5)              [RAGAS-style]
    2. Grounding         — citation coverage + LLM-as-judge faithfulness
    3. Retrieval quality — recall@k / MRR / nDCG@k vs. gold ids
    4. Prediction        — Phase-3 Macro-F1 (read from its metrics file)
    5. Latency & cost    — per-answer wall-clock + API $ / tokens (BudgetTracker)
    6. Failure analysis  — per-intent missing-info / zero-citation / low-score rates

A non-LLM lexical baseline is included as the control the brief asks for. When
JUDGE_MODE=local/free/paid a model scores relevance & faithfulness; with a tiny
hand-labeled set we also report judge–human agreement (Spearman). Design follows
common RAG-eval practice (RAGAS faithfulness/answer-relevance + IR recall/MRR/nDCG).
"""
from __future__ import annotations

import json
import logging
import re
import time

import numpy as np
import pandas as pd

from collections import Counter

from .. import config
from ..core.llm import BudgetTracker, judge_llm
from ..core import persian_text as pt
from ..phase2_assistant import prompts
from ..phase2_assistant.assistant import LexicalBaseline, build_assistant
from ..phase2_assistant.router import extract_filters

log = logging.getLogger("digikala.phase4")
_CITE = re.compile(r"\[(?:محصول|بازبینی)\s*\d+\]")
_CITE_P_EVAL = re.compile(r"\[محصول\s*(\d+)\]")
_CITE_R_EVAL = re.compile(r"\[بازبینی\s*(\d+)\]")
_GENERIC_VALUES = {"", "نامشخص", "سایر", "متفرقه"}


# ---- retrieval metrics --------------------------------------------------
def _dcg(rels):
    return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))


def ranking_metrics(retrieved_ids, relevant_ids, k=None) -> dict:
    k = k or len(retrieved_ids)
    retrieved = retrieved_ids[:k]
    hits = [1 if i in relevant_ids else 0 for i in retrieved]
    recall = sum(hits) / max(1, len(relevant_ids))
    mrr = next((1.0 / r for r, i in enumerate(retrieved, 1) if i in relevant_ids), 0.0)
    ideal = sorted([1] * min(len(relevant_ids), k), reverse=True)
    ndcg = _dcg(hits) / max(_dcg(ideal), 1e-9)
    return {"recall@k": round(recall, 4), "mrr": round(mrr, 4), "ndcg@k": round(ndcg, 4)}


def evaluate_retrieval(assistant, n_queries: int = 30, k: int = 10) -> dict:
    """Auto-labeled retrieval eval: a product's title is the query, its id the gold.

    Uses the most-reviewed products so the queries are realistic and each has a
    single known-relevant id.
    """
    products = assistant.c.products
    pids = (products.sort_values("comment_count", ascending=False)["product_id"]
            .head(n_queries).astype(int).tolist())
    rows = []
    for pid in pids:
        title = assistant.c.product(pid)["title_fa"]
        hits = assistant.pidx.search(title, k=k)
        rows.append(ranking_metrics([h["product_id"] for h in hits], {pid}, k=k))
    df = pd.DataFrame(rows)
    return {"n_queries": len(rows), "k": k,
            "recall@k": round(df["recall@k"].mean(), 4),
            "mrr": round(df["mrr"].mean(), 4),
            "ndcg@k": round(df["ndcg@k"].mean(), 4)}


def retrieval_ablation(assistant, n_queries: int = 30, k: int = 10) -> dict:
    """Quantify hybrid retrieval vs. its two single-method halves (dense-only,
    BM25-only), using the same title->own-id auto-labels as evaluate_retrieval,
    so the three numbers are directly comparable (bonus: 'Hybrid Search with
    quantified improvement').

    Caveat, reported not hidden: this auto-labeled benchmark uses each product's
    own TITLE as the query, which is a near-exact lexical-match task -- it
    structurally favors BM25 and is not a fair test of hybrid's actual value on
    fuzzy natural-language queries (which the discovery/QA intents actually get).
    On this specific benchmark BM25-only can legitimately beat hybrid; that is
    a real, measured result about the benchmark's bias, not a failure to fix."""
    products = assistant.c.products
    pids = (products.sort_values("comment_count", ascending=False)["product_id"]
            .head(n_queries).astype(int).tolist())
    per_method = {}
    for method in ("dense", "bm25", "hybrid"):
        rows = []
        for pid in pids:
            title = assistant.c.product(pid)["title_fa"]
            hits = assistant.pidx.search(title, k=k, method=method)
            rows.append(ranking_metrics([h["product_id"] for h in hits], {pid}, k=k))
        df = pd.DataFrame(rows)
        per_method[method] = {"recall@k": round(df["recall@k"].mean(), 4),
                               "mrr": round(df["mrr"].mean(), 4),
                               "ndcg@k": round(df["ndcg@k"].mean(), 4)}
    return {"n_queries": len(pids), "k": k, "by_method": per_method,
            "hybrid_vs_best_single_mrr_lift": round(
                per_method["hybrid"]["mrr"] - max(per_method["dense"]["mrr"], per_method["bm25"]["mrr"]), 4)}


# ---- natural-language retrieval benchmark (fixes the title-query bias) --
def build_natural_retrieval_cases(assistant, n_queries: int = 20):
    """Build a diverse Persian pseudo-gold retrieval benchmark using paraphrased
    brand+category+distinguishing-title-token queries, NOT each product's exact
    title (which `evaluate_retrieval`/`retrieval_ablation` use and which is a
    near-exact lexical match that structurally favors BM25). Gold ids are still
    programmatically derived from catalogue metadata -- explicitly not human
    annotation -- but the query text itself is a realistic paraphrase, and at
    most 4 cases come from the same category so one high-volume category can't
    dominate the benchmark."""
    p = assistant.c.products.copy()
    for col in ("product_id", "title_norm", "brand_norm", "category1_norm", "category2_norm", "sub_category_norm"):
        if col not in p:
            p[col] = ""
    p = p[p["product_id"].notna() & p["brand_norm"].notna()
         & (p["brand_norm"].astype(str).str.strip() != "") & (~p["brand_norm"].isin(_GENERIC_VALUES))].copy()
    if "comment_count" in p:
        p = p[p["comment_count"] > 0]
    common = {"مدل", "سری", "اصل", "اورجینال", "جدید", "محصول", "عدد", "بسته", "مناسب", "کیفیت", "خوب", "رنگ", "طرح"}
    candidates, seen_queries, seen_targets = [], set(), set()
    specs = [("brand_norm", "category1_norm"), ("brand_norm", "category2_norm"), ("brand_norm", "sub_category_norm")]
    max_per_category = max(2, min(4, int(np.ceil(n_queries / 5))))
    category_counts = Counter()

    for brand_col, cat_col in specs:
        tmp = p[p[cat_col].notna() & (p[cat_col].astype(str).str.strip() != "")
               & (~p[cat_col].isin(_GENERIC_VALUES))].copy()
        grouped = []
        for (brand, category), g in tmp.groupby([brand_col, cat_col], sort=False):
            if len(g) < 2 or len(g) > 60:
                continue
            strength = int(g.get("comment_count", pd.Series(0, index=g.index)).sum())
            grouped.append((strength, str(brand), str(category), g))
        grouped.sort(key=lambda x: x[0], reverse=True)

        for _, brand, category, g in grouped:
            category_fa = category if re.search(r"[آ-ی]", category) else "کالا"
            if category_counts[category_fa] >= max_per_category:
                continue
            g = g.sort_values(["comment_count", "rate_count"], ascending=False)
            token_df = Counter()
            for title in g["title_norm"].fillna(""):
                token_df.update(set(pt.tokenize_norm(title)))
            forbidden = set(pt.tokenize_norm(brand)) | set(pt.tokenize_norm(category)) | common

            for target in g.head(min(6, len(g))).itertuples():
                target_id = int(target.product_id)
                if target_id in seen_targets:
                    continue
                title_tokens = [t for t in pt.tokenize_norm(str(target.title_norm))
                                if len(t) >= 2 and not t.isdigit() and t not in forbidden and re.search(r"[آ-ی]", t)]
                if not title_tokens:
                    continue
                title_tokens = sorted(dict.fromkeys(title_tokens), key=lambda t: (token_df.get(t, 999999), -len(t)))
                desc = " ".join(title_tokens[:2])
                q = f"یک {category_fa} از برند {brand} می‌خواهم؛ ترجیحاً مدلی که مشخصهٔ «{desc}» را داشته باشد."
                if q in seen_queries:
                    continue
                seen_queries.add(q); seen_targets.add(target_id); category_counts[category_fa] += 1
                candidates.append({"query": q, "relevant_ids": [target_id], "brand": brand,
                                   "category": category, "target_title": str(target.title_norm)})
                break
            if len(candidates) >= n_queries:
                break
        if len(candidates) >= n_queries:
            break
    return candidates[:n_queries]


def evaluate_retrieval_natural(assistant, n_queries: int = 20, k: int = 10) -> dict:
    """Hybrid vs. lexical-baseline retrieval quality on the natural-language
    benchmark above -- this is the fair test of hybrid's value (unlike the
    title-exact-match benchmark, which structurally favors BM25). Reports
    whichever side actually wins, honestly, per query and in aggregate."""
    cases = build_natural_retrieval_cases(assistant, n_queries=n_queries)
    if not cases:
        return {"n_queries": 0, "note": "could not build a natural-language benchmark from the current sample"}
    baseline = LexicalBaseline(assistant.c)
    rows = []
    for case in cases:
        q, relevant = case["query"], set(case["relevant_ids"])
        filters = extract_filters(assistant.c, q)
        hybrid_ids = [int(x["product_id"]) for x in assistant.pidx.search(q, filters=filters, k=k)]
        hm = ranking_metrics(hybrid_ids, relevant, k=k)
        base_ids = [int(x) for x in baseline.discover(q, k=k)["product_id"].tolist()]
        bm = ranking_metrics(base_ids, relevant, k=k)
        rows.append({"query": q, "hybrid_recall@k": hm["recall@k"], "hybrid_mrr": hm["mrr"],
                     "hybrid_ndcg@k": hm["ndcg@k"], "baseline_recall@k": bm["recall@k"],
                     "baseline_mrr": bm["mrr"], "baseline_ndcg@k": bm["ndcg@k"]})
    df = pd.DataFrame(rows)
    hybrid_ndcg, baseline_ndcg = float(df["hybrid_ndcg@k"].mean()), float(df["baseline_ndcg@k"].mean())
    verdict = ("hybrid_better" if hybrid_ndcg > baseline_ndcg + 1e-9
              else "tie" if abs(hybrid_ndcg - baseline_ndcg) <= 1e-9 else "lexical_baseline_better")
    return {"n_queries": len(df), "k": k,
           "benchmark_type": "reproducible_programmatic_pseudo_gold_not_human_labeled",
           "quality_verdict_by_ndcg": verdict,
           "hybrid": {"recall@k": round(float(df["hybrid_recall@k"].mean()), 4),
                     "mrr": round(float(df["hybrid_mrr"].mean()), 4), "ndcg@k": round(hybrid_ndcg, 4)},
           "lexical_baseline": {"recall@k": round(float(df["baseline_recall@k"].mean()), 4),
                               "mrr": round(float(df["baseline_mrr"].mean()), 4), "ndcg@k": round(baseline_ndcg, 4)},
           "per_query": df.to_dict("records")}


# ---- grounding / quality ------------------------------------------------
def citation_coverage(text: str) -> float:
    sents = [s for s in re.split(r"[.!؟\n]+", text) if s.strip()]
    if not sents:
        return 0.0
    return round(sum(1 for s in sents if _CITE.search(s)) / len(sents), 4)


def citation_validity(ans) -> float:
    """Of the citation ids the answer text actually contains, what fraction
    were in the retriever's own allowed set? (verify_citations already strips
    disallowed ids from LLM-tier answers, so this should normally be 1.0 --
    it's a direct check of that guarantee, not a duplicate of coverage.)"""
    cited_p = {int(x) for x in _CITE_P_EVAL.findall(ans.text)}
    cited_r = {int(x) for x in _CITE_R_EVAL.findall(ans.text)}
    allowed_p, allowed_r = {int(x) for x in ans.citations}, {int(x) for x in ans.review_citations}
    total = len(cited_p) + len(cited_r)
    if total == 0:
        return 1.0 if ans.missing_info else 0.0
    return float((len(cited_p & allowed_p) + len(cited_r & allowed_r)) / total)


def task_completion_proxy(query: str, ans) -> float:
    """Intent-specific, fully deterministic 0-5 task-completion score: does the
    answer contain the structures/evidence each Phase-2 capability actually
    requires? Complements the LLM judge with a reproducible, non-LLM signal --
    useful given how unreliable a small local judge has proven (see README)."""
    text = str(ans.text)
    n_p, n_r = len(set(_CITE_P_EVAL.findall(text))), len(set(_CITE_R_EVAL.findall(text)))
    if ans.missing_info:
        return 3.0 if ("اطلاعات کافی" in text or "یافت نشد" in text) else 1.0
    if ans.intent == "discovery":
        score = (2.0 if n_p >= 3 else (1.0 if n_p >= 1 else 0.0))
        score += 1.0 if "قیمت" in text else 0.0
        score += 1.0 if any(w in text for w in ("امتیاز", "توصیه", "رضایت")) else 0.0
        score += 1.0 if any(w in text for w in ("پیشنهاد", "برتر", "گزینه")) else 0.0
        return min(5.0, score)
    if ans.intent == "product_qa":
        score = 1.0 if n_p >= 1 else 0.0
        score += 2.0 if n_r >= 2 else (1.0 if n_r >= 1 else 0.0)
        score += 1.0 if any(w in text for w in ("بازبینی", "نظر")) else 0.0
        qn = pt.normalize(query)
        if any(x in qn for x in ("مشکل", "ایراد", "ضعف", "بد")):
            score += 1.0 if any(x in text for x in ("ایراد", "مشکل", "منفی", "ضعف")) else 0.0
        else:
            score += 1.0 if any(x in text for x in ("مثبت", "قوت", "توصیه", "راضی", "خوب")) else 0.0
        return min(5.0, score)
    if ans.intent == "comparison":
        score = (1.0 if n_p >= 2 else 0.0) + (1.0 if n_r >= 2 else 0.0)
        score += 0.75 if "واقعیت‌های مستقیم" in text else 0.0
        score += 0.75 if "نقاط قوت" in text else 0.0
        score += 0.75 if "نقاط ضعف" in text else 0.0
        score += 0.75 if any(w in text for w in ("جمع‌بندی", "استنباط")) else 0.0
        return min(5.0, score)
    if ans.intent == "managerial":
        score = 1.0 if any(w in text for w in ("شکایت", "نارضایتی")) else 0.0
        score += 1.0 if "برند" in text else 0.0
        score += 1.0 if "نرخ توصیه" in text else 0.0
        score += 1.0 if any(w in text for w in ("تعداد بازبینی", "تعداد محصول")) else 0.0
        score += 1.0 if (n_p >= 1 or n_r >= 1) else 0.0
        return min(5.0, score)
    return 0.0


_PROXY_STOP = {"یک", "و", "در", "از", "به", "را", "برای", "با", "می", "است",
              "این", "آن", "چه", "آیا", "کدام", "محصول", "کالا", "میخواهم", "می‌خواهم"}


def _content_tokens(text: str) -> set:
    return {t for t in pt.tokenize(text) if len(t) > 1 and t not in _PROXY_STOP}


def deterministic_quality_proxy(query: str, ans) -> dict:
    """Transparent, non-LLM proxy scores in [0, 5] -- a second, reproducible
    quality/grounding axis alongside (not instead of) the LLM judge. Explicitly
    NOT a substitute for human evaluation, just a useful control when the judge
    is unavailable or (as measured this session) unreliable on some axes."""
    q_tokens = _content_tokens(query)
    evidence_tokens = _content_tokens(_sources_text(ans))
    answer_tokens = _content_tokens(ans.text)
    relevance = min(5.0, 5.0 * (len(q_tokens & (answer_tokens | evidence_tokens)) / len(q_tokens))) if q_tokens else 0.0
    validity = citation_validity(ans)
    coverage = citation_coverage(ans.text)
    grounding = (5.0 if validity == 1.0 else 2.5) if ans.missing_info else \
        5.0 * (0.75 * validity + 0.25 * min(1.0, coverage / 0.5))
    return {"proxy_relevance_0_5": round(float(relevance), 3),
            "task_completion_proxy_0_5": round(float(task_completion_proxy(query, ans)), 3),
            "proxy_grounding_0_5": round(float(grounding), 3),
            "citation_validity": round(float(validity), 3), "citation_coverage": round(float(coverage), 3)}


def _judge_score(judge, system, user) -> int | None:
    text = judge.generate(system, user)
    if not text:
        return None
    m = re.search(r"\b([0-5])\b", text)
    return int(m.group(1)) if m else None


def _sources_text(ans) -> str:
    if ans.intent == "product_qa":
        return prompts.evidence_reviews(ans.sources)
    if ans.intent == "discovery":
        return prompts.evidence_products(ans.sources)
    return ans.text


# ---- end-to-end generative eval ----------------------------------------
def evaluate_generative(assistant, queries, judge=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for q in queries:
        t0 = time.time()
        a = assistant.answer(q)
        wall = round(time.time() - t0, 3)
        proxy = deterministic_quality_proxy(q, a)
        rel = faith = None
        if judge is not None and judge.available():
            rel = _judge_score(judge, prompts.JUDGE_REL_SYS,
                               prompts.JUDGE_USER.format(query=q, answer=a.text, sources=""))
            faith = _judge_score(judge, prompts.JUDGE_FAITH_SYS,
                                 prompts.JUDGE_USER.format(query=q, answer=a.text, sources=_sources_text(a)))
        rows.append({"query": q, "intent": a.intent, "tier": a.tier, "latency_s": wall,
                     "cost_usd": round(a.cost_usd, 6), "missing_info": bool(a.missing_info),
                     "citation_coverage": citation_coverage(a.text),
                     "n_citations": len(a.citations) + len(a.review_citations),
                     **proxy,
                     "relevance": rel, "faithfulness": faith})
    per_query = pd.DataFrame(rows)
    by_intent = per_query.groupby("intent").agg(
        n=("query", "count"), mean_latency_s=("latency_s", "mean"),
        mean_citation_coverage=("citation_coverage", "mean"),
        zero_citation_rate=("n_citations", lambda s: float((s == 0).mean())),
        missing_info_rate=("missing_info", "mean"),
        mean_proxy_relevance=("proxy_relevance_0_5", "mean"),
        mean_task_completion=("task_completion_proxy_0_5", "mean"),
        mean_proxy_grounding=("proxy_grounding_0_5", "mean"),
        mean_relevance=("relevance", "mean"),
        mean_faithfulness=("faithfulness", "mean")).round(3).reset_index()
    return per_query, by_intent


def build_response_eval_queries(assistant, n_contexts: int = 2):
    """A response-quality query set SEPARATE from the demo queries, built from
    high-support categories in the current sample (not human-labeled, but not
    just re-scoring the exact examples shown in the demo either). Each query's
    expected intent is checked against the router before returning -- this is a
    lightweight regression test embedded in the eval itself: if a known-intent
    query mis-routes, that's caught here rather than silently producing a
    plausible-looking but wrong-tier answer."""
    p = assistant.c.products.copy()
    if "comment_count" not in p:
        p["comment_count"] = 0
    p["comment_count"] = pd.to_numeric(p["comment_count"], errors="coerce").fillna(0).astype(int)
    p = p[p["category1_norm"].fillna("").astype(str).str.contains(r"[آ-ی]", regex=True)].copy()

    stats = (p.groupby("category1_norm").agg(n_products=("product_id", "nunique"),
                                              n_reviews=("comment_count", "sum")).reset_index())
    stats = stats[(stats["n_products"] >= 8) & (stats["n_reviews"] >= 80)].sort_values(
        ["n_reviews", "n_products"], ascending=False)

    queries, expected, contexts = [], [], []
    for row in stats.itertuples(index=False):
        cat = pt.normalize(row.category1_norm)
        if not cat:
            continue
        g = p[p["category1_norm"] == row.category1_norm]
        pair = (g[g["comment_count"] >= 3].sort_values(
            ["comment_count", "rate_count", "product_rate_clean"], ascending=False)
            ["product_id"].head(2).astype(int).tolist())
        if len(pair) < 2:
            continue
        queries += [f"در دستهٔ {cat} چند محصول با رضایت خوب کاربران معرفی کن",
                   f"آیا محصول {pair[0]} از نظر کیفیت و تجربهٔ کاربران ارزش خرید دارد؟",
                   f"مهم‌ترین ایرادها و نقاط ضعف محصول {pair[1]} از نظر کاربران چیست؟",
                   f"محصول {pair[0]} و محصول {pair[1]} را از نظر قیمت، رضایت کاربران و نقاط قوت و ضعف مقایسه کن",
                   f"در دستهٔ {cat} شکایت‌های پرتکرار چیست و کدام محصولات با نظر کافی نرخ توصیهٔ پایین‌تری دارند؟"]
        expected += ["discovery", "product_qa", "product_qa", "comparison", "managerial"]
        contexts.append({"category": cat, "product_ids": pair,
                         "n_products": int(row.n_products), "n_reviews": int(row.n_reviews)})
        if len(contexts) >= n_contexts:
            break

    if not contexts:
        queries, expected, contexts = default_queries(assistant), None, [{"fallback_to_demo": True}]

    routing = [{"query": q, "expected": exp, "actual": assistant.router.route(q).intent,
               "passed": assistant.router.route(q).intent == exp}
              for q, exp in zip(queries, expected or [])]
    failures = [r for r in routing if not r["passed"]]
    if failures:
        log.warning("response-eval routing regression: %s", failures)

    meta = {"source": ("held_out_programmatic_multi_category_not_human_labeled"
                       if not contexts[0].get("fallback_to_demo") else "demo_fallback"),
           "n_queries": len(queries), "n_contexts": len(contexts), "contexts": contexts,
           "routing_checks_passed": not failures, "routing": routing}
    return queries, meta


def failure_analysis(assistant, judge=None, n_retrieval: int = 40, k: int = 10) -> dict:
    """Collect concrete failure examples across the system, per the brief's
    'Failure Analysis' requirement: retrieval misses, and generation failures
    (missing-info, zero-citation, or low judge faithfulness), with likely causes.
    """
    products = assistant.c.products
    # --- retrieval misses: the product's title should retrieve its own id ---
    pids = (products.sort_values("comment_count", ascending=False)["product_id"]
            .head(n_retrieval).astype(int).tolist())
    retrieval_failures = []
    for pid in pids:
        title = assistant.c.product(pid)["title_fa"]
        hits = [h["product_id"] for h in assistant.pidx.search(title, k=k)]
        if pid not in hits:
            retrieval_failures.append({
                "query": title, "gold_product_id": pid, "top_returned": hits[:3],
                "cause": "generic/short title or many near-duplicate listings dilute the match"})

    # --- generation failures on probe queries (some designed to fail) ---
    two = pids[:2] if len(pids) >= 2 else pids
    probes = [
        "یک محصول کاملاً بی‌ربط و ناموجود مثلا سفینه فضایی مریخ‌نورد می‌خواهم",  # out-of-catalog
        "نظر کاربران درباره محصول 999999999 چیست؟",                              # non-existent id
        "این دو تا رو مقایسه کن",                                               # comparison w/o ids
        f"آیا محصول {two[0]} ارزش خرید دارد؟" if two else "آیا ارزش خرید دارد؟",
    ]
    gen_failures = []
    for q in probes:
        a = assistant.answer(q)
        n_cit = len(a.citations) + len(a.review_citations)
        faith = None
        if judge is not None and judge.available():
            faith = _judge_score(judge, prompts.JUDGE_FAITH_SYS,
                                 prompts.JUDGE_USER.format(query=q, answer=a.text, sources=_sources_text(a)))
        reasons = []
        if a.missing_info:
            reasons.append("missing_info: no grounded evidence matched the query")
        if n_cit == 0:
            reasons.append("zero citations: nothing to ground an answer on")
        if a.needs_clarification:
            reasons.append("needs clarification: under-specified request")
        if faith is not None and faith <= 2:
            reasons.append(f"low faithfulness ({faith}/5) from the judge")
        if reasons:
            gen_failures.append({"query": q, "intent": a.intent, "tier": a.tier,
                                 "answer": a.text[:200], "reasons": reasons})
    return {
        "retrieval": {"n_checked": len(pids), "n_failed": len(retrieval_failures),
                      "examples": retrieval_failures[:5]},
        "generation": {"n_probes": len(probes), "n_failed": len(gen_failures),
                       "examples": gen_failures},
        "mitigations": [
            "extractive fallback guarantees a grounded ($0) answer when the LLM is unsure",
            "verify_citations strips any id the retriever did not return (no fabricated refs)",
            "the router asks for clarification instead of guessing on under-specified comparisons",
        ],
    }


def human_eval_query_set(assistant) -> list[str]:
    """A fixed, diverse ~16-query set for human-vs-judge evaluation: the 6 default
    queries (one per intent-ish scenario) plus harder/adversarial probes so the
    comparison isn't only on easy cases."""
    base = default_queries(assistant)
    pids = (assistant.c.products.sort_values("comment_count", ascending=False)["product_id"]
            .head(4).astype(int).tolist())
    extra = [
        "یک محصول کاملاً بی‌ربط و ناموجود مثلا سفینه فضایی مریخ‌نورد می‌خواهم",
        "نظر کاربران درباره محصول 999999999 چیست؟",
        "این دو تا رو مقایسه کن",
        f"بهترین قیمت برای محصول {pids[2] if len(pids) > 2 else pids[0]} چقدره؟",
        f"آیا محصول {pids[-1]} ارزش خرید دارد؟",
        "چه گوشی موبایلی زیر ۱۰ میلیون تومان با باتری خوب پیشنهاد میدی؟",
        f"محصول {pids[0]} چه ایرادهایی داره؟",
        "کیفیت ساخت این محصولات چطوره و کاربرا راضی بودن؟",
        f"محصول {pids[1]} و محصول {pids[2] if len(pids) > 2 else pids[0]} کدوم بهتره؟",
        "پرتکرارترین مشکلات کاربران در این دسته از محصولات چیه؟",
    ]
    seen, out = set(), []
    for q in base + extra:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:16]


def _answer_hash(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def build_human_eval_candidates(assistant, judge=None) -> list[dict]:
    """Generate the candidate set for human labeling: query, the system's answer,
    the evidence it was grounded on, and (if a judge is configured) the judge's
    own relevance/faithfulness scores for the same items -- so the human labels
    can be directly compared without re-running the judge later.

    Each row carries an `answer_hash` (of the exact answer text). A label is
    only ever compared against the judge score for the SAME answer text it was
    actually written for -- if the assistant's logic changes and the answer to
    the same query changes, the old label is now scoring a different, no
    longer existing response. Comparing it to a fresh judge score anyway would
    silently conflate "the judge disagrees with a human" with "the system's
    answer changed since this was labeled," which is a real methodological
    trap this project hit and is guarding against here, not hypothetically."""
    rows = []
    for q in human_eval_query_set(assistant):
        a = assistant.answer(q)
        rel = faith = None
        if judge is not None and judge.available():
            rel = _judge_score(judge, prompts.JUDGE_REL_SYS,
                               prompts.JUDGE_USER.format(query=q, answer=a.text, sources=""))
            faith = _judge_score(judge, prompts.JUDGE_FAITH_SYS,
                                 prompts.JUDGE_USER.format(query=q, answer=a.text, sources=_sources_text(a)))
        rows.append({"query": q, "intent": a.intent, "answer": a.text, "answer_hash": _answer_hash(a.text),
                     "evidence": _sources_text(a)[:1500], "judge_relevance": rel, "judge_faithfulness": faith})
    return rows


def human_eval_comparison(candidates: list[dict], labels: dict) -> dict:
    """Compare a human's 0-5 relevance/faithfulness labels (keyed by query) against
    the judge scores already attached to `candidates`. `labels` format:
    {query: {"relevance": int, "faithfulness": int, "answer_hash": str}}.
    A label without a matching `answer_hash` (or without one at all, e.g. an
    older label file predating this check) is treated as STALE -- scored
    against an answer the system no longer produces -- and excluded from
    agreement, counted separately in `n_stale` rather than silently mixed in."""
    rel_h, rel_j, faith_h, faith_j = [], [], [], []
    matched, stale = [], []
    for c in candidates:
        lab = labels.get(c["query"])
        if not lab:
            continue
        if lab.get("answer_hash") != c.get("answer_hash"):
            stale.append(c["query"])           # missing hash (older label file) also counts as stale
            continue
        matched.append(c["query"])
        if c.get("judge_relevance") is not None and lab.get("relevance") is not None:
            rel_h.append(lab["relevance"]); rel_j.append(c["judge_relevance"])
        if c.get("judge_faithfulness") is not None and lab.get("faithfulness") is not None:
            faith_h.append(lab["faithfulness"]); faith_j.append(c["judge_faithfulness"])
    return {"n_labeled": len(matched), "n_stale": len(stale), "queries_labeled": matched,
            "stale_queries": stale,
            "relevance_agreement": judge_human_agreement(rel_h, rel_j),
            "faithfulness_agreement": judge_human_agreement(faith_h, faith_j)}


def judge_human_agreement(human: list[float], judge: list[float]) -> dict:
    """Spearman correlation between judge scores and a small hand-labeled set."""
    if len(human) < 3 or len(human) != len(judge):
        return {}
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(human, judge)
        return {"spearman_rho": round(float(rho), 4), "p_value": round(float(p), 4), "n": len(human)}
    except Exception:
        r = float(np.corrcoef(human, judge)[0, 1])
        return {"pearson_r": round(r, 4), "n": len(human)}


# ---- default eval query set --------------------------------------------
def default_queries(assistant) -> list[str]:
    pids = (assistant.c.products.sort_values("comment_count", ascending=False)["product_id"]
            .head(2).astype(int).tolist())
    cat = assistant.c.products["category1_norm"].replace("نامشخص", np.nan).dropna().value_counts().index[0]
    return [
        "یک محصول اقتصادی و باکیفیت برای استفاده روزمره پیشنهاد بده",
        "یک کالای مناسب زیر ۵۰۰ هزار تومان می‌خواهم",
        f"آیا محصول {pids[0]} کیفیت خوبی دارد و کاربران راضی بودند؟",
        f"مشکلات و ایرادهای محصول {pids[1]} چیست؟",
        f"محصول {pids[0]} و محصول {pids[1]} را از نظر کیفیت مقایسه کن",
        f"پرتکرارترین شکایت‌ها و نقاط ضعف در دستهٔ {cat} چیست؟",
    ]


def run(n_retrieval: int = 30) -> dict:
    """Run the full Phase-4 suite and write artifacts/metrics/phase4_metrics.*"""
    budget = BudgetTracker()
    assistant = build_assistant()
    judge = judge_llm(budget=budget)
    log.info("assistant backend=%s | judge mode=%s", assistant.llm.backend, config.JUDGE_MODE)

    retrieval = evaluate_retrieval(assistant, n_queries=n_retrieval)
    log.info("retrieval: %s", retrieval)

    ablation_r = retrieval_ablation(assistant, n_queries=n_retrieval)
    log.info("retrieval ablation: %s", ablation_r["by_method"])

    natural = evaluate_retrieval_natural(assistant, n_queries=n_retrieval)
    log.info("natural-language retrieval benchmark: hybrid=%s lexical=%s verdict=%s",
             natural.get("hybrid"), natural.get("lexical_baseline"), natural.get("quality_verdict_by_ndcg"))

    queries, response_eval_meta = build_response_eval_queries(assistant, n_contexts=2)
    per_query, by_intent = evaluate_generative(assistant, queries, judge=judge)

    # baseline control
    baseline = LexicalBaseline(assistant.c)
    base_hits = baseline.discover("یک کالای ارزان و باکیفیت", k=5)

    # failure analysis (concrete examples + likely causes)
    failures = failure_analysis(assistant, judge=judge)
    log.info("failures: retrieval %d/%d, generation %d/%d",
             failures["retrieval"]["n_failed"], failures["retrieval"]["n_checked"],
             failures["generation"]["n_failed"], failures["generation"]["n_probes"])

    # human-vs-judge eval: generate/refresh the candidate set every run (so it's
    # always current), then compare against hand labels if the user has scored them
    candidates = build_human_eval_candidates(assistant, judge=judge)
    (config.METRICS_DIR / "human_eval_candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    labels_file = config.METRICS_DIR / "human_eval_labels.json"
    if labels_file.exists():
        labels = json.loads(labels_file.read_text(encoding="utf-8"))
        human_eval = human_eval_comparison(candidates, labels)
    else:
        human_eval = {"n_labeled": 0, "note": "no human_eval_labels.json yet -- "
                      "see artifacts/metrics/human_eval_candidates.json to label it"}
    log.info("human-eval agreement: %s", human_eval)

    # Phase-3 macro-F1 + leakage ablation (if trained)
    p3 = {}
    p3_file = config.METRICS_DIR / "phase3_metrics.json"
    if p3_file.exists():
        p3 = json.loads(p3_file.read_text(encoding="utf-8"))

    metrics = {
        "run_mode": config.RUN_MODE, "judge_mode": config.JUDGE_MODE,
        "retrieval_quality": retrieval,
        "retrieval_ablation": ablation_r,
        "retrieval_quality_natural": natural,
        "generation": {
            "n_queries": int(len(per_query)),
            "evaluation_set": response_eval_meta,
            "mean_latency_s": round(float(per_query["latency_s"].mean()), 3),
            "total_cost_usd": round(float(per_query["cost_usd"].sum()), 6),
            "mean_citation_coverage": round(float(per_query["citation_coverage"].mean()), 3),
            "mean_proxy_relevance_0_5": round(float(per_query["proxy_relevance_0_5"].mean()), 3),
            "mean_task_completion_proxy_0_5": round(float(per_query["task_completion_proxy_0_5"].mean()), 3),
            "mean_proxy_grounding_0_5": round(float(per_query["proxy_grounding_0_5"].mean()), 3),
            "mean_citation_validity": round(float(per_query["citation_validity"].mean()), 3),
            "mean_relevance": (None if per_query["relevance"].isna().all()
                               else round(float(per_query["relevance"].mean()), 3)),
            "mean_faithfulness": (None if per_query["faithfulness"].isna().all()
                                  else round(float(per_query["faithfulness"].mean()), 3)),
            "by_intent": by_intent.to_dict("records"),
        },
        "prediction_macro_f1": p3.get("test_macro_f1"),
        "prediction_grouped_macro_f1": p3.get("grouped_macro_f1"),
        "prediction_primary_macro_f1": p3.get("primary_macro_f1", p3.get("grouped_macro_f1")),
        "prediction_naive_split_product_overlap_pct": p3.get("naive_split_product_overlap_pct"),
        "prediction_leakage_ablation": p3.get("leakage_ablation"),
        "failure_analysis": failures,
        "human_eval_agreement": human_eval,
        "cost": budget.summary(),
        "baseline_control": {"query": "یک کالای ارزان و باکیفیت",
                             "top_product_ids": [int(x) for x in base_hits["product_id"].tolist()]},
    }
    (config.METRICS_DIR / "phase4_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    per_query.to_csv(config.METRICS_DIR / "phase4_per_query.csv", index=False, encoding="utf-8-sig")
    by_intent.to_csv(config.METRICS_DIR / "phase4_by_intent.csv", index=False, encoding="utf-8-sig")
    log.info("wrote phase4 metrics to %s", config.METRICS_DIR)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(run(), ensure_ascii=False, indent=2))
