"""Phase 2 — the grounded shopping assistant (the four capabilities).

`ShoppingAssistant.answer(query)` routes the query, retrieves evidence, and produces
an answer in two tiers:
  * LLM tier  — the configured model phrases the reply, then `verify_citations`
    strips any [محصول id] / [بازبینی id] the retriever didn't actually return, and
    a set of output-quality guards (`_gen`) rejects truncated, citation-free/hollow,
    or evidence-contradicting ("all reviews...") completions in favor of the
    extractive fallback.
  * extractive tier — when no LLM is configured (or it errors/empties/fails a
    guard), the reply is rendered deterministically from the evidence, so it is
    provably grounded and $0.

Aggregates (review stats, managerial complaint terms) are computed directly from the
cleaned tables, keeping hard facts separate from opinions per the brief. Evidence
selection for Q&A/comparison is polarity-aware (a "what's wrong" query surfaces
negative evidence, not whatever ranked highest on raw similarity) and validated
against placeholder/wrong-polarity text before being cited.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core import persian_text as pt
from ..core.llm import LLM
from . import prompts
from .retrieval import _minmax
from .router import Catalog, IntentRouter

# ---- managerial aggregates ---------------------------------------------
_COMPLAINT_TERMS = ["خراب", "عیب", "ایراد", "مشکل", "بوی", "شکست", "شکسته", "پاره",
                    "افتضاح", "پشیمان", "بی کیفیت", "تقلبی", "فیک", "معیوب", "جنس بد",
                    "کیفیت بد", "حساسیت", "جوش", "ترک", "خش", "لک", "چروک", "بو میده"]
_NORM_TERMS = sorted({pt.normalize(t) for t in _COMPLAINT_TERMS if pt.normalize(t)}, key=len, reverse=True)
_GENERIC_VALUES = {"", "نامشخص", "سایر", "متفرقه"}


def _rate_low_threshold(rate: pd.Series) -> float:
    """Digikala rate columns can be on a 0..5-like or 0..100-like scale
    depending on source; pick the threshold that matches the observed range
    instead of assuming one scale (a real bug class if assumed wrong)."""
    finite = pd.to_numeric(rate, errors="coerce").dropna()
    if finite.empty:
        return 2.0
    return 2.0 if float(finite.max()) <= 5.0 else 40.0


def review_stats(catalog: Catalog, product_id, light: bool = False) -> dict:
    """Aggregate one product's reviews. light=True skips the top advantages/
    disadvantages groupby — used by managerial, which ranks products but doesn't
    need per-product pros/cons (much faster over hundreds of products)."""
    rev = catalog.comments_by_product.get(int(product_id))
    if rev is None or rev.empty:
        return {"product_id": int(product_id), "n_reviews": 0, "n_labeled_recommendation": 0,
                "rec_rate": None, "not_rec_rate": None, "no_idea_rate": None, "avg_rate": None,
                "complaint_count": 0, "top_advantages": [], "top_disadvantages": []}
    n = len(rev)
    status = rev["recommendation_status"].fillna("").astype(str)
    valid_status = status.isin(("recommended", "not_recommended", "no_idea"))
    n_labeled = int(valid_status.sum())
    rec = int((status == "recommended").sum())
    not_rec = int((status == "not_recommended").sum())
    no_idea = int((status == "no_idea").sum())
    avg = float(rev["rate_clean"].dropna().mean()) if rev["rate_clean"].notna().any() else None

    def _top(col, k=3):
        if col not in rev:
            return []
        col_s = rev[col].fillna("")
        s = col_s[col_s.str.len() > 0]                # non-empty, no deprecated replace()
        if s.empty:
            return []
        ranked = (s.to_frame().assign(_l=rev.loc[s.index, "likes"].fillna(0))
                  .groupby(col, sort=False)["_l"].sum().sort_values(ascending=False))
        rows = []
        for text in ranked.head(k).index:
            candidates = rev.loc[s[s == text].index].sort_values("likes", ascending=False)
            r = candidates.iloc[0]
            rows.append({"text": str(text), "comment_id": int(r["comment_id"]),
                        "product_id": int(r["product_id"]),
                        "rate": float(r["rate_clean"]) if pd.notna(r["rate_clean"]) else None,
                        "recommendation_status": str(r["recommendation_status"]),
                        "likes": int(r["likes"]) if pd.notna(r["likes"]) else 0,
                        "is_buyer": bool(r["is_buyer"]) if pd.notna(r["is_buyer"]) else False})
        return rows

    low = _rate_low_threshold(rev["rate_clean"])
    neg = (status == "not_recommended") | (pd.to_numeric(rev["rate_clean"], errors="coerce") <= low)
    out = {"product_id": int(product_id), "n_reviews": n, "n_labeled_recommendation": n_labeled,
           "rec_rate": round(rec / n_labeled, 3) if n_labeled else None,
           "not_rec_rate": round(not_rec / n_labeled, 3) if n_labeled else None,
           "no_idea_rate": round(no_idea / n_labeled, 3) if n_labeled else None,
           "avg_rate": round(avg, 1) if avg is not None else None,
           "complaint_count": int(neg.fillna(False).sum()), "top_advantages": [], "top_disadvantages": []}
    if not light:
        out["top_advantages"] = _top("advantages_norm")
        out["top_disadvantages"] = _top("disadvantages_norm")
    return out


def _top_complaint_terms(catalog: Catalog, pids, k: int = 8):
    per = {t: re.compile(r"(?:^|\s)" + re.escape(t) + r"(?:$|\s)") for t in _NORM_TERMS}
    counts: dict = {}
    for pid in pids:
        rev = catalog.comments_by_product.get(int(pid))
        if rev is None:
            continue
        low = _rate_low_threshold(rev["rate_clean"])
        for r in rev.itertuples():
            dis = pt.normalize(getattr(r, "disadvantages_norm", "") or "")
            is_neg = str(r.recommendation_status) == "not_recommended" or (
                pd.notna(r.rate_clean) and float(r.rate_clean) <= low)
            if not is_neg and not dis.strip():
                continue
            text = " " + dis + (" " + pt.normalize(getattr(r, "body_norm", "") or "") if is_neg else "") + " "
            for term, rx in per.items():
                m = len(rx.findall(text))
                if m:
                    counts[term] = counts.get(term, 0) + m
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"term": t, "count": c} for t, c in top]


def managerial_summary(catalog: Catalog, scope, min_comments: int = 5) -> dict:
    """Category/brand analytics. Rates are REVIEW-WEIGHTED (not a plain mean of
    per-product rates, which would let a 1-review product outweigh a 500-review
    one), brand satisfaction requires a minimum sample, and the low-recommendation
    threshold falls back to a smaller minimum when the strict one yields nothing
    rather than silently returning an empty list."""
    kind, value = scope.get("kind"), pt.normalize(scope.get("value", ""))
    prods = catalog.products
    if not (kind and kind in prods):
        return {"scope": scope, "n_products": 0}
    scoped = prods[prods[kind].fillna("").str.strip() == value]
    if scoped.empty:
        return {"scope": scope, "n_products": 0}
    reviewed = set(catalog.comments_by_product)
    pids = [int(x) for x in scoped["product_id"].dropna().tolist() if int(x) in reviewed]
    if not pids:
        return {"scope": scope, "n_products": 0}
    rows = []
    for pid in pids:
        s = review_stats(catalog, pid, light=True)
        row = catalog.product(pid) or {}
        s.update({"title": row.get("title_fa", ""), "price": row.get("price_clean"),
                  "brand": row.get("brand_norm", ""), "rate": row.get("product_rate_clean")})
        rows.append(s)
    df = pd.DataFrame(rows)

    total_reviews = int(df["n_reviews"].sum())
    total_complaints = int(df["complaint_count"].sum())
    total_labeled = int(df["n_labeled_recommendation"].sum())
    weighted_rec_rate = (float((df["rec_rate"].fillna(0) * df["n_labeled_recommendation"]).sum() / total_labeled)
                         if total_labeled else None)
    product_mean_rec_rate = float(df["rec_rate"].mean()) if df["rec_rate"].notna().any() else None

    brand_sat = []
    if kind in ("category1_norm", "category2_norm", "sub_category_norm"):
        for brand, g in df.groupby("brand", dropna=True):
            brand = str(brand).strip()
            if not brand or brand in _GENERIC_VALUES:
                continue
            n_reviews_b = int(g["n_reviews"].sum())
            n_labeled_b = int(g["n_labeled_recommendation"].sum())
            if len(g) < 2 or n_reviews_b < 5 or n_labeled_b < 5:
                continue                              # too little data to trust a brand comparison
            rec_rate_b = float((g["rec_rate"].fillna(0) * g["n_labeled_recommendation"]).sum() / n_labeled_b)
            valid_rate = g["avg_rate"].notna()
            avg_rate_b = (float(np.average(g.loc[valid_rate, "avg_rate"],
                                           weights=g.loc[valid_rate, "n_reviews"].clip(lower=1)))
                         if valid_rate.any() else None)
            brand_sat.append({"brand": brand, "n_products": int(len(g)), "n_reviews": n_reviews_b,
                              "n_labeled_recommendation": n_labeled_b,
                              "review_weighted_rec_rate": round(rec_rate_b, 3),
                              "review_weighted_avg_rate": round(avg_rate_b, 2) if avg_rate_b is not None else None})
        brand_sat = sorted(brand_sat, key=lambda x: (x["n_reviews"], x["review_weighted_rec_rate"]),
                           reverse=True)[:8]

    threshold = int(min_comments)
    low = pd.DataFrame()
    for candidate in [threshold, 3, 2]:
        if candidate > threshold:
            continue
        cand_df = df[(df["n_reviews"] >= candidate) & df["rec_rate"].notna()].sort_values(
            ["rec_rate", "n_reviews"], ascending=[True, False])
        if len(cand_df):
            low, threshold = cand_df.head(8), candidate
            break
    low_list = [{"product_id": int(r.product_id), "title": r.title, "rec_rate": r.rec_rate,
                 "n_reviews": int(r.n_reviews), "rate": r.rate, "price": r.price}
                for r in low.itertuples(index=False)]

    return {"scope": scope, "product_ids": pids, "n_products": int(len(df)),
            "n_reviews": total_reviews, "n_labeled_recommendations": total_labeled,
            "total_complaints": total_complaints,
            "complaints_per_100_reviews": round(100 * total_complaints / total_reviews, 1) if total_reviews else None,
            "avg_rate": round(float(df["avg_rate"].mean()), 1) if df["avg_rate"].notna().any() else None,
            "avg_rec_rate_product_mean": round(product_mean_rec_rate, 3) if product_mean_rec_rate is not None else None,
            "review_weighted_rec_rate": round(weighted_rec_rate, 3) if weighted_rec_rate is not None else None,
            # kept for dashboard/back-compat: same as the weighted figure above
            "avg_rec_rate": round(weighted_rec_rate, 3) if weighted_rec_rate is not None else None,
            "low_recommendation_min_reviews": int(threshold) if len(low) else None,
            "brand_satisfaction": brand_sat, "low_recommendation_products": low_list,
            "top_complaint_terms": _top_complaint_terms(catalog, pids)}


# ---- comparison/QA evidence quality guards -------------------------------
# A raw disadvantages/advantages field is often just "ندارد" ("none") -- citing
# that as if it were real evidence, or citing an empty/wrong-polarity string as
# a "weakness", is a real correctness bug the reference implementation caught.
_PROCON_EMPTY_PHRASES = {"", "ندارد", "نداره", "ندارم", "نداشت", "نداشتم", "هیچ", "هیچی",
                         "موردی ندارد", "مورد خاصی ندارد", "نکته منفی ندارد",
                         "نقطه ضعف ندارد", "عیبی ندارد", "ایرادی ندارد", "مشکلی ندارد"}
_POSITIVE_EVIDENCE_HINTS = ("خوب", "عالی", "راضی", "رضایت", "مناسب", "جذاب", "ارزش خرید",
                            "پیشنهاد", "با کیفیت", "باکیفیت", "خوشش اومد", "خوشش آمد")
_STRONG_NEGATIVE_EVIDENCE_HINTS = ("بد", "ضعیف", "مشکل", "ایراد", "عیب", "خراب", "معیوب",
                                   "ناراضی", "نامناسب", "تقلبی", "فیک", "شکسته", "شکست",
                                   "پاره", "افتضاح", "بی کیفیت", "بی‌کیفیت", "کیفیت پایین",
                                   "نمی ارزد", "نمی‌ارزد", "ارزش نداره", "ارزش ندارد")


def _procon_flags(row: dict) -> dict:
    text = pt.normalize(row.get("text", "") or "")
    compact = re.sub(r"[\s،,؛;.!؟?]+", " ", text).strip()
    placeholder = compact in _PROCON_EMPTY_PHRASES or compact.replace(" ", "") in {
        "ندارد", "ندارم", "نداشت", "نداشتم", "نداردندارد", "هیچ", "هیچی"}
    has_pos = any(h in compact for h in _POSITIVE_EVIDENCE_HINTS)
    has_strong_neg = any(h in compact for h in _STRONG_NEGATIVE_EVIDENCE_HINTS)
    rate = row.get("rate")
    status = str(row.get("recommendation_status", ""))
    metadata_positive = status == "recommended" and rate is not None and pd.notna(rate) and float(rate) >= 4.0
    metadata_negative = status == "not_recommended" or (rate is not None and pd.notna(rate) and float(rate) <= 2.5)
    return {"text": compact, "placeholder": bool(placeholder), "has_positive_language": bool(has_pos),
            "has_strong_negative_language": bool(has_strong_neg), "metadata_positive": bool(metadata_positive),
            "metadata_negative": bool(metadata_negative), "evidence_type": str(row.get("evidence_type", ""))}


def _accept_positive_evidence(row: dict) -> bool:
    f = _procon_flags(row)
    if not f["text"] or f["placeholder"]:
        return False
    if f["has_strong_negative_language"] and not f["has_positive_language"]:
        return False                                  # never label a negative sentence as a strength
    if f["evidence_type"] == "advantage":
        return bool(f["has_positive_language"] or f["metadata_positive"] or not f["has_strong_negative_language"])
    return bool(f["has_positive_language"] or (f["metadata_positive"] and not f["has_strong_negative_language"]))


def _accept_negative_evidence(row: dict) -> bool:
    f = _procon_flags(row)
    if not f["text"] or f["placeholder"]:
        return False
    if f["has_positive_language"] and not f["has_strong_negative_language"]:
        return False                                  # never let a positive review appear under "weaknesses"
    if f["evidence_type"] == "retrieved_review":
        return bool(f["has_strong_negative_language"] or (f["metadata_negative"] and not f["has_positive_language"]))
    if f["evidence_type"] == "disadvantage":
        return bool(not f["has_positive_language"] and not f["placeholder"])
    return bool(f["has_strong_negative_language"] or (f["metadata_negative"] and not f["has_positive_language"]))


def _dedupe_rows(rows, k=3):
    out, seen = [], set()
    for row in rows:
        cid = int(row["comment_id"])
        if cid in seen:
            continue
        seen.add(cid)
        out.append(row)
        if len(out) >= k:
            break
    return out


# ---- citation verification + answer container ---------------------------
_CITE_P = re.compile(r"\[محصول\s*(\d+)\]")
_CITE_R = re.compile(r"\[بازبینی\s*(\d+)\]")
_CITE_ANY = re.compile(r"\[(?:محصول|بازبینی)\s*\d+\]")
_MISSING = ("اطلاعات کافی موجود نیست", "موجود نیست")
_TRAILING_CONNECTORS = {"و", "یا", "که", "اما", "با", "از", "به", "در", "برای"}
_UNIVERSAL_CLAIM_RE = re.compile(
    r"(?:همه|تمام)\s*(?:ی|ٔ)?\s*(?:بازبینی|بازبینی‌ها|نظر|نظرها)|هیچ\s*(?:بازبینی|نظر)")


def verify_citations(text, allowed_products, allowed_reviews) -> str:
    text = _CITE_P.sub(lambda m: m.group(0) if int(m.group(1)) in allowed_products else "", text)
    text = _CITE_R.sub(lambda m: m.group(0) if int(m.group(1)) in allowed_reviews else "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@dataclass
class Answer:
    intent: str
    query: str
    text: str
    citations: list = field(default_factory=list)
    review_citations: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    missing_info: bool = False
    tier: str = "extractive"
    latency_s: float = 0.0
    cost_usd: float = 0.0
    needs_clarification: bool = False


class ShoppingAssistant:
    def __init__(self, catalog: Catalog, product_index, review_retriever, llm=None, final_k: int = 8):
        self.c = catalog
        self.pidx = product_index
        self.rrev = review_retriever
        self.llm = llm or LLM(mode="extractive")
        self.router = IntentRouter(catalog)
        self.final_k = final_k
        self.generated_rejections = 0                  # how often the LLM tier failed a quality guard

    def answer(self, query) -> Answer:
        t0 = time.time()
        route = self.router.route(query)
        if route.needs_clarification:
            a = Answer(route.intent, query, "برای مقایسه، دو محصول را با شناسه مشخص کنید.",
                       needs_clarification=True)
        elif route.intent == "discovery":
            a = self._discover(query, route.filters)
        elif route.intent == "product_qa":
            a = self._qa(query, route.product_id)
        elif route.intent == "comparison":
            a = self._compare(query, route.product_ids)
        else:
            a = self._managerial(query, route.scope)
        a.latency_s = round(time.time() - t0, 3)
        a.cost_usd = getattr(self.llm, "last_cost_usd", 0.0)
        return a

    def _gen(self, system, user, extractive, allowed_p, allowed_r):
        """Generate with the LLM tier, or fall back to the deterministic extractive
        answer if the reply is empty, hollow (citation-only), missing a required
        citation, truncated mid-sentence, or asserts an absolute claim ("all
        reviews...") -- a real failure mode caught in testing where the model
        claimed every cited review rated >=3 while one cited review was a 0."""
        g = self.llm.generate(system, user)
        if not g or not g.strip():
            return extractive, "extractive"
        clean = verify_citations(g, allowed_p, allowed_r)

        without_cites = _CITE_ANY.sub(" ", clean)
        meaningful_tokens = [t for t in pt.tokenize(without_cites) if len(t) > 1]
        if len(meaningful_tokens) < 8:
            self.generated_rejections += 1
            return extractive, "extractive"

        if (allowed_p or allowed_r) and not _CITE_ANY.search(clean):
            self.generated_rejections += 1
            return extractive, "extractive"

        tail = pt.normalize(clean).rstrip()
        tail_tokens = pt.tokenize(tail)
        last_token = tail_tokens[-1] if tail_tokens else ""
        if (tail and tail[-1] in "،؛,:-") or last_token in _TRAILING_CONNECTORS:
            self.generated_rejections += 1            # looks token-truncated mid-sentence
            return extractive, "extractive"

        if _UNIVERSAL_CLAIM_RE.search(tail):
            self.generated_rejections += 1             # absolute claim, risk of contradicting evidence
            return extractive, "extractive"

        return clean, "llm"

    def _discover(self, query, filters):
        hits = self.pidx.search(query, filters, k=max(40, self.final_k * 5))
        if not hits:
            return Answer("discovery", query,
                          "هیچ محصولی با این فیلترها یافت نشد (اطلاعات کافی موجود نیست).", missing_info=True)

        qn = pt.normalize(query)
        cheap_pref = any(w in qn for w in ("اقتصادی", "ارزان", "ارزون", "قیمت مناسب", "مقرون به صرفه"))
        satisfaction_pref = any(w in qn for w in ("رضایت", "راضی", "توصیه کاربران", "پیشنهاد کاربران",
                                                  "نظر کاربران خوب", "خریداران راضی"))
        quality_pref = any(w in qn for w in ("باکیفیت", "کیفیت خوب", "بهترین", "خوب")) or satisfaction_pref

        # Retrieval finds relevant candidates; when the request mentions price,
        # quality, or user satisfaction, rerank with those as real data signals
        # (price/rating/review-recommend-rate) instead of leaving it to text match.
        if cheap_pref or quality_pref:
            retrieval_score = _minmax(np.array([h["score"] for h in hits], dtype=float))

            prices = np.array([float(h["price"]) if h["price"] not in (None, 0) else np.nan for h in hits])
            price_score = np.full(len(hits), 0.5, dtype=float)
            good_price = np.isfinite(prices)
            if good_price.sum() >= 2:
                price_score[good_price] = 1.0 - _minmax(np.log1p(prices[good_price]))

            rates = np.array([float(h["rate"]) if h["rate"] is not None else np.nan for h in hits])
            quality_score = np.full(len(hits), 0.5, dtype=float)
            good_rate = np.isfinite(rates)
            if good_rate.sum() >= 2:
                quality_score[good_rate] = _minmax(rates[good_rate])

            counts = np.log1p(np.array([h.get("rate_count", 0) for h in hits], dtype=float))
            confidence = _minmax(counts)

            review_rec_rates, review_support = [], []
            for h in hits:
                stats = review_stats(self.c, int(h["product_id"]))
                rec = stats.get("rec_rate")
                n_lab = int(stats.get("n_labeled_recommendation", 0) or 0)
                review_rec_rates.append(np.nan if rec is None else float(rec))
                review_support.append(n_lab)
                h["review_rec_rate"] = rec
                h["review_labeled_count"] = n_lab
            review_rec_rates = np.array(review_rec_rates, dtype=float)
            satisfaction_score = np.full(len(hits), 0.5, dtype=float)
            good_rec = np.isfinite(review_rec_rates)
            if good_rec.sum() >= 2:
                satisfaction_score[good_rec] = _minmax(review_rec_rates[good_rec])
            review_confidence = _minmax(np.log1p(np.array(review_support, dtype=float)))

            if satisfaction_pref:
                final_score = 0.60 * retrieval_score + 0.18 * satisfaction_score + 0.07 * review_confidence
                if cheap_pref:
                    final_score += 0.08 * price_score
                if quality_pref:
                    final_score += 0.05 * quality_score
                final_score += 0.02 * confidence
            else:
                final_score = 0.72 * retrieval_score
                if cheap_pref and quality_pref:
                    final_score += 0.13 * price_score + 0.10 * quality_score + 0.05 * confidence
                elif cheap_pref:
                    final_score += 0.23 * price_score + 0.05 * confidence
                else:
                    final_score += 0.23 * quality_score + 0.05 * confidence

            order = np.argsort(-final_score, kind="stable")
            hits = [hits[int(i)] for i in order]
            for rank, h in enumerate(hits, 1):
                h["rank"] = rank
                reasons = []
                if cheap_pref and h["price"] is not None:
                    reasons.append("قیمت مناسب‌تر در میان نامزدهای بازیابی‌شده")
                if quality_pref and h["rate"] is not None:
                    reasons.append(f"امتیاز {h['rate']} با {h.get('rate_count', 0)} رأی")
                if satisfaction_pref:
                    rec, n_lab = h.get("review_rec_rate"), int(h.get("review_labeled_count", 0) or 0)
                    reasons.append(f"{rec:.0%} توصیه در {n_lab} بازبینی دارای وضعیت معتبر" if rec is not None and n_lab
                                   else "شواهد وضعیت پیشنهاد کاربران در نمونه کافی نیست")
                h["preference_reason"] = "؛ ".join(reasons)

        ev = hits[:self.final_k]
        allowed = {h["product_id"] for h in ev}
        lines = []
        for h in ev:
            reason = h.get("preference_reason", "")
            reason_text = f"؛ دلیل: {reason}" if reason else ""
            lines.append(f"{h['rank']}. [محصول {h['product_id']}] {h['title']} — امتیاز "
                        f"{h['rate'] if h['rate'] is not None else 'نامشخص'}، قیمت {pt.format_toman(h['price'])} تومان،"
                        f" برند {h['brand'] or 'نامشخص'}{reason_text}")
        extractive = "پیشنهادهای برتر:\n" + "\n".join(lines)

        text, tier = self._gen(prompts.DISCOVERY_SYSTEM.format(k=len(ev)),
                               f"درخواست: {query}\nفیلترها: {filters}\n\nمدارک:\n"
                               f"{prompts.evidence_products(ev)}\n\nپاسخ:",
                               extractive, allowed, set())
        return Answer("discovery", query, text, citations=list(allowed), sources=ev, tier=tier,
                      missing_info=any(p in text for p in _MISSING))

    def _qa(self, query, pid):
        if pid is None:
            return Answer("product_qa", query,
                          "محصول شناسایی نشد؛ شناسه یا نام کامل را ذکر کنید.", missing_info=True)
        prod = self.c.product(pid) or {}
        title = prod.get("title_fa", "")
        candidates = self.rrev.retrieve(query, pid, k=max(20, self.final_k * 3))
        polarity = self.rrev._polarity(query)

        def _flags(row):
            txt = pt.normalize(row.get("text", ""))
            has_neg = any(c in txt for c in self.rrev._NEGATIVE_CUES) or row.get("has_disadvantage", False)
            has_pos = any(c in txt for c in self.rrev._POSITIVE_CUES) or row.get("has_advantage", False)
            return has_pos, has_neg

        if polarity == "negative":
            hits = [h for h in candidates if _flags(h)[1]
                   or (h.get("recommendation_status") == "not_recommended"
                       and h.get("rate") is not None and float(h["rate"]) <= 3.0)][:self.final_k]
        elif polarity == "positive":
            hits = []
            for h in candidates:
                has_pos, has_neg = _flags(h)
                metadata_positive = (h.get("recommendation_status") == "recommended"
                                     and h.get("rate") is not None and float(h["rate"]) >= 4.0)
                if (has_pos or metadata_positive) and not has_neg:  # never reuse a negative review as positive evidence
                    hits.append(h)
            hits = hits[:self.final_k]
        else:
            hits = candidates[:self.final_k]

        if not hits:
            if candidates and polarity in ("negative", "positive"):
                direction = "منفی" if polarity == "negative" else "مثبت"
                return Answer("product_qa", query,
                              f"برای [محصول {pid}] بازبینی مرتبط وجود دارد، اما شواهد {direction} کافی برای پاسخ "
                              f"مطمئن پیدا نشد (اطلاعات کافی موجود نیست).",
                              citations=[pid], sources=candidates[:self.final_k], missing_info=True)
            return Answer("product_qa", query,
                          f"برای [محصول {pid}] بازبینی‌ای نیست (اطلاعات کافی موجود نیست).",
                          citations=[pid], missing_info=True)

        allowed_r = {h["comment_id"] for h in hits}
        facts = review_stats(self.c, pid)
        if facts["n_reviews"] and facts.get("n_labeled_recommendation"):
            head = (f"دربارهٔ [محصول {pid}] ({title}): {facts['n_reviews']} بازبینی در نمونه داریم؛ "
                    f"از {facts['n_labeled_recommendation']} بازبینی دارای وضعیت پیشنهاد، "
                    f"{facts['rec_rate']:.0%} توصیه، {facts['not_rec_rate']:.0%} عدم توصیه و "
                    f"{facts['no_idea_rate']:.0%} بدون نظر قطعی بوده‌اند. [محصول {pid}]")
        elif facts["n_reviews"]:
            head = (f"دربارهٔ [محصول {pid}] ({title}): {facts['n_reviews']} بازبینی در نمونه داریم، اما وضعیت "
                    f"پیشنهاد معتبر کافی ثبت نشده است. [محصول {pid}]")
        else:
            head = f"[محصول {pid}] ({title}):"

        section = ("ایرادها، مشکلات و نکات منفی مرتبط:" if polarity == "negative" else
                  "نقاط قوت و تجربه‌های مثبت مرتبط:" if polarity == "positive" else "بازبینی‌های مرتبط با پرسش:")
        extractive = head + "\n" + section + "\n" + "\n".join(
            f"- «{h['text']}» [بازبینی {h['comment_id']}]" for h in hits)

        text, tier = self._gen(prompts.QA_SYSTEM.format(max_lines=8),
                               f"محصول: [محصول {pid}] {title}\nپرسش: {query}\n\nمدارک:\n"
                               f"{prompts.evidence_reviews(hits)}\n\nپاسخ:",
                               extractive, {pid}, allowed_r)
        return Answer("product_qa", query, text, citations=[pid], review_citations=list(allowed_r),
                      sources=hits, tier=tier, missing_info=any(p in text for p in _MISSING))

    def _compare(self, query, pids):
        facts, positive_by_product, negative_by_product, all_review_rows = [], {}, {}, []

        def _review_polarity(row):
            text_n = pt.normalize(row.get("text", ""))
            rate, status = row.get("rate"), row.get("recommendation_status")
            has_neg = (any(c in text_n for c in self.rrev._NEGATIVE_CUES) or row.get("has_disadvantage", False)
                      or status == "not_recommended" or (rate is not None and pd.notna(rate) and float(rate) <= 2.5))
            has_pos = (any(c in text_n for c in self.rrev._POSITIVE_CUES) or row.get("has_advantage", False)
                      or (status == "recommended" and rate is not None and pd.notna(rate) and float(rate) >= 4.0))
            return "negative" if has_neg else ("positive" if has_pos else "neutral")

        for pid in pids:
            p = self.c.product(pid) or {}
            stats = review_stats(self.c, pid)
            facts.append({"product_id": int(pid), "title": p.get("title_fa", "نامشخص"),
                          "price": p.get("price_clean"), "rate": p.get("product_rate_clean"),
                          "brand": p.get("brand_norm", ""), "stats": stats})

            positives = [dict(r, evidence_type="advantage") for r in stats.get("top_advantages", [])]
            negatives = [dict(r, evidence_type="disadvantage") for r in stats.get("top_disadvantages", [])]
            for row in self.rrev.retrieve(query, pid, k=max(6, min(10, self.final_k + 2))):
                pol = _review_polarity(row)
                if pol == "negative":
                    negatives.append(dict(row, evidence_type="retrieved_review"))
                elif pol == "positive":
                    positives.append(dict(row, evidence_type="retrieved_review"))

            positives = _dedupe_rows([r for r in positives if _accept_positive_evidence(r)], k=3)
            negatives = _dedupe_rows([r for r in negatives if _accept_negative_evidence(r)], k=3)
            positive_by_product[int(pid)], negative_by_product[int(pid)] = positives, negatives
            all_review_rows += positives + negatives

        review_rows, seen = [], set()
        for row in all_review_rows:
            cid = int(row["comment_id"])
            if cid not in seen:
                seen.add(cid)
                review_rows.append(row)

        allowed_p = {f["product_id"] for f in facts}
        allowed_r = {r["comment_id"] for r in review_rows}

        fact_lines = ["۱) واقعیت‌های مستقیم محصول"]
        for f in facts:
            rec = f["stats"].get("rec_rate")
            fact_lines.append(f"- [محصول {f['product_id']}] {f['title']} | برند: {f['brand'] or 'نامشخص'} | "
                              f"قیمت: {pt.format_toman(f['price'])} تومان | امتیاز محصول: "
                              f"{f['rate'] if f['rate'] is not None else 'نامشخص'} | "
                              f"تعداد بازبینی: {f['stats']['n_reviews']} | "
                              f"نرخ توصیه: {f'{rec:.0%}' if rec is not None else 'نامشخص'} | "
                              f"بازبینی‌های دارای نشانهٔ شکایت: {int(f['stats'].get('complaint_count', 0) or 0)}")

        evidence_lines = ["۲) شواهد بازبینی‌های کاربران"]
        for f in facts:
            pid = f["product_id"]
            evidence_lines.append(f"- [محصول {pid}] نقاط قوت / تجربه‌های مثبت:")
            pos = positive_by_product.get(pid, [])
            evidence_lines += [f"  - «{r['text']}» [بازبینی {r['comment_id']}]" for r in pos] or \
                ["  - شواهد مثبت کافی در نمونه پیدا نشد."]
            evidence_lines.append(f"- [محصول {pid}] نقاط ضعف / تجربه‌های منفی:")
            neg = negative_by_product.get(pid, [])
            evidence_lines += [f"  - «{r['text']}» [بازبینی {r['comment_id']}]" for r in neg] or \
                ["  - نقطهٔ ضعف قابل اتکای کافی در نمونه پیدا نشد."]

        inference_lines = ["۳) جمع‌بندی / استنباط از داده‌های بالا"]
        rec_candidates = [f for f in facts if f["stats"].get("rec_rate") is not None]
        if rec_candidates:
            best = max(f["stats"]["rec_rate"] for f in rec_candidates)
            winners = [f for f in rec_candidates if abs(f["stats"]["rec_rate"] - best) < 1e-12]
            if len(winners) == 1:
                w = winners[0]
                inference_lines.append(f"- اگر رضایت کاربران اولویت اصلی باشد، [محصول {w['product_id']}] در "
                                       f"دادهٔ فعلی نرخ توصیهٔ بالاتری دارد ({w['stats']['rec_rate']:.0%}).")
            else:
                tied = " و ".join(f"[محصول {f['product_id']}]" for f in winners)
                inference_lines.append(f"- از نظر نرخ توصیه، {tied} در دادهٔ فعلی برابرند ({best:.0%})؛ شواهد "
                                       f"قوت/ضعف و قیمت برای تصمیم نهایی مهم می‌شوند.")
        priced = [f for f in facts if f["price"] is not None and pd.notna(f["price"]) and float(f["price"]) > 0]
        if priced:
            cheapest = min(priced, key=lambda x: float(x["price"]))
            inference_lines.append(f"- اگر قیمت اولویت اصلی باشد، [محصول {cheapest['product_id']}] با قیمت "
                                   f"{pt.format_toman(cheapest['price'])} تومان ارزان‌تر است.")
        complaint_candidates = [f for f in facts if int(f["stats"].get("n_reviews", 0) or 0) > 0]
        if len(complaint_candidates) >= 2:
            rates = sorted(((int(f["stats"].get("complaint_count", 0) or 0)
                            / max(1, int(f["stats"].get("n_reviews", 0) or 0)), f) for f in complaint_candidates),
                           key=lambda x: x[0])
            low_rate, low_item = rates[0]
            high_rate = rates[-1][0]
            if high_rate - low_rate >= 0.10:
                inference_lines.append(f"- در نمونهٔ فعلی، [محصول {low_item['product_id']}] سهم کمتری از "
                                       f"بازبینی‌های دارای نشانهٔ شکایت دارد ({low_rate:.0%}).")
        if len(inference_lines) == 1:
            inference_lines.append("- برای نتیجه‌گیری قطعی اطلاعات کافی موجود نیست.")

        # Comparison stays fully deterministic: every fact and inference above is
        # computed directly from selected data, not phrased by an LLM.
        text = "\n".join(fact_lines + [""] + evidence_lines + [""] + inference_lines)
        return Answer("comparison", query, text, citations=list(allowed_p), review_citations=list(allowed_r),
                      sources={"facts": facts, "positive_reviews": positive_by_product,
                              "negative_reviews": negative_by_product},
                      tier="extractive", missing_info=any(p in text for p in _MISSING))

    def _managerial(self, query, scope):
        summary = managerial_summary(self.c, scope)
        if not summary.get("n_products"):
            return Answer("managerial", query,
                          f"برای دامنهٔ {scope} داده‌ای نیست (اطلاعات کافی موجود نیست).", missing_info=True)
        terms = " ".join(t["term"] for t in summary.get("top_complaint_terms", [])[:3])
        low_pids = [int(p["product_id"]) for p in summary.get("low_recommendation_products", [])]
        pids = low_pids + [int(p) for p in summary.get("product_ids", []) if int(p) not in set(low_pids)]

        def _is_negative_complaint(row):
            text_n = pt.normalize(row.get("text", ""))
            rate = row.get("rate")
            low_rate = rate is not None and pd.notna(rate) and float(rate) <= 2.5
            return bool(any(c in text_n for c in self.rrev._NEGATIVE_CUES) or row.get("has_disadvantage", False)
                       or row.get("recommendation_status") == "not_recommended" or low_rate)

        complaint_reviews, seen = [], set()
        if terms and pids:
            for pid in pids[:20]:
                for r in self.rrev.retrieve(terms, pid, k=2):
                    cid = int(r["comment_id"])
                    if cid in seen or not _is_negative_complaint(r):
                        continue
                    seen.add(cid)
                    complaint_reviews.append(r)
        complaint_reviews = complaint_reviews[:8]

        allowed_p = {p["product_id"] for p in summary["low_recommendation_products"]}
        allowed_r = {r["comment_id"] for r in complaint_reviews}
        weighted_rec = summary.get("review_weighted_rec_rate")
        threshold = summary.get("low_recommendation_min_reviews")

        ev = [f"دامنه: {scope.get('value')}",
             f"تعداد محصول دارای بازبینی: {summary['n_products']} | تعداد بازبینی: {summary['n_reviews']} | "
             f"وضعیت پیشنهاد معتبر: {summary.get('n_labeled_recommendations', 0)} | "
             f"میانگین امتیاز بازبینی: {summary['avg_rate']} | "
             f"نرخ توصیهٔ وزن‌دار در بین وضعیت‌های معتبر: {f'{weighted_rec:.1%}' if weighted_rec is not None else 'نامشخص'}",
             "شکایت‌های پرتکرار: " + (", ".join(f"{t['term']}({t['count']})"
                                                for t in summary["top_complaint_terms"]) or "نامشخص")]
        brands = summary.get("brand_satisfaction") or []
        if brands:
            ev.append("الگوی رضایت برندها:")
            ev += [f"- {b['brand']} | {b['n_products']} محصول | {b['n_reviews']} بازبینی | "
                  f"نرخ توصیهٔ وزن‌دار {b['review_weighted_rec_rate']:.1%}" for b in brands[:5]]
        ev.append(f"محصولات با نرخ توصیهٔ پایین (حداقل {threshold} بازبینی در نمونه):" if threshold is not None
                  else "محصولات با نرخ توصیهٔ پایین: دادهٔ کافی برای آستانهٔ حداقل دو بازبینی موجود نیست.")
        ev += [f"- [محصول {p['product_id']}] {p['title']} | نرخ توصیه {p['rec_rate']:.1%} | {p['n_reviews']} بازبینی"
              for p in summary["low_recommendation_products"]]
        evidence_text = "\n".join(ev)
        extractive = evidence_text + ("\n\nنمونهٔ شکایت‌های قابل ردیابی:\n" + "\n".join(
            f"- «{r['text']}» [بازبینی {r['comment_id']}]" for r in complaint_reviews) if complaint_reviews else "")

        # Deterministic by design: prevents unsupported generalizations about
        # price/quality while cutting API cost and latency for analytics queries.
        return Answer("managerial", query, extractive, citations=list(allowed_p), review_citations=list(allowed_r),
                      sources=[summary], tier="extractive",
                      missing_info=any(p in extractive for p in _MISSING))


# ---- non-LLM lexical baseline (evaluation control) ----------------------
class LexicalBaseline:
    """Control: token-overlap product retrieval + arithmetic, no embeddings, no LLM."""

    def __init__(self, catalog: Catalog):
        self.c = catalog

    def discover(self, query, k: int = 5):
        from .router import extract_filters
        from .retrieval import product_filter_mask
        f = extract_filters(self.c, query)
        mask = product_filter_mask(self.c.products, f)
        sub = self.c.products[mask]
        q = set(pt.tokenize(query))
        sub = sub.assign(_ov=sub["product_text_norm"].map(lambda t: len(q & set(pt.tokenize(t)))))
        sub = sub.sort_values(["_ov", "product_rate_clean", "rate_count"], ascending=False).head(k)
        return sub[["product_id", "title_fa", "price_clean", "product_rate_clean"]]


def build_assistant(llm=None, final_k: int = 8) -> ShoppingAssistant:
    """Wire a ready-to-use assistant from the cached index + cleaned comments."""
    from . import retrieval
    idx = retrieval.ProductIndex.load()
    by_product = retrieval.load_comments_by_product()
    catalog = Catalog.build(idx.products, by_product)
    rrev = retrieval.ReviewRetriever(by_product)
    return ShoppingAssistant(catalog, idx, rrev, llm=llm, final_k=final_k)
