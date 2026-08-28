"""Deterministic intent router + entity resolution for the assistant.

`Catalog` holds the lightweight lookups the router and assistant need (product-id
map, category/brand vocabularies, reviews grouped by product). `IntentRouter.route`
maps a Persian query to one of four intents and extracts its slots — product ids,
managerial scope, and structured filters (brand/category/price) — with no LLM call,
so routing is free and reproducible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .. import config
from ..core import persian_text as pt

_GENERIC = {"متفرقه", "نامشخص", "سایر"}
_CMP = ("مقایسه", "تفاوت", "فرق", "کدوم بهتر", "کدام بهتر", "بهتره", "بهتر است", "vs", "در برابر")
_QA = ("آیا", "چطور", "چگونه", "چرا", "چند", "چقدر", "مشکل", "سایز", "اندازه", "جنس",
       "کیفیت", "مناسب", "خوب", "بد", "رضایت", "ارزش خرید", "باتری", "دوام", "نظر")
# NOTE: deliberately no bare "دسته" ("category") here -- `scope` already detects
# a category/brand mention, so if "دسته" alone counted as an analytical cue,
# every discovery request that names a category ("در دستهٔ X چند محصول خوب
# هست؟") would incorrectly route to managerial. These cues represent genuine
# analytical/complaint intent, not just "a category was named".
_MNG = ("شکایت", "شکایات", "عملکرد", "تحلیل", "برند", "مشکلات", "پرفروش",
        "کمترین", "بیشترین", "محبوب", "نرخ", "نارضایتی", "بازار", "مدیر")


@dataclass
class Catalog:
    products: pd.DataFrame
    comments_by_product: dict
    category_values: dict = field(default_factory=dict)
    brand_values: set = field(default_factory=set)
    _by_id: pd.DataFrame = None                       # products indexed by product_id
    reviewed_title_tokens: dict = field(default_factory=dict)  # pid -> token set (reviewed only)

    @classmethod
    def build(cls, products: pd.DataFrame, comments_by_product: dict) -> "Catalog":
        products = products.reset_index(drop=True)
        by_id = products.dropna(subset=["product_id"]).copy()
        by_id["product_id"] = by_id["product_id"].astype(int)
        by_id = by_id.set_index("product_id", drop=False)
        cats = {c: {v for v in products[c].dropna().unique() if str(v).strip() and str(v) not in _GENERIC}
                for c in ("category1_norm", "category2_norm", "sub_category_norm") if c in products}
        brands = {v for v in products["brand_norm"].dropna().unique() if str(v).strip() and str(v) not in _GENERIC}
        # only reviewed products are candidates for fuzzy name->id resolution (QA needs reviews)
        reviewed = {}
        for pid in comments_by_product:
            if pid in by_id.index:
                reviewed[int(pid)] = set(pt.tokenize_norm(by_id.at[int(pid), "title_norm"]))
        return cls(products, comments_by_product, cats, brands, by_id, reviewed)

    @property
    def product_lookup(self):                          # kept for compatibility with callers
        return self._by_id

    def product(self, pid):
        pid = int(pid)
        if self._by_id is not None and pid in self._by_id.index:
            return self._by_id.loc[pid].to_dict()
        return None


def match_known_value(query, values, min_ratio: float = 0.6):
    # `values` come from already-normalized *_norm columns, so avoid re-running the
    # (heavy) hazm normalizer per value — normalize the query once, split the rest.
    t = pt.normalize(query)
    t_tokens = set(pt.tokenize_norm(t))
    best, best_score = None, 0.0
    for v in values:
        vn = str(v).strip()
        if not vn or len(vn) < 3 or vn in _GENERIC:
            continue
        v_tokens = set(pt.tokenize_norm(vn))
        if not v_tokens:
            continue
        if vn in t:
            score = len(vn) + 100.0
        else:
            ratio = len(t_tokens & v_tokens) / max(1, len(v_tokens))
            if ratio < min_ratio:
                continue
            score = ratio * len(v_tokens)
        if score > best_score:
            best, best_score = vn, score
    return best


def extract_product_ids(catalog: Catalog, query) -> list[int]:
    t = pt.normalize(query)
    found: list[int] = []
    for m in re.finditer(r"\b(\d{6,9})\b", t):
        pid = int(m.group(1))
        if pid in catalog._by_id.index and pid not in found:
            found.append(pid)
    return found


def resolve_product_id(catalog: Catalog, text):
    """Fuzzy map a description to a product id, searching only reviewed products
    (a Q&A only makes sense for a product that actually has reviews)."""
    t = pt.normalize(text)
    m = re.search(r"\b(\d{6,9})\b", t)
    if m and int(m.group(1)) in catalog._by_id.index:
        return int(m.group(1))
    q_tokens = set(pt.tokenize(t))
    if not q_tokens:
        return None
    best_pid, best = None, 0.0
    for pid, title_tokens in catalog.reviewed_title_tokens.items():
        if not title_tokens:
            continue
        score = len(q_tokens & title_tokens) / max(1, len(title_tokens))
        if score > best:
            best_pid, best = pid, score
    return best_pid if best >= 0.5 else None


def resolve_scope(catalog: Catalog, text) -> dict:
    best, best_score = None, 0.0
    for kind, values in catalog.category_values.items():
        v = match_known_value(text, values)
        if v and len(v) > best_score:
            best, best_score = (kind, v), len(v)
    v = match_known_value(text, catalog.brand_values)
    if v and len(v) > best_score:
        best, best_score = ("brand_norm", v), len(v)
    return {"kind": best[0], "value": best[1]} if best else {}


def extract_filters(catalog: Catalog, query) -> dict:
    filters: dict = {}
    best_cat, best_len = None, 0
    for values in catalog.category_values.values():
        v = match_known_value(query, values)
        if v and len(v) > best_len:
            best_cat, best_len = v, len(v)
    if best_cat:
        filters["category"] = best_cat
    b = match_known_value(query, catalog.brand_values)
    if b:
        filters["brand"] = b
    filters.update(pt.extract_price_constraint(query, config.TOMAN_TO_RIAL))
    return filters


@dataclass
class Route:
    intent: str
    product_id: int | None = None
    product_ids: list = field(default_factory=list)
    scope: dict = field(default_factory=dict)
    filters: dict = field(default_factory=dict)
    needs_clarification: bool = False


class IntentRouter:
    def __init__(self, catalog: Catalog):
        self.c = catalog

    def route(self, query) -> Route:
        t = pt.normalize(query)
        ids = extract_product_ids(self.c, query)
        scope = resolve_scope(self.c, query)
        if len(ids) >= 2 or (any(w in t for w in _CMP) and len(ids) >= 1):
            if len(ids) >= 2:
                return Route("comparison", product_ids=ids)
            return Route("comparison", product_ids=ids, needs_clarification=True)
        has_qa_cue = any(w in t for w in _QA)
        # An explicit numeric id is unambiguous -> product_qa regardless of scope.
        if ids and has_qa_cue:
            return Route("product_qa", product_id=ids[0])
        # A confident managerial-scope match takes priority over the FUZZY (no
        # explicit id) name->product resolution below: fuzzy token-overlap
        # matching can accidentally hit some unrelated reviewed product's title
        # just because a generic QA cue word (like "چند") appears in a query
        # that is actually asking about a whole category, e.g. "در دستهٔ اسباب
        # بازی چند محصول ... معرفی کن" -- caught by the response-eval routing
        # regression check.
        if scope and any(w in t for w in _MNG):
            return Route("managerial", scope=scope)
        if has_qa_cue:
            pid = resolve_product_id(self.c, query)
            if pid is not None:
                return Route("product_qa", product_id=pid)
        return Route("discovery", filters=extract_filters(self.c, query))
