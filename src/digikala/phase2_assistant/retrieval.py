"""Phase 2 retrieval — hybrid dense + BM25 over products, per-product review search.

Design (chosen for full-data scale): every product is embedded once and the vectors
are cached to disk, so product discovery searches the whole catalogue. Reviews are
*not* embedded globally (6M is impractical on a 6 GB GPU); instead a product's own
reviews are embedded on demand when the user asks about that product — a product has
at most a few hundred reviews, so this is fast and memory-flat.

Dense cosine similarity and a from-scratch BM25 are fused with Reciprocal Rank
Fusion (RRF); structured filters (price/brand/category/exclude-fake) are applied
*before* ranking, and reviews get a likes/buyer/rating signal rerank.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..core import persian_text as pt

log = logging.getLogger("digikala.retrieval")
_model = None


# ---- embedding model (lazy, GPU with CPU fallback) ----------------------
def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        device = config.EMBEDDING_DEVICE
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"
        log.info("loading %s on %s", config.EMBEDDING_MODEL, device)
        _model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    return _model


def embed(texts, batch_size: int = 256) -> np.ndarray:
    texts = [t or "" for t in texts]
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    return get_model().encode(texts, batch_size=batch_size, convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=len(texts) > 5000).astype("float32")


# ---- BM25 (Okapi) — the sparse half of hybrid retrieval -----------------
class BM25Okapi:
    """Okapi BM25 backed by a scipy-sparse term-document matrix.

    A CountVectorizer builds the doc×term counts once (C-optimized), so this scales
    to ~1M products: scoring a query touches only the columns of its terms, and the
    whole index (matrix + idf + doc lengths + vocab) saves/loads from disk in
    seconds instead of rebuilding a Python inverted index every process start.
    """

    def __init__(self, counts, doc_len, idf, vocab, k1: float = 1.5, b: float = 0.75):
        from scipy.sparse import csc_matrix
        self.k1, self.b = k1, b
        self.counts = counts.tocsc() if not isinstance(counts, csc_matrix) else counts
        self.doc_len = doc_len.astype(np.float32)
        self.n = self.counts.shape[0]
        self.avgdl = float(self.doc_len.mean()) if self.n else 1.0
        self.idf = idf.astype(np.float32)
        self.vocab = vocab                            # term -> column index

    @classmethod
    def from_texts(cls, texts, k1: float = 1.5, b: float = 0.75):
        from sklearn.feature_extraction.text import CountVectorizer
        # corpus texts are already normalized (*_norm columns) -> fast regex tokenizer
        vec = CountVectorizer(tokenizer=pt.tokenize_norm, token_pattern=None,
                              lowercase=False, preprocessor=None)
        counts = vec.fit_transform(texts)             # docs x terms, CSR
        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        n = counts.shape[0]
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        return cls(counts, doc_len, idf, vec.vocabulary_, k1, b)

    def get_scores(self, query) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        avgdl = max(self.avgdl, 1e-6)
        denom_len = self.k1 * (1 - self.b + self.b * self.doc_len / avgdl)
        for term in set(pt.tokenize(query)):
            col = self.vocab.get(term)
            if col is None:
                continue
            c = self.counts.getcol(col)               # sparse column of term freqs
            rows = c.indices
            freq = c.data.astype(np.float32)
            scores[rows] += self.idf[col] * freq * (self.k1 + 1) / (freq + denom_len[rows])
        return scores

    def save(self, path):
        from scipy.sparse import save_npz
        path = Path(path)
        save_npz(path / "bm25_counts.npz", self.counts)
        np.save(path / "bm25_doc_len.npy", self.doc_len)
        np.save(path / "bm25_idf.npy", self.idf)
        import json
        (path / "bm25_vocab.json").write_text(json.dumps(self.vocab), encoding="utf-8")

    @classmethod
    def load(cls, path, k1: float = 1.5, b: float = 0.75):
        from scipy.sparse import load_npz
        import json
        path = Path(path)
        counts = load_npz(path / "bm25_counts.npz")
        doc_len = np.load(path / "bm25_doc_len.npy")
        idf = np.load(path / "bm25_idf.npy")
        vocab = json.loads((path / "bm25_vocab.json").read_text(encoding="utf-8"))
        return cls(counts, doc_len, idf, vocab, k1, b)


def rrf_fuse(rank_lists, k: int = 60, weights=None) -> dict:
    """Reciprocal-rank fusion over short candidate lists. Weights are optional
    and let sparse lexical evidence count slightly more for exact product
    attributes while keeping dense semantic retrieval in the mix."""
    if weights is None:
        weights = [1.0] * len(rank_lists)
    scores: dict = {}
    for ranks, weight in zip(rank_lists, weights):
        for rank, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + float(weight) / (k + rank + 1)
    return scores


def _minmax(v):
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi - lo < 1e-9 else (v - lo) / (hi - lo)


def product_filter_mask(products: pd.DataFrame, filters: dict) -> np.ndarray:
    mask = np.ones(len(products), dtype=bool)
    f = filters or {}
    if f.get("category"):
        cat = str(f["category"])
        col_mask = np.zeros(len(products), dtype=bool)
        for col in ("category1_norm", "category2_norm", "sub_category_norm"):
            if col in products:
                col_mask |= products[col].fillna("").map(lambda v: cat in str(v)).to_numpy()
        mask &= col_mask
    if f.get("brand"):
        mask &= products["brand_norm"].fillna("").map(lambda v: str(f["brand"]) in str(v)).to_numpy()
    price = products["price_clean"].to_numpy(dtype=float)
    if f.get("price_min") is not None:
        mask &= np.where(np.isnan(price), False, price >= f["price_min"])
    if f.get("price_max") is not None:
        mask &= np.where(np.isnan(price), False, price <= f["price_max"])
    if f.get("exclude_fake", True) and "is_fake" in products:
        mask &= ~products["is_fake"].fillna(False).astype(bool).to_numpy()
    return mask


# ---- the retriever ------------------------------------------------------
class ProductIndex:
    """Full-catalogue product retrieval: dense vectors + BM25, fused with RRF."""

    def __init__(self, products: pd.DataFrame, vectors: np.ndarray, bm25: BM25Okapi, rrf_k: int = 60):
        self.products = products.reset_index(drop=True)
        # keep as-is (a memmap stays on disk) unless the dtype needs converting
        self.vectors = vectors if vectors.dtype == np.float32 else vectors.astype("float32")
        self.bm25 = bm25
        self.rrf_k = rrf_k

    @classmethod
    def build(cls, products: pd.DataFrame) -> "ProductIndex":
        texts = products["product_text_norm"].fillna("").tolist()
        log.info("embedding %d products", len(texts))
        vectors = embed(texts)
        bm25 = BM25Okapi.from_texts(texts)
        return cls(products, vectors, bm25)

    def save(self, path=config.PRODUCT_INDEX_DIR):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        self.products.to_parquet(path / "products.parquet", index=False)
        self.bm25.save(path)                          # persist the sparse BM25 so load is fast
        log.info("saved product index (%d) to %s", len(self.products), path)

    @classmethod
    def load(cls, path=config.PRODUCT_INDEX_DIR) -> "ProductIndex":
        path = Path(path)
        products = pd.read_parquet(path / "products.parquet")
        vectors = np.load(path / "vectors.npy", mmap_mode="r")
        bm25 = BM25Okapi.load(path)
        return cls(products, vectors, bm25)

    def search(self, query: str, filters: dict | None = None, k: int = config.TOP_K,
               method: str = "hybrid") -> list[dict]:
        """method: 'hybrid' (dense+BM25 RRF, default) | 'dense' | 'bm25' -- the
        latter two exist for the retrieval ablation (quantifying hybrid's lift)."""
        mask = product_filter_mask(self.products, filters)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return []
        if method == "dense":
            qv = embed([query])[0]
            dense = self.vectors @ qv
            order = valid[np.argsort(-dense[valid], kind="stable")][:k]
            top = [(idx, float(dense[idx])) for idx in order]
        elif method == "bm25":
            sparse = self.bm25.get_scores(query)
            order = valid[np.argsort(-sparse[valid], kind="stable")][:k]
            top = [(idx, float(sparse[idx])) for idx in order]
        else:
            qv = embed([query])[0]
            dense = self.vectors @ qv
            sparse = self.bm25.get_scores(query)
            # cap the candidate pool so RRF stays cheap at ~1M-product scale
            pool = min(int(valid.size), max(int(config.RRF_CANDIDATE_POOL), int(k) * 20))
            d_order = valid[np.argsort(-dense[valid], kind="stable")[:pool]]
            sparse_valid = valid[sparse[valid] > 0]     # drop zero-score lexical noise
            if sparse_valid.size:
                s_order = sparse_valid[np.argsort(-sparse[sparse_valid], kind="stable")[:pool]]
                fused = rrf_fuse([d_order.tolist(), s_order.tolist()], k=self.rrf_k,
                                 weights=[config.PRODUCT_RRF_DENSE_WEIGHT, config.PRODUCT_RRF_SPARSE_WEIGHT])
            else:
                fused = rrf_fuse([d_order.tolist()], k=self.rrf_k)
            top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out = []
        for idx, score in top:
            r = self.products.iloc[idx]
            out.append({
                "rank": len(out) + 1, "product_id": int(r["product_id"]),
                "title": r["title_fa"], "brand": r["brand_norm"],
                "price": float(r["price_clean"]) if pd.notna(r["price_clean"]) else None,
                "rate": float(r["product_rate_clean"]) if pd.notna(r["product_rate_clean"]) else None,
                "rate_count": int(r["rate_count"]) if pd.notna(r["rate_count"]) else 0,
                "comment_count": int(r.get("comment_count", 0)),
                "category": r.get("category1_norm", ""), "score": round(float(score), 5),
            })
        return out


class ReviewRetriever:
    """Per-product hybrid review search with query-intent-aware reranking.

    Beyond plain RRF, reviews are reranked by whether the QUERY is asking about
    problems ("مشکل چیه؟") vs. satisfaction ("خوبه؟") — a negative-intent query
    should surface reviews that actually read negative (low rating / not
    recommended / has a disadvantages field), not just whatever ranked highest
    on raw text similarity. This directly improves QA precision: without it, a
    "what's wrong with this product" query could return the top-BM25-scoring
    review even if it happens to be glowingly positive.
    """

    _NEGATIVE_CUES = ("مشکل", "ایراد", "عیب", "بد", "ضعف", "منفی", "ناراضی",
                      "خراب", "معیوب", "نمی ارزد", "نمی‌ارزد", "عدم توصیه",
                      "نقاط ضعف", "معایب")
    _POSITIVE_CUES = ("خوب", "مثبت", "مزیت", "مزایا", "رضایت", "راضی",
                      "ارزش خرید", "پیشنهاد", "نقاط قوت")

    def __init__(self, comments_by_product, rrf_k: int = 60):
        self.by_product = comments_by_product
        self.rrf_k = rrf_k

    @staticmethod
    def _rating_norm(rate: np.ndarray) -> np.ndarray:
        rate = np.asarray(rate, dtype=float)
        if rate.size == 0:
            return rate
        finite = np.isfinite(rate)
        if not finite.any():
            return np.full(len(rate), 0.5, dtype=float)
        vals = rate.copy()
        vals[~finite] = float(np.nanmedian(vals[finite]))
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if hi - lo < 1e-9:
            return np.full(len(vals), 0.5, dtype=float)
        return (vals - lo) / (hi - lo)

    def _polarity(self, query: str) -> str:
        q = pt.normalize(query)
        if any(x in q for x in self._NEGATIVE_CUES):
            return "negative"
        if any(x in q for x in self._POSITIVE_CUES):
            return "positive"
        return "neutral"

    def retrieve(self, query: str, product_id: int, k: int = config.TOP_K, rerank: bool = True) -> list[dict]:
        rev = self.by_product.get(int(product_id))
        if rev is None or rev.empty:
            return []
        rev = rev.reset_index(drop=True)
        texts = rev["comment_text_norm"].fillna("").tolist()
        qv = embed([query])[0]
        dense = embed(texts) @ qv
        sparse = BM25Okapi.from_texts(texts).get_scores(query)

        pool = min(len(rev), max(int(config.REVIEW_CANDIDATE_POOL), int(k) * 10))
        order_d = np.argsort(-dense, kind="stable")[:pool].tolist()
        positive_sparse = np.flatnonzero(sparse > 0)
        if positive_sparse.size:
            order_s = positive_sparse[np.argsort(-sparse[positive_sparse], kind="stable")[:pool]].tolist()
            fused = rrf_fuse([order_d, order_s], k=self.rrf_k, weights=[1.0, 1.05])
        else:
            fused = rrf_fuse([order_d], k=self.rrf_k)

        if rerank and fused:
            idxs = list(fused)
            base = _minmax(np.array([fused[i] for i in idxs], dtype=float))
            likes = rev["likes"].fillna(0).to_numpy(dtype=float)
            buyer = rev["is_buyer"].fillna(False).astype(float).to_numpy()
            lk = _minmax(np.log1p(likes[idxs]))
            rate_all = pd.to_numeric(rev["rate_clean"], errors="coerce").to_numpy(dtype=float)
            rate_norm = self._rating_norm(rate_all)
            status = rev["recommendation_status"].fillna("").astype(str).to_numpy()
            has_disadv = (rev["disadvantages_norm"].fillna("").astype(str).str.len().gt(0).to_numpy()
                         if "disadvantages_norm" in rev else np.zeros(len(rev), dtype=bool))
            has_adv = (rev["advantages_norm"].fillna("").astype(str).str.len().gt(0).to_numpy()
                      if "advantages_norm" in rev else np.zeros(len(rev), dtype=bool))

            polarity = self._polarity(query)
            intent_score = np.full(len(idxs), 0.5, dtype=float)
            if polarity == "negative":
                for j, idx in enumerate(idxs):
                    status_score = 1.0 if status[idx] == "not_recommended" else (0.5 if status[idx] == "no_idea" else 0.0)
                    intent_score[j] = 0.55 * status_score + 0.30 * (1.0 - rate_norm[idx]) + 0.15 * float(has_disadv[idx])
            elif polarity == "positive":
                for j, idx in enumerate(idxs):
                    status_score = 1.0 if status[idx] == "recommended" else (0.5 if status[idx] == "no_idea" else 0.0)
                    intent_score[j] = 0.60 * status_score + 0.25 * rate_norm[idx] + 0.15 * float(has_adv[idx])

            final_scores = {}
            for j, idx in enumerate(idxs):
                if polarity == "neutral":
                    score = 0.84 * base[j] + 0.09 * lk[j] + 0.07 * buyer[idx]
                else:
                    iw = float(config.REVIEW_NEGATIVE_INTENT_WEIGHT if polarity == "negative"
                              else config.REVIEW_POSITIVE_INTENT_WEIGHT)
                    score = (0.93 - iw) * base[j] + 0.07 * lk[j] + 0.03 * buyer[idx] + iw * intent_score[j]
                final_scores[idx] = float(score)
            fused = final_scores

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out = []
        for idx, score in top:
            r = rev.iloc[idx]
            out.append({
                "rank": len(out) + 1, "comment_id": int(r["comment_id"]),
                "product_id": int(r["product_id"]), "text": r["comment_text_norm"],
                "rate": float(r["rate_clean"]) if pd.notna(r["rate_clean"]) else None,
                "recommendation_status": str(r["recommendation_status"]),
                "likes": int(r["likes"]) if pd.notna(r["likes"]) else 0,
                "is_buyer": bool(r["is_buyer"]) if pd.notna(r["is_buyer"]) else False,
                "has_advantage": bool(str(r.get("advantages_norm", "") or "").strip()),
                "has_disadvantage": bool(str(r.get("disadvantages_norm", "") or "").strip()),
                "score": round(float(score), 5),
            })
        return out


# ---- build helpers ------------------------------------------------------
def _prepare_products(products: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    products = products.copy()
    counts = comments.groupby("product_id").size()
    products["comment_count"] = products["product_id"].map(counts).fillna(0).astype(int)
    return products


def build_product_index(sample_comments: int | None = None) -> ProductIndex:
    """Build + persist the product index from the cleaned tables (run.py index)."""
    products = pd.read_parquet(config.PRODUCTS_CLEAN)
    comments = pd.read_parquet(config.COMMENTS_CLEAN, columns=["product_id"])
    products = _prepare_products(products, comments)
    idx = ProductIndex.build(products)
    idx.save()
    return idx


class GroupedComments:
    """Per-product review access backed by ONE index-sorted DataFrame.

    Behaves like a dict of {product_id: reviews-DataFrame} (`.get`, `in`, iteration,
    `len`) but never materializes a separate frame per product, so it stays memory-
    lean on the full corpus (millions of reviews) instead of holding N sub-frames.
    """

    def __init__(self, df: pd.DataFrame):
        df = df.dropna(subset=["product_id"]).copy()
        df["product_id"] = df["product_id"].astype(int)
        self.df = df.set_index("product_id", drop=False).sort_index()
        self._ids = set(self.df.index.unique())

    def get(self, pid, default=None):
        pid = int(pid)
        if pid not in self._ids:
            return default
        # reset to a clean RangeIndex — callers do .loc/.assign that break on the
        # duplicated product_id index
        return self.df.loc[[pid]].reset_index(drop=True)

    def __contains__(self, pid):
        return int(pid) in self._ids

    def __iter__(self):
        return iter(self._ids)

    def __len__(self):
        return len(self._ids)


_REVIEW_COLS = ["comment_id", "product_id", "comment_text_norm", "body_norm",
                "advantages_norm", "disadvantages_norm", "recommendation_status",
                "rate_clean", "likes", "is_buyer", "has_text"]


def load_comments_by_product(only_with_text: bool = True) -> GroupedComments:
    """Load cleaned comments (needed columns only) for on-demand review retrieval."""
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(config.COMMENTS_CLEAN).schema.names)
    cols = [c for c in _REVIEW_COLS if c in have]
    comments = pd.read_parquet(config.COMMENTS_CLEAN, columns=cols)
    if only_with_text and "has_text" in comments:
        comments = comments[comments["has_text"].astype(bool)]
    return GroupedComments(comments)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_product_index()
