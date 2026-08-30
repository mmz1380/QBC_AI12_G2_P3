"""Phase 1 — clean the raw Digikala data into a documented, analysis-ready schema.

The output column names use `_norm` / `_clean` suffixes so Phase 2 (retrieval) and
Phase 3 (the recommendation classifier) read exactly the same tables. Every
`clean_*` step returns a report dict of what it changed, because the brief grades
*justified* cleaning decisions, not just clean data.

Products (~1.28M rows) are cleaned in memory. Comments (~6M rows) are streamed
chunk-by-chunk and appended to a Parquet file, so peak memory stays flat and the
full dataset can be processed on a 15 GB machine.
"""
from __future__ import annotations

import ast
import json
import logging
import re

import numpy as np
import pandas as pd
from persiantools.jdatetime import JalaliDate

from .. import config
from ..core import dataio
from ..core import persian_text as pt

log = logging.getLogger("digikala.phase1")

_JALALI_MONTHS = {
    "فروردین": 1, "اردیبهشت": 2, "خرداد": 3, "تیر": 4, "مرداد": 5, "شهریور": 6,
    "مهر": 7, "آبان": 8, "آذر": 9, "دی": 10, "بهمن": 11, "اسفند": 12,
}
_TRUE = {"1", "true", "yes", "بله", "t", "y"}
_FALSE = {"0", "false", "no", "خیر", "f", "n"}
_GENERIC = "نامشخص"


# ---- small typed converters --------------------------------------------
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_bool(s: pd.Series) -> pd.Series:
    def conv(v):
        if pd.isna(v):
            return pd.NA
        t = str(v).strip().lower()
        return True if t in _TRUE else False if t in _FALSE else pd.NA
    return s.map(conv).astype("boolean")


def _parse_list_field(v) -> str:
    """advantages/disadvantages arrive as list-literal strings, e.g.
    "['جنسش خوبه\\r', 'خوش رنگه']" — pull the items out and join them."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).replace("\\r", " ").replace("\r", " ")
    try:
        parsed = ast.literal_eval(s)
        items = [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]
    except Exception:
        items = re.split(r"',\s*'", s.strip("[]"))
    items = [pt.normalize(it.strip(" '\"")) for it in items]
    return " ، ".join(it for it in items if it)


def _dedup(df: pd.DataFrame, report: dict, id_col: str) -> pd.DataFrame:
    n = len(df)
    df = df.drop_duplicates()
    report["dropped_exact_duplicates"] = n - len(df)
    if id_col in df.columns:
        n = len(df)
        df = df.drop_duplicates(subset=id_col, keep="first")
        report["dropped_duplicate_ids"] = n - len(df)
    return df


# ---- products -----------------------------------------------------------
def clean_products(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {"input_rows": len(df)}
    df = _dedup(df.copy(), report, id_col="id")
    out = pd.DataFrame()

    out["product_id"] = _to_num(df.get("id")).astype("Int64")
    out["title_fa"] = df.get("title_fa", "").map(pt.normalize)
    out["title_norm"] = out["title_fa"]

    for src, dst in [("Brand", "brand_norm"), ("Category1", "category1_norm"),
                     ("Category2", "category2_norm"), ("sub_category", "sub_category_norm"),
                     ("Seller", "seller_norm")]:
        out[dst] = df.get(src, pd.Series(index=df.index)).fillna(_GENERIC).map(pt.normalize)

    out["price_clean"] = _to_num(df.get("Price"))
    out["min_price_last_month"] = _to_num(df.get("min_price_last_month"))
    out["product_rate_clean"] = _to_num(df.get("Rate"))          # 0..100 scale in this dataset
    out["rate_count"] = _to_num(df.get("Rate_cnt")).fillna(0).astype("Int64")
    out["is_fake"] = _to_bool(df.get("Is_Fake", pd.Series(index=df.index)))

    # price 0 / missing means "not for sale", not a real price — flag it, keep the row
    report["zero_or_missing_price"] = int((out["price_clean"].isna() | (out["price_clean"] == 0)).sum())
    out["price_available"] = out["price_clean"].notna() & (out["price_clean"] > 0)

    # embedding text: title + categories + brand, skipping the generic placeholder
    def _ptext(r):
        parts = [r["title_norm"], r["category1_norm"], r["category2_norm"],
                 r["sub_category_norm"], r["brand_norm"]]
        return " ".join(p for p in parts if p and p != _GENERIC)
    out["product_text_norm"] = out.apply(_ptext, axis=1)

    out = out.dropna(subset=["product_id"]).reset_index(drop=True)
    report["output_rows"] = len(out)
    return out, report


# ---- comments -----------------------------------------------------------
def clean_comments(df: pd.DataFrame, valid_product_ids: set | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean one comments dataframe (a chunk or the whole notebook sample)."""
    report: dict = {"input_rows": len(df)}
    df = _dedup(df.copy(), report, id_col="id")
    out = pd.DataFrame()

    def col(name):     # always a Series aligned to df, even if the column is absent
        return df[name] if name in df.columns else pd.Series(index=df.index, dtype="object")

    out["comment_id"] = _to_num(col("id")).astype("Int64")
    out["product_id"] = _to_num(col("product_id")).astype("Int64")
    out["title_norm"] = col("title").map(pt.normalize)
    out["body_norm"] = col("body").map(pt.normalize)
    out["advantages_norm"] = col("advantages").map(_parse_list_field)
    out["disadvantages_norm"] = col("disadvantages").map(_parse_list_field)

    # combined text used by retrieval and the classifier
    out["comment_text_norm"] = (
        out[["title_norm", "body_norm", "advantages_norm", "disadvantages_norm"]]
        .agg(" ".join, axis=1).map(pt.normalize))
    out["has_text"] = out["comment_text_norm"].map(pt.is_meaningful)

    out["rate_clean"] = _to_num(col("rate"))
    out["likes"] = _to_num(col("likes")).fillna(0).astype("Int64")
    out["dislikes"] = _to_num(col("dislikes")).fillna(0).astype("Int64")
    out["is_buyer"] = _to_bool(col("is_buyer"))
    out["true_to_size_rate"] = _to_num(col("true_to_size_rate"))

    if "created_at" in df.columns:
        out["created_at"] = _parse_datetime(df["created_at"])

    # recommendation label: validate against the 3 allowed classes
    rs = df.get("recommendation_status", pd.Series(index=df.index)).astype("string").str.strip().str.lower()
    out["recommendation_status"] = rs.where(rs.isin(config.RECOMMENDATION_CLASSES))
    out["recommendation_valid"] = out["recommendation_status"].isin(config.RECOMMENDATION_CLASSES)

    # does the comment point at a real product? (needed for RAG grounding + joins)
    if valid_product_ids is not None:
        out["has_product_match"] = out["product_id"].isin(valid_product_ids)
    else:
        out["has_product_match"] = out["product_id"].notna()

    report["rows_without_text"] = int((~out["has_text"]).sum())
    report["invalid_recommendation_labels"] = int((~out["recommendation_valid"]).sum())
    report["rows_without_product_match"] = int((~out["has_product_match"]).sum())
    out = out.dropna(subset=["comment_id"]).reset_index(drop=True)

    # Pin dtypes so every streamed chunk produces an identical parquet schema
    # (otherwise an all-null column in one chunk can mismatch a later one).
    if "created_at" not in out:
        out["created_at"] = pd.NaT
    out = out.astype({
        "comment_id": "Int64", "product_id": "Int64", "likes": "Int64", "dislikes": "Int64",
        "rate_clean": "float64", "true_to_size_rate": "float64",
        "title_norm": "string", "body_norm": "string", "advantages_norm": "string",
        "disadvantages_norm": "string", "comment_text_norm": "string",
        "recommendation_status": "string", "has_text": "boolean",
        "recommendation_valid": "boolean", "has_product_match": "boolean",
    })
    out["is_buyer"] = out["is_buyer"].astype("boolean")
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
    report["output_rows"] = len(out)
    return out, report


def _parse_datetime(s: pd.Series) -> pd.Series:
    """created_at is a Jalali string like "23 شهریور 1402"; convert to Gregorian.
    Cached per distinct string since there are relatively few unique dates."""
    cache: dict[str, pd.Timestamp] = {}

    def parse_one(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return pd.NaT
        key = str(v).strip()
        if key in cache:
            return cache[key]
        result = pd.NaT
        parts = key.split()
        if len(parts) == 3 and parts[1] in _JALALI_MONTHS:
            try:
                day = int(parts[0].translate(pt._DIGIT_MAP))
                year = int(parts[2].translate(pt._DIGIT_MAP))
                result = pd.Timestamp(JalaliDate(year, _JALALI_MONTHS[parts[1]], day).to_gregorian())
            except Exception:
                result = pd.NaT
        cache[key] = result
        return result
    return s.map(parse_one)


# ---- driver -------------------------------------------------------------
def _merge_reports(total: dict, chunk: dict) -> None:
    for k, v in chunk.items():
        if isinstance(v, (int, float)):
            total[k] = total.get(k, 0) + v


def build(full: bool = True) -> dict:
    """Clean products (in memory) then stream-clean comments to Parquet.

    full=True streams the entire comments CSV chunk-by-chunk (run.py default).
    full=False cleans only the notebook sample from config.COMMENTS_SAMPLE_SIZE.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataio.download_raw()

    log.info("cleaning products")
    products, p_rep = clean_products(dataio.load_products())
    products.to_parquet(config.PRODUCTS_CLEAN, index=False)
    valid_ids = set(int(x) for x in products["product_id"].dropna())
    log.info("saved %s (%d rows)", config.PRODUCTS_CLEAN.name, len(products))

    c_total: dict = {}
    writer = None
    if full:
        # `_dedup` inside clean_comments only sees one chunk at a time, so a
        # comment id that happens to repeat in the RAW csv further apart than
        # one CHUNK_SIZE (a real, observed pattern in this dataset) survives
        # into the output as a cross-chunk duplicate -- silently doubling that
        # review's text in every downstream citation/quote. `seen_ids` makes
        # the dedup global across the whole stream, not just per chunk.
        seen_ids: set[int] = set()
        log.info("streaming + cleaning comments in chunks of %d", config.CHUNK_SIZE)
        for i, chunk in enumerate(dataio.iter_comment_chunks()):
            cleaned, rep = clean_comments(chunk, valid_ids)
            is_new = ~cleaned["comment_id"].isin(seen_ids)
            rep["dropped_cross_chunk_duplicate_ids"] = int((~is_new).sum())
            cleaned = cleaned[is_new]
            rep["output_rows"] = len(cleaned)
            seen_ids.update(int(x) for x in cleaned["comment_id"].dropna())
            _merge_reports(c_total, rep)
            table = pa.Table.from_pandas(cleaned, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(config.COMMENTS_CLEAN, table.schema)
            writer.write_table(table)
            if i % 5 == 0:
                log.info("  chunk %d done (%d rows cumulative)", i, c_total.get("output_rows", 0))
        if writer is not None:
            writer.close()
    else:
        comments, c_total = clean_comments(dataio.load_comments(), valid_ids)
        comments.to_parquet(config.COMMENTS_CLEAN, index=False)
    log.info("saved %s (%d rows)", config.COMMENTS_CLEAN.name, c_total.get("output_rows", 0))

    report = {"products": p_rep, "comments": c_total, "full": full}
    config.PHASE1_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(build(full=False), ensure_ascii=False, indent=2))
