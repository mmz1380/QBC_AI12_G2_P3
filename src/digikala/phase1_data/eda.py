"""Phase 1 — exploratory data analysis, all figures in Plotly.

Each `fig_*` function takes a cleaned dataframe and returns a plotly Figure, so the
notebook and the Streamlit dashboard render the exact same charts. `run()` writes
them to artifacts/figures as standalone HTML plus a summary.json of headline stats.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .. import config

log = logging.getLogger("digikala.eda")
_TEMPLATE = "plotly_white"


# ---- headline numbers ---------------------------------------------------
def summary_stats(products: pd.DataFrame, comments: pd.DataFrame) -> dict:
    return {
        "n_products": int(len(products)),
        "n_comments": int(len(comments)),
        "n_brands": int(products["brand_norm"].nunique()),
        "n_categories": int(products["category1_norm"].nunique()),
        "pct_products_priced": round(100 * products["price_available"].mean(), 1),
        "pct_comments_with_text": round(100 * comments["has_text"].mean(), 1),
        "pct_comments_labeled": round(100 * comments["recommendation_valid"].mean(), 1),
        "median_price_toman": (None if products["price_clean"].dropna().empty
                               else int(products.loc[products["price_available"], "price_clean"].median() // 10)),
        "avg_comments_per_product": round(
            len(comments) / max(1, comments["product_id"].nunique()), 1),
    }


# ---- figures ------------------------------------------------------------
def fig_recommendation_balance(comments: pd.DataFrame) -> go.Figure:
    vc = comments.loc[comments["recommendation_valid"], "recommendation_status"].value_counts()
    fig = px.bar(x=vc.index, y=vc.values, template=_TEMPLATE,
                 labels={"x": "recommendation_status", "y": "count"},
                 title="Phase 3 target — recommendation_status balance", text=vc.values)
    fig.update_traces(marker_color=["#2ca02c", "#d62728", "#7f7f7f"][:len(vc)])
    return fig


def fig_top_categories(products: pd.DataFrame, n: int = 15) -> go.Figure:
    vc = products["category1_norm"].value_counts().head(n)[::-1]
    return px.bar(x=vc.values, y=vc.index, orientation="h", template=_TEMPLATE,
                  labels={"x": "products", "y": "category"},
                  title=f"Top {n} level-1 categories")


def fig_top_brands(products: pd.DataFrame, n: int = 15) -> go.Figure:
    vc = products.loc[products["brand_norm"] != "نامشخص", "brand_norm"].value_counts().head(n)[::-1]
    return px.bar(x=vc.values, y=vc.index, orientation="h", template=_TEMPLATE,
                  labels={"x": "products", "y": "brand"}, title=f"Top {n} brands")


def fig_price_distribution(products: pd.DataFrame) -> go.Figure:
    priced = products.loc[products["price_available"], "price_clean"] / 10  # Rials -> Toman
    priced = priced[priced > 0]
    fig = px.histogram(np.log10(priced), nbins=60, template=_TEMPLATE,
                       title="Price distribution (log10 Toman)",
                       labels={"value": "log10(price, Toman)"})
    fig.update_layout(showlegend=False)
    return fig


def fig_rating_distribution(products: pd.DataFrame) -> go.Figure:
    rate = products["product_rate_clean"].dropna()
    return px.histogram(rate, nbins=50, template=_TEMPLATE,
                        title="Product rating distribution (0–100)",
                        labels={"value": "product_rate_clean"})


def fig_comments_per_product(comments: pd.DataFrame) -> go.Figure:
    counts = comments.groupby("product_id").size()
    counts = counts[counts > 0]
    fig = px.histogram(np.log10(counts), nbins=50, template=_TEMPLATE,
                       title="Reviews per product (log10)",
                       labels={"value": "log10(reviews per product)"})
    fig.update_layout(showlegend=False)
    return fig


def fig_text_length(comments: pd.DataFrame) -> go.Figure:
    lens = comments.loc[comments["has_text"], "comment_text_norm"].str.split().map(len)
    lens = lens[lens <= lens.quantile(0.99)]        # trim the long tail for readability
    return px.histogram(lens, nbins=60, template=_TEMPLATE,
                        title="Review length (words, 99th pct clipped)",
                        labels={"value": "words per review"})


def fig_missingness(products: pd.DataFrame, comments: pd.DataFrame) -> go.Figure:
    rows = []
    for name, df in [("products", products), ("comments", comments)]:
        for col in df.columns:
            rows.append({"table": name, "column": col,
                         "missing_%": round(100 * df[col].isna().mean(), 1)})
    m = pd.DataFrame(rows)
    m = m[m["missing_%"] > 0].sort_values("missing_%")
    return px.bar(m, x="missing_%", y="column", color="table", orientation="h",
                  template=_TEMPLATE, title="Missing values by column")


ALL_FIGURES = {
    "recommendation_balance": fig_recommendation_balance,
    "top_categories": fig_top_categories,
    "top_brands": fig_top_brands,
    "price_distribution": fig_price_distribution,
    "rating_distribution": fig_rating_distribution,
    "comments_per_product": fig_comments_per_product,
    "text_length": fig_text_length,
    "missingness": fig_missingness,
}


def run() -> dict:
    """Load the cleaned tables, build every figure, and write them to artifacts."""
    products = pd.read_parquet(config.PRODUCTS_CLEAN)
    comments = pd.read_parquet(config.COMMENTS_CLEAN)
    stats = summary_stats(products, comments)

    for name, fn in ALL_FIGURES.items():
        arg = (products,) if fn in (fig_top_categories, fig_top_brands,
                                    fig_price_distribution, fig_rating_distribution) else \
              (comments,) if fn in (fig_recommendation_balance, fig_comments_per_product,
                                    fig_text_length) else (products, comments)
        try:
            fig = fn(*arg)
            fig.write_html(config.FIGURES_DIR / f"{name}.html", include_plotlyjs="cdn")
        except Exception as e:
            log.warning("figure %s failed: %s", name, e)

    (config.FIGURES_DIR / "eda_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("EDA figures + summary written to %s", config.FIGURES_DIR)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(run(), ensure_ascii=False, indent=2))
