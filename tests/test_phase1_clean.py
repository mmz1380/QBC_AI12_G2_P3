"""Phase-1 cleaning produces the canonical schema with stable dtypes."""
import pandas as pd

from digikala.phase1_data import clean


def test_clean_products_schema():
    # rows 0,1 share id=1 but differ (Price) -> caught as a duplicate id, not exact dup
    raw = pd.DataFrame({"id": [1, 1, 2], "title_fa": ["الف", "الف", "ب"],
                        "Brand": ["x", "x", None], "Category1": ["c", "c", "d"],
                        "Price": [1000, 2000, 0], "Rate": [90, 90, 50], "Rate_cnt": [3, 3, 1],
                        "Is_Fake": [0, 0, 1]})
    out, rep = clean.clean_products(raw)
    assert {"product_id", "product_text_norm", "price_clean", "brand_norm"} <= set(out.columns)
    assert rep["dropped_duplicate_ids"] == 1          # the repeated id=1
    assert (out["price_available"] == (out["price_clean"] > 0)).all()


def test_clean_comments_labels_and_flags():
    raw = pd.DataFrame({"id": [1, 2, 3], "product_id": [10, 10, 99],
                        "title": ["t", "", "x"], "body": ["خوب بود", "", "بد"],
                        "advantages": ["['خوب']", "[]", "[]"], "disadvantages": ["[]", "[]", "['بد']"],
                        "rate": [5, 3, 1], "likes": [2, 0, 1], "is_buyer": [1, 0, 1],
                        "recommendation_status": ["recommended", "bogus", "not_recommended"]})
    out, rep = clean.clean_comments(raw, valid_product_ids={10})
    # invalid label nulled out; product 99 not matched
    assert out.loc[out["comment_id"] == 2, "recommendation_valid"].iloc[0] == False
    assert out.loc[out["comment_id"] == 3, "has_product_match"].iloc[0] == False
    # dtypes pinned for stable streaming schema
    assert str(out["recommendation_status"].dtype) == "string"
    assert str(out["has_text"].dtype) == "boolean"
