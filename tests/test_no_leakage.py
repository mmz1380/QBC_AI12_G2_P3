"""Guardrail tests: the recommendation model must not use the leaky features
(rate/likes/is_buyer) and splits must not share review text."""
import inspect

import pandas as pd

from digikala.phase3_predict import recommend


def test_final_model_is_text_only():
    # the persisted bundle + reported metrics must declare text-only features
    src = inspect.getsource(recommend._train)
    assert '"features": ["comment_text_norm"]' in src
    assert 'numeric=False' in src            # final + baselines + grouped are text-only


def test_predict_signature_has_no_metadata_args():
    params = list(inspect.signature(recommend.predict).parameters)
    assert params == ["texts"]               # no rate/likes/is_buyer


def test_prep_dedups_comment_text():
    df = pd.DataFrame({
        "recommendation_valid": [True, True, True],
        "has_text": [True, True, True],
        "product_id": [1, 1, 2],
        "comment_text_norm": ["دوپ", "دوپ", "یکتا"],   # first two identical
        "is_buyer": [True, False, True], "rate_clean": [5, 5, 1], "likes": [1, 0, 2],
    })
    out = recommend._prep(df)
    assert len(out) == 2                      # duplicate text collapsed -> no cross-split leak


def test_numeric_features_only_in_ablation():
    # NUMERIC_FEATURES exist for the ablation, but the excluded list names them
    src = inspect.getsource(recommend._train)
    assert '"excluded_to_avoid_leakage": NUMERIC_FEATURES' in src
    assert set(recommend.NUMERIC_FEATURES) == {"rate_clean", "likes", "is_buyer_num"}


def test_grouped_split_is_the_reported_primary_metric():
    # the product-grouped Macro-F1 is the headline number, not the naive random split
    src = inspect.getsource(recommend._train)
    assert '"primary_macro_f1": grouped_macro_f1' in src
    assert '"primary_split": "product_grouped"' in src


def test_naive_split_reports_product_overlap():
    # the naive random split's residual leakage risk (same product in train+test
    # via a different review) must be quantified, not hidden
    src = inspect.getsource(recommend._train)
    assert "naive_overlap_pct" in src
    assert '"naive_split_product_overlap_pct": naive_overlap_pct' in src


def test_grouped_split_has_zero_product_overlap():
    gss_src = inspect.getsource(recommend._train)
    assert "GroupShuffleSplit" in gss_src
    assert "set(g_train[\"product_id\"]) & set(g_test[\"product_id\"])" in gss_src


def test_vectorizer_fit_only_on_train_via_pipeline():
    # TfidfVectorizer must be fit only on X_train (inside sklearn Pipeline.fit),
    # never on the full/combined data -- that would leak vocabulary/idf into test
    src = inspect.getsource(recommend._train)
    assert "_pipeline(clf, numeric=False).fit(X_train, y_train)" in src
    assert "_pipeline(_logreg(), numeric=False).fit(X_train, y_train)" in src
