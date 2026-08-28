"""Phase 3 — predict recommendation_status (recommended / not_recommended / no_idea).

The brief asks us to predict a review's recommendation *from its textual content*,
so the model is **text-only** (TF-IDF 1–2gram + a linear classifier). This is a
deliberate anti-leakage choice:

  * `rate_clean` (the 1–5 star rating) is a second expression of the *same*
    sentiment the user recorded in the same act — it is a near-duplicate of the
    label, so feeding it in leaks the target and trivializes the task.
  * `likes` accrue *after* the review is posted (other users vote over time), so
    they are not available at prediction time — temporal leakage.
  * `is_buyer` is non-textual metadata outside the "predict from text" spec.

We still *report* a text+numeric ablation to quantify exactly how much those
features inflate the score (that lift is the leakage), but the saved/final model
and every reported headline number use text only.

Anti-leakage guards also kept: dedup by comment text so no review appears in two
splits, a majority-class baseline, and a product-grouped split (no product in both
train and test). Primary metric: Macro-F1.

Headline number is the **product-grouped** Macro-F1, not the naive random split.
A random row-level split can still let the model see other reviews of the *same*
product in training (brand names, model-specific phrases) even after exact-text
dedup — a residual, subtler leakage channel than rate/likes. The grouped split
holds entire products out, which is the honest test of whether the model
generalizes from review language rather than memorizing per-product patterns.
Both numbers are reported; `naive_split_product_overlap_pct` quantifies exactly
how much product overlap the random split has, so the difference (if any) between
the two Macro-F1s is explained by data, not asserted.
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .. import config

log = logging.getLogger("digikala.phase3")

RANDOM_STATE = config.RANDOM_SEED
FA_TOKEN_PATTERN = r"[؀-ۿ0-9A-Za-z]+"
# Non-textual features. NOT used by the model — kept only for the leakage ablation.
NUMERIC_FEATURES = ["rate_clean", "likes", "is_buyer_num"]
MAX_PER_CLASS = 30_000

PERSIAN_STOPWORDS = [
    "و", "در", "به", "از", "که", "این", "با", "را", "برای", "رو", "هم", "یک", "ها",
    "است", "نیز", "شد", "شود", "می", "خواهد", "بر", "آن", "تا", "کرد", "دارد", "بود",
    "اما", "اگر", "هر", "همه", "خیلی", "بیشتر", "کمتر", "مثل", "مانند", "حتی",
]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Keep labeled, text-bearing, product-matched rows; dedup by text; fill numerics.
    Shared by the full-data loader and the in-memory demo path so both agree."""
    mask = (df["recommendation_valid"].fillna(False) & df["has_text"].fillna(False)
            & df["product_id"].notna())
    df = df.loc[mask].drop_duplicates(subset="comment_text_norm", keep="first").copy()
    df["is_buyer_num"] = df["is_buyer"].fillna(False).astype(int)
    df["rate_clean"] = df["rate_clean"].fillna(0)
    df["likes"] = df["likes"].fillna(0)
    return df


def _load() -> pd.DataFrame:
    cols = ["comment_id", "product_id", "comment_text_norm", "recommendation_status",
            "recommendation_valid", "has_text", "rate_clean", "likes", "is_buyer"]
    return _prep(pd.read_parquet(config.COMMENTS_CLEAN, columns=cols))


def _stratified_cap(df: pd.DataFrame, col: str, cap: int) -> pd.DataFrame:
    parts = [g.sample(n=min(len(g), cap), random_state=RANDOM_STATE)
             for _, g in df.groupby(col, sort=False)]
    return pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def _vectorizer(max_features: int = 50_000) -> TfidfVectorizer:
    return TfidfVectorizer(token_pattern=FA_TOKEN_PATTERN, ngram_range=(1, 2),
                           min_df=5, max_df=0.8, sublinear_tf=True,
                           max_features=max_features, stop_words=PERSIAN_STOPWORDS)


def _pipeline(clf, numeric: bool = False) -> Pipeline:
    """Text-only by default. numeric=True adds the leaky metadata (ablation only)."""
    trans = [("text", _vectorizer(), "comment_text_norm")]
    if numeric:
        trans.append(("num", StandardScaler(), NUMERIC_FEATURES))
    return Pipeline([("preprocessor", ColumnTransformer(trans)), ("clf", clf)])


def _xy(df: pd.DataFrame, numeric: bool = False):
    cols = ["comment_text_norm"] + (NUMERIC_FEATURES if numeric else [])
    return df[cols], df["target_encoded"]


def _logreg():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0,
                              solver="saga", random_state=RANDOM_STATE)


def prepare_split(df: pd.DataFrame):
    """Shared label-encode + stratified-cap + product-grouped split. Deterministic
    (fixed RANDOM_STATE) so this reproduces the exact same split _train() uses
    internally -- used by the LoRA fine-tune script so it's compared against the
    TF-IDF baseline on the identical held-out product-grouped test set."""
    le = LabelEncoder()
    df = df.copy()
    df["target_encoded"] = le.fit_transform(df["recommendation_status"])
    sample = _stratified_cap(df, "target_encoded", MAX_PER_CLASS)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    gtr, gte = next(gss.split(sample, groups=sample["product_id"]))
    g_train, g_test = sample.iloc[gtr].copy(), sample.iloc[gte].copy()
    assert not (set(g_train["product_id"]) & set(g_test["product_id"]))
    return sample, g_train, g_test, le


def _train(df: pd.DataFrame) -> tuple[dict, dict]:
    """Train text-only model + baselines + grouped split + leakage ablation on a
    prepped frame. Returns (model_bundle, metrics). Persists nothing — callers do."""
    log.info("training rows: %d", len(df))
    le = LabelEncoder()
    df["target_encoded"] = le.fit_transform(df["recommendation_status"])
    inv = {i: c for i, c in enumerate(le.classes_)}

    sample = _stratified_cap(df, "target_encoded", MAX_PER_CLASS)
    train_df, temp = train_test_split(sample, test_size=0.40, stratify=sample["target_encoded"],
                                      random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp, test_size=0.50, stratify=temp["target_encoded"],
                                       random_state=RANDOM_STATE)
    # leakage guard: no shared comment text across splits
    assert not (set(train_df["comment_text_norm"]) & set(test_df["comment_text_norm"]))

    # quantify the residual leakage risk of the naive random split: how many test
    # products were *also* seen (via a different review) in training?
    train_pids, test_pids = set(train_df["product_id"]), set(test_df["product_id"])
    naive_overlap_pct = round(100 * len(train_pids & test_pids) / max(1, len(test_pids)), 2)

    # ---- text-only (the real model) ----
    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_test, y_test = _xy(test_df)

    # baselines (majority + logistic regression), text-only, on the validation split
    baselines = {}
    for name, clf in [("majority", DummyClassifier(strategy="most_frequent")),
                      ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                    C=1.0, random_state=RANDOM_STATE))]:
        pipe = _pipeline(clf, numeric=False).fit(X_train, y_train)
        baselines[name] = round(f1_score(y_val, pipe.predict(X_val), average="macro"), 4)
    log.info("baselines (val macro-F1): %s", baselines)

    # final model: text-only logistic regression (saga) — cheap, strong, reproducible
    final = _pipeline(_logreg(), numeric=False).fit(X_train, y_train)
    y_pred = final.predict(X_test)
    test_macro_f1 = round(f1_score(y_test, y_pred, average="macro"), 4)
    labels = list(le.classes_)
    y_test_label = [inv[i] for i in y_test]
    y_pred_label = [inv[i] for i in y_pred]
    report = classification_report(y_test_label, y_pred_label,
                                   labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # failure analysis: which (true, pred) confusions are most common, and a
    # handful of concrete misclassified review texts to inspect
    err = pd.DataFrame({"comment_text": test_df["comment_text_norm"].reset_index(drop=True),
                        "true": y_test_label, "pred": y_pred_label})
    err = err[err["true"] != err["pred"]]
    error_pairs = (err.groupby(["true", "pred"]).size().reset_index(name="count")
                  .sort_values("count", ascending=False).to_dict("records"))
    failure_examples = (err.sample(n=min(12, len(err)), random_state=RANDOM_STATE)
                        .assign(comment_text=lambda x: x["comment_text"].str.slice(0, 350))
                        .to_dict("records") if len(err) else [])

    # product-grouped validation (no product in both train and test), text-only
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    gtr, gte = next(gss.split(sample, groups=sample["product_id"]))
    g_train, g_test = sample.iloc[gtr], sample.iloc[gte]
    assert not (set(g_train["product_id"]) & set(g_test["product_id"]))
    gpipe = _pipeline(_logreg(), numeric=False).fit(*_xy(g_train))
    grouped_macro_f1 = round(f1_score(g_test["target_encoded"], gpipe.predict(_xy(g_test)[0]),
                                      average="macro"), 4)

    # ---- leakage ablation: text-only vs text + numeric (rate/likes/is_buyer) ----
    numeric_pipe = _pipeline(_logreg(), numeric=True).fit(*_xy(train_df, numeric=True))
    numeric_test_f1 = round(f1_score(y_test, numeric_pipe.predict(_xy(test_df, numeric=True)[0]),
                                     average="macro"), 4)
    ablation = {"text_only_macro_f1": test_macro_f1,
                "text_plus_numeric_macro_f1": numeric_test_f1,
                "leakage_lift": round(numeric_test_f1 - test_macro_f1, 4),
                "note": "The lift from adding rate_clean/likes/is_buyer is leakage: rate "
                        "restates the label, likes are post-hoc. The final model excludes them."}
    log.info("ablation text-only %.4f vs text+numeric %.4f (leak +%.4f)",
             test_macro_f1, numeric_test_f1, ablation["leakage_lift"])

    bundle = {"pipeline": final, "label_encoder": le, "labels": labels,
              "features": ["comment_text_norm"]}
    metrics = {"n_rows": int(len(df)), "n_sampled": int(len(sample)),
               "features_used": ["comment_text_norm"],
               "excluded_to_avoid_leakage": NUMERIC_FEATURES,
               "baselines_val_macro_f1": baselines,
               "primary_macro_f1": grouped_macro_f1,
               "primary_split": "product_grouped",
               "test_macro_f1": test_macro_f1,
               "grouped_macro_f1": grouped_macro_f1,
               "naive_split_product_overlap_pct": naive_overlap_pct,
               "leakage_ablation": ablation,
               "labels": labels, "confusion_matrix": cm, "classification_report": report,
               "error_pairs": error_pairs, "failure_examples": failure_examples}
    log.info("TEXT-ONLY test macro-F1 %.4f | grouped(PRIMARY) %.4f | naive-split product overlap %.2f%%",
             test_macro_f1, grouped_macro_f1, naive_overlap_pct)
    return bundle, metrics


def train_and_save() -> dict:
    """Full-data Phase-3 run: train on the cleaned parquet and persist model+metrics."""
    bundle, metrics = _train(_load())
    model_path = config.MODELS_DIR / "recommendation_model.pkl"
    joblib.dump(bundle, model_path)
    metrics["model_path"] = str(model_path)
    (config.METRICS_DIR / "phase3_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("saved %s", model_path.name)
    return metrics


def train_from_frame(comments_df: pd.DataFrame) -> tuple[dict, dict]:
    """Train from an in-memory cleaned-comments frame (the shared demo path used by
    the notebook and `run.py demo`). Returns (bundle, metrics); persists nothing so
    the full-data artifacts are never clobbered."""
    return _train(_prep(comments_df))


def fig_confusion(metrics: dict):
    """Plotly confusion-matrix heatmap for the dashboard/notebook."""
    import plotly.express as px
    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["labels"]
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig = px.imshow(cm_norm, x=labels, y=labels, text_auto=".2f", color_continuous_scale="Blues",
                    labels={"x": "predicted", "y": "true", "color": "row-normalized"},
                    title=f"Confusion matrix (product-grouped Macro-F1 = {metrics.get('primary_macro_f1', metrics.get('grouped_macro_f1'))})")
    return fig


def load_model():
    return joblib.load(config.MODELS_DIR / "recommendation_model.pkl")


def predict_with(bundle, texts) -> list[str]:
    """Classify text(s) with a given model bundle (text-only)."""
    if isinstance(texts, str):
        texts = [texts]
    from ..core import persian_text as pt
    X = pd.DataFrame({"comment_text_norm": [pt.normalize(t) for t in texts]})
    preds = bundle["pipeline"].predict(X)
    return [bundle["label_encoder"].inverse_transform([p])[0] for p in preds]


def predict(texts) -> list[str]:
    """Classify raw review text(s) from TEXT ONLY — used by the dashboard 'Try it!'."""
    return predict_with(load_model(), texts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(train_and_save()["test_macro_f1"], ensure_ascii=False))
