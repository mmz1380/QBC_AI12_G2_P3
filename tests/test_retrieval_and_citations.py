import numpy as np

from digikala.phase2_assistant.retrieval import BM25Okapi, rrf_fuse, product_filter_mask
from digikala.phase2_assistant.assistant import verify_citations
from digikala.phase4_eval.evaluate import ranking_metrics


def test_bm25_ranks_matching_doc_first():
    bm = BM25Okapi.from_texts(["کیف چرم زنانه", "گوشی موبایل", "کفش ورزشی"])
    scores = bm.get_scores("کیف چرم")
    assert int(np.argmax(scores)) == 0
    assert scores[0] > 0


def test_rrf_fuse_rewards_agreement():
    fused = rrf_fuse([[0, 1, 2], [0, 2, 1]])
    assert max(fused, key=fused.get) == 0


def test_product_filter_mask_price(products):
    mask = product_filter_mask(products, {"price_max": 3_000_000})
    assert mask.tolist() == [True, False, False]   # only the 2M product passes


def test_verify_citations_strips_unknown_ids():
    text = "خوب است [محصول 101] ولی [محصول 999] نامعتبر [بازبینی 5]"
    out = verify_citations(text, allowed_products={101}, allowed_reviews={5})
    assert "[محصول 101]" in out
    assert "999" not in out
    assert "[بازبینی 5]" in out


def test_ranking_metrics_perfect():
    m = ranking_metrics([7, 1, 2], {7}, k=3)
    assert m["recall@k"] == 1.0 and m["mrr"] == 1.0
