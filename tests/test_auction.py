"""Sponsored Search Auction: economic invariants + the query-time allocator.

These are the same checks `validate_auction_system()` runs at scale (500
random trials), asserted directly so a regression fails the test suite, not
just a notebook printout.
"""
import numpy as np
import pytest

from digikala.phase5_auction import auction


def _random_trial(rng):
    bids = rng.integers(1000, 50001, size=3).astype(float)
    qualities = rng.uniform(0.2, 1.0, size=3)
    return bids, qualities


@pytest.mark.parametrize("seed", range(20))
def test_gsp_invariants_hold_across_random_trials(seed):
    rng = np.random.default_rng(seed)
    for _ in range(25):                      # 20 seeds x 25 = 500 trials total
        bids, qualities = _random_trial(rng)
        rows = auction.quality_adjusted_gsp(bids, qualities)

        vendor_ids = [r["vendor_index"] for r in rows]
        slots = [r["slot"] for r in rows]
        ranks = [r["ad_rank"] for r in rows]

        assert len(set(vendor_ids)) == 3, "each vendor allocated exactly one slot"
        assert slots == [1, 2, 3], "slots always 1,2,3 in rank order"
        assert all(ranks[i] >= ranks[i + 1] for i in range(len(ranks) - 1)), "descending ad rank"
        assert all(r["actual_cpc"] <= r["max_cpc"] + 1e-9 for r in rows), "actual CPC never exceeds max CPC"
        assert all(r["actual_cpc"] >= 0 for r in rows), "CPC never negative"


def test_gsp_rejects_wrong_vendor_count():
    with pytest.raises(ValueError):
        auction.quality_adjusted_gsp([100, 200], [0.5, 0.5])


def test_gsp_rejects_negative_bids():
    with pytest.raises(ValueError):
        auction.quality_adjusted_gsp([100, -1, 50], [0.5, 0.5, 0.5])


def test_highest_bid_does_not_automatically_win_on_quality():
    # Vendor 0 bids highest but has much lower quality than vendor 1.
    rows = auction.quality_adjusted_gsp(bids=[10000, 6000, 5000], qualities=[0.1, 0.9, 0.5])
    winner = next(r for r in rows if r["slot"] == 1)
    assert winner["vendor_index"] == 1, "quality-adjusted ad rank should beat a low-quality high bid"


def test_bid_only_baseline_ignores_quality():
    rows = auction.bid_only_auction(bids=[10000, 6000, 5000], qualities=[0.1, 0.9, 0.5])
    winner = next(r for r in rows if r["slot"] == 1)
    assert winner["vendor_index"] == 0, "the naive baseline ranks by raw bid only"


def test_validate_auction_system_reports_zero_violations_and_writes_artifact(tmp_path, monkeypatch):
    from digikala import config
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path)
    report = auction.validate_auction_system(n_invariant_trials=100, n_simulation_trials=200)
    assert report["violations"] == 0
    assert report["invariant_pass_rate"] == 1.0
    assert report["mentor_approved_new_problem"] is False        # default: not mentor-approved
    assert report["bonus_claim_supported"] is False               # gated even if technically sound
    assert (tmp_path / "auction_metrics.json").exists()


def test_run_query_auction_respects_active_flag_and_max_cpc():
    campaigns = [
        {"vendor_name": "A", "product_id": 1, "max_cpc": 8000, "active": True},
        {"vendor_name": "B", "product_id": 2, "max_cpc": 6000, "active": False},   # inactive: excluded
        {"vendor_name": "C", "product_id": 3, "max_cpc": 5000, "active": True},
    ]
    products = {1: {"product_rate_clean": 4.5}, 2: {"product_rate_clean": 4.9}, 3: {"product_rate_clean": 3.0}}
    stats = {1: {"recommendation_rate": 90}, 3: {"recommendation_rate": 60}}

    rows = auction.run_query_auction(campaigns, lambda pid: products.get(pid), stats,
                                     query_scores={1: 5.0, 3: 2.0})

    assert len(rows) == 2, "inactive campaign must be excluded from allocation"
    assert all(r["actual_cpc"] <= r["max_cpc"] for r in rows)
    assert {r["placement_position"] for r in rows} <= set(auction.SPONSORED_POSITIONS)


def test_run_query_auction_skips_unknown_product_id():
    campaigns = [{"vendor_name": "A", "product_id": 999999, "max_cpc": 5000, "active": True}]
    rows = auction.run_query_auction(campaigns, lambda pid: None, {})
    assert rows == []
