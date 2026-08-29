"""Sponsored Search Auction — a mentor-suggested optional business extension.

Three vendors can each register a (product_id, max_cpc, active) campaign. For a
given search query, eligible campaigns are ranked by Ad Rank = max_cpc *
quality * query_relevance (a simplified quality-adjusted Generalized Second
Price mechanism), NOT by raw bid alone -- so the highest bidder does not
automatically win. The quality factor is derived from data (product rating +
recommendation rate), never set by the advertiser. Winners are inserted at
fixed sponsored positions (1, 3, 5) and must always be labeled "Sponsored /
تبلیغ" -- never presented as organic recommendations, and "sponsored" must
never be treated as evidence a product is objectively better.

`validate_auction_system()` is the offline evidence: hundreds of randomized
trials checking the auction's economic invariants (unique allocation, correct
slot/rank ordering, actual CPC <= max CPC, non-negative CPC), plus a
simulation comparing this quality-adjusted mechanism against a naive
highest-bid-only baseline. Simulated CTR/revenue are explicitly proxy values,
never real production outcomes.

`mentor_approved`, off by default, gates whether this is claimable as an
approved new problem versus an unapproved experimental extension -- per the
project rule that IMPLEMENTED != SUCCESSFUL != CLAIMABLE.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger("digikala.auction")

SPONSORED_POSITIONS = [1, 3, 5]
RESERVE_CPC_TOMAN = 1000.0


def quality_adjusted_gsp(bids, qualities, reserve_cpc: float = RESERVE_CPC_TOMAN) -> list[dict]:
    """Rank exactly 3 vendors by Ad Rank = bid * quality; price via a
    quality-adjusted second-price rule so actual_cpc <= max_cpc always."""
    bids = np.asarray(bids, dtype=float)
    qualities = np.asarray(qualities, dtype=float)
    if len(bids) != 3 or len(qualities) != 3:
        raise ValueError("Exactly three vendors are required.")
    if np.any(bids < 0):
        raise ValueError("Bids cannot be negative.")

    qualities = np.clip(qualities, 0.05, 1.0)
    ad_rank = bids * qualities
    order = np.argsort(-ad_rank, kind="stable")

    rows = []
    for slot, idx in enumerate(order, start=1):
        next_rank = ad_rank[order[slot]] if slot < len(order) else reserve_cpc * qualities[idx]
        threshold_cpc = (next_rank / max(qualities[idx], 1e-9)) + 1.0
        actual_cpc = min(bids[idx], max(reserve_cpc, threshold_cpc))
        rows.append({
            "vendor_index": int(idx), "slot": int(slot),
            "max_cpc": round(float(bids[idx]), 2), "quality": round(float(qualities[idx]), 4),
            "ad_rank": round(float(ad_rank[idx]), 4), "actual_cpc": round(float(actual_cpc), 2),
        })
    return rows


def bid_only_auction(bids, qualities, reserve_cpc: float = RESERVE_CPC_TOMAN) -> list[dict]:
    """Naive baseline: rank only by bid, ignoring quality/relevance entirely."""
    bids = np.asarray(bids, dtype=float)
    qualities = np.asarray(qualities, dtype=float)
    order = np.argsort(-bids, kind="stable")

    rows = []
    for slot, idx in enumerate(order, start=1):
        next_bid = bids[order[slot]] if slot < len(order) else reserve_cpc
        actual_cpc = min(bids[idx], max(reserve_cpc, next_bid + 1.0))
        rows.append({
            "vendor_index": int(idx), "slot": int(slot),
            "max_cpc": round(float(bids[idx]), 2), "quality": round(float(qualities[idx]), 4),
            "ad_rank": round(float(bids[idx]), 4), "actual_cpc": round(float(actual_cpc), 2),
        })
    return rows


def _position_ctr(slot: int) -> float:
    """Transparent presentation-only CTR proxy for the offline simulation."""
    return {1: 0.12, 2: 0.075, 3: 0.05}.get(int(slot), 0.03)


def _simulated_market_metrics(rows: list[dict]) -> dict:
    weighted_quality = expected_revenue = 0.0
    for row in rows:
        ctr = _position_ctr(row["slot"]) * float(row["quality"])
        weighted_quality += ctr * float(row["quality"])
        expected_revenue += ctr * float(row["actual_cpc"])
    return {"quality_weighted_click_value": weighted_quality,
            "expected_revenue_per_impression_proxy": expected_revenue}


def validate_auction_system(seed: int = 42, n_invariant_trials: int = 500,
                            n_simulation_trials: int = 2000,
                            mentor_approved: bool | None = None) -> dict:
    """Randomized offline validation: economic invariants + a simulation
    comparing quality-adjusted GSP against a naive highest-bid baseline.
    Writes `artifacts/metrics/auction_metrics.json`."""
    if mentor_approved is None:
        mentor_approved = config.MENTOR_APPROVED_AUCTION
    rng = np.random.default_rng(seed)
    violations = []

    for trial in range(n_invariant_trials):
        bids = rng.integers(1000, 50001, size=3).astype(float)
        qualities = rng.uniform(0.2, 1.0, size=3)
        rows = quality_adjusted_gsp(bids, qualities, reserve_cpc=RESERVE_CPC_TOMAN)
        ranks = [r["ad_rank"] for r in rows]
        slots = [r["slot"] for r in rows]
        vendor_ids = [r["vendor_index"] for r in rows]
        checks = {
            "three_unique_winners": len(set(vendor_ids)) == 3,
            "slots_are_1_2_3": slots == [1, 2, 3],
            "rank_is_descending": all(ranks[i] >= ranks[i + 1] for i in range(len(ranks) - 1)),
            "actual_cpc_never_exceeds_bid": all(r["actual_cpc"] <= r["max_cpc"] + 1e-9 for r in rows),
            "nonnegative_cpc": all(r["actual_cpc"] >= 0 for r in rows),
        }
        if not all(checks.values()):
            violations.append({"trial": trial, "checks": checks})

    qa_quality, base_quality, qa_revenue, base_revenue = [], [], [], []
    for _ in range(n_simulation_trials):
        bids = rng.integers(1000, 50001, size=3).astype(float)
        # "quality" here stands in for the combined product-quality/query-relevance
        # factor the live dashboard computes -- this is an offline proxy
        # simulation, not observed production CTR.
        qualities = np.clip(rng.beta(2.5, 1.8, size=3), 0.05, 1.0)
        qa_rows = quality_adjusted_gsp(bids, qualities, reserve_cpc=RESERVE_CPC_TOMAN)
        base_rows = bid_only_auction(bids, qualities, reserve_cpc=RESERVE_CPC_TOMAN)
        qa_m = _simulated_market_metrics(qa_rows)
        base_m = _simulated_market_metrics(base_rows)
        qa_quality.append(qa_m["quality_weighted_click_value"])
        base_quality.append(base_m["quality_weighted_click_value"])
        qa_revenue.append(qa_m["expected_revenue_per_impression_proxy"])
        base_revenue.append(base_m["expected_revenue_per_impression_proxy"])

    mean_qa_quality, mean_base_quality = float(np.mean(qa_quality)), float(np.mean(base_quality))
    mean_qa_revenue, mean_base_revenue = float(np.mean(qa_revenue)), float(np.mean(base_revenue))
    quality_lift = mean_qa_quality / max(mean_base_quality, 1e-12) - 1.0
    revenue_lift = mean_qa_revenue / max(mean_base_revenue, 1e-12) - 1.0

    demo = quality_adjusted_gsp(bids=[8000, 6200, 5400], qualities=[0.72, 0.93, 0.81],
                                reserve_cpc=RESERVE_CPC_TOMAN)

    report = {
        "available": True,
        "mentor_requested": True,
        "mentor_approved_new_problem": bool(mentor_approved),
        "mentor_approval_status": "approved" if mentor_approved else "pending_confirmation",
        "problem": "three_vendor_sponsored_search_auction",
        "auction_type": "quality-adjusted generalized second-price",
        "vendors": 3,
        "sponsored_search_positions": SPONSORED_POSITIONS,
        "reserve_cpc_toman": RESERVE_CPC_TOMAN,
        "invariant_trials": int(n_invariant_trials),
        "violations": int(len(violations)),
        "invariant_pass_rate": round(1.0 - len(violations) / n_invariant_trials, 4),
        "offline_simulation": {
            "trials": int(n_simulation_trials),
            "baseline": "highest-bid ranking + second-price CPC",
            "proposed": "quality-adjusted GSP",
            "mean_quality_weighted_click_value_baseline": round(mean_base_quality, 6),
            "mean_quality_weighted_click_value_proposed": round(mean_qa_quality, 6),
            "quality_proxy_lift_pct": round(100 * quality_lift, 2),
            "mean_expected_revenue_proxy_baseline": round(mean_base_revenue, 4),
            "mean_expected_revenue_proxy_proposed": round(mean_qa_revenue, 4),
            "revenue_proxy_lift_pct": round(100 * revenue_lift, 2),
            "important_limitation": ("CTR/revenue values are transparent offline simulation "
                                     "proxies, not observed production advertising outcomes."),
        },
        "demo_allocation": demo,
        "tested_invariants": ["unique allocation", "slot ordering", "descending ad rank",
                              "actual CPC <= max CPC", "nonnegative CPC"],
    }
    report["technical_result_supported"] = bool(len(violations) == 0 and quality_lift > 0)
    report["bonus_claim_supported"] = bool(mentor_approved and report["technical_result_supported"])

    out = config.METRICS_DIR / "auction_metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("auction validation: %d/%d invariant trials clean, quality lift %.1f%%",
             n_invariant_trials - len(violations), n_invariant_trials, 100 * quality_lift)
    return report


# ---- live per-query allocation (used by the dashboard's Auction tab) ------
def compute_quality(rate_0_5: float | None, recommendation_rate_pct: float | None,
                    query_relevance_0_1: float) -> float:
    """Data-derived quality factor -- never set by the advertiser. Blends
    product rating, the review-based recommendation rate, and how relevant
    this product is to the current search query."""
    rate_norm = float(np.clip(rate_0_5 / 5.0, 0, 1)) if rate_0_5 is not None and pd.notna(rate_0_5) else 0.5
    rec_norm = (float(np.clip(recommendation_rate_pct / 100.0, 0, 1))
                if recommendation_rate_pct is not None and pd.notna(recommendation_rate_pct) else 0.5)
    rel_norm = float(np.clip(query_relevance_0_1, 0, 1))
    return float(np.clip(0.40 * rate_norm + 0.35 * rec_norm + 0.25 * rel_norm, 0.05, 1.0))


def run_query_auction(campaigns: list[dict], product_lookup, stats_lookup: dict,
                      query_scores: dict[int, float] | None = None,
                      reserve_cpc: float = RESERVE_CPC_TOMAN) -> list[dict]:
    """Allocate sponsored slots for one search query.

    `campaigns`: up to 3 dicts with vendor_name/product_id/max_cpc/active.
    `product_lookup(pid)` -> dict with at least 'product_rate_clean', or None
    if the product id doesn't exist in the sampled catalogue.
    `stats_lookup`: {product_id: {"recommendation_rate": pct}} from review stats.
    `query_scores`: optional {product_id: raw_retrieval_score} for relevance;
    scores are normalized against the max positive score seen this query.
    """
    query_scores = query_scores or {}
    max_score = max((s for s in query_scores.values() if s > 0), default=0.0)

    ranked = []
    for campaign in campaigns[:3]:
        if not campaign.get("active", True):
            continue
        pid = int(campaign["product_id"])
        prod = product_lookup(pid)
        if prod is None:
            continue
        raw_rel = float(query_scores.get(pid, 0.0))
        rel = raw_rel / max_score if max_score > 0 else 0.5
        stat = stats_lookup.get(pid, {})
        quality = compute_quality(prod.get("product_rate_clean"), stat.get("recommendation_rate"), rel)
        bid = max(0.0, float(campaign["max_cpc"]))
        ranked.append({**campaign, "product_id": pid, "quality": quality,
                       "query_relevance": float(np.clip(rel, 0, 1)), "ad_rank": bid * quality})

    ranked.sort(key=lambda x: x["ad_rank"], reverse=True)
    for i, item in enumerate(ranked):
        next_rank = ranked[i + 1]["ad_rank"] if i + 1 < len(ranked) else reserve_cpc * item["quality"]
        threshold = next_rank / max(item["quality"], 1e-9) + 1.0
        item["slot"] = i + 1
        item["placement_position"] = SPONSORED_POSITIONS[min(i, len(SPONSORED_POSITIONS) - 1)]
        item["actual_cpc"] = min(float(item["max_cpc"]), max(reserve_cpc, threshold))
    return ranked
