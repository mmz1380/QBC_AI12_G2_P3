"""Raw-data acquisition and streaming IO.

The comments CSV is ~6M rows / 1.3 GB, so we never load it whole unless asked:
`iter_comment_chunks` streams it for the chunked cleaner, and `reservoir_sample`
draws a uniform sample in a single pass with O(k) memory for the notebook.
"""
from __future__ import annotations

import logging
import os
import shutil

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger("digikala.dataio")

# The HF "Xet" transfer backend stalls on the big comments file; force plain HTTP.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def download_raw() -> None:
    """Fetch both CSVs from the pinned HF revision into data/raw if missing."""
    from huggingface_hub import hf_hub_download

    targets = {
        "digikala-products.csv": config.PRODUCTS_CSV,
        "digikala-comments.csv": config.COMMENTS_CSV,
    }
    for filename, dest in targets.items():
        if dest.exists() and dest.stat().st_size > 0:
            log.info("%s already present, skipping", dest.name)
            continue
        log.info("downloading %s @ %s", filename, config.HF_REVISION[:8])
        cached = hf_hub_download(repo_id=config.HF_REPO_ID, repo_type="dataset",
                                 filename=filename, revision=config.HF_REVISION)
        shutil.copy2(cached, dest)                   # copy out of the cache to keep the repo self-contained
        log.info("saved %s (%.0f MB)", dest.name, dest.stat().st_size / 1024**2)


def load_products() -> pd.DataFrame:
    """Products (~1.28M rows) fit in memory, so read the whole file."""
    return pd.read_csv(config.PRODUCTS_CSV, low_memory=False)


def iter_comment_chunks(chunksize: int | None = None):
    """Yield the comments CSV in row chunks so the cleaner never holds it all."""
    chunksize = chunksize or config.CHUNK_SIZE
    yield from pd.read_csv(config.COMMENTS_CSV, chunksize=chunksize, low_memory=False)


def load_comments(sample_size: int | None = "default") -> pd.DataFrame:
    """Read comments for the notebook: a reservoir sample by default, all if None."""
    if sample_size == "default":
        sample_size = config.COMMENTS_SAMPLE_SIZE
    if not sample_size:
        return pd.read_csv(config.COMMENTS_CSV, low_memory=False)
    return reservoir_sample(config.COMMENTS_CSV, sample_size, config.RANDOM_SEED)


def reservoir_sample(path, k: int, seed: int) -> pd.DataFrame:
    """Exact uniform k-row sample in one streaming pass, vectorized (fast even
    over 6M rows): give every row an i.i.d. random priority and keep the k
    smallest seen so far. Equivalent guarantee to classic reservoir sampling
    (every row equally likely to end up in the sample) but chunk-vectorized
    instead of a per-row Python loop, so it doesn't bottleneck a 6M-row scan."""
    if k <= 0:
        raise ValueError("sample size must be positive")
    rng = np.random.default_rng(seed)
    best = None
    for chunk in pd.read_csv(path, chunksize=config.CHUNK_SIZE, low_memory=False):
        chunk = chunk.copy()
        chunk["_sample_priority"] = rng.random(len(chunk))
        best = chunk if best is None else pd.concat([best, chunk], ignore_index=True)
        best = best.nsmallest(min(k, len(best)), "_sample_priority")
    if best is None:
        return pd.DataFrame()
    return (best.drop(columns="_sample_priority")
                .sample(frac=1, random_state=seed)     # shuffle away the priority order
                .reset_index(drop=True))


priority_sample = reservoir_sample                      # alias (same algorithm)
