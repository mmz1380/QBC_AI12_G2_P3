"""Central configuration: paths, run modes, sample sizes, model + budget settings.

Everything the pipeline needs to know lives here so the package, the notebook and
the dashboard all read the same knobs. Override any value with an environment
variable of the same name (e.g. DIGIKALA_RUN_MODE=free) or by editing this file.

API keys are read from a local `.env` file (see `.env.sample`) via python-dotenv,
never hardcoded or committed.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # populates os.environ from a local .env, if present; no-op otherwise


def _env(name: str, default: str) -> str:
    return os.environ.get(f"DIGIKALA_{name}", default)


# ---- paths --------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # repo root (E:\QBC13_AI_G6_Project3)
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"

for _d in (RAW_DIR, PROCESSED_DIR, INDEX_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- raw dataset (pinned so it can't change under us) -------------------
HF_REPO_ID = "RadeAI/Digikala_comments_products"
HF_REVISION = "89c3133b169c8d3793db8834f56f32fee33d9db0"
PRODUCTS_CSV = RAW_DIR / "digikala-products.csv"
COMMENTS_CSV = RAW_DIR / "digikala-comments.csv"

# ---- cleaned outputs ----------------------------------------------------
PRODUCTS_CLEAN = PROCESSED_DIR / "products_clean.parquet"
COMMENTS_CLEAN = PROCESSED_DIR / "comments_clean.parquet"
PHASE1_REPORT = PROCESSED_DIR / "phase1_report.json"

# ---- retrieval indexes (products embedded fully; reviews on demand) ------
PRODUCT_INDEX_DIR = INDEX_DIR / "products"

# ---- data volume --------------------------------------------------------
# Cleaning/EDA/Phase-3 stream the full comments file in chunks, so they don't
# need a sample. COMMENTS_SAMPLE_SIZE only caps what the *notebook* loads so a
# reviewer can "Run All" in minutes; set it to None (or DIGIKALA_SAMPLE=full) to
# process everything. The package's run.py always works on the full data.
COMMENTS_SAMPLE_SIZE: int | None = (
    None if _env("SAMPLE", "").lower() in {"full", "none", "0"}
    else int(_env("SAMPLE", "200000"))
)
CHUNK_SIZE = 200_000            # rows per streaming chunk when cleaning comments
RANDOM_SEED = 42

RECOMMENDATION_CLASSES = ("recommended", "not_recommended", "no_idea")
TOMAN_TO_RIAL = 10             # prices are stored in Rials; users talk in Tomans

# ---- embeddings ---------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = _env("EMBED_DEVICE", "cuda")     # falls back to cpu automatically
TOP_K = 8

# hybrid retrieval tuning: candidate pools keep RRF cheap at ~1M-product scale;
# weights let lexical (sparse) evidence count slightly more for exact product
# attributes while keeping dense semantic retrieval in the mix
RRF_CANDIDATE_POOL = 200
REVIEW_CANDIDATE_POOL = 80
PRODUCT_RRF_DENSE_WEIGHT = 0.65
PRODUCT_RRF_SPARSE_WEIGHT = 1.35
# review reranking: how much a detected pos/neg query intent (vs base RRF score)
# shifts which reviews get selected as evidence -- see phase2_assistant/retrieval.py
REVIEW_NEGATIVE_INTENT_WEIGHT = 0.32
REVIEW_POSITIVE_INTENT_WEIGHT = 0.28

# ---- LLM run mode -------------------------------------------------------
# Modes, per the brief:
#   local        -> a local HF model (Qwen) or an Ollama server        ($0)
#   free         -> a free-tier hosted API (Groq / OpenRouter)         ($0, rate-limited)
#   paid         -> the $5 credit on an OpenAI-compatible gateway      (tracked)
#   hosted_auto  -> auto-detect whichever hosted key is present (groq, then paid);
#                   this is what a .env with GROQ_API_KEY/PAID_API_KEY set switches
#                   the assistant to automatically, no code change needed
# extractive is the always-available no-LLM tier the assistant falls back to.
#
# If DIGIKALA_RUN_MODE is not explicitly set, auto-detect from .env: a hosted key
# present -> hosted_auto; otherwise -> local (the $0, always-available default).
_explicit_run_mode = os.environ.get("DIGIKALA_RUN_MODE")
if _explicit_run_mode:
    RUN_MODE = _explicit_run_mode
elif os.environ.get("GROQ_API_KEY") or os.environ.get("PAID_API_KEY"):
    RUN_MODE = "hosted_auto"
else:
    RUN_MODE = "local"

# local
LOCAL_BACKEND = _env("LOCAL_BACKEND", "transformers")   # transformers | ollama
HF_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"        # fits the 6 GB GPU next to the embedder
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

# hosted providers (OpenAI-compatible chat/completions). Keys via .env only.
GROQ_MODEL_PRICES = {                               # $/1M tokens (input, output)
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}
PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
             "price_per_m": GROQ_MODEL_PRICES.get(
                 os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"), (0.075, 0.30)),
             "price_note": "Estimated list cost from configured per-million-token rates, "
                            "not a Groq invoice.",
             # Groq's free tier is $0; only mark it "billed" (counts against the
             # $5 cap) if you explicitly opt in via DIGIKALA_GROQ_BILLED=1.
             "billed": _env("GROQ_BILLED", "0").strip().lower() in {"1", "true", "yes"},
             "api_key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "model": "meta-llama/llama-3.3-70b-instruct:free",
                   "price_per_m": (0.0, 0.0), "price_note": "OpenRouter's :free tier.",
                   "billed": False,
                   "api_key_env": "OPENROUTER_API_KEY"},
    "paid": {"base_url": _env("PAID_BASE_URL", "https://api.openai.com/v1"),
             "model": _env("PAID_MODEL", "gpt-4o-mini"),
             "price_per_m": (float(_env("PAID_IN_PRICE", "0.15")),
                              float(_env("PAID_OUT_PRICE", "0.60"))),
             "price_note": "Estimated list cost from configured per-million-token rates.",
             "billed": True,
             "api_key_env": "PAID_API_KEY"},
}
FREE_PROVIDER = _env("FREE_PROVIDER", "groq")      # which entry above the "free" mode uses
HOSTED_PROVIDER_ORDER = ("groq", "paid")           # try order for RUN_MODE=hosted_auto

# bounded retry/network-fallback for hosted calls (some Windows/VPN setups have
# an environment proxy that hangs while a direct, no-proxy path works fine)
API_MAX_ATTEMPTS = 3
API_RETRY_BASE_S = 1.5
API_CONNECT_TIMEOUT_S = 30
API_READ_TIMEOUT_S = 120

LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.0

# ---- budget fence (the $5 rule) -----------------------------------------
BUDGET_USD = 5.0
BUDGET_LOG = METRICS_DIR / "budget_log.jsonl"
# rough $/1M tokens for cost accounting on paid calls (input, output)
PAID_PRICE_PER_M = (float(_env("PAID_IN_PRICE", "0.15")), float(_env("PAID_OUT_PRICE", "0.60")))

# ---- evaluation judge ---------------------------------------------------
# Which model scores answers in Phase 4. Local by default (free, offline);
# switch to "free"/"paid"/"hosted_auto" to use a hosted judge.
JUDGE_MODE = _env("JUDGE_MODE", "local")           # local | free | paid | hosted_auto | none
