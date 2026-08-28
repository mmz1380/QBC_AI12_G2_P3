# Digikala AI Shopping Assistant — QBC13 AI · Group 6 · Project 3

A grounded, Persian-language shopping assistant over the real Digikala catalogue
(≈1.28 M products, ≈6 M reviews). It covers all four required capabilities, a
recommendation-status classifier, a full evaluation suite, and an interactive
dashboard — with a strict **$0-by-default** design and a **$5 API budget fence**.

## What it does (the four phases)

| Phase | What | Where |
|---|---|---|
| **1 · Data** | Stream-clean the raw CSVs into a documented `_norm`/`_clean` schema; Plotly EDA | `src/digikala/phase1_data/` |
| **2 · Assistant** | Discovery · review Q&A · comparison · manager analytics — hybrid retrieval, grounded citations | `src/digikala/phase2_assistant/` |
| **3 · Prediction** | Classify `recommendation_status` (recommended / not_recommended / no_idea), **Macro-F1** | `src/digikala/phase3_predict/` |
| **4 · Evaluation** | Retrieval (recall/MRR/nDCG), grounding, response quality (LLM-judge), latency, cost, failure analysis | `src/digikala/phase4_eval/` |

Everything is also packaged as a **self-contained notebook**
(`notebooks/digikala_project3_standalone.ipynb`) and an **interactive Streamlit
dashboard** (`dashboard/app.py`).

## Architecture

```
run.py                      startup script (setup/clean/eda/index/train/eval/dashboard/all/menu)
src/digikala/
  config.py                 all paths, run-mode, sample size, budget
  core/                     persian_text · dataio (streaming) · llm (3 modes + BudgetTracker)
  phase1_data/              clean (chunked full-data) · eda (Plotly)
  phase2_assistant/         retrieval (dense+BM25+RRF) · router · prompts · assistant
  phase3_predict/           recommend (TF-IDF + linear, Macro-F1)
  phase4_eval/              evaluate (six-axis suite)
dashboard/app.py            Streamlit UI (light/dark, to-top, "Try it!" panels, Plotly)
data/{raw,processed,index}  raw CSVs · cleaned parquet · product embedding index
artifacts/{figures,metrics,models}
```

**Key design choices (all defensible per the brief):**
- **Hybrid retrieval**: dense multilingual-MiniLM embeddings **+** a from-scratch
  Okapi BM25 (scipy-sparse), fused with **Reciprocal Rank Fusion**; structured
  filters (price/brand/category/exclude-fake) are applied *before* ranking.
- **Products embedded fully, reviews retrieved per-product on demand** — embedding
  all 6 M reviews is impractical on a 6 GB GPU, and a product has at most a few
  hundred reviews, so this scales while staying grounded.
- **Zero-hallucination contract**: every claim must cite `[محصول <id>]` /
  `[بازبینی <id>]`; `verify_citations` strips any id the retriever didn't return.
  When no LLM is configured the answer is rendered **extractively** from the
  evidence — provably grounded and **$0**.
- **Deterministic intent router** (no LLM) → free, reproducible routing.

## Install

Python **3.11** (hazm needs it). From the repo root:

```bash
python -m venv .venv
./.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
./.venv/Scripts/python -m pip install -r requirements.txt
```

## Run

```bash
python run.py setup      # download the pinned dataset revision into data/raw
python run.py clean      # Phase 1: stream-clean the FULL data  (add --sample for a fast subset)
python run.py eda        # Phase 1: write Plotly EDA figures
python run.py index      # Phase 2: build + cache the product embedding index
python run.py train      # Phase 3: train the classifier, report Macro-F1
python run.py eval       # Phase 4: evaluation suite (+ failure analysis, retrieval ablation)
python run.py lora       # bonus: LoRA fine-tune vs. the TF-IDF baseline
python run.py dashboard  # launch the Streamlit dashboard
python run.py all        # end-to-end (add --sample for the quick path)
python run.py demo       # deterministic sample pipeline (mirrors the notebook exactly)
python run.py test       # run the test suite (tests/)
python run.py menu       # interactive menu
```

**No notebook-vs-app discrepancy.** `python run.py demo --sample-size 20000` and the
standalone notebook call the *same* functions on the *same* deterministic sample, so
their substantive outputs (retrieval metrics, Macro-F1, predictions, citations) are
byte-identical. The full run (`run.py all`) is the same code at scale.

## No data leakage (Phase 3)

The recommendation model is **text-only**. `rate_clean`, `likes` and `is_buyer` are
excluded: `rate` is a second copy of the same sentiment as the label, `likes` accrue
*after* posting (not available at prediction time), and the brief asks us to predict
from the *text*. `train` reports a text-vs-text+numeric **ablation** that quantifies
the leak (≈ +0.04 Macro-F1).

A second, subtler leakage channel was also audited and fixed: a **naive random
row-level split** can still put two different reviews of the *same* product in
both train and test (brand names, model-specific phrases the model could memorize
instead of generalizing sentiment language) — even after exact-duplicate-text dedup
removes identical rows. So the **headline metric is now the product-grouped
Macro-F1** (`primary_macro_f1` in `phase3_metrics.json`), where no product appears
in both splits. The naive random split is still reported alongside it together with
`naive_split_product_overlap_pct`, so any gap between the two numbers is explained
by measured data, not asserted away. Other guards: dedup by review text across
splits (`_prep`), a majority-class baseline, and the TF-IDF vectorizer is fit only
inside the training fold via `sklearn.Pipeline.fit(X_train, y_train)` — never on
combined train+test data, which would otherwise leak vocabulary/idf statistics.

**Sampling justification (Phase 3):** training uses a class-stratified cap
(`MAX_PER_CLASS = 30,000` per class) rather than all ~4M labeled reviews. This is a
resource-constrained but reasoned choice, not an arbitrary one: recommendation
sentiment is a comparatively easy, saturating text-classification signal (a
majority-class baseline already sits at 0.167 Macro-F1; the trained model reaches
0.72+ well before consuming the full corpus), so the marginal value of more rows is
low relative to the extra fit/eval time on a laptop GPU.

### Run modes, `.env`, and the $5 budget

Copy `.env.sample` to `.env` and fill in whatever hosted key you have (or none —
everything works at **$0** either way):

```bash
cp .env.sample .env
```

```bash
# .env (git-ignored, never committed)
GROQ_API_KEY=...
PAID_API_KEY=...
OPENROUTER_API_KEY=...
```

`config.py` loads `.env` via `python-dotenv` and **auto-detects the run mode**
if you don't set `DIGIKALA_RUN_MODE` explicitly: a hosted key present →
`hosted_auto` (tries Groq, then the paid gateway, in `HOSTED_PROVIDER_ORDER`);
no key → `local` (the $0 Qwen/Ollama tier). This is the ".env → auto mode
switch" behavior — put a key in `.env` and the assistant picks it up on the
next run, no code change needed.

```bash
# override explicitly instead of auto-detecting:
DIGIKALA_RUN_MODE=extractive   # $0, always-grounded, no model at all
DIGIKALA_RUN_MODE=local        # local Qwen (transformers) or Ollama
DIGIKALA_RUN_MODE=hosted_auto  # auto-detect: groq -> paid (this is the .env-driven default)
DIGIKALA_RUN_MODE=free         # Groq / OpenRouter free tier  (set DIGIKALA_FREE_PROVIDER)
DIGIKALA_RUN_MODE=paid         # $5 credit, OpenAI-compatible gateway (tracked)

# data volume for the notebook / quick runs:
DIGIKALA_SAMPLE=full           # or an integer, e.g. 200000
```

The hosted client (`core/llm.py`) retries transient failures (429/5xx) and
falls back from the normal "environment proxy" network path to a direct,
no-proxy path — some Windows/VPN setups have an environment proxy that hangs
while direct works fine, so a single dead network path doesn't take down
hosted mode. Every hosted call passes through `BudgetTracker`, which tracks
attempted/successful/failed calls and tokens, logs them to
`artifacts/metrics/budget_log.jsonl`, and distinguishes **tracked cost**
(counts against the $5 cap) from **estimated list cost** (always reported,
even for Groq's free tier, which costs $0 unless you set
`DIGIKALA_GROQ_BILLED=1`). `LLM.diagnose()` runs a bounded preflight (model
listing, then a tiny fully-accounted chat probe if needed) so a dead key or
wrong model name is caught before a real run.

> Tip: after the models are cached, set `HF_HUB_OFFLINE=1` to skip HuggingFace
> network checks and load faster.

## Evaluation

`python run.py eval` writes `artifacts/metrics/phase4_metrics.json` (+ per-query /
per-intent CSVs) covering the six axes the brief requires: **response quality**
(LLM-judge relevance + a deterministic non-LLM proxy), **grounding** (citation
coverage/validity + LLM-judge faithfulness + a deterministic proxy),
**retrieval quality** (recall@k / MRR / nDCG on two benchmarks — see below —
plus a hybrid-vs-single-method **ablation**), **prediction** (Macro-F1),
**latency & cost**, and **failure analysis** (per-intent missing-info /
zero-citation rates + concrete adversarial probes + a router-regression check).
The judge model is chosen with `DIGIKALA_JUDGE_MODE` (local by default). A
non-LLM lexical baseline is included as the control throughout.

**Two retrieval benchmarks, on purpose.** `evaluate_retrieval`/`retrieval_ablation`
query each product by its own *title* — a near-exact lexical match that
structurally favors BM25. `evaluate_retrieval_natural` instead builds
brand+category+partial-title *paraphrase* queries (`build_natural_retrieval_cases`,
category-diversity-capped so one category can't dominate) — the fair test of
hybrid's actual value on realistic natural-language requests. **Both
benchmarks currently show the lexical/BM25 baseline beating hybrid on nDCG**,
and this specific submission's own reference implementation
(`reference/digikala_project3_submission/`) measured the same thing
independently on a different sample — this is reported as a real, corroborated
property of this embedding model on this catalogue, not a bug or a benchmark
artifact. Hybrid still clearly beats dense-only on both benchmarks.

**Deterministic quality proxy, alongside the LLM judge.**
`deterministic_quality_proxy`/`task_completion_proxy` give a fully reproducible,
non-LLM, intent-specific 0–5 score (does the answer actually contain the
citations/structure each capability requires?) — a second axis that doesn't
depend on the local judge's reliability. `citation_validity` separately checks
that every citation the answer text contains was actually in the retriever's
allowed set (should be ~1.0 given `verify_citations`; this is a direct check
of that guarantee, not a duplicate of citation coverage).

**Response-eval queries are separate from the demo queries.**
`build_response_eval_queries` builds a query set from held-out, high-support
categories (not the same examples shown in the Phase-2 demo cells) and checks
each query's actual router intent against its expected intent — a lightweight
regression test embedded in evaluation itself. It caught a real router bug
this session (see "Router regression caught" below) rather than raising a hard
error, since the project's failure philosophy throughout is to degrade
gracefully and report, not crash.

### Router regression caught (real bug, found by the eval's own routing check)

`build_response_eval_queries`'s routing check caught two real intent-router
bugs this session:
1. The generic word `"چند"` ("how many") was a `product_qa` cue; for a query
   with no explicit product id it triggered a *fuzzy* title-token-overlap
   search (`resolve_product_id`) over all reviewed products, which could
   coincidentally match an unrelated product and misroute a plain "how many
   good products are in category X" request to `product_qa`.
2. The bare word `"دسته"` ("category") was itself a managerial cue — but
   `scope` resolution already detects a category mention, so *any* query
   naming a category (`"در دستهٔ X ..."`), including plain discovery requests,
   satisfied the managerial check and got misrouted.
Fix (`router.py`): an explicit numeric id + QA cue always wins (unambiguous);
a confident managerial-scope match now takes priority over the fuzzy,
no-id product resolution; `"دسته"` was removed from the managerial cue list
since `scope` already captures that signal on its own. Verified on the
held-out response-eval query set: 3/10 routing mismatches → 0/10 (occasional
category-specific token-overlap coincidences may still surface; see
`phase4_metrics.json["generation"]["evaluation_set"]["routing"]` for the
current, honestly-reported pass rate on each run).

### LLM-as-judge: criteria, prompts, and limitations

Two axes are scored by an LLM judge (0–5 integer), selected by `DIGIKALA_JUDGE_MODE`
(default `local`, i.e. the same local Qwen2.5-1.5B model used for local-mode
generation — $0, deterministic-ish, no extra API cost):

- **Relevance** ("response quality") — system prompt:
  `"تو یک ارزیاب هستی. «مفید و مرتبط بودن پاسخ» را بسنج: آیا مستقیم و مرتبط به سؤال جواب داده؟ فقط یک عدد ۰ تا ۵ بنویس."`
- **Faithfulness** ("grounding") — system prompt:
  `"تو یک ارزیاب هستی. «پایبندی به منبع» را بسنج: آیا هر ادعای پاسخ در منابع ارجاع‌شده پشتیبانی می‌شود؟ فقط یک عدد ۰ تا ۵ بنویس."`
- Both share one user template (`prompts.JUDGE_USER`):
  `"سؤال: {query}\n\nپاسخ سیستم:\n{answer}\n\nمنابع:\n{sources}\n\nنمره (۰ تا ۵):"`
  — for faithfulness, `sources` is the actual retrieved evidence (`evidence_products`/
  `evidence_reviews`); for relevance it's empty (relevance only needs the query+answer).
  The judge's reply is regex-parsed for a single `0-5` digit (`evaluate._judge_score`);
  an unparseable reply is recorded as `None`, not silently coerced to a number.

**Known limitations (measured, not hypothetical):** a 1.5B local model is a weak
judge. In the full-corpus run, **`relevance` came back unparseable (`None`) on
all 16/16** human-eval candidate queries — the model never emitted a bare 0–5
digit for that prompt, so `mean_relevance` is `null` in `phase4_metrics.json`
rather than a fabricated number. `faithfulness` parsed on 14/16. A single 0–5
integer per axis is also a coarse rubric with no inter-rater reliability of its
own (only one "rater," the judge itself). This is exactly why a
**human-vs-judge comparison** is run: `python run.py eval` (re)generates
`artifacts/metrics/human_eval_candidates.json` — a fixed 16-query set (6
realistic + 10 harder/adversarial) with the system's answer, its evidence, and
the judge's own scores. A human labeler scores the same queries into
`artifacts/metrics/human_eval_labels.json`
(`{"query": {"relevance": n, "faithfulness": n, "answer_hash": "..."}, ...}`);
`evaluate.human_eval_comparison` reports Spearman ρ as
`phase4_metrics.json["human_eval_agreement"]`, shown on the dashboard's
Evaluation page. Each candidate carries an `answer_hash` (of its exact answer
text); a label only counts if its `answer_hash` still matches the current
candidate — otherwise it's a **stale** label (the code changed since it was
scored, so it's no longer scoring the answer the system currently produces)
and is excluded from agreement, not silently compared against a different
answer than the one actually labeled. This is a real trap the project hit:
after this session's retrieval/evidence-guard rewrite changed the assistant's
answers, blindly recomputing agreement against the old labels produced a
misleading **ρ ≈ -0.05 (not significant)** — not because the judge got worse,
but because the comparison was apples-to-oranges. With staleness detection,
that run correctly reports `n_labeled: 0, n_stale: 16` instead.

**Earlier measured result** (before that rewrite, 16 hand labels against the
pre-rewrite answers): faithfulness agreement was **ρ = 0.524 (p = 0.054,
n = 14)** — a moderate positive correlation, borderline significance;
relevance agreement was empty (`{}`) because the judge never produced a
comparable score for that axis. Re-labeling against the current answers is a
reasonable next step but wasn't repeated this session (large scope already
covered) — the dashboard's Evaluation page shows the live, current status.

## Results (full corpus: 948,352 products / 6,155,711 comments)

| Metric | Value |
|---|---|
| Product retrieval (title benchmark) recall@10 / MRR / nDCG@10 | **1.0 / 0.957 / 0.967** |
| Retrieval ablation: dense / BM25 / hybrid MRR (title benchmark) | 0.822 / **0.983** / 0.957 — see caveat below |
| Retrieval, natural-language benchmark: hybrid / lexical nDCG@10 | 0.279 / **0.343** — lexical wins here too, corroborated |
| Recommendation Macro-F1, **text-only**, **primary = product-grouped** | **0.7291** (naive random split: 0.7200, with 47.76% train/test product overlap — see leakage section) |
| Leakage ablation (text vs text+numeric) | +0.042 (excluded from the final model) |
| Grounding: judge faithfulness / citation coverage / citation validity | **3.71/5 / 0.66 / 1.0** |
| Response quality: judge relevance / deterministic task-completion proxy | 4.0/5 / 4.80/5 (judge relevance parsed here; parsing is query-set-dependent — see judge limitations) |
| Judge-vs-human agreement | 16/16 prior labels now STALE (answers changed this session) — see below |
| Failure analysis | 0/40 retrieval misses; 2/4 adversarial generation probes correctly flagged; 1-2/10 router edge cases (see "Router regression caught") |
| LoRA (ParsBERT) vs. TF-IDF baseline | 0.7049 vs. **0.7291** (baseline wins on this capped comparison) — `phase3_lora_metrics.json` / dashboard Bonus page |
| API cost / budget | **$0.00 / $5.00** |

**Retrieval ablation caveat:** the title-benchmark queries are each product's
own title — a near-exact lexical match that structurally favors BM25, so
BM25-only beating hybrid there is a property of that benchmark. The
**natural-language benchmark** (brand+category+partial-title paraphrases, not
exact titles) is the fairer test — and lexical *still* wins there too, which
is a real, corroborated property of this embedding model on this catalogue
(the reference submission measured the same thing independently), not a
benchmark artifact. Hybrid clearly beats dense-only on both benchmarks.

*(Live numbers: `artifacts/metrics/phase3_metrics.json` and `phase4_metrics.json`;
dashboard's Evaluation + Bonus & Engineering pages render the current run.)*

## Bonus work

Full self-assessment + evidence pointers: `artifacts/metrics/engineering_notes.json`
and the dashboard's **🏆 Bonus & Engineering** page.

- **Router (no-LLM intent routing).** The deterministic 4-way `IntentRouter`
  (discovery/product_qa/comparison/managerial) costs $0 and ~0ms per query instead
  of an extra hosted call — `src/digikala/phase2_assistant/router.py`.
- **Caching & optimization**, with measured before/after numbers: BM25 rebuilt as
  scipy-sparse + disk-persisted (minutes/6GB → 9.2s build / 0.4s load); a fast
  regex tokenizer (`tokenize_norm`) for already-normalized text instead of re-running
  hazm; `match_known_value` fixed from re-normalizing the whole vocab per query
  (assistant load 13.6s→3.1s, discovery 3.4s→0.04s); `managerial_summary` direct
  string compare instead of per-row hazm (79.5s→2.1s); memory-mapped product vectors;
  `GroupedComments` for O(1) per-product review access instead of N sub-frames;
  `st.cache_resource` in the dashboard.
- **Hybrid retrieval, quantified.** `python run.py eval` now also runs
  `retrieval_ablation` — the same title→own-id auto-labeled queries scored under
  `dense`-only, `bm25`-only, and `hybrid` (RRF) retrieval, reporting hybrid's MRR
  lift over the best single method. See `phase4_metrics.json["retrieval_ablation"]`.
- **LoRA fine-tune vs. the TF-IDF baseline.** `python run.py lora` LoRA-tunes
  `HooshvareLab/bert-fa-base-uncased` (ParsBERT, r=8 adapters on query/value,
  <1% of parameters trainable) on the **identical product-grouped split** the
  TF-IDF baseline uses (`recommend.prepare_split`, same seed), and reports the
  Macro-F1 delta in `artifacts/metrics/phase3_lora_metrics.json`. The training set
  is further capped for laptop-GPU feasibility — documented in the module
  docstring as a reasoned resource trade-off, not treated as a full-data result.
- **Interactive dashboard** covers all 4 phases with live "Try it!" panels, plus
  a dedicated Evaluation page (retrieval ablation, failure analysis, judge-vs-human
  agreement) and this Bonus & Engineering page.
- **Presentation storyline** — the dashboard's Bonus & Engineering page opens with
  a problem → decisions → experiments → results → failures narrative tying the
  above together.
- **Not pursued:** a new mentor-approved problem (no mentor channel available in
  this workflow).

### Comparison against a second reference implementation

`reference/digikala_project3_submission/` is a teammate's independently-built,
notebook-only submission (no package/dashboard/tests, ~100k-comment sample vs.
our full 948k/6.16M-row run). It was read cell-by-cell and compared against
this repo; several of its ideas were genuinely more advanced and were ported in:

- **Robust hosted LLM client** (`core/llm.py`): `hosted_auto` provider
  auto-detection from `.env`, retry-on-transient-failure, environment-proxy-vs-
  direct network fallback, and the tracked-vs-estimated-cost distinction — see
  "Run modes, `.env`, and the $5 budget" above.
- **Weighted, candidate-pool-capped RRF** (`retrieval.py`) instead of uniform
  RRF, plus **query-intent-aware reranking**: discovery blends retrieval score
  with price/rating/review-recommend-rate when the query signals a price/
  quality/satisfaction preference; review retrieval reranks by detected query
  *polarity* (a "what's wrong" query surfaces reviews that actually read
  negative, not just the top text-similarity match).
- **Evidence-polarity correctness guards** (`assistant.py`:
  `_procon_flags`/`_accept_positive_evidence`/`_accept_negative_evidence`):
  prevent citing a placeholder ("ندارد") or wrong-polarity review as evidence
  in comparison/QA — a real correctness class, not just a style choice.
  Q&A also filters retrieved evidence by detected query polarity.
- **LLM-output quality guards** (`_gen`): reject truncated, citation-free/hollow,
  or evidence-contradicting ("all reviews...") completions in favor of the
  extractive fallback — comparison and managerial analytics were also switched
  to always-deterministic (no LLM tier) to eliminate unsupported-claim risk
  there entirely, matching the reference's design rationale.
- **Managerial analytics correctness fixes**: review-count-*weighted*
  recommendation rate (not a mean-of-per-product-means), brand satisfaction
  with a minimum-sample requirement, an adaptive low-recommendation threshold
  fallback, and a **scale-aware** negative-review detector (Digikala rate
  columns appear on both 0–5-like and 0–100-like scales depending on source;
  assuming one would be a real bug).
- **Two Phase-4 evaluation upgrades**: the natural-language retrieval benchmark
  and the deterministic quality/grounding proxy, both described above, plus
  Phase-3 `error_pairs`/`failure_examples` for confusion-direction analysis.
- **Faster unbiased sampling** (`core/dataio.py`): the existing `reservoir_sample`
  was already statistically unbiased but used a slow per-row Python loop;
  rewritten as a vectorized one-pass priority-sample (same guarantee, chunk-
  vectorized), and `demo.py`'s sample is now drawn uniformly across the whole
  CSV instead of just the first N rows.
- **Not adopted as-is:** the reference's Metis provider (course-specific
  gateway we don't have a key for) and its `raise` on routing regressions
  (kept as a logged warning instead, consistent with this project's
  never-crash-the-pipeline philosophy elsewhere).

Corroborating, not copied: the reference's own retrieval benchmark
independently found the same lexical-beats-hybrid result — see "Two retrieval
benchmarks, on purpose" above.

## Team

Phase 1 & 2 (this coding style): **Mesbah** · integrated notebook: **Hossein** ·
Phase 3: **Alireza, Mohammadali, Erfan**. Working language: English.

Teammates' original source notebooks are preserved under `reference/` for provenance;
their ideas are integrated into `src/digikala/`.
