# Project Context — QBC13 AI · Group 6 · Project 3 (Digikala Assistant)

Working notes for resuming this session after context compaction. Read this first,
then spot-check against actual files (this is a snapshot, not a live source of truth).

## Who / what / language

- User writes in Persian, wants ALL code/comments/responses in **English**.
- Project: Persian Digikala shopping-assistant AI project, 4 phases per `QBC12 _ AI _ Project 3.pdf`
  (in repo root). $5 total API budget per group.
- Team file provenance (preserved in `reference/`, ideas merged into `src/digikala/`):
  - `reference/digikala_final_section2.ipynb` — Hossein's integrated Phase 1+2 notebook
  - `reference/part3.ipynb` — Alireza/Mohammadali/Erfan's Phase 3 (TF-IDF+LogReg+SMOTE)
  - User (**Mesbah**) did his own Phase 1+2 in a flat-file style — this repo's coding style
    descends from that (now reorganized into a package).

## Repo layout (E:\QBC13_AI_G6_Project3, NOT a git repo)

```
QBC12 _ AI _ Project 3.pdf   README.md   requirements.txt   run.py   context.md (this file)
src/digikala/
  config.py                        paths, RUN_MODE, sample size, budget, JUDGE_MODE
  core/
    persian_text.py                normalize (hazm-based), tokenize, tokenize_norm (fast,
                                    no hazm — for ALREADY-normalized text), extract_price_constraint,
                                    format_toman
    dataio.py                      download_raw, load_products, iter_comment_chunks,
                                    load_comments, reservoir_sample
    llm.py                         LLM class: 3 run modes (local/free/paid) + extractive
                                    fallback; BudgetTracker; judge_llm()
  phase1_data/
    clean.py                       clean_products, clean_comments (dtype-pinned!), build()
                                    (streams full comments in chunks -> one parquet schema)
    eda.py                         Plotly EDA figures, summary_stats, run()
  phase2_assistant/
    retrieval.py                   BM25Okapi (scipy-sparse, FAST build/load), ProductIndex
                                    (embed+BM25+RRF over ALL products), ReviewRetriever
                                    (per-product on-demand review embedding),
                                    GroupedComments (memory-lean per-product review access,
                                    replaces old dict-of-frames), rrf_fuse, product_filter_mask
    router.py                      Catalog (set_index-based, O(1) product lookup,
                                    reviewed_title_tokens only for QA resolution),
                                    IntentRouter (discovery/product_qa/comparison/managerial),
                                    match_known_value (uses tokenize_norm — FIXED, was hazm-heavy),
                                    extract_filters, resolve_scope, resolve_product_id
    prompts.py                     SYSTEM_CORE (zero-hallucination contract), per-intent
                                    system prompts, evidence_products/evidence_reviews,
                                    JUDGE_FAITH_SYS/JUDGE_REL_SYS/JUDGE_USER
    assistant.py                   ShoppingAssistant (route->retrieve->LLM-or-extractive->
                                    verify_citations), review_stats (light= flag for managerial
                                    speed), managerial_summary, LexicalBaseline, build_assistant()
  phase3_predict/
    recommend.py                   TEXT-ONLY model (leakage fix — see below). _prep, _load,
                                    _train (returns bundle,metrics, persists nothing),
                                    train_and_save (full-data, persists), train_from_frame
                                    (in-memory demo path), predict_with(bundle,texts), predict,
                                    fig_confusion
  phase4_eval/
    evaluate.py                    ranking_metrics, evaluate_retrieval, evaluate_generative,
                                    citation_coverage, failure_analysis (NEW), run()
  demo.py                          NEW — shared deterministic demo pipeline: sample_frames(),
                                    build_assistant(), run(). Single source of truth used by
                                    BOTH `run.py demo` and the standalone notebook, so their
                                    outputs are byte-identical (proven).
dashboard/app.py                   Streamlit: light/dark sun-moon toggle, to-top button,
                                    "Try it!" panels per phase, failure-analysis panel,
                                    text-only predictor (leakage-ablation callout shown)
notebooks/digikala_project3_standalone.ipynb   self-contained (41 cells), generator script
                                    lives OUTSIDE repo at:
                                    C:/Users/HP/AppData/Local/Temp/claude/E--QBC13-AI-G6-Project3/
                                    4571b1f4-546b-4879-8053-50a3c1a695c8/scratchpad/build_nb.py
                                    (embeds real src/ code verbatim per module, strips
                                    intra-package imports, adds SimpleNamespace aliases)
reference/                         teammates' original notebooks (preserved, not deleted)
data/{raw,processed,index}         raw CSVs, cleaned parquet, product embedding index
artifacts/{figures,metrics,models} EDA figures, phase3/4 metrics+CSVs, trained model
tests/                             pytest suite (22 tests, all passing) — see below
```

## Canonical schema (unifies phase2 zip-derived code + phase3 teammates' code)

**products_clean.parquet**: product_id, title_fa, title_norm, brand_norm,
category1_norm, category2_norm, sub_category_norm, price_clean, product_rate_clean,
rate_count, is_fake, price_available, product_text_norm, min_price_last_month,
seller_norm

**comments_clean.parquet**: comment_id, product_id, title_norm, body_norm,
advantages_norm, disadvantages_norm, comment_text_norm, has_text, rate_clean,
likes, dislikes, is_buyer, true_to_size_rate, created_at, recommendation_status,
recommendation_valid, has_product_match

Dtypes are PINNED in `clean_comments` (Int64/string/boolean/float64) so streamed
chunks all produce the identical Arrow schema when written via ParquetWriter.

## Architecture decisions (user-approved, don't relitigate)

- **Package** (src/digikala/) for the app + **self-contained standalone notebook**
  (all code embedded, runs top-to-bottom) — both required deliverables.
- **Full-data** pipeline by default (streamed chunked cleaning), notebook uses a
  smaller deterministic sample (SAMPLE_SIZE, default 20000) for fast "Run All".
- **Products embedded fully** (~948k, cached vectors.npy, mmap'd), **reviews
  retrieved per-product on demand** (can't globally embed 6M reviews on 6GB GPU).
- **Phase 3: TEXT-ONLY** classical TF-IDF + linear model (leakage fix, see below).
- **3 LLM run modes**: `local` (Qwen/transformers or Ollama), `free` (Groq/OpenRouter),
  `paid` ($5 credit, OpenAI-compatible) + `extractive` ($0 always-available fallback).
  Judge model separately configurable via `DIGIKALA_JUDGE_MODE` (default `local`).
- Zero-hallucination contract: every claim cites `[محصول id]`/`[بازبینی id]`;
  `verify_citations` strips any id the retriever didn't actually return.

## Key engineering fixes (performance + correctness), chronological

1. **BM25 rewrite**: pure-Python inverted index (minutes, 6GB RAM to build/load) →
   scipy-sparse CountVectorizer-based (`BM25Okapi` in retrieval.py). Build 9s, load
   0.4s. Persisted to disk (bm25_counts.npz + idf/doc_len/vocab.json).
2. **tokenize_norm**: regex-only tokenizer (no hazm) for text that's ALREADY
   normalized (the `*_norm` columns). Using the heavy hazm-based `tokenize` on
   pre-normalized corpus text was a huge unnecessary cost.
3. **Catalog refactor** (router.py): dict-of-948k-rows → pandas set_index +
   `.loc` lookups. `reviewed_title_tokens` (for fuzzy product-name resolution in
   QA) built ONLY over reviewed products (35k, not 948k), using tokenize_norm.
4. **managerial_summary**: category mask was calling `pt.normalize` (hazm) per row
   over 948k products → direct string comparison against the already-normalized
   column. 79s → 2s.
5. **GroupedComments** (retrieval.py): replaced a dict-of-N-subframes (OOM risk on
   6M comments) with one index-sorted DataFrame + `.get()` that does `.loc[[pid]]`
   then `.reset_index(drop=True)` (the reset was needed — first version crashed
   with "cannot reindex on an axis with duplicate labels").
6. **match_known_value** (router.py): was running hazm `pt.normalize` over the
   ENTIRE vocab on EVERY query. Fixed to use pre-normalized values + tokenize_norm.
   This was the single biggest latency fix: assistant load 13.6s→3.1s, discovery
   3.4s→0.04s (post-fix, on full data).
7. **clean_comments column access**: `df.get(name)` returns `None` (not an empty
   Series) when the column is absent, and `_to_num(None)` returns a bare
   `numpy.float64`, which then crashes on `.fillna()`. A pytest test caught this.
   Fixed with a `col(name)` helper that always returns a Series aligned to `df.index`.

## Leakage fix (Phase 3) — IMPORTANT, don't reintroduce

`recommend.py` originally used `rate_clean` (star rating), `likes`, `is_buyer` as
numeric features alongside TF-IDF text. This is data leakage:
- `rate_clean` is a second expression of the SAME sentiment as the label
  (recorded in the same review-submission act) — near-duplicate of the target.
- `likes` accrue AFTER the review is posted — not available at prediction time
  (temporal leakage).
- Brief explicitly asks to predict from the review's TEXTUAL content.

**Fix**: final model is TEXT-ONLY (`comment_text_norm` via TF-IDF only).
`NUMERIC_FEATURES = ["rate_clean","likes","is_buyer_num"]` still exists but is
ONLY used in a labeled **ablation** (`_train` fits a second text+numeric pipeline,
reports `leakage_ablation: {text_only_macro_f1, text_plus_numeric_macro_f1,
leakage_lift}`) to quantify and document the leak, not to use it.

`predict(texts)` and `predict_with(bundle, texts)` take ONLY text — no rate/likes/
is_buyer params anymore (was previously `predict(texts, rate=, likes=, is_buyer=)`).

Dashboard predictor page updated to match (no sliders for rate/likes/buyer, shows
the leakage-ablation number as an info callout).

**Full-data results (verified, current)**:
- Baselines (val): majority 0.1667, logreg(text-only) 0.7202
- Final TEXT-ONLY: test Macro-F1 **0.7200**, grouped **0.7291**
- Ablation: text+numeric would be 0.7624 → **+0.0424 is leakage** (documented, excluded)
- (Earlier, BEFORE the leakage fix, numbers were 0.762/0.773 — those were LEAKY,
  now superseded. Don't cite 0.762 as the model's real performance.)

## Failure analysis (new, phase4_eval/evaluate.py)

`failure_analysis(assistant, judge=None, n_retrieval=40, k=10)`:
- Retrieval misses: for the top-N most-reviewed products, search the product's own
  title and check if its own id comes back in top-k. Full data: **1/40 missed**
  (a generic-title beard-oil product).
- Generation failure probes (deliberately adversarial): out-of-catalog nonsense
  query, non-existent product id, under-specified comparison, a normal QA query.
  Flags: missing_info, zero_citations, needs_clarification, low judge faithfulness
  (≤2/5). Full data run: nonsense query got faithfulness 0/5, non-existent id got
  2/5 — both correctly flagged (2/4 failed in that run; later notebook run with a
  smaller/different sample showed 0/4 failed — probe outcomes are sample-dependent,
  that's expected/fine).
- Wired into `evaluate.run()` → `phase4_metrics.json["failure_analysis"]`.
- Displayed in dashboard's Evaluation page (`page_eval` in dashboard/app.py) and in
  the standalone notebook (cell after the baseline-control demo).

## No notebook-vs-application discrepancy (proven, don't relitigate)

`src/digikala/demo.py` is the SHARED deterministic pipeline:
- `sample_frames(sample_size=20000, seed=42)`: head-reads N comments, finds the
  products they reference, cleans both with the real Phase-1 functions.
- `build_assistant(products, comments, llm=None)`: builds the SAME classes
  (ProductIndex, GroupedComments, Catalog, ShoppingAssistant) the packaged app uses.
- `run(sample_size, seed, judge=None)`: full demo, returns a dict with
  n_products/n_comments/retrieval_quality/prediction_test_macro_f1/
  prediction_grouped_macro_f1/prediction_leakage_ablation/demos (3 sample answers
  with sorted citation lists).

The notebook's Phase-2/3 "build + demo" cells and the "no discrepancy" summary cell
(section 4.1) call these EXACT same functions (embedded verbatim by the generator).

**Verified**: `python run.py demo --sample-size 20000` output vs. the notebook's
printed summary — compared field by field (n_products, n_comments,
retrieval_quality, both macro-F1 values, demos dict) — **all identical**. This is
because both paths use nrows=20000 head-sampling (not random), fixed RANDOM_SEED=42
throughout, and identical code paths.

`run.py` commands: `setup clean eda index train eval dashboard all menu demo test`.
`demo` takes `--sample-size` (default 20000, must match notebook's SAMPLE_SIZE to
compare). `test` runs `pytest tests/`.

## Test suite (tests/, 22 tests, all passing)

- `conftest.py`: fixtures `products`/`comments`/`catalog` — tiny synthetic data,
  product ids are 6-digit (100101 etc.) because the router's product-id regex
  requires `\b(\d{6,9})\b` — DON'T use 3-digit ids in test fixtures, router won't match.
- `test_persian_text.py`: normalize, price constraint parsing (both directions,
  bare-number-is-not-a-constraint), format_toman, tokenize_norm.
- `test_router.py`: discovery w/ price filter, comparison (2 ids), product_qa
  (id + QA cue word), managerial (scope resolution), extract_filters (brand match).
- `test_retrieval_and_citations.py`: BM25 ranks correctly, RRF fusion,
  product_filter_mask, verify_citations strips unknown ids, ranking_metrics.
- `test_no_leakage.py`: asserts `_train` source contains `numeric=False` for the
  final/baseline/grouped paths and `"features": ["comment_text_norm"]`; asserts
  `predict` signature is `["texts"]` only (no rate/likes/is_buyer params); asserts
  `_prep` dedups by comment_text_norm; asserts NUMERIC_FEATURES are documented as
  excluded-to-avoid-leakage.
- `test_phase1_clean.py`: clean_products schema + duplicate-id detection,
  clean_comments label validation + product-match flag + pinned dtypes.

Run via `python run.py test` or `pytest tests/ -q` (needs `sys.path.insert(0,'src')`,
handled by conftest.py).

## Dashboard (dashboard/app.py)

Streamlit app, pages: Overview (Phase1 EDA), 🔎Discovery, 💬Review Q&A, ⚖️Compare,
📊Manager analytics, 🤖Recommendation predictor (Phase3, text-only now), 🧪Evaluation
(Phase4, now includes failure-analysis panel). Sidebar: sun/moon theme toggle
(session_state.dark), LLM run-mode selectbox, section radio. Floating to-top link
(`#top` anchor). Cached via `st.cache_resource` (_load_tables, _load_assistant,
_load_p3). Recently sed-replaced all `use_container_width=True` → `width="stretch"`
(Streamlit deprecation, harmless mechanical change — verify no width= collisions
if editing that file further).

Verified via headless `streamlit.testing.v1.AppTest` (script was in scratchpad,
NOT committed to repo — recreate if needed): loads dashboard/app.py, sets the
section radio, asserts `not at.exception`. Both predictor and eval pages passed.
Also boot-tested live via `streamlit run --server.headless true` + browser pane
(showed Products 948,352 / Comments 6,155,711 correctly on Overview).

## Environment / run mode reference

```bash
DIGIKALA_RUN_MODE=extractive|local|free|paid     # LLM backend for answers
DIGIKALA_JUDGE_MODE=none|local|free|paid          # LLM backend for Phase-4 judge
DIGIKALA_SAMPLE=full|<int>                        # comments to load (config.COMMENTS_SAMPLE_SIZE)
DIGIKALA_FREE_PROVIDER=groq|openrouter
GROQ_API_KEY / OPENROUTER_API_KEY / PAID_API_KEY  # hosted API keys, never committed
```
Python 3.11 (.venv at E:\QBC13_AI_G6_Project3\.venv — hazm needs <=3.11).
GPU: RTX 3060, used for embeddings (MiniLM) + local Qwen2.5-1.5B-Instruct.
Budget: $5 total, tracked via `core.llm.BudgetTracker`, logs to
`artifacts/metrics/budget_log.jsonl`, currently **$0.00 spent** (only extractive/
local/free-untested modes used so far — paid mode never invoked).

## What's done vs. what's not

**Done & verified**: Phase 1 (full 6.16M-comment streaming clean), Phase 2 (4
grounded intents, citation verification, hybrid retrieval, fast BM25), Phase 3
(text-only, leakage-free, ablation reported), Phase 4 (6-axis eval incl. failure
analysis), dashboard (renders, tested headless), standalone notebook (41 cells,
0 errors, byte-identical to `run.py demo`), test suite (22 passing), repo cleaned
(superseded root .py files deleted, teammates' notebooks moved to reference/,
.gitignore updated for new artifact dirs).

**NOT yet done / could still do**:
- A `run.py all` full end2end AFTER the leakage fix hasn't been re-run (only
  `train` and `eval` were individually re-run post-fix on full data — clean/eda/
  index don't depend on Phase 3 so they're still valid from the earlier full run).
- Paid-mode ($5 credit) showcase run — budget is untouched, one env-var away
  (`DIGIKALA_RUN_MODE=paid`, needs `PAID_API_KEY`/`PAID_BASE_URL` set).
- Free-tier (Groq/OpenRouter) judge run — currently only local Qwen judge tested.
- No git repo exists yet — if the user wants version control, `git init` + first
  commit is still pending. Be careful: "no undo" currently for destructive ops.
- `context.md` (this file) and any scratchpad build scripts are NOT part of the
  formal deliverable — the scratchpad `build_nb.py` generator lives OUTSIDE the
  repo (in the Claude temp scratchpad dir) and would need to be recreated or
  relocated into the repo if the user wants notebook regeneration to be
  self-service without Claude.

## Memory files (persistent across sessions)

`C:\Users\HP\.claude\projects\E--QBC13-AI-G6-Project3\memory\full-project-build.md`
has a similar (slightly less detailed) summary; `MEMORY.md` in that same dir
indexes it. This context.md is the fuller, in-repo, session-resumable version.

## Session 3 (2026-08-28): PDF-vs-repo gap analysis, leakage re-audit, full bonus pass

Extracted the full PDF text (pypdf), built a to-be checklist against the actual
spec text, and compared it to the repo. Required sections were all substantively
done; real gaps found: no human-vs-judge comparison was ever run, no full
`run.py all` since the leakage fix, and the user flagged possible remaining
leakage. User chose: pursue ALL bonus items (including LoRA), self-label the
human-eval set, and re-run the full pipeline immediately.

**Leakage re-audit (second, deeper pass) — real finding.** The naive random
row-level split can still put two *different* reviews of the *same* product in
both train and test (exact-text dedup doesn't catch this). Measured on full data:
**47.76% product overlap** in the naive split. Fix: `recommend.py` now reports
`primary_macro_f1`/`primary_split="product_grouped"` as the headline number, with
`naive_split_product_overlap_pct` shown alongside (not hidden). Interesting
result: despite 47.76% overlap, the grouped Macro-F1 (0.7291) came out *higher*
than the naive split (0.7200) — no evidence the overlap actually inflated the
naive number this run. Also confirmed (not a bug): the TF-IDF vectorizer was
already leak-safe, fit only inside `sklearn.Pipeline.fit(X_train, y_train)`.
New `recommend.prepare_split()` factors out the label-encode+stratified-cap+
grouped-split logic (deterministic, same seed) so the LoRA script can train/eval
on the *identical* split as the baseline. 4 new tests in `test_no_leakage.py`
(26 total, all pass).

**Bonus work added, each with evidence persisted as an artifact (not just
claimed):**
- **Retrieval ablation** — `ProductIndex.search(..., method="dense"|"bm25"|"hybrid")`
  + `evaluate.retrieval_ablation()`. Honest measured result on full data: dense
  MRR 0.822, BM25 MRR 0.983, hybrid MRR 0.937 — **hybrid actually underperforms
  BM25-only here** (lift -0.046). Root cause understood and documented: the
  auto-labeled benchmark uses each product's own *title* as the query, a
  near-exact lexical match that structurally favors BM25 — not a fair test of
  hybrid's value on fuzzy natural-language queries. Reported as a genuine,
  reasoned finding, not spun into a false "win." Hybrid still clearly beats
  dense-only. Dashboard shows a `st.warning` with this caveat when the lift is
  negative.
- **Human-eval-vs-judge** — `evaluate.build_human_eval_candidates` /
  `human_eval_query_set` / `human_eval_comparison` (+ `judge_human_agreement`,
  pre-existing but previously unused). `run()` always (re)writes
  `artifacts/metrics/human_eval_candidates.json` (16 fixed queries: 6 realistic +
  10 adversarial, with the assistant's real answer + judge scores). User
  hand-labeled all 16 in-chat (relevance+faithfulness 0-5); labels saved to
  `artifacts/metrics/human_eval_labels.json`. **Measured result**: judge
  `relevance` was unparseable (`None`) on **16/16** — a real, total failure of
  that axis for the 1.5B local judge, stronger than the earlier "some prompts"
  wording (README updated accordingly). `faithfulness` agreement: **Spearman
  ρ=0.524, p=0.054, n=14** — moderate, borderline-significant. Computed via a
  standalone script (not a full `run.py eval` re-run) and patched into the
  existing `phase4_metrics.json["human_eval_agreement"]`.
- **LoRA fine-tune bonus** — new `src/digikala/phase3_predict/lora_finetune.py`:
  LoRA (r=8, alpha=16, query+value adapters, peft library) on
  `HooshvareLab/bert-fa-base-uncased` (ParsBERT), trained/evaluated on the
  IDENTICAL product-grouped split as the TF-IDF baseline via
  `recommend.prepare_split()`. Capped to 6000 train / 1500 test rows for
  laptop-GPU feasibility — documented in the module docstring as a reasoned
  resource trade-off (general project rule explicitly allows justified
  subsetting), not a full-data comparison. `python run.py lora` writes
  `artifacts/metrics/phase3_lora_metrics.json`. Added `peft>=0.11` to
  requirements.txt. [Was running in the background as of the last message in
  this session — check `artifacts/metrics/phase3_lora_metrics.json` for whether
  it completed; if missing/stale, re-run `python run.py lora`.]
- **New dashboard page "🏆 Bonus & Engineering"** (`dashboard/app.py:page_bonus`):
  storyline (problem→decisions→experiments→results→failures), a bonus scorecard
  table (self-assessed, evidence-linked, explicitly NOT a claimed grade), and
  router/caching/LoRA writeups with the real numbers. Reads NEW artifact
  `artifacts/metrics/engineering_notes.json` (router bonus, caching bonus with
  the before/after perf numbers that used to live only in this file, retrieval
  ablation bonus incl. the honest caveat, human-eval bonus incl. the honest
  caveat, and the scorecard).
- **README**: new "LLM-as-judge: criteria, prompts, and limitations" section
  (exact `JUDGE_FAITH_SYS`/`JUDGE_REL_SYS`/`JUDGE_USER` prompt text quoted
  verbatim), new "Bonus work" section, Phase-3 sampling-justification paragraph,
  expanded no-leakage section, refreshed Results table with the real post-fix
  numbers (all of the above).

**Full `run.py all` re-run** (`DIGIKALA_RUN_MODE=extractive DIGIKALA_JUDGE_MODE=local`)
completed successfully after all the code changes above (products 948,352 /
comments 6,155,711; Phase3 primary/grouped 0.7291, naive 0.7200, overlap 47.76%;
Phase4 recall@10=1.0/MRR=0.937/nDCG=0.952; retrieval ablation as above; failures
1/40 retrieval + 2/4 generation probes; cost $0/$5). Verified via headless
`AppTest` that all 8 dashboard pages (including the two new sections) render
with zero exceptions on the real post-run artifacts.

**Standalone notebook regenerated** from the (still scratchpad-external)
`build_nb.py` generator, updated to: print the primary/grouped/naive-overlap
framing in the Phase-3 cell, call `retrieval_ablation` in the Phase-4 cell (with
the same honest caveat noted inline), add a "4.0.1 Human-vs-judge evaluation"
markdown cell pointing at the full-data artifact (human-eval isn't re-run on the
notebook's small sample), and updated the 4.1 no-discrepancy summary dict to
include `prediction_primary_macro_f1`/`prediction_naive_split_product_overlap_pct`
— matching new fields added to `src/digikala/demo.py`'s `run()` return dict for
the same reason. 42 cells now (was 41). **Not yet re-executed headlessly this
session** as of the last message — do that (`jupyter nbconvert --execute
--inplace`, absolute paths, run from repo root) before calling the notebook
verified again, and only after the LoRA background job (GPU-heavy) has finished
to avoid VRAM contention on the 6GB GPU.

**Still open at end of session:**
1. Confirm `phase3_lora_metrics.json` was written (LoRA job was running in the
   background when this note was written) and surface the number to the user.
2. Re-execute the standalone notebook headlessly and confirm 0 errors on the
   regenerated 42-cell version.
3. Re-verify the dashboard's Bonus & Engineering + Evaluation pages render the
   LoRA numbers once `phase3_lora_metrics.json` exists.
4. Git init still not done (unchanged from session 2) — ask before doing it.
5. Paid-mode / free-tier judge runs still untested (unchanged from session 2).

## Session 3, part 2: full reference-notebook comparison + port (same session, continued)

All items 1-5 above are now DONE. Full sequence: LoRA re-ran successfully after
fixing a torch/transformers safetensors incompatibility (pinned
`revision="refs/pr/2"` on the HF model repo — see `lora_finetune.py`); compared
`reference/digikala_project3_submission/digikala_project3_final.ipynb`
(teammate's independent, more mature, notebook-only submission) cell-by-cell
against this repo; user approved porting EVERYTHING found better + Groq/paid
`hosted_auto` (skip Metis). See the persistent memory file
(`C:\Users\HP\.claude\projects\E--QBC13-AI-G6-Project3\memory\full-project-build.md`)
for the full, detailed list of what was ported — it's long (hosted LLM client
rewrite, weighted/reranked retrieval, evidence-polarity guards, LLM-output
quality guards, managerial correctness fixes incl. a real rec_rate divide-by-
wrong-denominator bug, two Phase-4 eval upgrades, faster unbiased sampling,
Phase-3 failure examples) — read that file, don't re-derive from git history.

**Two real bugs found and fixed via the new eval-embedded routing-regression
check** (not found any other way): (1) `"چند"` as a QA cue could fuzzy-match
an unrelated product on a plain category query; (2) bare `"دسته"` as a
managerial cue meant ANY query naming a category (including discovery ones)
routed to managerial. Both fixed in `router.py`; verified 3/10 → 1/10 routing
mismatches on held-out queries (residual: a 5-digit product id not matched by
the `\d{6,9}` regex — documented, not chased further, out of scope).

**A third real methodological bug caught after the full rewrite**: re-running
`run.py eval` recomputed `human_eval_agreement` against the OLD (pre-rewrite)
human labels but the NEW (post-rewrite, different) assistant answers — produced
a misleading `rho=-0.05, not significant`, which would have looked like "the
judge doesn't track humans" when actually it was scoring two different things.
Fixed properly: `build_human_eval_candidates` now stamps each candidate with
`answer_hash` (sha1 of the answer text); `human_eval_comparison` only counts a
label if its `answer_hash` matches, else buckets it as `stale` (`n_stale`)
instead of silently mixing it in. Current state: `n_labeled=0, n_stale=16` —
honest, pending re-labeling (not done this session, would need the user again).
Old ρ=0.524 result documented as "earlier, pre-rewrite" in the README, not
presented as current.

**Full-data `run.py train` + `run.py eval` re-run** (clean/index untouched,
not part of what changed) — confirmed via fresh numbers: Phase3 primary/naive
0.7291/0.7200 (unchanged, Phase 3 code didn't functionally change besides
adding error_pairs); Phase4 title-benchmark hybrid MRR 0.9567 (bm25 wins
0.9833, still); NEW natural-language benchmark: hybrid nDCG 0.279 vs lexical
0.343 — **lexical wins here too**, independently corroborating the
reference's own finding on a completely different query construction method;
failure_analysis retrieval misses dropped to 0/40 (was 1/40); generation
judge relevance actually parsed this time (4.0/5) on the `build_response_eval_queries`
set — the earlier "16/16 unparseable" claim was specific to the human-eval
16-query set, not universal (README updated to note this nuance).

**New file**: `docs/COOKBOOK.md` — ~450-line "zero to hero" theory document
(normalization → tokenization → cleaning → embeddings/cosine → TF-IDF/BM25
with real formulas → RRF → RAG/grounding/citations → rule-based routing → all
3 leakage types found+fixed with real numbers → Macro-F1 → LoRA math →
multi-axis eval philosophy → LLM-judge pitfalls → cost management → a full
"one query start to finish" trace → glossary). Written for zero prior
background, every example drawn from this repo's real code.

**Also done**: `.env` (real, blank, gitignored) + `.env.sample` at repo root;
`.gitignore` +`.pytest_cache/`; dashboard's Evaluation page now shows the
natural-language benchmark + a staleness warning banner for human-eval;
README extensively updated (`.env`/hosted_auto run-mode section, "Two
retrieval benchmarks, on purpose", "Router regression caught", human-eval
staleness explanation, refreshed Results table, new "Comparison against a
second reference implementation" bonus subsection).

**Verified end-to-end after ALL changes**: 26/26 pytest passing; standalone
notebook regenerated + executed headlessly TWICE, 0 errors both times (42/42
cells) — 2nd time was needed because the dataio.py unbiased-sampling fix
created a fresh notebook-vs-demo discrepancy that had to be caught and fixed
in the SAME session (notebook's Phase-1 cell was still using the old nrows
head-read); `run.py demo --sample-size 20000` output now byte-matches the
notebook's summary exactly (n_products=17043, primary_macro_f1=0.4924,
naive_overlap=14.34% — both, confirmed identical); dashboard verified via
headless AppTest TWICE — first all 8 pages load with 0 exceptions against
fresh artifacts, second all 5 "Try it!" buttons (Search/Ask/Compare/Analyze/
Predict) actually CLICKED and exercised the real rewritten assistant code
paths against the full 948k-product index, all OK.

**Genuinely still open, nothing pretended done:**
1. Human-eval re-labeling against current answers (optional, user's call).
2. Git init still not done.
3. **Real API-key end2end testing (local GPU + both hosted modes) — the
   user explicitly deferred this to their NEXT message. Do not preempt it or
   assume it's been done.**
4. `engineering_notes.json`'s prose scorecard wasn't mechanically re-synced
   with every new number in this part-2 pass (README is the current source of
   truth for exact numbers; engineering_notes.json's qualitative claims still
   hold).
