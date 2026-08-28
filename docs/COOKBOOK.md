# The Digikala Assistant Cookbook — Zero to Hero

A from-scratch explanation of every idea this project uses, with a worked example
from the actual codebase for each one. You don't need any prior ML/NLP background
to follow this; each section builds on the last. Code references point at real
files in `src/digikala/` so you can go read the working version after understanding
the idea here.

If you only read one thing before touching the code, read **§8 (RAG & grounding)**
and **§10 (leakage)** — they're the two ideas most likely to bite you if skipped.

---

## Table of contents

1. [What problem is this solving?](#1-what-problem-is-this-solving)
2. [Persian text normalization](#2-persian-text-normalization)
3. [Tokenization](#3-tokenization)
4. [Data cleaning fundamentals](#4-data-cleaning-fundamentals)
5. [Embeddings & dense retrieval](#5-embeddings--dense-retrieval)
6. [TF-IDF & BM25 (sparse/lexical retrieval)](#6-tf-idf--bm25-sparselexical-retrieval)
7. [Hybrid retrieval & Reciprocal Rank Fusion](#7-hybrid-retrieval--reciprocal-rank-fusion)
8. [RAG, grounding, and citation verification](#8-rag-grounding-and-citation-verification)
9. [Intent routing without an LLM](#9-intent-routing-without-an-llm)
10. [Data leakage — what it is and how it hides](#10-data-leakage--what-it-is-and-how-it-hides)
11. [Text classification & the Macro-F1 metric](#11-text-classification--the-macro-f1-metric)
12. [Parameter-efficient fine-tuning (LoRA)](#12-parameter-efficient-fine-tuning-lora)
13. [Evaluating a system that has no single right answer](#13-evaluating-a-system-that-has-no-single-right-answer)
14. [LLM-as-judge — power and pitfalls](#14-llm-as-judge--power-and-pitfalls)
15. [Cost & resource management](#15-cost--resource-management)
16. [Putting it all together — one query, start to finish](#16-putting-it-all-together--one-query-start-to-finish)
17. [Glossary](#17-glossary)

---

## 1. What problem is this solving?

Digikala (an Iranian e-commerce site) has ~1.28M products and ~6M user reviews.
A shopper can't read 300 reviews before buying a phone case; a category manager
can't manually skim ten thousand reviews to find "what are people complaining
about this month." The whole point of this project is: **turn a pile of raw
text into answers a person can act on, without making anything up.**

That constraint — *without making anything up* — is the single hardest part,
and almost every design choice in this codebase exists to serve it. A fluent,
convincing-sounding paragraph that's wrong is worse than no answer at all,
because the reader has no way to tell it's wrong just by reading it. Sections
8 and 14 are about exactly this problem.

The system has four capabilities (discovery, product Q&A, comparison, manager
analytics — see `src/digikala/phase2_assistant/assistant.py`), a classifier
that predicts whether a review is a "recommend" (`phase3_predict/recommend.py`),
and an evaluation suite that tries to catch the system lying to you
(`phase4_eval/evaluate.py`).

---

## 2. Persian text normalization

**The problem.** The same word can be written many different ways: Arabic vs.
Persian keyboard characters that look identical (`ي` vs `ی`), half-spaces vs.
full spaces vs. no space (`می‌خواهم` vs `می خواهم` vs `میخواهم`), Eastern
Arabic-Indic digits vs. Western digits (`۱۲۳` vs `123`), and inconsistent use
of diacritics. If you don't normalize, your code sees `کیفیت` and `كيفيت` as
two completely different words, when a human reads them as identical.

**The idea.** Pick one canonical form for every character/digit/spacing
variant, and rewrite everything into it before any comparison, search, or
model ever sees the text. This has to happen *once*, consistently, everywhere
— if cleaning normalizes one way and search normalizes another way, nothing
will ever match.

**In this codebase:** `src/digikala/core/persian_text.py::normalize()`.
It uses the `hazm` library's `Normalizer` (Arabic→Persian character mapping,
half-space fixing, digit folding) with a regex fallback if `hazm` isn't
installed, so the pipeline degrades gracefully instead of crashing.

```python
from digikala.core import persian_text as pt
pt.normalize("كيفيت خوبي داره، ۱۲۳ تومن")
# -> "کیفیت خوبی داره، 123 تومن"   (Arabic chars -> Persian, digits folded)
```

Every `_norm` column in `products_clean.parquet`/`comments_clean.parquet`
(`title_norm`, `body_norm`, `comment_text_norm`, ...) is the output of this
function — see the schema in [README.md](../README.md).

---

## 3. Tokenization

**The problem.** Once text is normalized, you still need to break it into
units (tokens) to count word overlap, build a vocabulary for TF-IDF, or feed
a model. Persian has clitics, compound words, and half-space-joined
morphemes, so naive `text.split()` on whitespace under- or over-splits.

**Two tokenizers, on purpose:**
- `pt.tokenize(text)` — hazm's proper tokenizer, linguistically aware, but
  slower because it re-runs normalization internally.
- `pt.tokenize_norm(text)` — a fast regex split (`\w+`-style) used **only**
  on text that is *already* normalized (the `_norm` columns). Since the
  heavy normalization work was already done once during cleaning, re-running
  hazm's tokenizer on every single query/document at search time would be
  pure waste. This one performance decision was worth **~100x** at query
  time (see the "match_known_value" fix in the README's engineering notes:
  13.6s → 3.1s assistant load, 3.4s → 0.04s per discovery query).

**Lesson generalized:** know the cost of your preprocessing step, and don't
pay it twice. If a transformation is idempotent and you've already applied
it once (at cleaning time), a downstream consumer that trusts that
invariant can skip straight to a cheaper operation.

---

## 4. Data cleaning fundamentals

Raw data is never ready to use. The brief explicitly warns: no guarantee of
complete, unique, balanced, outlier-free, or model-ready values. Concretely,
in this dataset:

- **Missing values**: not every review has a `rate`; not every product has a
  `brand`. Decision: keep the row, mark the field `NaN`/`None`, and let
  downstream code branch on presence rather than silently coercing to 0
  (coercing a missing rating to 0 would make an unrated product look like
  the worst-rated one — a real bug class).
- **Duplicates**: the same comment id can appear twice in a raw export.
  `_dedup()` in `phase1_data/clean.py` drops exact duplicate ids.
- **Invalid categorical values**: `recommendation_status` should only ever be
  one of `recommended`/`not_recommended`/`no_idea`; anything else (typos,
  other languages, empty) is marked invalid rather than guessed at
  (`recommendation_valid` flag) — see `config.RECOMMENDATION_CLASSES`.
- **Type coercion**: prices, ratings, and dates arrive as loosely-typed CSV
  strings. `_to_num()`/`_to_bool()`/`_parse_datetime()` convert them to
  proper numeric/boolean/datetime dtypes, using `errors="coerce"` so a
  malformed value becomes `NaN` instead of crashing the whole pipeline.
- **Schema stability across chunks**: the comments file is streamed in
  200k-row chunks (`config.CHUNK_SIZE`) so 6M rows never sit in memory at
  once. If chunk A happens to have an all-null column and chunk B doesn't,
  pandas can infer *different dtypes per chunk*, which breaks writing them
  all into one Parquet file. Fix: pin every output column's dtype explicitly
  at the end of `clean_comments()`, so every chunk produces an identical
  Arrow schema regardless of what values happened to be in it.

**Lesson generalized:** cleaning decisions are not neutral — they encode
assumptions about what "wrong" data means. Document every one (this project's
README has a "canonical schema" section for exactly this reason), because a
grader (or a future you) needs to be able to tell *what* was decided and
*why*, not just read the code and reverse-engineer it.

---

## 5. Embeddings & dense retrieval

**The idea.** An embedding model converts text into a fixed-length vector of
numbers (here, 384 dimensions) such that texts with *similar meaning* end up
as *nearby vectors* — even if they don't share any of the same words. This is
what lets a search for "ارزون و باکیفیت" ("cheap and good quality") find a
product titled "مقرون به صرفه" ("affordable") even though the words are
completely different.

**How similarity is measured.** Once text is a vector, "similar meaning" =
"small angle between vectors" = **cosine similarity**. If the vectors are
already normalized to length 1 (unit vectors), cosine similarity is just a
dot product — which is why `embed()` in `retrieval.py` passes
`normalize_embeddings=True` to the encoder, and product search is literally
`self.vectors @ qv` (a matrix-vector product = 948k dot products at once).

```python
# src/digikala/phase2_assistant/retrieval.py
def embed(texts, batch_size=256):
    return get_model().encode(texts, batch_size=batch_size,
                              convert_to_numpy=True, normalize_embeddings=True)
```

Model used: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — a
small (≈120M parameter), multilingual sentence encoder, chosen because it
fits comfortably on a 6GB laptop GPU alongside everything else, and supports
Persian without a Persian-specific fine-tune.

**Why not embed all 6M reviews too?** Because a per-product review search
only ever needs *that one product's* reviews (at most a few hundred), while
product discovery genuinely needs to search the *whole* catalogue. So: all
~948k products are embedded once and cached to disk (`vectors.npy`,
memory-mapped so it doesn't all sit in RAM); a product's reviews are embedded
**on demand**, only when someone asks about that specific product. This is a
resource-management decision, not a modeling one — see §15.

**The honest limitation, measured not assumed:** dense embeddings are not
automatically better than lexical search. On this catalogue, with this
model, our own retrieval ablation shows BM25-only *beating* the dense-only
and even the hybrid retriever on nDCG (see `phase4_metrics.json`, and the
"Two retrieval benchmarks" section of the README) — reported honestly
rather than assumed away. This is exactly why §6/§7 matter: you need both
tools, and you need to *measure* which one is actually pulling its weight.

---

## 6. TF-IDF & BM25 (sparse/lexical retrieval)

**TF-IDF (Term Frequency–Inverse Document Frequency)** scores how important a
word is to a specific document *relative to the whole corpus*: a word that
appears often in this document (high TF) but rarely across all documents
(high IDF) is a strong signal for that document. "کیفیت" (quality) appears
everywhere, so it gets down-weighted; "نامسبک" (a specific brand mistyping)
appears in three reviews, so it's very distinctive when it does. This
codebase uses TF-IDF for **Phase 3's text classifier** (`recommend.py`,
`TfidfVectorizer`) — turning each review's raw text into a sparse numeric
vector a linear model can learn from.

**BM25 (Best Matching 25)** is TF-IDF's more careful cousin, purpose-built
for *ranking documents by relevance to a query* rather than as generic
features for a classifier. Its formula (`retrieval.py::BM25Okapi.get_scores`)
adds two refinements TF-IDF doesn't have:
- **Term-frequency saturation**: the 10th occurrence of a word in a document
  shouldn't count 10x as much as the 1st — BM25's `(k1+1)*freq / (freq+...)`
  term flattens out as `freq` grows.
- **Document-length normalization**: a long document naturally contains more
  word repeats just by being long, not because it's more relevant — the `b`
  parameter corrects for that using `doc_len / avgdl`.

```python
score(term, doc) = idf(term) * freq(term, doc) * (k1 + 1) /
                    (freq(term, doc) + k1 * (1 - b + b * doc_len / avgdl))
```

**Why scipy-sparse, not a hand-rolled inverted index?** The textbook way to
implement BM25 is a Python dict mapping `term -> [(doc_id, freq), ...]`. Over
948k product titles that's slow to build (minutes) and memory-heavy (6GB+).
This codebase instead builds one `CountVectorizer` term-document matrix
(C-optimized, one pass) and does all the arithmetic as sparse-matrix column
slicing — 9.2s to build, 0.4s to load from a disk cache. Same algorithm,
100x+ faster implementation — a lesson that *how* you implement a classic
algorithm matters as much as which algorithm you pick.

---

## 7. Hybrid retrieval & Reciprocal Rank Fusion

Dense (§5) and sparse (§6) retrieval fail on different kinds of queries:
dense is good at "similar meaning, different words"; sparse is good at exact
attribute matches (a model number, a brand name) that an embedding might
blur together with similar-but-wrong products. **Hybrid retrieval** runs
both and *fuses* their rankings into one.

**Reciprocal Rank Fusion (RRF)** is the fusion method used here. For each
ranked list, a document's contribution is `1 / (k + rank)` — so a document
ranked #1 in either list contributes a lot, and the contributions from both
lists are summed. Crucially, RRF only needs *ranks*, not raw scores — which
sidesteps the problem that a cosine similarity (range -1..1) and a BM25
score (range 0..∞, uncalibrated) aren't directly comparable numbers.

```python
# src/digikala/phase2_assistant/retrieval.py
def rrf_fuse(rank_lists, k=60, weights=None):
    scores = {}
    for ranks, weight in zip(rank_lists, weights or [1.0]*len(rank_lists)):
        for rank, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + weight / (k + rank + 1)
    return scores
```

This project's RRF is **weighted** (sparse counts slightly more,
`PRODUCT_RRF_SPARSE_WEIGHT=1.35` vs `PRODUCT_RRF_DENSE_WEIGHT=0.65` in
`config.py`) and **candidate-pool-capped** (`RRF_CANDIDATE_POOL=200`) — only
the top-200 candidates from each side are fused, not the full 948k-length
ranking, which keeps a single search fast at catalogue scale.

**A genuinely honest finding, not a bug to hide:** on this project's own
measured retrieval benchmarks (both the title-exact-match one *and* the
harder natural-language paraphrase one — see `phase4_eval/evaluate.py`),
lexical/BM25-only retrieval currently **beats** the hybrid fusion on nDCG.
Hybrid still clearly beats dense-only. The lesson: adding a second retrieval
signal isn't automatically an improvement — you have to *measure* the fusion
against each half separately (that's what `retrieval_ablation()` does), or
you're just guessing that combining two things made them better.

---

## 8. RAG, grounding, and citation verification

**Retrieval-Augmented Generation (RAG)**: instead of asking a language model
to answer purely from what it memorized during training (which it might get
wrong, or which might be true for a *different* product than the one asked
about), you first *retrieve* the actual relevant data (§5–7), then hand that
data to the model and ask it to answer **using only that**.

RAG alone does not prevent hallucination — a model can still be handed five
real reviews and still assert something none of them say. This project treats
that as the actual threat model and adds a second layer: **citation
verification.**

**The contract:** every factual claim the system makes must carry an inline
citation, `[محصول 12345]` for a product fact or `[بازبینی 67890]` for a
specific review. After the model generates text, `verify_citations()`
regex-strips any citation id that the retriever *didn't actually return* —
so even if the model invents a citation to a plausible-looking id, it gets
deleted before the user ever sees it:

```python
# src/digikala/phase2_assistant/assistant.py
def verify_citations(text, allowed_products, allowed_reviews):
    text = _CITE_P.sub(lambda m: m.group(0) if int(m.group(1)) in allowed_products else "", text)
    text = _CITE_R.sub(lambda m: m.group(0) if int(m.group(1)) in allowed_reviews else "", text)
    return text
```

**The $0 fallback tier.** If no LLM is configured (or the LLM errors, or its
output fails a quality check — see below), the answer is instead **rendered
directly from the evidence** by plain string formatting — no model involved
at all. This "extractive" tier is provably grounded (it's *literally* the
retrieved data, formatted) and costs nothing. It's not a lesser fallback
bolted on as an afterthought — it's the foundation the LLM tier is layered
on top of, which is why the system works even with `RUN_MODE=extractive`
and zero API keys.

**Output-quality guards beyond citation-stripping** (`ShoppingAssistant._gen`
in `assistant.py`): a generated answer is rejected back to the extractive
fallback if it's citation-free/hollow (fewer than 8 meaningful tokens once
citations are stripped out), looks truncated mid-sentence (ends on a
connector like "و"/"یا"), or asserts an absolute claim ("all reviews say...")
that risks contradicting evidence that doesn't universally agree. That last
one came from a real observed failure: a model once claimed every cited
review rated ≥3 while one of the actual cited reviews was rated 0.

---

## 9. Intent routing without an LLM

Before you can answer a query, you need to know *what kind* of question it
is: is this person looking for a product (discovery), asking about a
specific one they've already found (Q&A), comparing two (comparison), or
doing category-level analysis (managerial)? You could ask an LLM to classify
this — but that costs a hosted API call (money + latency) for every single
query, just to decide what to do next.

Instead, `IntentRouter.route()` (`router.py`) uses **plain rules**: regex to
pull out explicit product ids (`\b(\d{6,9})\b`), keyword-cue lists per intent
(`_CMP`, `_QA`, `_MNG`), and fuzzy token-overlap matching against known
category/brand vocabularies (`match_known_value`) — all pure Python, no
model call, effectively free and instant.

**This is a real engineering trade-off, not a shortcut.** A rule-based router
is more brittle than an LLM classifier on genuinely ambiguous queries, and
this project's own evaluation caught two real routing bugs from overly broad
cue words (see the README's "Router regression caught" section) — but it's
$0, deterministic (the same query always routes the same way, which matters
for reproducibility and debugging), and every intent then gets a
purpose-built retrieval + prompt path instead of one generic prompt trying
to do everything.

**Priority order matters.** An explicit numeric product id is unambiguous
evidence and should win over any fuzzy match. A *fuzzy* (no-explicit-id)
match is inherently uncertain and should never outrank a *confident* signal
(like an already-resolved category scope) just because a generic word like
"چند" (how many) happened to also be a Q&A cue — see `router.py`'s comments
for the exact ordering this project settled on after finding the bug.

---

## 10. Data leakage — what it is and how it hides

**The core idea.** A model's test score is only meaningful if the test data
represents information the model genuinely couldn't have seen at prediction
time. **Leakage** is any way information from outside that boundary sneaks
in — the model then looks great on your test set but will fail in
production, because production doesn't grant it that same illicit access.

This project found and fixed **three separate, genuinely different kinds** of
leakage risk in Phase 3 (predicting `recommendation_status` from review text)
— worth walking through all three, because they're different failure modes
that each need a different fix:

**1. Feature leakage — a feature that restates the label.**
`rate_clean` (the review's own 1–5 star rating) was originally used as a
model feature. But the star rating and the recommendation label were both
recorded by the *same person in the same act of writing the review* — they're
two expressions of the same underlying sentiment, not independent evidence.
Feeding the rating in lets the model trivially "predict" a near-copy of the
label instead of learning anything about the review's *text*. `likes` has a
second, different problem: it accrues **after** the review is posted, so at
true prediction time (when the review is first submitted) it doesn't exist
yet — using it is *temporal* leakage.
**Fix:** the final model is text-only. **Detection tool:** a labeled
ablation — train the same model *with* those features too, and see how much
the score inflates (`recommend.py`'s `leakage_ablation`: text-only 0.72 vs.
text+numeric 0.76 — the +0.04 gap **is** the leak, made visible rather than
silently avoided).

**2. Train/test split leakage — the same example appears on both sides.**
If you split rows randomly, the exact same review text could theoretically
end up in both train and test (duplicate rows in the raw data). The model
would "predict" that review perfectly at test time simply because it
memorized it during training — a fake win.
**Fix:** deduplicate by review text *before* splitting (`_prep()` in
`recommend.py`), so no identical text can appear on both sides.

**3. Group leakage — a subtler, related-but-different example on both sides.**
Even after deduplicating identical text, a **different** review of the
**same product** can still end up in train and in test. The model can then
learn product-specific shortcuts (a particular brand name, a recurring
phrase about that one product) rather than generalizable sentiment language
— and get an inflated score from memorizing the product, not understanding
the review. This is the leak this project's *second* leakage audit caught
(triggered by a direct question: "make sure there is no leakage" — worth
re-auditing even code you already fixed once, because leakage has more than
one channel). **Fix:** a **grouped split** (`GroupShuffleSplit` on
`product_id`) that guarantees no product appears in both train and test.
This is now the **primary, headline metric** (`primary_macro_f1` in
`phase3_metrics.json`), not the naive random split — with the naive split's
measured product-overlap percentage (`naive_split_product_overlap_pct`)
reported alongside it, so any gap between the two numbers is explained by
data, not asserted away. (Interesting real result: on this run the overlap
was 47.76%, yet the grouped score came out *higher* than the naive one — no
evidence the overlap actually inflated the score this time. That's still
worth reporting, because "we checked and it happened not to matter here" is
a different, more honest claim than "we didn't check.")

**4. A leak that turned out NOT to be one, worth checking anyway.**
The TF-IDF vectorizer's vocabulary/IDF statistics are fit *inside* the
sklearn `Pipeline`, only ever called with `.fit(X_train, y_train)` — never
on the combined train+test data. If it *had* been fit on everything before
splitting, the vocabulary and document frequencies would encode information
about the test set's word distribution — a subtle, easy-to-miss leak many
projects get wrong. Verifying this was already correct (rather than
assuming it) is itself part of doing a leakage audit properly.

**Lesson generalized:** "no leakage" isn't a single checkbox. Ask, for every
feature and every split: (a) could this value only exist *after* the thing
I'm predicting, (b) does this example share hidden structure (same
product/user/session) with an example on the other side of the split, and
(c) is any statistic (vocabulary, scaler mean, PCA components...) computed
over data the model isn't supposed to see yet.

---

## 11. Text classification & the Macro-F1 metric

Phase 3 is a 3-class text classification problem: given a review's text,
predict `recommended` / `not_recommended` / `no_idea`. The model
(`recommend.py`) is TF-IDF features (§6) → `LogisticRegression` — a simple,
fast, interpretable linear model, deliberately not something fancier, because
a simple baseline you can actually explain beats a complex model you can't,
*unless* the complex model demonstrably does better (which is exactly what
the LoRA bonus, §12, checks rather than assumes).

**Why Macro-F1, not accuracy?** If 70% of reviews are "recommended," a model
that *always* predicts "recommended" gets 70% accuracy while being useless —
it never once identifies a "not_recommended" review. **F1** per class is the
harmonic mean of precision (of everything I labeled X, how much was actually
X?) and recall (of everything that was actually X, how much did I catch?).
**Macro-F1** averages the per-class F1 scores *unweighted* — so the rare
class counts exactly as much as the common one, and a model can't hide behind
the majority class. This project's own majority-class baseline (`DummyClassifier`)
makes the floor explicit: 0.167 Macro-F1 (three balanced-in-eval classes,
scored near-randomly) vs. the trained model's 0.72+ — the gap is real signal.

```
precision = true_positives / (true_positives + false_positives)
recall    = true_positives / (true_positives + false_negatives)
F1        = 2 * precision * recall / (precision + recall)
Macro-F1  = mean(F1_recommended, F1_not_recommended, F1_no_idea)   # unweighted
```

---

## 12. Parameter-efficient fine-tuning (LoRA)

**The problem fine-tuning solves.** A pretrained language model (like
ParsBERT, a Persian BERT) already understands Persian broadly, but hasn't
learned *this specific task* (3-class recommendation classification).
Fine-tuning adapts it to the task by continuing training on labeled examples.

**Why not fine-tune the whole model?** ParsBERT has ~118M+ parameters.
Updating all of them needs a lot of GPU memory (gradients + optimizer state
for every parameter) and risks **catastrophic forgetting** — the model can
overwrite its broad language understanding while overfitting to a small
task-specific dataset.

**LoRA (Low-Rank Adaptation)** freezes the entire pretrained model and
instead injects small, trainable "adapter" matrices into specific layers
(here: the attention query and value projections). The math: instead of
learning a full-size weight update `ΔW` (huge), LoRA learns two small
matrices `A` and `B` such that `ΔW ≈ B·A`, where `A` and `B` have a small
inner dimension `r` ("rank"). This project uses `r=8` — and the result is
**297,219 trainable parameters out of 163,140,870 total (0.18%)**
(`phase3_lora_metrics.json`). Training touches almost nothing, so it's fast,
memory-light, and can't catastrophically forget what the base model already
knows about Persian.

```python
# src/digikala/phase3_predict/lora_finetune.py
lora_cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                      lora_dropout=0.1, target_modules=["query", "value"])
model = get_peft_model(base_model, lora_cfg)
```

**Fair comparison matters.** LoRA is only meaningful evidence if it's
compared against the baseline on the **exact same split** — otherwise a
different split could explain any score difference, not the modeling choice.
`recommend.prepare_split()` factors out the label-encode + stratified-cap +
grouped-split logic so both the TF-IDF baseline and the LoRA script train and
evaluate on identical data. **Honest result, measured not assumed:** on a
resource-capped comparison (6000 train rows, 2 epochs — a reasoned trade-off
for laptop-GPU feasibility, documented as such, not silently treated as a
full-data result), LoRA scored **0.7049** vs. the baseline's **0.7291** — the
simple linear model actually won here. That's a legitimate finding: a small
amount of fine-tuning on a small subset doesn't automatically beat a strong,
well-regularized linear baseline, and reporting a loss honestly is more
valuable evidence of engineering rigor than only reporting wins would be.

---

## 13. Evaluating a system that has no single right answer

Classification has an unambiguous right answer per example (§11's F1 works
because you can check the true label). "Is this a good answer to a shopping
question?" doesn't have one ground truth — which is why Phase 4 measures
along **several separate axes** instead of collapsing everything into one
number:

- **Retrieval quality** (recall@k / MRR / nDCG) — did the search actually
  surface the relevant item? See §5–7's discussion of the two benchmarks.
- **Grounding** — are the claims backed by what was retrieved? (citation
  coverage/validity, §8)
- **Response quality** — does the answer actually address the question? (the
  deterministic `task_completion_proxy` and the LLM judge's relevance score)
- **Prediction** — Macro-F1 for Phase 3 (§11)
- **Latency & cost** — how long did it take, how much did it cost (§15)
- **Failure analysis** — concrete cases where the system got something wrong,
  and why

**Why two retrieval benchmarks and two quality-scoring methods (§14) instead
of one?** Any single evaluation method has blind spots specific to how it was
built. A title-exact-match retrieval benchmark structurally favors lexical
search; a paraphrase benchmark is fairer but its "gold" labels are still
programmatically derived, not human-verified. An LLM judge can be
unreliable (§14); a deterministic proxy can only check for *surface*
structure (does the answer contain a citation?), not whether the prose is
actually good. Using more than one lens and *reporting where they disagree*
is more honest than picking whichever single number looks best.

---

## 14. LLM-as-judge — power and pitfalls

**The idea.** Use a language model to *score* another model's output (e.g.,
"rate this answer's faithfulness to the evidence, 0–5") instead of requiring
a human to read and score every single response. It's much cheaper than
human evaluation and scales to hundreds of queries.

**The catch, measured in this project rather than assumed:** a small local
judge model (Qwen2.5-1.5B) is not a reliable evaluator. In this project's
own 16-query human-eval set, the judge's `relevance` score came back
**unparseable on 16 out of 16 queries** — the model never even emitted a
plain 0–5 digit in the expected format for that particular prompt.
`faithfulness` parsed on 14/16, and where it did, agreement with a human
labeler was only moderate and borderline-significant (Spearman
**ρ = 0.524, p = 0.054, n = 14**) — see the README's "LLM-as-judge" section
for the full numbers.

**What to do about it, concretely, rather than just trusting the number:**
1. **Document the exact prompt** the judge sees (this project quotes the
   literal system/user prompt text in the README) — a judge score is only
   interpretable if you know precisely what it was asked.
2. **Never silently coerce an unparseable reply into a number.** An
   unparseable score is recorded as `None`/`null` and excluded from the
   mean, not treated as a 0 (that would be a fabricated, misleadingly
   confident-looking data point).
3. **Compare against real human labels on a small sample** when possible
   (`human_eval_comparison`, Spearman correlation) — this is the only way to
   know whether the judge's scores track human judgment at all, rather than
   just assuming an LLM saying "4/5" means what a person would mean by "4/5."
4. **Add a second, non-LLM axis** (`deterministic_quality_proxy`, §13) that
   doesn't depend on judge reliability at all, as a hedge.

**Lesson generalized:** "we used an LLM to evaluate it" is not, by itself,
evidence of anything. The evaluation method needs its own evaluation.

---

## 15. Cost & resource management

The brief caps hosted API spend at **$5 total** and explicitly says managing
compute resources is itself part of the engineering problem, not a side
concern. A few concrete techniques this project uses:

- **A $0 tier is always available** (the extractive fallback, §8) — the
  system never *requires* a paid call to function at all.
- **Track attempted, successful, and failed calls separately**
  (`core/llm.py::BudgetTracker`), and distinguish **tracked cost** (counts
  against the $5 cap) from **estimated list cost** (always reported, even
  for a $0 free-tier call) — so the final report can honestly say both "we
  spent $0" and "here's what it would have cost at list price."
- **Retry with backoff, and fall back between network paths** — a transient
  429/5xx shouldn't fail the whole run, and (a real problem this project
  hit) a hanging environment proxy shouldn't either, if a direct connection
  works.
- **Cache anything expensive that doesn't change.** The BM25 index and
  product embedding vectors are built once and persisted to disk
  (`ProductIndex.save`/`.load`); vectors are memory-mapped (`mmap_mode="r"`)
  so 1.45GB of embeddings doesn't have to sit fully in RAM; Streamlit's
  `st.cache_resource` loads the assistant once per dashboard server process
  instead of on every page interaction.
- **Sample deliberately, and say why.** Full-data runs use all 948k
  products / 6.16M comments; the notebook/demo path uses a smaller,
  *unbiased* sample (§10's "group leakage" section explains why a biased
  head-read sample would be worse) so a reviewer can "Run All" in minutes —
  and MAX_PER_CLASS=30,000 caps Phase-3 training because the task
  saturates well before consuming the full corpus (see the README's
  "Sampling justification").

---

## 16. Putting it all together — one query, start to finish

Say a user types: **"محصول 4332510 چه ایرادهایی داره؟"**
("What's wrong with product 4332510?")

1. **Routing** (§9): `IntentRouter.route()` finds the explicit product id
   `4332510` via regex, sees the Q&A cue "چه" isn't matched but "ایراد"
   ("flaw") is in `_QA` — routes to `product_qa` with that product id. No
   model call, ~instant.
2. **Retrieval** (§5–7): `ReviewRetriever.retrieve()` embeds the query,
   embeds *only this product's* reviews (on demand), scores them with both
   dense cosine similarity and BM25, fuses the two rankings with weighted
   RRF. `_polarity()` (§7/§9) detects this is a **negative-intent** query
   ("ایراد" is a negative cue) and reranks candidates toward reviews that
   actually read negative (low rating, `not_recommended` status, a non-empty
   disadvantages field) — not just whatever ranked highest on raw text
   similarity.
3. **Evidence assembly**: `review_stats()` computes the product's real
   recommend/not-recommend rates from the labeled reviews (§4's cleaned,
   typed data). The selected reviews become the evidence the model (or the
   extractive renderer) is allowed to cite.
4. **Generation** (§8): if a hosted/local LLM is configured, it's given the
   evidence and asked to answer, citing `[بازبینی id]` for each claim.
   `verify_citations()` strips any id it didn't actually receive.
   `_gen()`'s quality guards reject a hollow, truncated, or absolute-claim
   answer back to the deterministic extractive rendering.
5. **Answer**: text + a list of the actual cited review ids + latency + cost
   (`$0` if extractive/local, tracked if hosted) are returned as an `Answer`
   object (`assistant.py`).
6. **Evaluation** (§13–14), separately and later: this exact kind of query
   is one of the probes in `failure_analysis()`, scored by the deterministic
   proxy and (if configured) the LLM judge, with results written to
   `artifacts/metrics/phase4_metrics.json` and shown on the dashboard's
   Evaluation page.

Every one of those steps is deterministic and inspectable except step 4's
LLM call — and even that one is fenced by verification on both sides
(citation stripping before, quality-guard rejection after). That's the whole
design philosophy in one query.

---

## 17. Glossary

| Term | Plain-language meaning |
|---|---|
| **Normalization** | Rewriting text into one canonical spelling/character/digit convention so identical meanings compare equal. |
| **Tokenization** | Splitting text into countable units (roughly, words). |
| **Embedding** | A vector of numbers representing a piece of text's meaning; similar meanings → nearby vectors. |
| **Cosine similarity** | A measure of the angle between two vectors; 1 = identical direction, 0 = unrelated, -1 = opposite. |
| **Dense retrieval** | Search using embeddings/cosine similarity — good at "similar meaning, different words." |
| **Sparse retrieval** | Search using exact term matching (TF-IDF/BM25) — good at exact attributes. |
| **TF-IDF** | A score weighting a word by how often it appears in one document vs. how rare it is across all documents. |
| **BM25** | A refined ranking function built on TF-IDF ideas, with term-frequency saturation and document-length normalization. |
| **RRF (Reciprocal Rank Fusion)** | Combines multiple ranked lists into one, using only each item's *rank* (not raw score) in each list. |
| **RAG** | Retrieval-Augmented Generation: retrieve real data first, then generate an answer grounded in it. |
| **Grounding** | The property that every claim in an answer is actually supported by retrieved evidence. |
| **Hallucination** | A model stating something as fact that isn't actually supported by anything real. |
| **Intent routing** | Deciding what *kind* of request a query is, before deciding how to answer it. |
| **Data leakage** | Information reaching a model (via a feature or a data split) that wouldn't genuinely be available at real prediction time. |
| **Train/val/test split** | Dividing labeled data into non-overlapping sets so a model's test score reflects genuine generalization, not memorization. |
| **Grouped split** | A split that keeps all examples sharing some group key (e.g. product id) entirely on one side, preventing group-level leakage. |
| **Macro-F1** | The unweighted average of per-class F1 scores; treats rare and common classes equally. |
| **Fine-tuning** | Continuing to train a pretrained model on task-specific labeled data. |
| **LoRA** | A fine-tuning method that trains small low-rank adapter matrices instead of the full model's weights. |
| **LLM-as-judge** | Using a language model to score another model's output instead of (or alongside) a human. |
| **Ablation** | Removing or isolating one component/feature and re-measuring, to find out how much that specific piece actually mattered. |
| **Pseudo-gold benchmark** | An evaluation benchmark whose "correct answers" are derived programmatically from metadata, not verified by a human — useful, but explicitly not the same as human-labeled ground truth. |

---

*This document lives at `docs/COOKBOOK.md`. It explains the theory; the
actual, current, executable implementation is always the code in
`src/digikala/` — if the two ever disagree, trust the code and consider this
file stale until updated.*
