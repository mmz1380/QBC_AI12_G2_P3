# Digikala AI Shopping Assistant — Project 3

## Files

- `digikala_project3_final.ipynb` — complete executed project notebook.
- `README.md` — dependencies, execution, architecture, evaluation and final results.

The raw dataset is not bundled because it is the public course dataset and is
much larger than the project code.

Dataset: `RadeAI/Digikala_comments_products`

Pinned revision:
`89c3133b169c8d3793db8834f56f32fee33d9db0`

Required raw files:
- `digikala-products.csv`
- `digikala-comments.csv`

## Dependencies

The final executed run used Python 3.11.0.

```bash
pip install pandas numpy pyarrow scikit-learn plotly hazm persiantools sentence-transformers torch huggingface-hub scipy joblib requests
```

`HF_TOKEN` is optional.

## Reproduce

1. Put the two CSV files beside the notebook or inside `data/raw/`.
2. Keep the hosted API key in the operating-system environment; never paste the key into the notebook.
3. The final run used `GROQ_API_KEY` and `openai/gpt-oss-20b`.
4. Restart VS Code/Jupyter after any environment or VPN/proxy change.
5. Use **Restart Kernel → Run All**.
6. Save the executed notebook.

Run artifacts are written under a fresh `nb_artifacts/run_*` directory.

## Phase 1 — data preparation

The notebook scans the full raw CSVs, normalizes Persian text, handles missing,
duplicate and invalid values, and creates a reproducible uniform random review
sample with a fixed seed.

Final executed run:
- raw product rows scanned: **1,283,496**
- raw comment rows scanned: **6,156,289**
- sampled/clean comments: **99,999**
- Phase-2 products represented in the sample: **55,374**

Sampling limitation: Phase 2 indexes only products referenced by sampled
reviews, so its catalogue is not the complete million-product catalogue.

## Phase 2 — shopping assistant

Implemented:
1. Product Discovery — structured filters + dense retrieval + BM25 + weighted RRF + reranking.
2. Review-based Product QA — grounded in real reviews with review IDs.
3. Product Comparison — direct facts, positive/negative review evidence, review IDs and deterministic inference.
4. Managerial Analytics — category/brand complaint patterns, weighted recommendation rate, brand satisfaction and low-recommendation products.

The final comparison evidence diagnostic passed with no invalid positive or
negative evidence.

Hosted generation is used selectively for Discovery and Product-QA. Comparison
and Managerial analytics remain deterministic/grounded to reduce unsupported
claims, latency and API cost.

## Phase 3 — recommendation prediction

Classes:
- `recommended`
- `not_recommended`
- `no_idea`

Main model: text-only TF-IDF + Logistic Regression.

Product IDs are used for grouped train/validation/test splitting to reduce
leakage.

Final results:
- Majority baseline Macro-F1: **0.2585**
- Grouped test Macro-F1: **0.6905**

## Phase 4 — evaluation

Response evaluation is separate from the visible demos and contains 10
programmatically generated queries across 2 contexts/categories.

Final response metrics:
- mean task-completion proxy: **4.4 / 5**
- mean grounding proxy: **4.938 / 5**
- citation validity: **1.0**
- citation coverage: **0.74**

These are deterministic proxy metrics, not human evaluation or
LLM-as-a-Judge.

### Retrieval benchmark

The retrieval benchmark is reproducible programmatic pseudo-gold, not
human-labeled.

Final results:
- Hybrid Recall@10: **0.70**
- Lexical Recall@10: **0.70**
- Hybrid nDCG@10: **0.3313**
- Lexical nDCG@10: **0.4255**
- Hybrid latency speedup vs lexical baseline: **7.98×**

Therefore the project does not claim Hybrid improves ranking quality on this
benchmark. The lexical baseline has better measured nDCG, while Hybrid is
materially faster.

## Hosted API accounting

The final executed run successfully used the hosted API.

Provider/model:
- provider: **Groq**
- model: **openai/gpt-oss-20b**
- key source: **`GROQ_API_KEY`**; the secret value is never stored or printed

Network diagnostic:
- environment/proxy path: transport timeout
- direct no-environment-proxy path: successful (`HTTP 200`)

Final API accounting:
- preflight HTTP requests: **2**
- Chat requests: **7**
- successful Chat requests: **7**
- failed Chat requests: **0**
- unresolved requests: **0**
- total HTTP requests reported: **9**
- input tokens: **6,243**
- output tokens: **2,476**
- estimated list cost: **$0.001211**
- project API budget cap: **$5**
- within budget: **yes**

The cost is explicitly an estimated list cost from configured token rates, not
a provider invoice.

## Failure analysis

The notebook keeps real retrieval, response-quality, classifier and hosted
provider/network failure cases and reports probable causes, severity and next
steps rather than hiding them.

## Final readiness

The final executed notebook:
- completed Run All
- has continuous code-cell execution counts
- contains no code-cell error output
- passed router regression checks
- passed comparison evidence polarity checks
- reports retrieval results honestly
- uses response evaluation separate from demos
- reports Phase-3 Macro-F1
- passed grounding/task-completion/citation checks
- completed real hosted API calls
- reports request count, tokens and estimated cost
- stayed well below the `$5` API cap

Final mechanical result:

`submission_checks_passed = true`
