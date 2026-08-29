# Digikala Project 3 — FINAL CLEAN READY TO RUN

This package is the clean execution candidate. It intentionally does **not** include the two large raw CSV files.

## Put these files in the same folder

1. `Digikala_Project3_FINAL_CLEAN_READY_TO_RUN.ipynb`
2. `human_eval_labels.csv` (the already completed 10-sample human evaluation)
3. `digikala-products.csv`
4. `digikala-comments.csv`

## Before Run All

- Use Python 3.11.
- Make sure `METIS_API_KEY` is available as an environment variable.
- Do **not** paste the API key into the notebook.
- Restart the Jupyter kernel.

## Run

Use **Restart Kernel → Run All** and wait until the final cells finish. The first complete run can take noticeably longer because optional LoRA/quantization benchmarks and LLM-as-a-Judge are enabled. Optional bonus failures are caught and should not break mandatory project completion.

The notebook will generate and launch the Streamlit dashboard automatically. Test both light/dark modes, comparison search, manager dashboard, model/evaluation page, and Auction demo.

## Final acceptance checklist

- `submission_checks_passed == True`
- all mandatory submission checks are `True`
- Metis provider/model are correct and total estimated API cost is safely below the $5 course credit
- `phase4_cost_attribution_consistent == True`
- no code-cell errors
- Streamlit opens and the main tabs work in both themes
- `human_eval_validation.json` reports matching human labels (ideally all 10)
- Auction remains `mentor_approved = false` until explicit mentor approval; this must not be falsely claimed
- save the executed notebook before sending it for the final audit

## Important

Do not submit the full ~100 MB cache/artifact directory unless the platform explicitly requests it. The notebook and compact evidence/report files are the intended normal submission evidence.
