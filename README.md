# ECSE 551 Assignment 1 — Task 3

The active implementation is **corrected protocol v2**. The original test results were already observed before these corrections. Finalists and hyperparameters are unchanged; this is not a fresh held-out test or a new model-selection exercise. The report is intentionally left unchanged until all assignment tasks are finished.

## Inputs and environment

Place the five actual CSVs in `instructions-data/`:

- `eq_returns.csv`
- `eq_indicator.csv`
- `eq_stock_metadata.csv`
- `psych_bullish.csv`
- `psych_bearish.csv`

`.webloc` files are web shortcuts and cannot replace CSV contents. `instructions-data/` and prediction caches are ignored by Git. No course dataset should be added to version control.

Use Python with the packages in `requirements-task3.txt`. Every executed run records actual dependency versions and SHA-256 input fingerprints. The historical log records a different environment; numerical changes versus history can therefore reflect dependency versions as well as the protocol correction.

## Reproduce

The portable notebook is `notebooks/task3_agent.ipynb`. Set its single `DATA_DIR` variable to your CSV folder (use `../instructions-data` when the notebook kernel starts in `notebooks/`) and run every cell in order. The notebook contains readable implementation cells, exact configuration and experiment definitions, historical comparison numbers, the original agent prompt and log, and the additional ChatGPT disclosure. It needs no repository modules or pre-existing result files. Outputs, plots, checks, and training records are generated during execution. Allow several hours for all 62 development definitions and finalist refits.

To regenerate notebook code after changing a maintained module:

```sh
python3 scripts/build_task3_notebook.py
```

This clears execution outputs deliberately. To run it in isolation, with only the notebook and copies of the five CSVs:

```sh
python3 scripts/execute_task3_notebook.py --data-dir instructions-data
```

This preserves the executed notebook and copies corrected outputs back into `results/corrected_v2/`. Temporary execution directories are ignored under `.task3-runtime/`.

Alternatively, run the identical maintained implementation directly:

```sh
python3 src/task3_corrected.py all --data-dir instructions-data
```

Stages `checks`, `development`, `finalists`, and `figures` are also available. The old phase entry points now route to corrected stages and cannot write the historical results. Stage runs are fresh recomputations, not cache resumes. `finalists` retains the already chosen configurations, regardless of corrected development scores. Test evaluation is explicitly a correction after the original results were observed.

## Timing and metrics

- Features at close **t** predict the next observed trading day. Explicit realization dates handle holidays and weekends. Development outcomes are in 1990–2010; test outcomes are in 2011–2013. The December 31, 2010 prediction belongs to test evaluation.
- At each refit **t**, training outcomes must be observed strictly **before t**. A 756-day window includes the most recent 756 eligible feature dates, normally ending two trading days before the refit. Refit records store the last feature and outcome date.
- The original scheduled refit calendar is preserved. A separately marked warmup fit predicts the two preceding days to cover the first outcome and delayed-execution diagnostic. Its random generator does not advance the scheduled-fit generator.
- MLP epoch selection uses the newest 20% of eligible feature days. Earlier training rows whose targets reach that holdout are purged. The probe scaler sees only the older training subset; the final scaler/network use all eligible rows.
- The primary market benchmark is the equal-weight next-day **raw excess return**, among prediction-day members. The old average volatility-adjusted benchmark is retained as a labeled secondary diagnostic. Test-period OLS hedges are descriptive, not deployable hedging rules.
- Holding period uses matched consecutive valid prediction pairs inside each evaluation period. Gaps and membership exits are not bridged. Constant nonzero positions give infinity; all-zero positions are undefined. Per-stock pairs, eligibility, and undefined/infinite counts are saved. Entries/exits and transaction costs are not modeled.
- Every expected scored trading day must have at least 100 predictions. Missing prediction dates are errors, not silently dropped observations. Daily P&L and IC files use realization dates.

The inherited static sector classifications and sentiment identifier mapping based on development-period availability are retained. Truncation tests and stricter target timing do not establish that these metadata were available point by point historically.

The fixed replay definitions are in `config/task3_experiments_v2.json`; all 62 original source identifiers, including controls and post-processing dependencies, are preserved. The Lasso penalties use their original exact values. No additional grid or finalist selection is performed.

## Verification and submission

```sh
python3 -m pytest -q tests
python3 src/task3_truncation_check.py lasso 2012-06-29 --data-dir instructions-data
```

The truncation command requires saved corrected finalist predictions and removes future raw data before rebuilding features and refitting. Daily-series verification checks serialized P&L, IC, annual/sub-window Sharpes, and their saved summaries. The run manifest records success or failure; the notebook execution metadata distinguishes an executed artifact from an unexecuted template.

Keep historical artifacts in `results/` and `notebooks/task3.ipynb` unchanged. Corrected metrics, comparisons, and figures belong in `results/corrected_v2/`. Integrate the portable Task 3 cells into the group's final single notebook with Tasks 1 and 2, use the required surname naming convention, and complete the report later. The separate `docs/chatgpt_disclosure.tex` snippet supplements the existing disclosure of other AI assistance; its validation wording must match the work actually completed.
