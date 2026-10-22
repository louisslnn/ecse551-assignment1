# Gap-reset validation

Validated on 22 October 2026 on `feature/task3-gap-burn-in`, based directly on main
`ecfa4870cf8c41fa9b60bd2ce5af27edc11291bd`. The active notebook retains main's function and variable names.

## Change and execution scope

The volatility function now resets mean, variance and observation count on the 21st
consecutive missing trading row, then requires 100 newly observed returns. Shorter
gaps retain the previous behavior. Volatility first appears at observation 100;
the lagged adjusted return first appears at observation 101. Missing returns, not
index membership or elapsed calendar days, trigger the reset.

Only Task 3 was freshly executed. All Task 1/2 cells, outputs and metadata are
preserved exactly from main. Historical modules, result files outside this compact
directory, and the report remain unchanged. The tests' outdated function references
were mechanically aligned with main; no naming refactor was made in the notebook.

## Runs and checks

- A fresh unchanged-main baseline and two independent changed-study runs each used
  isolated kernels with only the Task 3 notebook cells and five supplied CSV inputs.
  The pinned notebook dependencies were verified before running.
- Each run repeated all 16 development rows and the existing selection rule, followed
  by all six test strategies, including smoothing and delayed execution. No grids,
  seeds, feature definitions, training windows or refit schedules changed.
- The two changed runs match at `rtol=1e-6`, `atol=1e-8` for development, selection and
  test tables, and full-precision daily P&L and rank IC. Maximum differences are
  recorded in `validation_summary.json`.
- All 16 Task 3 code cells executed without error outputs. The combined notebook's
  remaining 25 code cells retain their previously saved outputs. Five Task 3 figures
  were regenerated and visually checked.
- 39 compact-notebook tests passed, covering the existing timing/metric/source checks
  and the new gap boundary, fresh-state, stock independence, burn-in and causality cases.
- Each run checked 319 refits with `last_training_outcome < refit_date`. All 754 test
  outcome days were scored, with at least 491 predictions and scored stocks per day.
- Raw-data truncation at 2010-06-30 and 2013-12-30 produced exactly unchanged volatility,
  features and Lasso predictions. The independent truncation refits also passed timing guards.
- The sandbox logged an ipykernel process-enumeration warning during shutdown, after
  successful execution/export. Notebook cells had zero error outputs and all runs completed.

Measured Task 3 runtimes, excluding final export/kernel shutdown:

- baseline: 8.74 minutes.
- changed: 8.47 minutes.
- repeat: 5.77 minutes.

## Baseline comparison

The data contain 16 resumed gaps of at least 21 missing trading rows across 15 stock
series. Restarting burn-in creates no additional unavailable adjusted member returns
in this dataset. Selected configurations remain unchanged:

- lasso: `{"alpha": 0.0001, "model": "lasso"}`.
- forest: `{"depth": 8, "leaf": 200, "model": "forest"}`.
- mlp: `{"epochs": 40, "hidden": 32, "model": "mlp"}`.

| Strategy | Baseline test Sharpe | Reset test Sharpe | Change |
|---|---:|---:|---:|
| lasso | 1.312907 | 1.312907 | -1.023e-09 |
| forest | 3.189437 | 3.189437 | -1.321e-09 |
| mlp | 2.343392 | 2.343392 | +1.055e-09 |
| MLP smoothed | 1.241707 | 1.241707 | +3.069e-09 |
| 5-day reversal | 0.727769 | 0.727769 | +2.168e-09 |
| 12–1 momentum | 0.779349 | 0.779349 | -4.096e-08 |

Test scoring coverage is unchanged. The largest absolute Sharpe change is
4.096e-08; performance is effectively unchanged.
The reset is a missing-data handling improvement, not evidence of better performance.

`gap_reset_comparison.csv` contains full-precision before/after settings, development
Sharpes for selected models, test metrics, yearly/delayed Sharpes, coverage and gap counts.
The existing three published CSV tables retain notebook display precision; the JSON
summary records base/notebook/source/input hashes, runtimes, gap counts and validation.

## Preserved limitations

This study follows previously viewed test results and is not a fresh blind test.
Same-close execution and omitted trading costs remain limitations. Task 1/2 were
preserved, not corrected or certified: their existing classification split and Lasso
gradient-update issues remain. The assignment separately requires a report PDF;
the report was outside this change.
