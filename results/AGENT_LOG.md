# Task 3 agent log

Running record of decisions, problems, and questions for ECSE 551 Assignment 1, Task 3.
Entries are added in order and never rewritten. Dates are in 2026.

## Commit hashes (made by a group member, not by the agent)

| What | Hash |
|---|---|
| Phase 0 frozen harness (`src/task3_data.py`, `src/task3_eval.py`) | `cceeb15e720d5b637c2082d1da9a74c5f101496a` |
| Phase 3 frozen finalists (`results/final_configs.json`) | pending |

## Environment

Conda `base`, Python 3.14.6, macOS, 8 cores, 8 GB RAM. Packages installed with pip on Oct 9 with the group's permission:

| Package | Version |
|---|---|
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| pytest | 9.1.1 (already present) |
| nbformat | 5.11.1 |
| ipykernel | 7.4.0 |

## Before Phase 0 (Oct 9)

The agent read `AGENT_PROMPT.md`, inspected the repository and `notebook/task3.ipynb` without changing anything, and asked four questions.

### Questions and answers

| # | Question | Answer |
|---|---|---|
| Q1 | The prompt requires perfect-foresight Sharpe above 50, but the notebook's own output is 16.6 on 2011-2013. Lower the threshold (suggested: above 10)? | Not answered in chat yet. Phase 0 reports the result against the threshold as written. See Problem 1. |
| Q2 | The current environment has no scikit-learn. May I use the `ecgfm` conda environment that the notebook used? | No. Stay in the current environment; installing numpy, pandas, scipy, scikit-learn, matplotlib, pytest, nbformat and ipykernel is allowed. Versions recorded above. |
| Q3 | The prompt says `notebooks/`, the repository has `notebook/`. Keep `notebook/`? | Not answered in chat yet. Nothing renamed. |
| Q4 | The repository has no commits. What goes in the first commit? | The agent never stages, commits, or pushes. At each commit point it lists the files, suggests a message, and waits. A group member commits and sends back the hash. In Phase 3 nothing runs on the test period until the group confirms `final_configs.json` is committed and gives its hash. |

### Defaults proposed by the agent, all accepted

1. `evaluate(P, Y, mem, start, end, windows, demean, label, plot)`: `Y` and `mem` become the second and third arguments, body unchanged.
2. In Phases 1 and 2 every coverage statistic, fill fraction and prediction stops at 2010-12-31, including sentiment. The Phase 0 pipeline checks use the full history because the expected values require it.
3. The log has two window columns (1990-1999, 2000-2010). The four selection windows per configuration are saved separately in `results/dev_windows.csv`.
4. Lasso alpha grid: six log-spaced values chosen from the first training window only.
5. The expanding training window starts on the first date all return features exist (about mid-1981).
6. MLP epochs are chosen on the most recent 20% of each training window by date, then the model is refit on the whole window with that number of epochs.

## Phase 0 (Oct 9)

### What was done

- `src/task3_data.py`: `load_data()`, `ewma_vol()`, `build_targets()`, copied from notebook cells 1 to 3.
- `src/task3_eval.py`: `sharpe()`, `evaluate()`, copied from notebook cell 3.
- `src/task3_log.py`: `append_log()`, an append-only writer for `results/experiment_log.csv`.
- `src/task3_phase0.py`: the four harness controls.
- `tests/test_task3_pipeline.py`: 12 tests, all passing.

### Decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Package versions pinned to the ones in the notebook's environment (numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, scikit-learn 1.9.0, matplotlib 3.11.1). | The notebook's numbers were verified with those versions, so the extraction can be checked exactly. |
| D2 | The notebook's exploratory `print` calls in the loading cell were not copied into `load_data()`. | They are checks, not logic. The same checks are now assertions in the pipeline test. |
| D3 | `data_dir` is an argument of `load_data()` and `build_targets()`, default `"data"` (repository root). | The notebook used the global `DATA = Path("../data")`. Turning a global into an argument is the one change the prompt allows. |
| D4 | The scripts select the matplotlib `Agg` backend themselves; the harness is untouched. | `evaluate()` calls `plt.show()`, which would open a window and block a script. |
| D5 | The Phase 0 controls are logged as rows E001 to E004 with `phase = 0` and an empty `dev_sharpe`. Their Sharpe is in the `notes` column. | They are not development-period results (three are on the test period, one on 1985-2010), so they must not look like candidates. |
| D6 | The test for "fraction above 5" uses an absolute tolerance of 5e-6 instead of 1e-3 relative. | The expected value 0.00076 is given to two significant digits; the computed value is 0.000755. |

### Problems

**Problem 1: perfect-foresight control is below the prompt's threshold.** `P = Y` gives Sharpe 16.598 on 2011-2013, identical to the notebook, but the prompt requires above 50. The harness was copied unchanged and was not modified. The value is far beyond any realistic strategy (rank IC is exactly 1.0), so the control still shows that predictions and returns are aligned. It cannot reach 50 because the daily P&L is the sum of squared returns across stocks, which varies a lot from day to day. Status: reported as failing the threshold as written; waiting for the group to set the threshold.

**Problem 2: the notebook's tail fractions use the wrong denominator.** The notebook reports that 0.076% of member vol-adjusted returns exceed 5 in absolute value. That number divides by all 8577 x 1314 = 11,270,178 cells, because in pandas 3 `.stack()` keeps NaN cells (non-members and burn-in days). Among the 4,227,347 actual member values the fraction is 8,509 / 4,227,347 = 0.201%. The same applies to the other two thresholds:

| Threshold | Count | Notebook (all cells) | Among member values |
|---|---|---|---|
| abs > 3 | 55,538 | 0.493% | 1.314% |
| abs > 5 | 8,509 | 0.076% | 0.201% |
| abs > 10 | 933 | 0.008% | 0.022% |

This is in an exploratory print only. The harness, the count, the standard deviation and the kurtosis are not affected (those functions skip NaN). The test asserts both numbers. Status: reported to the group; the report draft currently quotes 0.08%.

### Phase 0 results: pipeline checks

| Check | Expected | Actual | Pass |
|---|---|---|---|
| duplicated permno columns renamed `_b` | 21 | 21 | yes |
| empty columns dropped | 21 | 21 | yes |
| securities remaining | 1314 | 1314 | yes |
| metadata index equals return columns | True | True | yes |
| mean members per day | 499.79 | 499.7938 | yes |
| min members per day | 489 | 489 | yes |
| member cells with NaN return | 0 | 0 | yes |
| non-member cells with a return | 2,190,592 | 2,190,592 | yes |
| non-NaN vol-adjusted returns on member cells | 4,227,347 | 4,227,347 | yes |
| std of those | 1.0574 | 1.05744 | yes |
| excess kurtosis | 22.75 | 22.7496 | yes |
| fraction with abs value > 5 | 0.00076 | 0.000755 over all cells; 0.002013 over member values | yes, see Problem 2 |

### Phase 0 results: harness controls

| Control | Period | Expected | Actual | Pass |
|---|---|---|---|---|
| perfect foresight `P = Y`, Sharpe | 2011-2013 | above 50 | 16.598 | no, see Problem 1 |
| random N(0,1), seed 42, absolute Sharpe | 2011-2013 | below 1.5 | 0.314 | yes |
| always long `P = 1`, correlation with market | 2011-2013 | above 0.9 | 1.000 | yes |

### Phase 0 results: 5-day reversal baseline, demeaned, 1985-2010 only

| Metric | Value |
|---|---|
| Sharpe 1985-2010 | 2.327 |
| Sharpe 1985-1989 | 3.25 |
| Sharpe 1990-1994 | 4.09 |
| Sharpe 1995-1999 | 2.59 |
| Sharpe 2000-2004 | 1.39 |
| Sharpe 2005-2010 | 1.25 |
| mean rank IC | 0.0255 |
| IC t-stat | 18.49 |
| holding period (days) | 1.57 |
| correlation with market | 0.159 |
| min predictions per day | 493 |

The yearly Sharpes for 1985 to 2010 match the notebook's values exactly.

### Frozen-harness diff

Not yet possible: there is no commit to diff against. It will be run as soon as the group provides the Phase 0 hash.

## Phase 0 follow-up (Oct 9, after the checkpoint)

### Answers received

| # | Answer |
|---|---|
| Q1 | Confirmed: the perfect-foresight threshold is lowered from Sharpe above 50 to Sharpe above 10. |
| Q3 | Rename `notebook/` to `notebooks/`, as the prompt's layout assumes. |
| new | Add `__pycache__/` and `.DS_Store` to `.gitignore`. |

### Changes made

| # | Change | Reason |
|---|---|---|
| C1 | `src/task3_phase0.py`: the perfect-foresight check is now `Sharpe > 10` (was `> 50`), with a comment pointing here. | Group decision on Q1. The threshold only has to separate a leaking predictor from a realistic one; 16.6 with rank IC 1.0 does that. The evaluation code was not changed. |
| C2 | `results/phase0_controls.csv`: first row edited by hand from `> 50, False` to `> 10, True`. The measured Sharpe (16.598) is unchanged. | The controls were not re-run, because a second run would append four duplicate rows to the experiment log. |
| C3 | `notebook/` renamed to `notebooks/` with `mv`. `task3.ipynb` itself was not opened or edited. | Group decision on Q3. |
| C4 | The module docstrings of `src/task3_data.py` and `src/task3_eval.py` now say `notebooks/task3.ipynb`. No code line changed. | Follows C3. Done before the frozen commit, so the freeze is not affected. |
| C5 | `.gitignore` now contains `data`, `__pycache__/`, `.DS_Store`. | Group request. |

Problem 1 is closed by C1. Updated control row:

| Control | Period | Expected | Actual | Pass |
|---|---|---|---|---|
| perfect foresight `P = Y`, Sharpe | 2011-2013 | above 10 (lowered from 50) | 16.598 | yes |

The experiment log was not touched: rows E001 to E004 never recorded a threshold or a pass flag.

After these changes the 12 pipeline tests were run again and all pass.

## Phase 0 freeze (Oct 9)

A group member committed Phase 0 as `cceeb15e720d5b637c2082d1da9a74c5f101496a` ("task3 phase 0: initial commit with frozen harness"). The agent wrote that hash to `results/FROZEN_COMMIT.txt` and filled it into the table at the top of this file (the only earlier line changed).

From this commit on, `src/task3_data.py` and `src/task3_eval.py` are frozen. First frozen-harness check, run right after the hash was recorded:

```
$ git diff $(cat results/FROZEN_COMMIT.txt) -- src/task3_data.py src/task3_eval.py
(empty)
```
