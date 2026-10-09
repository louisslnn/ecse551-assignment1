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

## Phase 1: features and leakage tests (Oct 9)

Started after the group replied "continue, phase 1 now". No model was fitted and no configuration was evaluated, so the experiment log is unchanged (4 rows).

### What was done

- `src/task3_features.py`: 24 return features and 13 sentiment features.
- `src/task3_train.py`: `fit_model()`, which asserts that every training row is dated strictly before the prediction date. Every Phase 2 fit will go through it.
- `src/task3_phase1.py`: writes `results/feature_table.csv` (statistics stop at 2010-12-31).
- `tests/test_leakage.py`: 6 tests, all passing (about 3 minutes).

### Decisions

| # | Decision | Reason |
|---|---|---|
| D7 | Windows: `rev_5` needs at least 3 of 5 values, `rev_21` at least 15 of 21, `mom_12_1` all 231. | `rev_5` then equals the group's 5-day reversal baseline. A mean with a few missing days is still comparable; a partial sum is not. Gaps are rare (5,231 cells in total up to 2010). |
| D8 | All 8 base return features get smoothed versions (half-lives 5 and 10), giving 24 return features. | Literal reading of the prompt. See Problem 4 for the six that are near-duplicates. |
| D9 | Smoothing is applied to the ranked feature AFTER missing ranks are set to 0, using the plain recursion `s[t] = a x[t] + (1 - a) s[t-1]`. | One rule for all features. A stock with no recent information drifts back to neutral instead of keeping a stale value forever, which matters for sentiment (90% missing). Cost: a stock that just joined the index has smoothed features shrunk toward 0 for its first days. |
| D10 | Smoothed features are not re-ranked. | The prompt says "EWMA over time of the ranked feature". They stay inside [-0.5, 0.5]. |
| D11 | A stock alone in its sector on a given day gets a sector-relative value of 0. Three GICS sector codes (18, 28, 46) contain one stock each. | Its value minus its own mean is 0, which ranks near the middle. |
| D12 | Sentiment from several calendar days landing on the same trading day is averaged, ignoring NaN. | The scores are intensities between 0 and 4, so a mean keeps Monday on the same scale as other days. A sum would make Monday about three times larger. |
| D13 | `sent_net` and `sent_act` are NaN when the stock had no messages. | Up to 2010 bullish and bearish are always missing together, and 0 is a real score (9,370 bullish zeros, 24,165 bearish zeros), so NaN is never treated as 0 before ranking. |
| D14 | `sent_flag` stays 0/1 and is not ranked. | It is already on a fixed scale. |
| D15 | Sentiment features are stored only for trading days from 2009-07-13 on. | Nothing exists before; saves memory. |
| D16 | Features are stored as float32. | 37 tables of 8577 x 1314 would need 3.3 GB in float64 on an 8 GB machine. A rank among 500 stocks needs three digits. The truncation test still passes at 1e-10. |
| D17 | Truncation test dates: one random day from each of 20 equal blocks of 1985-2010 (seed 42), which guarantees the spread. Plus 30 extra random days in 2009-07 to 2010 for sentiment only. | Sentiment exists for only 1.5 of the 26 years, so plain random dates would test it once at most. With the blocks, one of the 20 dates falls in the sentiment period; the 30 extra dates include Mondays. |
| D18 | Three tests beyond the two mandatory ones: the truncation check must fail on a deliberately leaky feature; the Friday/Saturday/Sunday example of Section 2.5 on a toy table; `mom_12_1` must ignore days t-20..t and day t-252. | A leakage test is only convincing if it is shown to catch a leak, and the two trickiest time shifts get an explicit check. |

### Sentiment mapping for duplicated permnos

606 sentiment columns. 18 belong to securities dropped as empty. 572 are ordinary permnos. Of the 19 duplicated permnos that remain, 16 have a sentiment column:

| Action | Permnos |
|---|---|
| mapped to the first security (only it has returns) | 48653, 77157, 79129 |
| mapped to the `_b` security (only it has returns) | 25232, 25582, 32707, 46703 |
| excluded, both have returns | 12431, 23536, 79237, 90537 |
| unused, neither has returns in the sentiment period | 12650, 19166, 21979, 45356, 79057 |

In total 579 columns are mapped to an equity id.

### Problems and disclosures

**Problem 3: the sentiment mapping looks at whether returns exist after 2010.** The group's rule says "has returns during the sentiment period", which runs to 2013. The function therefore checks whether each of the two securities has any return from 2009-07-10 on, test period included, and during exploration the agent printed the number of non-NaN return cells per duplicated permno for 2009-07 to 2010 and for 2011 to 2013. No return value, statistic or prediction from the test period was looked at. It changes one case: for 12431 the second security only has returns from 2011, so with data up to 2010 the sentiment would be mapped to the first one, and with the full period it is excluded. The effect is to remove information (31 member-days up to 2010), not to add any. The mapping is a fixed cleaning step, like dropping the 21 empty columns, and is the one input the truncation test does not recompute.

**Problem 4: six smoothed features are near-duplicates of their base feature.** Correlation with the base feature over member cells, 1990-2010: `mom_12_1_hl5` 0.98, `mom_12_1_hl10` 0.97, `vol_level_hl5` 0.99, `vol_level_hl10` 0.97, and the same 0.98 / 0.97 for the two smoothed `mom_12_1_sec`. The base features already move slowly, so smoothing them changes little. Lasso will pick between near-identical columns almost arbitrarily, which makes the "which coefficients are nonzero" table harder to read. Status: kept, question put to the group.

**Problem 5: sentiment is sparse.** In 2010 only 270 of 500 members have a sentiment column at all, and on an average day only 12% of members (about 60 stocks) have a usable score. The raw sentiment features are 0 for the other 88%. Smoothed versions are nonzero for 51%. This is below the 100 predictions per day the protocol asks for if a model used raw sentiment alone, so sentiment can only be an addition to the return features. Status: reported.

**Limitation: sector codes are static.** The metadata gives one GICS code per security for the whole of 1980-2013, and GICS did not exist before 1999. The sector-relative features in the early years therefore use a classification assigned later. This is a property of the data provided; it is not something the truncation test can detect.

**Fix during the phase:** the first run of the truncation test crashed for dates before 2009-07 because the truncated sentiment table was empty. The test now checks that no sentiment is dated before such a date and skips the recomputation. `task3_features.py` was not changed.

**Fix during the phase:** the first version of the feature table had an empty correlation column, because `.stack()` keeps NaN in pandas 3 (the same cause as Problem 2). It now uses a boolean mask.

### Feature table (statistics up to 2010-12-31)

Coverage is the share of members with a real, not filled, value, averaged from the first available date to 2010-12-31. For smoothed features it is the share of members whose value is not exactly 0. Fill fraction is 1 minus coverage. Full table with formulas: `results/feature_table.csv`.

| Feature | Lookback (trading days) | First available | Mean coverage | Fill fraction |
|---|---|---|---|---|
| `rev_1` | 1 | 1980-05-23 | 0.9978 | 0.0022 |
| `rev_5`, `rev_5_sec` | 5 | 1980-05-28 | 0.9978 | 0.0022 |
| `rev_21`, `rev_21_sec` | 21 | 1980-06-13 | 0.9975 | 0.0025 |
| `mom_12_1`, `mom_12_1_sec` | 252 | 1981-05-22 | 0.9869 | 0.0131 |
| `vol_level` | all past, 100-day burn-in | 1980-05-22 | 0.9979 | 0.0021 |
| smoothed return features (`_hl5`, `_hl10`) | base + EWMA | same as base | same as base (momentum: 0.9907) | same as base (momentum: 0.0093) |
| `sent_bull`, `sent_bear`, `sent_net`, `sent_act`, `sent_flag` | 1 to 4 calendar days | 2009-07-13 | 0.0986 (2010 only: 0.1197) | 0.9014 |
| smoothed sentiment features (`_hl5`, `_hl21`) | base + EWMA | 2009-07-13 | 0.437 (2010 only: 0.515) | 0.563 |

The expanding training window (default 5) therefore starts on 1981-05-22.

### Tests

`tests/test_task3_pipeline.py`: 12 passed. `tests/test_leakage.py`: 6 passed. The truncation test passes at a tolerance of 1e-10 on all 20 + 30 dates; on the one date inspected by hand (2000-06-15) the largest difference was exactly 0.0.

### Frozen-harness diff at the Phase 1 checkpoint

```
$ git diff $(cat results/FROZEN_COMMIT.txt) -- src/task3_data.py src/task3_eval.py
(empty)
```

## Phase 1 follow-up (Oct 9, after the checkpoint)

### Instructions received

1. Drop the six near-duplicate smoothed features and use the remaining 18 return features.
2. Decide the duplicate-permno sentiment mapping using data up to 2010-12-31 only, and freeze it.
3. Re-run all tests, regenerate `results/feature_table.csv`, and stop.

### Changes made

| # | Change | Reason |
|---|---|---|
| C6 | Removed `mom_12_1_hl5`, `mom_12_1_hl10`, `mom_12_1_sec_hl5`, `mom_12_1_sec_hl10`, `vol_level_hl5`, `vol_level_hl10`. `task3_features.py` now smooths only the features listed in `RETURN_SMOOTHED` (`rev_1`, `rev_5`, `rev_21`, `rev_5_sec`, `rev_21_sec`). 18 return features remain: 8 base and 10 smoothed. | Group decision. The six had a correlation of 0.97 to 0.99 with their base feature (Problem 4), because momentum and volatility already move slowly. Near-identical columns would make the Lasso choice between them arbitrary. This replaces decision D8 and closes Problem 4. |
| C7 | `sentiment_column_map()` now checks for returns between 2009-07-10 and 2010-12-31 only (`SENT_MAP_END`). | Group decision, so that no decision depends on test-period data. This replaces the rule described in Problem 3. |
| C8 | The mapping is frozen in `results/sentiment_column_map.json`. A new test asserts that the function returns exactly that mapping, and that the result is identical when every row after 2010-12-31 is removed from the returns. | So the mapping cannot drift, and so its independence from the test period is checked, not just stated. |

### Disclosure on the earlier mapping rule

During Phase 1 exploration the agent printed, for each of the 16 duplicated permnos that have a sentiment column, the number of non-NaN return cells and the number of member days in 2011 to 2013, next to the same counts for 2009-07 to 2010. These are availability counts only. No return value, statistic, plot or prediction from the test period was viewed or computed. The first version of the mapping rule used that availability ("any return from 2009-07-10 on"). The rule was then changed (C7) so that no decision depends on test-period data.

The change affects one permno. 12431: the first security has returns in 2009-07 to 2010 and the second does not, so the sentiment is now mapped to `12431` (it was excluded under the first rule). The other 15 are unchanged.

Known consequence, stated because the availability counts were already seen: the second security `12431_b` starts trading during the test period, so from then on the sentiment column of permno 12431 stays attached to `12431` although two securities share the permno. This is accepted as the price of a rule that uses development data only.

### Frozen sentiment mapping (data up to 2010-12-31)

| Action | Permnos |
|---|---|
| mapped to the first security (only it has returns) | 12431, 48653, 77157, 79129 |
| mapped to the `_b` security (only it has returns) | 25232, 25582, 32707, 46703 |
| excluded, both have returns | 23536, 79237, 90537 |
| unused, neither has returns | 12650, 19166, 21979, 45356, 79057 |

580 of the 606 sentiment columns are mapped to an equity id (572 ordinary permnos and 8 duplicated ones).

### Regenerated feature table

31 features: 18 return and 13 sentiment. The return rows are unchanged apart from the six removed. The sentiment coverage moved slightly because of 12431: 0.0988 up to 2010 (was 0.0986) and 0.1200 in 2010 (was 0.1197).

### Tests

All 19 tests pass: 12 in `tests/test_task3_pipeline.py` and 7 in `tests/test_leakage.py` (the 6 from the checkpoint plus the new mapping test).

### Frozen-harness diff

```
$ git diff $(cat results/FROZEN_COMMIT.txt) -- src/task3_data.py src/task3_eval.py
(empty)
```
