# Task 3 agent log

Running record of decisions, problems, and questions for ECSE 551 Assignment 1, Task 3.
Entries are added in order and never rewritten. Dates are in 2026.

## Commit hashes (made by a group member, not by the agent)

| What | Hash |
|---|---|
| Phase 0 frozen harness (`src/task3_data.py`, `src/task3_eval.py`) | `cceeb15e720d5b637c2082d1da9a74c5f101496a` |
| Phase 3 frozen finalists (`results/final_configs.json`) | `f817d04c028b0cec2b804a5cfde2a4ca89df9564` |

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

## Phase 2: walk-forward experiments on the development period (Oct 9)

Started after the group replied "continue". The Phase 1 commit is `e463be1`.

### What was done

- `src/task3_walkforward.py`: target, stacked panel of member rows, walk-forward loop, prediction smoothing.
- `src/task3_models.py`: Lasso, random forest and MLP behind one function, `fit_config()`. Every fit first calls the training-window assertion.
- `src/task3_experiments.py`: `score_and_log()` runs the frozen `evaluate()` and appends one row to the experiment log. It refuses any evaluation ending after 2010-12-31.
- `src/task3_phase2.py`, `src/task3_phase2b.py`: the stages (baselines, Lasso grid, random forest, MLP, variants and controls, sentiment, post-processing).
- `src/task3_phase2_report.py`: `results/phase2_results.csv` and six figures in `results/figures/`.
- `tests/test_walkforward.py`: 7 tests.

51 configurations were evaluated in Phase 2 (log rows E005 to E055). With the 4 Phase 0 rows the log has 55 rows. No run failed. The slowest configuration took 492 seconds.

| Group | Configurations | Log rows |
|---|---|---|
| Rule-based baselines | 2 | E005, E006 |
| Lasso, 6 penalties x 3 windows | 18 | E007 to E024 |
| Random forest, 3 depths x 2 leaf sizes | 6 | E025 to E030 |
| MLP, 2 sizes x 2 penalties | 4 | E031 to E034 |
| Lasso with raw target | 1 | E035 |
| Negative and positive control | 2 | E036, E037 |
| Sentiment study (2010 only) | 6 | E038 to E043 |
| Post-processing of the best Lasso, RF, MLP | 12 | E044 to E055 |

### Decisions

| # | Decision | Reason |
|---|---|---|
| D19 | The panel of training rows ends on 2010-12-31, and periods are defined by PREDICTION date. The prediction made on 2010-12-31 is scored by the frozen harness against the return of 2011-01-03, and the test period will start with the prediction made on 2011-01-03 (scored on the return of 2011-01-04). | This is how the frozen `evaluate()` slices dates. No return is used in both periods. The last refit in Phase 2 is on 2010-12-29, so the latest training row is 2010-12-28 and its target is the return of 2010-12-29. |
| D20 | Features are standardized at each refit with the mean and standard deviation of the training rows only, for all three models. | Smoothed features have a much smaller spread than raw ranks, and a Lasso penalty or an MLP treats small-scale features differently. Using training rows only avoids lookahead. Trees are unaffected; one code path for all models is simpler. |
| D21 | Lasso penalty grid: `alpha_max * 0.5 * 0.4**k`, k = 0..5, where `alpha_max` = 0.0331 is the smallest penalty that zeroes every coefficient on the first expanding training window (1981-05-22 to 1989-12-29, 1,085,244 rows). Values: 1.66e-2, 6.63e-3, 2.65e-3, 1.06e-3, 4.24e-4, 1.70e-4. Saved in `results/lasso_alpha_grid.json`. | Accepted default 4: chosen from the first training window only. |
| D22 | Random forest uses `max_features = 1/3`. | The prompt does not set it. The scikit-learn default for regression uses all features at every split, which is bagging, not a random forest. One third is the textbook value for regression. |
| D23 | One random subsample of 200,000 training rows per refit is shared by the six forests (seed 42). | Same data for every forest at a given refit, so differences come from the hyperparameters. |
| D24 | MLP: Adam with scikit-learn defaults, batch size 1024, one epoch = one `partial_fit` pass. Epochs are chosen between 1 and 30 by the squared error on the most recent 20% of the training days; a fresh network is then trained on all training days for that many epochs. | Accepted default 6. `early_stopping=True` is not used. The cap was first 10; it was raised to 30 before any development run, see the smoke-test disclosure below. |
| D25 | Models that share features, target, window and refit schedule are fitted in one pass over time, on the same training rows. `runtime_s` is the fit-and-predict time of that configuration, without the shared data preparation. | Identical results to separate runs, much less time. |
| D26 | Training window for the chosen Lasso, and for random forest and MLP: 3 years. | Among the 18 Lasso runs, 3 years has the highest development Sharpe (4.06, against 4.00 for 5 years and 3.09 for expanding) and the best last window (1.81, against 1.54 and 1.32). The gap between 3 and 5 years is small compared with the standard error (about 0.22 over 21 years), so this is a weak preference; 3 years is also the cheapest. Expanding is clearly worse, which suggests the relationship drifts over time. |
| D27 | Best of each family for post-processing: Lasso E012 (3 years, smallest penalty), random forest E029 (depth 8, leaf 200), MLP E033 (32 units, alpha 1e-4). | Highest development Sharpe in its family. E012 and E029 are also best or tied in at least three of the four windows. E033 and E034 are tied (5.16 and 5.14). |
| D28 | Sentiment study, 6 configurations on 2010 only: Lasso on return features alone and on return plus sentiment features, each at two penalties (grid index 5, the main choice, and index 2, about 16 times larger); Lasso on sentiment alone; random forest on return plus sentiment (depth 4, leaf 1000). All train on an expanding window from 2009-07-13. | The matched return-only runs on the same short history are the fair comparison for "does sentiment add anything". The larger penalty is near the noise level of a six-month sample. The forest is shallow because the first fit has only about 60,000 rows. |
| D29 | Prediction smoothing uses `ewm(halflife).mean()` on the stored development predictions and is scored with daily demeaning. "Demean off" variants are scored without it. | Post-processing only: no model is refitted. |
| D30 | Prediction tables are stored in `results/cache/` (885 MB). They are not meant to be committed. | Needed for post-processing and figures without refitting. |
| D31 | Figure colours: Lasso blue, random forest orange, MLP green, baselines grey, the same in every figure; fonts 14 pt or larger; PDF. The three colours were checked with a colour-blindness validator. | Prompt requirement on fonts and format; consistent identity across figures. |

### Smoke tests outside the development period (not logged as configurations)

Before the first logged run, the engine was tried on prediction dates 1986-01-01 to 1987-12-31, which are before the development period. Eight specs were run there (six Lasso penalties, one forest, one MLP) plus the two reversal rules, and their Sharpe ratios on 1986-1987 were printed. One setting was changed as a result: the MLP epoch cap went from 10 to 30, because the holdout error was still falling at epoch 10. That decision used the holdout error curve, not a Sharpe ratio. The sentiment and control code paths were dry-run on Oct to Dec 2009 and on 1985 without scoring anything. None of this touched 1990-2010 or the test period.

One observation from that smoke test that matters for interpretation: on 1986-1987 the one-day reversal rule had a Sharpe of -3.5 while the five-day reversal rule had +3.3. In this data yesterday's return tends to continue for one more day, and the reversal appears over several days.

### Results: top configurations (development period, demeaned predictions)

Full table: `results/phase2_results.csv`. Per-window values: `results/dev_windows.csv`.

| Row | Configuration | Sharpe 1990-2010 | 1990-94 | 1995-99 | 2000-04 | 2005-10 | Holding period (days) | Mean IC |
|---|---|---|---|---|---|---|---|---|
| E029 | RF, depth 8, leaf 200 | 5.50 | 9.52 | 6.25 | 3.95 | 2.55 | 1.14 | 0.033 |
| E030 | RF, depth 8, leaf 1000 | 5.19 | 9.21 | 5.77 | 3.71 | 2.41 | 1.21 | 0.033 |
| E033 | MLP, 32 units, alpha 1e-4 | 5.16 | 8.56 | 6.08 | 3.73 | 2.56 | 1.29 | 0.033 |
| E034 | MLP, 32 units, alpha 1e-2 | 5.14 | 8.61 | 6.05 | 3.66 | 2.57 | 1.30 | 0.033 |
| E027 | RF, depth 6, leaf 200 | 5.14 | 8.95 | 5.87 | 3.74 | 2.27 | 1.19 | 0.032 |
| E012 | Lasso, 3 years, alpha 1.70e-4 | 4.06 | 7.48 | 4.77 | 2.60 | 1.81 | 1.39 | 0.030 |
| E018 | Lasso, 5 years, alpha 1.70e-4 | 4.00 | 7.49 | 4.31 | 2.92 | 1.54 | 1.35 | 0.029 |
| E024 | Lasso, expanding, alpha 1.70e-4 | 3.09 | 6.75 | 2.31 | 2.26 | 1.32 | 1.05 | 0.016 |
| E005 | Baseline: 5-day reversal | 2.18 | 4.09 | 2.59 | 1.39 | 1.25 | 1.56 | 0.023 |
| E006 | Baseline: 12-1 momentum | 0.52 | 0.85 | 1.22 | -0.01 | 0.23 | 10.91 | 0.012 |

All configurations have at least 487 predictions per day.

### Results: variants of the best models

| Variant | Lasso E012 | RF E029 | MLP E033 |
|---|---|---|---|
| As selected (demeaned, no smoothing): Sharpe / holding period | 4.06 / 1.39 | 5.50 / 1.14 | 5.16 / 1.29 |
| Scored without daily demeaning | 4.06 | 5.12 | 3.73 |
| Predictions smoothed, half-life 2 | 2.86 / 3.55 | 3.79 / 2.95 | 3.55 / 3.30 |
| Predictions smoothed, half-life 5 | 2.08 / 6.27 | 2.53 / 5.23 | 2.45 / 5.73 |
| Predictions smoothed, half-life 10 | 1.61 / 9.92 | 1.76 / 8.29 | 1.75 / 8.87 |

Training the Lasso on the raw target instead of the demeaned one (E035) gives the same Sharpe, 4.06.

### Results: controls

| Row | Control | Sharpe 1990-2010 | Mean IC | Expected | Verdict |
|---|---|---|---|---|---|
| E037 | Positive: features shifted one day into the future | 68.28 | 0.995 | dramatic jump | pass: 17 times the real Lasso (4.06) |
| E036 | Negative: training targets shuffled within each day | -0.02 | 0.003 | near 0 | pass |

Note on the negative control: its Sharpe by window is 1.22, -0.05, 0.06, -0.67. These are not close to 0 individually. The reason is that the shuffled-target Lasso still outputs a random mix of real features, and features like reversal have a strong true effect, so whichever random sign the fit lands on is carried for many consecutive refits (each 3-year window overlaps the next by 92%). Over the whole period the signs cancel.

### Results: sentiment study (2010 only, one year)

| Row | Model | Features | Sharpe 2010 | Mean IC |
|---|---|---|---|---|
| E038 | Lasso, alpha index 5 | return | 0.43 | 0.003 |
| E039 | Lasso, alpha index 5 | return + sentiment | 0.32 | 0.003 |
| E040 | Lasso, alpha index 2 | return | 0.37 | 0.003 |
| E041 | Lasso, alpha index 2 | return + sentiment | 0.37 | 0.003 |
| E042 | Lasso, alpha index 5 | sentiment only | -0.90 | -0.002 |
| E043 | RF, depth 4, leaf 1000 | return + sentiment | 1.32 | 0.004 |

The standard error of a one-year Sharpe is about 1.0, so none of these differ from each other or from zero. Adding sentiment to the Lasso does not help at either penalty, and sentiment alone is negative. For context, the long-history models using return features only have these 2010 Sharpe ratios (slices of already logged P&L, not new configurations): Lasso E012 1.66, RF E029 3.67, MLP E033 2.43, 5-day reversal 1.21. The sentiment models, trained on 6 to 18 months, are all below them. Sentiment does not look viable as a finalist.

### What the models use

Lasso E012 over its 85 refits (standardized features): `rev_1_hl5` is nonzero in every refit with a negative coefficient (mean -0.066, by far the largest): a stock that rose over the last week or so is predicted to fall. `rev_1` is nonzero in 95% of refits with a mostly positive coefficient (mean +0.013): yesterday's move partly continues. `vol_level` is nonzero in 99% of refits, mostly negative. On average 12.7 of the 18 coefficients are nonzero (minimum 9, maximum 16). Random forest E029: the largest mean importances are `rev_1` (0.11), `rev_1_hl5` (0.10), `vol_level` (0.09) and `rev_1_hl10` (0.08).

### Problems and cautions

**Problem 6: the best configuration of every family sits on the edge of its grid.** Lasso: the smallest penalty is best in all three windows, and the Sharpe is still rising as the penalty falls (3.96 to 4.06 on the last step), so the Lasso is close to ordinary least squares. Random forest: the deepest trees with the smallest leaves are best. MLP: the larger network is best, and 67% of the refits chose the maximum of 30 epochs, so the networks are probably under-trained. The grids were fixed in advance by the prompt or by the accepted defaults and were not extended. Status: reported; extending a grid is a group decision.

**Caution 1: the development Sharpe ratios are far above the assignment's benchmark of about 2 for one-day holding periods.** The positive and negative controls, the truncation tests and the training-window assertion give no sign of leakage, and the 5-day reversal rule alone, with nothing fitted, already reaches 2.18. The harness has no transaction costs, and a holding period of 1.1 to 1.4 days means the whole book is replaced almost daily.

**Caution 2: performance decays steadily.** Every model and the reversal baseline lose roughly three quarters of their Sharpe between 1990-1994 and 2005-2010 (RF 9.5 to 2.6, Lasso 7.5 to 1.8, reversal 4.1 to 1.3). The development total is dominated by the 1990s. The last window is the most relevant guide for the test period.

**Caution 3: selection and multiple testing.** 51 configurations were scored on the same 21 years. The standard error of a 21-year Sharpe is about 0.22, so the ranking inside a family (for example RF 5.50 against 5.19, MLP 5.16 against 5.14) is not reliable. The ordering RF and MLP above Lasso above reversal holds in every window.

**Caution 4: the strategies are not market neutral in the beta sense.** With daily demeaning the P&L still has a correlation of 0.21 to 0.29 with the equal-weight market return.

**Caution 5: smoothing costs a lot.** Because the signal lives at the one-day horizon, a half-life of 5 days halves the Sharpe for a holding period of about 5 to 6 days.

### Proposed finalists

| # | Row | Configuration | Why |
|---|---|---|---|
| 1 | E012 | Lasso, 18 return features, 3-year window, alpha 1.70e-4, refit every 63 days | Best Lasso overall and in the most recent window; the simplest model and the one whose coefficients can be read. |
| 2 | E029 | Random forest, depth 8, leaf 200, 3-year window, refit every 252 days | Best configuration overall and best in every window among the forests. |
| 3 | E033 | MLP, 32 units, alpha 1e-4, 3-year window, refit every 252 days | Best MLP; tied with E034, chosen for the higher total. |
| 4 (optional) | E054 | MLP E033 with predictions smoothed, half-life 5 | A slow variant (holding period 5.7 days, Sharpe 2.45) to compare against the assignment's long-holding-period benchmark of 0.6. Chosen over the forest with the same smoothing (E050, Sharpe 2.53) because it is steadier in the last window (1.46 against 0.79). |

No sentiment finalist is proposed.

### Figures (`results/figures/`, PDF, 14 pt fonts)

`phase2_lasso_grid.pdf`, `phase2_sharpe_by_window.pdf`, `phase2_sharpe_vs_holding.pdf`, `phase2_smoothing_tradeoff.pdf`, `phase2_controls.pdf`, `phase2_cumulative_pnl.pdf`.

### Tests

All 26 tests pass: 12 in `tests/test_task3_pipeline.py`, 7 in `tests/test_leakage.py`, 7 in `tests/test_walkforward.py`. The new file checks, on small synthetic data, that every refit trains on days strictly before the refit date, that predictions up to a date do not change when later data is removed, that the shuffle of the negative control keeps each day's values, and that smoothing only looks backwards. It also reads the refit records of the real Phase 2 runs (2,026 refits) and asserts that the last training day is before the refit date in every one.

### Frozen-harness diff at the Phase 2 checkpoint

```
$ git diff $(cat results/FROZEN_COMMIT.txt) -- src/task3_data.py src/task3_eval.py
(empty)
```

## Phase 2 follow-up: grid extension, delay diagnostic, freeze (Oct 9, after the checkpoint)

### Instructions received

Finalists E012, E029, E033 and E054 approved in principle. Before writing `final_configs.json`, on the development period only and with every run logged:

1. Grid extension, at most 5 configurations: Lasso with alpha = 0 on the 3-year window; random forest depth 10 and 12 with leaf 200; MLP 32 units with epoch cap 60; MLP 64 units. Rule fixed by the group before the runs: replace a finalist only if the extension beats it by more than 0.22 in development Sharpe; otherwise keep it and report a plateau.
2. Delayed-execution diagnostic: evaluate `P.shift(1)` on 1990-2010 for the finalists and the reversal baseline, from the cached predictions. Diagnostic only, not a selection criterion.
3. Add `results/cache/` to `.gitignore`.
4. Write the expected range of test Sharpe for each finalist, based on 2005-2010 performance.
5. Write `final_configs.json`, stop, and run nothing on the test period until the group sends the commit hash.

### Changes to the code

| # | Change | Reason |
|---|---|---|
| C9 | `task3_models.py`: new spec `{"model": "ols"}` (scikit-learn `LinearRegression`), and the MLP epoch cap can be set per spec with `max_epochs` (default still 30). | Needed for the extension. `Lasso(alpha=0)` is not used because scikit-learn advises against it; least squares is the same model. |
| C10 | `task3_phase2b.py`: stages `extension` and `delay`; the dict `FINALISTS`. `task3_phase2_report.py`: the window and cumulative P&L figures now show the final finalists. | Requested runs. |
| C11 | `.gitignore`: added `results/cache/`. | Requested. |
| C12 | New `src/task3_finalists.py`, which writes `results/final_configs.json`. | So the frozen file is generated by code and can be checked against the log. |

### Grid extension results (rows E056 to E060)

| Row | Extension | Sharpe 1990-2010 | 1990-94 | 1995-99 | 2000-04 | 2005-10 | Compared with | Difference | Rule (> 0.22) |
|---|---|---|---|---|---|---|---|---|---|
| E056 | Lasso alpha = 0 (least squares), 3 years | 4.199 | 7.65 | 4.93 | 2.69 | 2.01 | E012: 4.064 | +0.135 | keep E012 |
| E057 | RF depth 10, leaf 200 | 5.682 | 9.96 | 6.41 | 4.13 | 2.56 | E029: 5.500 | +0.182 | keep |
| E058 | RF depth 12, leaf 200 | 5.784 | 10.19 | 6.49 | 4.14 | 2.70 | E029: 5.500 | +0.284 | replace |
| E059 | MLP 32 units, epoch cap 60 | 5.394 | 9.20 | 6.33 | 3.86 | 2.69 | E033: 5.159 | +0.235 | replace |
| E060 | MLP 64 units, epoch cap 30 | 5.387 | 8.94 | 6.78 | 3.71 | 2.61 | E033: 5.159 | +0.228 | replace |

Slowest extension run: 484 seconds.

Applying the rule:

- **Lasso: E012 stays.** Removing the penalty completely adds 0.135, less than 0.22. Performance plateaus as the penalty goes to zero.
- **Random forest: E029 is replaced by E058 (depth 12).** The gain flattens (depth 8 to 10: +0.18, depth 10 to 12: +0.10), which is expected because a leaf of 200 rows already limits how deep the trees can usefully grow.
- **MLP: E033 is replaced by E059 (32 units, cap 60).** Two extensions pass the rule and are tied (5.394 and 5.387). E059 is taken because it has the higher total and is better in three of the four windows. It is the same network as E033 trained longer. With the cap at 60, 41% of the refits still stop at the cap (67% at cap 30); the mean chosen number of epochs is 53.
- **Slow variant: E054 stays as approved** (the predictions of E033 smoothed with a half-life of 5 days). No extension run was a smoothed model, so nothing beat it under the rule. Consequence: the slow finalist is built on the 30-epoch network E033, not on the new MLP finalist E059, so Phase 3 must also carry E033 forward as the base of the slow variant. Deriving the slow variant from E059 instead would need one more development run that the group did not ask for. Status: reported as an open choice for the group before the freeze is committed.

Both replacements beat the threshold by a small margin (0.284 and 0.235 against 0.22, where 0.22 is one standard error of a 21-year Sharpe), and all five extensions improve on their starting point. The honest reading is a slow, continuing improvement with more capacity, not a newly found optimum.

### Delayed-execution diagnostic (rows E061 to E065, not eligible for selection)

Every prediction is used one day late: the prediction made at the close of day t-1 is held over day t+1.

| Finalist | On time: Sharpe 1990-2010 | One day late | 1990-94 | 1995-99 | 2000-04 | 2005-10 | Share kept |
|---|---|---|---|---|---|---|---|
| Lasso E012 | 4.06 | 2.70 | 3.53 | 3.88 | 2.13 | 1.16 | 67% |
| RF E058 | 5.78 | 3.81 | 5.07 | 4.67 | 3.46 | 1.96 | 66% |
| MLP E059 | 5.39 | 3.55 | 5.23 | 4.57 | 2.45 | 2.19 | 66% |
| MLP slow E054 | 2.45 | 1.70 | 2.21 | 2.48 | 1.15 | 1.20 | 70% |
| 5-day reversal E005 | 2.18 | 2.29 | 5.27 | 1.97 | 1.41 | 1.34 | 105% |

The fitted models lose about a third of their Sharpe when execution is one day late, most of it in 1990-1994. The reversal rule loses nothing, which fits the earlier observation that yesterday's return tends to continue for a day: waiting a day removes the part of the rule that fights that continuation. A sizeable part of what the fitted models add over the rule therefore depends on trading at the very close on which the prediction is made.

### Totals before the freeze

65 rows in the experiment log: 4 Phase 0 harness checks and 61 Phase 2 rows (51 at the checkpoint, 5 grid extensions, 5 delay diagnostics). This is one more than the "about 60" budget of the prompt, because of the diagnostics the group asked for. No run failed.

### Expected test-period Sharpe, written before any test-period run

Basis, as asked: the 2005-2010 Sharpe of each finalist. Two reference points are used.

- **No further decay:** the test Sharpe equals the 2005-2010 Sharpe.
- **Decay continues:** between consecutive development windows the Sharpe of the finalists and of the reversal rule was multiplied by 0.64 on median (15 ratios, from 0.49 to 0.92). One more step of 0.65 is applied.

The standard error of a 3-year Sharpe is about 0.58, so the range is widened by one standard error on each side.

| Finalist | Sharpe 2005-2010 | Central expectation (decay continues .. no decay) | Expected range (central +/- 1 standard error) |
|---|---|---|---|
| Lasso E012 | 1.81 | 1.2 to 1.8 | 0.6 to 2.4 |
| RF E058 | 2.70 | 1.75 to 2.7 | 1.2 to 3.3 |
| MLP E059 | 2.69 | 1.75 to 2.7 | 1.2 to 3.3 |
| MLP slow E054 | 1.46 | 0.95 to 1.45 | 0.4 to 2.0 |
| Baseline 5-day reversal | 1.25 | 0.8 to 1.25 | 0.2 to 1.8 |
| Baseline 12-1 momentum | 0.23 | about 0.2 | -0.4 to 0.8 |

What would change the interpretation:

- A result below the range would mean the decay accelerated or the 2011-2013 regime differs. It would not by itself indicate a bug.
- A result above the range (for example a forest above 3.3) would be surprising given 20 years of decay and should be checked before it is believed.
- The year-by-year values inside 2005-2010 are very uneven (Lasso from 0.35 in 2008 to 4.29 in 2006; forest from 1.10 to 5.17), so a single 3-year number can land anywhere in these ranges without contradicting the development results.
- Ranking between the forest and the MLP on the test period carries no information: they are tied on 2005-2010 (2.70 and 2.69).

**Disclosure: this expectation is not fully blind for the reversal baseline.** The group's notebook, which the agent read in Phase 0 as the source of truth, contains the 5-day reversal Sharpe by year for 2011 (1.55), 2012 (-0.32) and 2013 (0.85), and 0.83 for the window 2010-2013. The agent has therefore seen that the reversal rule kept decaying in the test period. The ranges above were computed from 2005-2010 numbers by the stated formula, but the choice to treat "decay continues" as a serious scenario cannot be shown to be independent of that knowledge. No test-period result of any fitted model has been computed or seen.

### `results/final_configs.json`

Written by `src/task3_finalists.py`. Contents: the two periods, the walk-forward rules (panel start, refit dates counted from the first trading day of 1990 and continued unchanged through 2013, 3-year window, standardization, target, seed, subsample and smoothing rules), the 18 feature names, the four finalists with their exact hyperparameters, and the two baseline rules. Checked: the feature list equals the column order of the development panel, and each finalist's settings equal the log row it comes from.

| Finalist | From row | Model | Window | Refit | Smoothing |
|---|---|---|---|---|---|
| `lasso` | E012 | Lasso, alpha 1.697e-4 | 3 years | 63 days | none |
| `rf` | E058 | Random forest, depth 12, leaf 200, 100 trees, 200,000 rows per fit | 3 years | 252 days | none |
| `mlp` | E059 | MLP, 32 units, alpha 1e-4, epochs chosen in 1..60 | 3 years | 252 days | none |
| `mlp_slow` | E054 | MLP, 32 units, alpha 1e-4, epochs chosen in 1..30 (network of E033) | 3 years | 252 days | half-life 5 |

Nothing has been run on the test period. Phase 3 starts only after the group sends the hash of the commit that contains this file.

## Slow finalist changed before the freeze (Oct 9)

The Phase 2 commit made by the group is `ecf4534`.

### Instruction received

Replace `mlp_slow` with the predictions of E059 smoothed at half-life 5, computed from the cached development predictions. The choice is fixed regardless of its development Sharpe. Log the run and its expected test range by the same method as the others, update `final_configs.json`, and give the files for the freeze commit. In the Phase 3 output, also include the full development window Sharpes and holding periods for E058, E059 and the new slow variant.

### What was done

| # | Change | Reason |
|---|---|---|
| C13 | New stage `slow` in `task3_phase2b.py`. It smooths the cached development predictions of E059 with a half-life of 5 days and scores them with daily demeaning. Logged as row E066. | Group decision. This closes the open choice reported at the previous checkpoint: the slow finalist is now built on the MLP finalist itself, so Phase 3 no longer needs the 30-epoch network E033. |
| C14 | `task3_finalists.py` and `results/final_configs.json`: `mlp_slow` now points to E066 (network of E059, epochs chosen in 1..60, smoothing half-life 5). | Same. |

The earlier slow candidate E054 (E033 smoothed) is no longer a finalist. Its rows E054 and E064 stay in the log.

### Row E066: MLP E059 with predictions smoothed, half-life 5

| Sharpe 1990-2010 | 1990-94 | 1995-99 | 2000-04 | 2005-10 | Holding period (days) | Mean IC | Min predictions per day |
|---|---|---|---|---|---|---|---|
| 2.53 | 4.02 | 3.46 | 1.71 | 1.43 | 5.56 | 0.019 | 493 |

For comparison, the candidate it replaces (E054): 2.45, with 3.67, 3.35, 1.66, 1.46 by window and a holding period of 5.73 days. The two are equivalent within noise; the choice was fixed before this number was computed. No delayed-execution diagnostic was run for E066, because it was not asked for.

### Expected test-period Sharpe for the new slow finalist (same method)

| Finalist | Sharpe 2005-2010 | Central expectation (decay continues .. no decay) | Expected range (central +/- 1 standard error) |
|---|---|---|---|
| MLP slow E066 | 1.43 | 0.93 to 1.43 | 0.35 to 2.0 |

This replaces the "MLP slow E054" line of the earlier table. The other lines, and the disclosure about the reversal baseline, are unchanged.

### Final frozen configuration

| Finalist | From row | Model | Window | Refit | Smoothing | Dev Sharpe | Dev 2005-10 | Holding (days) |
|---|---|---|---|---|---|---|---|---|
| `lasso` | E012 | Lasso, alpha 1.697e-4 | 3 years | 63 days | none | 4.06 | 1.81 | 1.39 |
| `rf` | E058 | Random forest, depth 12, leaf 200, 100 trees, 200,000 rows per fit | 3 years | 252 days | none | 5.78 | 2.70 | 1.09 |
| `mlp` | E059 | MLP, 32 units, alpha 1e-4, epochs chosen in 1..60 | 3 years | 252 days | none | 5.39 | 2.69 | 1.27 |
| `mlp_slow` | E066 | the predictions of `mlp`, smoothed | 3 years | 252 days | half-life 5 | 2.53 | 1.43 | 5.56 |

66 rows are in the experiment log at the freeze: 4 Phase 0 checks and 62 Phase 2 rows. Nothing has been run on the test period.

### To do in Phase 3 (requested)

The Phase 3 output must include the development Sharpe by window and the holding period for E058, E059 and E066 next to the test results.

### Frozen-harness diff before the freeze

```
$ git diff $(cat results/FROZEN_COMMIT.txt) -- src/task3_data.py src/task3_eval.py
(empty)
```

## Phase 3: freeze recorded, then one test evaluation (Oct 9)

### Freeze

The group committed the frozen finalists as `f817d04c028b0cec2b804a5cfde2a4ca89df9564` ("task3: freeze finalists") and told the agent to start Phase 3. Before running anything the agent checked that this commit contains `results/final_configs.json` and `src/task3_finalists.py`, that the working copies of both are identical to the committed versions, that nothing else is modified apart from the group's own uncommitted edits in `report/`, and that `results/test_results.csv` does not exist yet. The hash was filled into the table at the top of this file. The commit before it, `3cb48b5`, holds the slow-finalist run E066.

### What was done

- `src/task3_phase3.py`, stage `predict`: walk-forward of the three fitted finalists from 1990 to 2013-12-31 with the frozen rules, the slow finalist derived from the MLP predictions, and the two baseline rules. Predictions were saved. Nothing after 2010 was evaluated in this stage.
- Reproduction check before touching the test period: the development Sharpe of all six was recomputed from the new predictions and compared with the frozen values. All six match to four decimals (4.0640, 5.7837, 5.3944, 2.5310, 2.1757, 0.5196).
- Stage `test`: one call of the frozen `evaluate()` per finalist and baseline on 2011-01-01 to 2013-12-31, with daily demeaning. The stage refuses to run if `results/test_results.csv` exists. It was run once, at 20:37 on Oct 9.
- `src/task3_phase3_report.py`: seven figures.
- `src/task3_truncation_check.py`: an end-to-end lookahead check on real data (see below).
- `results/FINALISTS_COMMIT.txt` holds the freeze hash, so a test can check that `final_configs.json` is unchanged.
- No configuration was changed after the test results were seen.

### Decisions

| # | Decision | Reason |
|---|---|---|
| D32 | The test evaluations are logged as six rows with `phase = 3` (E067 to E072). Their `dev_sharpe` columns hold development values; `mean_ic`, `ic_tstat`, `holding_period`, `corr_market` and `min_preds_per_day` hold TEST-period values, and the test Sharpe is in `notes`. The complete numbers are in `results/test_results.csv`. | The log has no test columns, and every evaluation must have a row. |
| D33 | Prediction and evaluation are two separate stages. | A crash during the walk-forward cannot leave the test period half evaluated, and the development reproduction check runs before any test number exists. |
| D34 | The walk-forward for 1990-2013 is one run with the refit dates counted from the first trading day of 1990, exactly as in development. | Stated in `final_configs.json`. It makes the development predictions reproduce exactly and avoids choosing a new refit calendar for the test period. |
| D35 | In the cumulative P&L figures each series is divided by its own daily standard deviation. | The models' predictions have different scales, so raw P&L is not comparable; after scaling, the slope of a line is proportional to its Sharpe. |

### Test results (2011-01-01 to 2013-12-31, 753 days, evaluated once)

| Finalist | From row | Test Sharpe | 2011 | 2012 | 2013 | Mean IC | IC t-stat | Holding period (days) | Corr. with market | Expected range | Inside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Lasso | E012 | 1.07 | 1.31 | 0.64 | 1.22 | 0.014 | 2.67 | 1.48 | 0.38 | 0.6 to 2.4 | yes |
| Random forest | E058 | 3.65 | 3.20 | 3.69 | 4.15 | 0.018 | 5.55 | 0.94 | 0.39 | 1.2 to 3.3 | **no, above** |
| MLP | E059 | 3.01 | 2.02 | 3.22 | 4.09 | 0.018 | 5.65 | 1.04 | 0.31 | 1.2 to 3.3 | yes |
| MLP, smoothed (half-life 5) | E066 | 1.40 | 0.11 | 1.54 | 2.85 | 0.007 | 1.84 | 4.51 | 0.40 | 0.35 to 2.0 | yes |
| Baseline: 5-day reversal | E005 | 0.73 | 1.55 | -0.32 | 0.85 | 0.008 | 1.87 | 1.61 | 0.15 | 0.2 to 1.8 | yes |
| Baseline: 12-1 momentum | E006 | 0.79 | 0.75 | 0.48 | 1.15 | 0.019 | 3.21 | 11.44 | 0.12 | -0.4 to 0.8 | yes |

At least 491 predictions per day for every row. The standard error of a 3-year Sharpe is about 0.58. The yearly values of the reversal baseline are identical to the ones in the group's notebook (1.55, -0.32, 0.85), which confirms that the harness scores the test period the same way as in Phase 0.

### Development values next to the test values (requested by the group)

| Finalist | Row | Dev Sharpe 1990-2010 | 1990-94 | 1995-99 | 2000-04 | 2005-10 | Dev holding period | Test Sharpe 2011-13 | Test holding period |
|---|---|---|---|---|---|---|---|---|---|
| Lasso | E012 | 4.06 | 7.48 | 4.77 | 2.60 | 1.81 | 1.39 | 1.07 | 1.48 |
| Random forest | E058 | 5.78 | 10.19 | 6.49 | 4.14 | 2.70 | 1.09 | 3.65 | 0.94 |
| MLP | E059 | 5.39 | 9.20 | 6.33 | 3.86 | 2.69 | 1.27 | 3.01 | 1.04 |
| MLP, smoothed | E066 | 2.53 | 4.02 | 3.46 | 1.71 | 1.43 | 5.56 | 1.40 | 4.51 |
| 5-day reversal | E005 | 2.18 | 4.09 | 2.59 | 1.39 | 1.25 | 1.56 | 0.73 | 1.61 |
| 12-1 momentum | E006 | 0.52 | 0.85 | 1.22 | -0.01 | 0.23 | 10.91 | 0.79 | 11.44 |

### The random forest is above the range written down before the test

Before the freeze this log said that a forest above 3.3 "would be surprising" and "should be checked before it is believed". The test value is 3.65. Checks made after seeing it, none of which changes a configuration or evaluates the test period again:

| Check | Result |
|---|---|
| Development Sharpe reproduced from the Phase 3 predictions | exact for all six |
| Baseline test values against the group's notebook | identical by year |
| Refit records of the 1990-2013 run (97 Lasso, 25 forest, 25 MLP refits): last training day before the refit date | true for all (test added) |
| `task3_truncation_check.py lasso 2012-06-29`: all raw data after 2012-06-29 removed, features, targets and walk-forward recomputed from scratch, predictions for 1990 to 2012-06-29 compared with the saved ones | identical, largest difference 0 over 5,672 days |
| `task3_truncation_check.py rf 2012-06-29`: the same for the forest | identical, largest difference 0 over 5,672 days |
| Frozen harness and `final_configs.json` | unchanged |

So the forest's predictions through the first half of the test period are provably independent of any later data. No sign of a bug was found.

A likely reason the range was too low, found after the fact and therefore weaker than a prediction: the "no decay" reference was the pooled 2005-2010 Sharpe (2.70), and that number is pulled down by 2008. The forest's daily P&L had a standard deviation of 1.86 in 2008 against 1.40 for 2005-2010 as a whole and 1.00 in the test period. Its yearly development Sharpes were 3.14, 5.17, 2.13, 1.10, 2.69 and 3.92 for 2005 to 2010 (mean 3.02), and 3.18 for 2009-2010 pooled. The test value of 3.65 is in line with the calm years. Its rank IC did not rise: 0.020 in 2005-2010 and 0.018 in the test period. The forest did not predict better in 2011-2013; its P&L was steadier.

How unusual is it: 3.65 is 0.35 above the top of the range, and 1.6 standard errors above the no-decay reference of 2.70.

### Interpretation

- **All four finalists beat both baselines on the test period.** The forest (3.65) and the MLP (3.01) are far above the assignment's benchmark of about 2 for one-day holding periods, and their IC t-statistics (5.6) are large for three years. The Lasso (1.07) is positive but only 0.34 above the reversal rule, about 0.6 standard errors.
- **The decay stopped for the non-linear models but not for the linear ones.** From 2005-2010 to the test period the Lasso went from 1.81 to 1.07 and the reversal rule from 1.25 to 0.73, both a factor of about 0.6, which is the "decay continues" scenario. The forest and the MLP went from 2.70 to 3.65 and from 2.69 to 3.01. The gap between linear and non-linear models, about 0.9 in 2005-2010, is about 2 to 2.6 on the test period. This is one 3-year sample.
- **Forest against MLP:** 3.65 against 3.01 is a difference of about one standard error and is not evidence that one is better.
- **The slow finalist** (holding period 4.5 days, Sharpe 1.40) is above the assignment's 0.6 benchmark for long holding periods, but almost all of it comes from 2012 and 2013 (0.11 in 2011), and its IC t-statistic is 1.84.
- **Momentum** has the highest test IC (0.019) but a Sharpe of only 0.79.

### Limitations that apply to the test numbers

1. **No transaction costs.** The forest's holding period on the test period is 0.94 days: the whole book is replaced every day. The harness has no cost model.
2. **Execution at the close.** On the development period, using predictions one day late cost the fitted models about a third of their Sharpe (rows E061 to E064). That diagnostic was NOT run on the test period, because it would be a second evaluation of the test period. Running it is a group decision and would have to be disclosed.
3. **Not market neutral.** The correlation of the P&L with the equal-weight market return is 0.31 to 0.40 on the test period for the finalists, higher than in development (0.24 to 0.29).
4. **One test period of three years.** Standard error about 0.58 per Sharpe.
5. **72 rows in the experiment log**, of which 62 development configurations and diagnostics were tried before the freeze. Two finalists were chosen at the edge of an extended grid by a margin of about one standard error.
6. **Static sector codes** and a sentiment mapping fixed with data up to 2010 (see Phase 1). Sentiment is not used by any finalist.

### Figures (`results/figures/`, PDF, 14 pt fonts)

| File | Content |
|---|---|
| `phase3_test_cumulative_pnl.pdf` | cumulative P&L on the test period, finalists and baselines |
| `phase3_test_cumulative_ic.pdf` | cumulative rank IC on the test period |
| `phase3_sharpe_by_window.pdf` | Sharpe in the four development windows and the test period |
| `phase3_sharpe_vs_holding.pdf` | all development runs in grey; each finalist from development (hollow) to test (filled) |
| `phase3_lasso_nonzero_frequency.pdf` | share of the 97 Lasso refits in which each coefficient is nonzero |
| `phase3_rf_feature_importance.pdf` | mean feature importance over the 25 forest refits |
| `phase3_full_cumulative_pnl.pdf` | cumulative P&L 1990-2013, development period labelled as used for selection, test period shaded |

### Problem 7: one unexplained failure of a leakage test (not reproduced)

After the test evaluation, one run of the full test suite reported `1 failed, 27 passed`. The failing test was `tests/test_leakage.py::test_truncation_all_features_20_dates`, which had passed in every earlier run. The agent had kept only the last two lines of that run's output, so the assertion message (which names the date and the feature) is lost. That was a mistake in how the run was recorded.

What was done to reproduce it, with no code change in between:

| Attempt | Result |
|---|---|
| The failing test alone | passed |
| The full suite, twice more | 28 passed, both times |
| `build_return_features` recomputed 12 times on the full data and compared bit for bit with a first run | all 18 tables identical every time |
| The logic of the failing test repeated 6 times (120 truncated recomputations of the return features, plus sentiment where it exists), printing every mismatch | 0 mismatches |
| `np.log(sig)` on 60 truncated tables against the full table (the one feature that uses a non-exact math function) | bit-identical |

Status: **not explained and not reproduced**. `task3_features.py` and `tests/test_leakage.py` have not changed since the Phase 1 commit. The code is deterministic as far as these checks can tell, and independent evidence agrees: the Phase 3 run reproduced all six development Sharpes to four decimals, and the two real-data truncation checks reproduced 5,672 days of Lasso and forest predictions with a largest difference of exactly 0. The agent's best guess is a transient problem of the machine during that run (it has 8 GB of memory and was under heavy memory pressure during Phase 3), but there is no evidence for that guess. If the test fails again, the full output must be kept.

### Tests at the Phase 3 checkpoint

28 tests: 12 in `tests/test_task3_pipeline.py`, 7 in `tests/test_leakage.py`, 9 in `tests/test_walkforward.py` (two new: the refit records of the 1990-2013 run, and `final_configs.json` unchanged since the freeze commit). Last two full runs: 28 passed. One earlier run had the unexplained failure described in Problem 7.

### Frozen files at the Phase 3 checkpoint

```
$ git diff $(cat results/FROZEN_COMMIT.txt) -- src/task3_data.py src/task3_eval.py
(empty)
$ git diff $(cat results/FINALISTS_COMMIT.txt) -- results/final_configs.json
(empty)
```
