# Task 3 Agent Prompt (ECSE 551, Assignment 1)

You are working in our group's repository for ECSE 551 Assignment 1, Task 3: predicting next-day volatility-adjusted returns of S&P 500 stocks from historical returns and sentiment signals.

Your job is to run a careful, honest experimental study under a fixed protocol. You are NOT optimizing for the highest possible number. A negative result is acceptable and must be reported honestly. Every member of our group will be questioned individually in an oral demo about every line of code and every result, so clarity and correctness matter more than cleverness.

Before doing anything else, read this entire file, then reply with a short plan and any questions. Do not start Phase 0 until we answer.

---

## 0. Ground rules (apply throughout, no exceptions)

1. **Frozen harness.** After Phase 0, `src/task3_data.py` and `src/task3_eval.py` are frozen. You may import them but never edit them. At every checkpoint, run `git diff $(cat results/FROZEN_COMMIT.txt) -- src/task3_data.py src/task3_eval.py` and show us the output. It must be empty. If you believe the harness has a bug, stop and tell us; do not fix it yourself.

2. **Test set lockout.** The test period is 2011-01-01 to 2013-12-31. During Phases 1 and 2 you must never compute any metric, plot, or statistic on test-period predictions or test-period returns. All model selection and hyperparameter tuning uses the development period only. The test period is evaluated exactly once, in Phase 3, only for finalists we approve, and only after the frozen configurations have been committed.

3. **No lookahead.** A prediction made at the close of day t may use only information available at the close of day t. Concretely:
   - Features at row t may only use data from rows up to and including t.
   - The training target for row s is the clipped vol-adjusted return on day s+1. That return is only known at the close of s+1. Therefore a model used to predict on day t may only be trained on rows s <= t-1. Enforce this with an `assert` in the training code, not just a comment.
   - Sentiment follows the timing rule in Section 2.5.
   - Leakage tests in Phases 1 and 2 are mandatory.

4. **Experiment log.** Every configuration you evaluate (including failed runs, controls, and anything you abandon) gets one row appended to `results/experiment_log.csv`. Never delete or edit rows. Columns: `exp_id, timestamp, phase, model, features, hyperparams, target_variant, demean, pred_smoothing_halflife, refit_every, train_window, dev_sharpe, dev_sharpe_1990_1999, dev_sharpe_2000_2010, mean_ic, ic_tstat, holding_period, corr_market, min_preds_per_day, runtime_s, notes`. We will report the total number of configurations tried, because that is the multiple-testing context for any result.

5. **Scope.** Only the model families the assignment asks for: sparse linear regression (Lasso), a basic MLP, and a random forest, plus the simple rule-based baselines listed below. scikit-learn implementations are allowed for Task 3. No gradient boosting libraries, no deep learning frameworks, no external data.

6. **Code we can explain.** Plain functions with short docstrings that say what the function does and why. Comments on every line that involves a time shift (`shift`, `rolling`, `ewm`, date slicing), stating which days the value depends on. No class hierarchies, decorators, or clever pandas one-liners where a clear three-line version exists. Keep each module under about 300 lines. Fixed random seed 42 everywhere. The final submission is a single notebook, so modules must be self-contained enough to paste into notebook cells.

7. **Checkpoints.** At the end of each phase, stop: summarize what you did in plain language, show results, list any decisions you made and why, show the frozen-harness diff, then end your turn and wait for us to reply "continue". Do not start the next phase on your own.

8. **Git hygiene.** Commit locally at the end of each phase with message `task3 phase N: <summary>`. Never commit anything in `data/`. Never push, force, rebase, or rewrite history.

9. **Boundaries.** Do not modify files for Tasks 1 and 2, the `report/` folder, or `notebooks/task3.ipynb`. Use the current Python environment; ask before installing any package.

10. **When in doubt, stop and ask.** If a rule blocks something you think is necessary, explain the problem instead of working around it.

11. **Running record.** Maintain `results/AGENT_LOG.md` throughout: each decision with its reason, each problem with its fix, and each question you asked us with our answer. This file is cited in our report.

---

## 1. Repository layout

```
data/                    CSVs (gitignored, read-only for you)
notebooks/task3.ipynb    our verified exploration, source of truth for Section 2 (do not edit)
src/                     modules (you create task3_*.py here)
tests/                   tests (you create)
results/                 logs, tables, figures (you create)
AGENT_PROMPT.md          this file
```

---

## 2. Established pipeline (built and verified by us)

`notebooks/task3.ipynb` contains our verified implementation of everything in this section. When extracting it in Phase 0, copy the logic exactly.

### 2.1 Data facts

- `eq_returns.csv`: rows are trading days (8577 days, 1980-01-02 to 2013-12-31), columns are permnos plus a `date` column. Values are daily excess returns as decimals.
- `eq_indicator.csv`: same shape. 1 if the stock is an S&P 500 member that day, else 0. About 500 members per day (min 489).
- `eq_stock_metadata.csv`: `permno, cname, gics` (8-digit GICS; sector = first 2 digits).
- `psych_bullish.csv`, `psych_bearish.csv`: rows are CALENDAR days (weekends included) starting 2009-07-10. Columns are permnos (606 stocks). NaN most likely means no messages that day, which is different from a score of 0.
- `psych_stock_metadata.csv`: `permno, ticker` (not needed for joining).

### 2.2 Cleaning (already decided, reproduce exactly)

- 21 permnos appear twice as column headers. pandas renames the second copy `XXXX.1`. We rename it `XXXX_b` and treat it as a distinct security, because the two series are never identical (correlation 0.06 to 0.47 on overlapping days) and are often both members on the same day. The metadata has matching duplicate rows in the same order; the second occurrence gets id `XXXX_b`.
- 21 columns with no data at all are dropped, leaving 1314 securities.
- Stock ids stay strings throughout (`"12431_b"` is not an integer).
- Non-member days have returns for about 255 stocks per day (2,190,592 cells). These are used ONLY for volatility estimation and feature construction, never as training targets, predictions, or evaluation. The dataset only contains stocks that were members at some point in 1980 to 2013, so training on pre-inclusion days would leak future index membership.

### 2.3 Volatility adjustment

Per stock, every day, in this order (EWMA filters, assignment eqs. (2) and (3)):

```
mu_t     = (1 - beta2) * r_t + beta2 * mu_{t-1}
a_t      = r_t - mu_t
sigma2_t = (1 - beta1) * a_t^2 + beta1 * sigma2_{t-1}
vadj_t   = r_t / sigma_{t-1}          # note t-1: implemented as sig.shift(1)
```

beta1 = 0.97, beta2 = 0.99, initial variance 0.02**2, burn-in of 100 observations per stock (sigma is NaN before that), state carried forward unchanged on NaN days.

### 2.4 Targets, masks, and evaluation

- `vadj_c = vadj.clip(-5, 5)`: for features and training targets only.
- `Y = vadj.shift(-1)`: UNclipped, for evaluation only. `Y[t]` is the realized vol-adjusted return on day t+1, earned by a prediction made at the close of day t.
- `mem = ind.astype(bool)`.
- `evaluate(P, start, end, windows, demean, label, plot)`: takes a prediction DataFrame `P` (dates x stocks, `P[t]` = prediction made at close of day t), restricts to members on day t, optionally demeans each day, and reports: annualized Sharpe of the daily P&L `sum_i P[t,i] * Y[t,i]`, Sharpe by year and by window, mean daily Spearman rank IC and its t-stat, cumulative P&L and cumulative IC plots, average holding period (assignment eq. (7), stocks with at least 20 predictions), correlation of the P&L with an equal-weight long-only portfolio of members, and the minimum number of predictions per day.

### 2.5 Sentiment timing rule

Sentiment for calendar day d may contain posts made after that day's market close. To be safe, sentiment dated d is first usable for the prediction made at the close of the first trading day STRICTLY AFTER d. Sentiment from non-trading days (weekends, holidays) is aggregated into that same first usable trading day. Example: Friday, Saturday, and Sunday sentiment all become usable at Monday's close; Monday's sentiment becomes usable at Tuesday's close.

Of the 19 duplicated permnos that remain, check which have sentiment columns. For each one, if exactly one of the two securities has returns during the sentiment period, map the sentiment to that one. If both do, exclude sentiment for both. Log what you did in `AGENT_LOG.md`.

---

## Phase 0: Extract, verify, freeze (then STOP)

1. Create `src/task3_data.py` with `load_data()`, `ewma_vol()`, and `build_targets()` (returning `ret, ind, meta, sig, vadj, vadj_c, Y, mem`), copying the logic from the notebook. The only allowed change is turning globals into function arguments or return values.
2. Create `src/task3_eval.py` with `sharpe()` and `evaluate()`, copied from the notebook. Same allowed change only: `Y` and `mem` become arguments.
3. Create `tests/test_task3_pipeline.py` that checks these values reproduce (tolerance 1e-3 relative unless exact):

   | Check | Expected |
   |---|---|
   | duplicated permno columns renamed `_b` | 21 |
   | empty columns dropped | 21 |
   | securities remaining | 1314 |
   | metadata index equals return columns | True |
   | mean members per day / min | 499.79 / 489 |
   | member cells with NaN return | 0 |
   | non-member cells with a return | 2,190,592 |
   | non-NaN vol-adjusted returns on member cells | 4,227,347 |
   | std of those | 1.0574 |
   | excess kurtosis | 22.75 (tolerance 0.05) |
   | fraction with abs value > 5 | 0.00076 |

4. Run the harness controls on 2011-01-01 to 2013-12-31 (this is the one allowed use of the test period before Phase 3, because these controls contain no model): perfect-foresight `P = Y` must give an absurd Sharpe (above 50); i.i.d. N(0,1) predictions with seed 42 must give an absolute Sharpe below 1.5; always-long `P = 1` must give market correlation above 0.9. Report the 5-day reversal baseline (`-vadj_c.rolling(5, min_periods=3).mean()`, demeaned) over 1985 to 2010 only.
5. Commit, write the commit hash to `results/FROZEN_COMMIT.txt`, commit that file too, and STOP with an expected-vs-actual table.

---

## Phase 1: Features and leakage tests (then STOP)

Create `src/task3_features.py`. All features are computed per stock from data up to and including row t, then transformed cross-sectionally each day among members: rank, scaled to [-0.5, 0.5]. Missing values after ranking become 0 (neutral); log the fraction filled per feature.

Return-based features (from `vadj_c` and `sig`):
- `rev_1`: vadj_c on day t
- `rev_5`: mean of vadj_c over days t-4..t
- `rev_21`: mean over t-20..t
- `mom_12_1`: sum over t-251..t-21 (skips the most recent month)
- `vol_level`: log(sig_t)
- sector-relative versions of `rev_5`, `rev_21`, `mom_12_1`: subtract the same-day mean among members of the same GICS sector (first 2 digits), before ranking
- smoothed versions: EWMA over time of the ranked feature with half-lives 5 and 10 days

Sentiment features (from 2009-07 onward, Section 2.5 timing): bullish, bearish, net = bullish - bearish, activity = bullish + bearish, each also EWMA-smoothed with half-lives 5 and 21 days, plus a 0/1 availability flag.

Leakage tests in `tests/test_leakage.py` (mandatory):
- **Truncation test:** for 20 random dates t (seed 42) spread across 1985 to 2010, recompute all features using only data up to t (returns, and sentiment respecting the timing rule) and assert row t matches the full-data computation to 1e-10.
- **Training-window assertion:** a unit test showing the training function raises an error if given any row s >= t when predicting on day t.

STOP with a feature table: name, formula, lookback in days, first available date, mean coverage among members.

---

## Phase 2: Walk-forward experiments on the development period (then STOP)

### Protocol

- **Training data:** member rows only. Target = `vadj_c.shift(-1)`, cross-sectionally demeaned each day (default), or raw (one variant only).
- **Walk-forward:** at each refit date d, fit on rows s in [d - window, d - 1], then predict every day from d until the next refit. Refit every 63 trading days for Lasso and every 252 trading days for random forest and MLP (compute budget). Training windows: 3 years, 5 years, expanding (try on Lasso; pick one for RF/MLP and log why).
- **Development period:** predictions from 1990-01-01 to 2010-12-31. Selection metric: dev Sharpe of demeaned predictions, but prefer configurations that are consistent across the windows 1990-1994, 1995-1999, 2000-2004, 2005-2010 over ones with a higher total driven by one window. Report holding period for everything.
- **Sentiment models:** sentiment starts 2009-07, so their development window is 2010-01-01 to 2010-12-31 with expanding training from 2009-07. That is very little data: use a tiny grid (at most 6 configurations) and flag the limitation clearly.
- **Coverage:** at least 100 predictions per day; report the minimum.

### Models

- Baselines: 5-day reversal, 12-1 momentum (rule-based, no fitting).
- Lasso (`sklearn.linear_model.Lasso`): alpha grid of about 6 values on a log scale. Record which coefficients are nonzero at each refit.
- Random forest (`RandomForestRegressor`): `n_estimators=100`, `max_depth` in {4, 6, 8}, `min_samples_leaf` in {200, 1000}, subsample 200,000 training rows per fit (seed 42), `n_jobs=-1`.
- MLP (`MLPRegressor`): one hidden layer of 16 or 32 units, alpha in {1e-4, 1e-2}. Do NOT use `early_stopping=True`: it splits randomly, which mixes future and past. Instead, choose the number of epochs with a time-ordered holdout (the most recent 20% of each training window by date).
- Post-processing variants on the best configurations: demean on/off, and EWMA smoothing of predictions over time with half-life in {0, 2, 5, 10} days to show the Sharpe vs holding-period trade-off.

### Budget and controls

- At most about 60 configurations in total for Phase 2. Every one goes in the log.
- **Positive control:** take one Lasso configuration and shift its features one day into the future (`shift(-1)`). The dev Sharpe must jump dramatically. This proves the pipeline is sensitive to leakage.
- **Negative control:** same configuration, but permute training targets across stocks within each day. Dev Sharpe should be near 0.
- Controls are logged but never eligible for selection.
- If any single configuration takes more than about 15 minutes, reduce cost (fewer refits, smaller subsample), log it, and continue.

STOP with: the full results table sorted by dev Sharpe, windowed Sharpe for the top configurations, the control results, and a proposal of at most 4 finalists (at least one Lasso, one random forest, one MLP, and a sentiment variant only if it looks viable), each with a short justification. Save figures to `results/figures/` as PDF with large fonts (at least 14 pt).

---

## Phase 3: Freeze, then test once (only after we approve the finalists)

1. Write the approved finalists to `results/final_configs.json` and commit with message `task3: freeze finalists` BEFORE running anything on the test period.
2. Run walk-forward predictions through 2013-12-31 with the same rules (training on rows up to t-1, same refit schedule) and evaluate each finalist plus the two baselines on 2011-01-01 to 2013-12-31, exactly once. Save to `results/test_results.csv`.
3. Do not change any configuration after seeing test results. If you discover a bug, stop and tell us; any fix and re-run must be disclosed in `AGENT_LOG.md` and the experiment log.
4. Also produce full walk-forward curves for 1990 to 2013, clearly marking 1990 to 2010 as the development period (used for selection, so not truly out-of-sample).
5. Figures (PDF, large fonts): cumulative P&L on the test period for all finalists and baselines; cumulative rank IC on the test period; Sharpe by window as a bar chart; Sharpe vs holding period scatter over all dev configurations plus finalists; Lasso nonzero-coefficient frequency across refits; random forest feature importances.

---

## Phase 4: Write-up support (then STOP)

1. `results/SUMMARY.md`: protocol, every number we need for the report, and an honest interpretation. Compare against the assignment's rough benchmarks: Sharpe of 0.6 or more is interesting for long holding periods, about 2 for one-day holding periods; standard error of the Sharpe is roughly 1/sqrt(years), so about 0.58 for the 3-year test period. State limitations plainly.
2. `results/EXPLAIN.md`: a walkthrough of every function in `src/task3_*.py`, in plain language, followed by about 15 likely demo questions with concise answers (e.g. why sigma_{t-1}, why train only on rows up to t-1, why demean, why rank features, what the controls show).
3. `notebooks/task3_agent.ipynb`: imports the modules and reproduces every final table and figure top to bottom, with outputs saved.
4. Final checks: frozen-harness diff empty, total configuration count from the log, all tests passing. Commit and STOP.