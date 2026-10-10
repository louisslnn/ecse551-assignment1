# Task 3 summary: predicting next-day volatility-adjusted returns

This file holds the protocol, every number needed for the report, and an interpretation. The step-by-step record of decisions and problems is `results/AGENT_LOG.md`. The code walkthrough and demo questions are in `results/EXPLAIN.md`.

## 1. Result in brief

- Four frozen models were evaluated once on the test period 2011-2013. All four beat the two rule-based baselines.
- The random forest (Sharpe 3.65) and the MLP (3.01) are above the assignment's benchmark of about 2 for one-day holding periods. The Lasso (1.07) is barely above the 5-day reversal rule (0.73).
- The forest landed above the range written down before the test (1.2 to 3.3). The checks found no lookahead or bug.
- Two analyses run after the test evaluation qualify the headline numbers:
  - **One day late:** if every position is entered one day late, the forest drops to 1.24 and the MLP to 1.02. About two thirds of their test Sharpe depends on trading at the same close the prediction is made on.
  - **Market hedged:** removing the market component changes little for the forest (3.50) and the MLP (2.81), and lowers the Lasso to 0.70.
- The harness has no transaction costs, and the best models replace their whole book about once a day.

## 2. Protocol

| Item | Value |
|---|---|
| Data | daily excess returns and S&P 500 membership, 8577 trading days, 1980-01-02 to 2013-12-31, 1314 securities after cleaning |
| Target | next-day volatility-adjusted return, clipped to [-5, 5], minus its same-day mean across members |
| Evaluation | frozen `evaluate()`: daily P&L = sum over members of prediction x unclipped next-day vol-adjusted return; annualized Sharpe; daily rank IC; holding period; correlation with the equal-weight market |
| Predictions | demeaned across members each day before scoring |
| Development period | prediction dates 1990-01-01 to 2010-12-31, used for all selection |
| Test period | prediction dates 2011-01-01 to 2013-12-31, evaluated once, after the finalists were committed |
| Walk-forward | at each refit date d, fit on member rows of the 756 trading days before d, predict from d until the next refit |
| Refit schedule | Lasso every 63 trading days; random forest and MLP every 252 |
| Model families | Lasso, random forest, one-hidden-layer MLP (scikit-learn), plus two rule-based baselines |
| Random seed | 42 everywhere |
| Frozen harness commit | `cceeb15e720d5b637c2082d1da9a74c5f101496a` |
| Frozen finalists commit | `f817d04c028b0cec2b804a5cfde2a4ca89df9564` |

The rules that prevent lookahead:

1. A feature at row t uses data up to day t only. Checked by recomputing every feature from truncated data on 20 dates (plus 30 dates for sentiment).
2. A model predicting on day t is trained only on rows s <= t-1, enforced by an `assert` in every fit.
3. Sentiment dated d is first usable at the close of the first trading day strictly after d.
4. Nothing was evaluated on the test period before the freeze, apart from the three no-model harness controls of Phase 0.

## 3. Pipeline facts (Phase 0)

| Check | Value |
|---|---|
| Duplicated permno columns renamed `_b` | 21 |
| Empty columns dropped | 21 |
| Securities remaining | 1314 |
| Members per day, mean / minimum | 499.79 / 489 |
| Non-member cells with a return (used for volatility and features only) | 2,190,592 |
| Vol-adjusted returns on member cells | 4,227,347 |
| Their standard deviation / excess kurtosis | 1.0574 / 22.75 |
| Share with absolute value above 5 | 0.201% (8,509 values) |

The notebook's earlier figure of 0.076% for the last line divided by all date-by-stock cells, including non-members; 0.201% is the share among actual member values. The same correction applies to the other thresholds: 1.314% above 3 and 0.022% above 10.

Harness controls on 2011-2013 (no model):

| Control | Result | Requirement |
|---|---|---|
| Perfect foresight, `P = Y` | Sharpe 16.60 | above 10 (lowered from 50 by the group) |
| Random N(0,1), seed 42 | Sharpe 0.31 | absolute value below 1.5 |
| Always long, `P = 1` | market correlation 1.000 | above 0.9 |

## 4. Features (Phase 1)

All features are ranked across members each day and scaled to [-0.5, 0.5]; a missing rank becomes 0.

| Group | Features | Count |
|---|---|---|
| Return, base | `rev_1`, `rev_5`, `rev_21`, `mom_12_1`, `vol_level` | 5 |
| Return, sector-relative | `rev_5_sec`, `rev_21_sec`, `mom_12_1_sec` | 3 |
| Return, smoothed (EWMA half-lives 5 and 10) | of `rev_1`, `rev_5`, `rev_21`, `rev_5_sec`, `rev_21_sec` | 10 |
| Sentiment, base | `sent_bull`, `sent_bear`, `sent_net`, `sent_act`, `sent_flag` | 5 |
| Sentiment, smoothed (half-lives 5 and 21) | of the first four | 8 |

- Return features: 18. Coverage among members is 99.8% (98.7% for momentum). All exist from 1981-05-22.
- Six smoothed momentum and volatility features were dropped because they had a correlation of 0.97 to 0.99 with their base feature.
- Sentiment: 13 features from 2009-07-13. On an average day in 2010 only 12% of members (about 60 stocks) have a usable score, and 270 of 500 members have a sentiment column at all.

## 5. Development results (Phase 2)

66 rows were in the experiment log at the freeze: 4 Phase 0 checks and 62 development rows. Full table: `results/phase2_results.csv`.

### 5.1 Main configurations

| Row | Configuration | Sharpe 1990-2010 | 1990-94 | 1995-99 | 2000-04 | 2005-10 | Holding (days) |
|---|---|---|---|---|---|---|---|
| E058 | RF, depth 12, leaf 200 | 5.78 | 10.19 | 6.49 | 4.14 | 2.70 | 1.09 |
| E057 | RF, depth 10, leaf 200 | 5.68 | 9.96 | 6.41 | 4.13 | 2.56 | 1.11 |
| E029 | RF, depth 8, leaf 200 | 5.50 | 9.52 | 6.25 | 3.95 | 2.55 | 1.14 |
| E059 | MLP, 32 units, epochs up to 60 | 5.39 | 9.20 | 6.33 | 3.86 | 2.69 | 1.27 |
| E060 | MLP, 64 units, epochs up to 30 | 5.39 | 8.94 | 6.78 | 3.71 | 2.61 | 1.29 |
| E033 | MLP, 32 units, epochs up to 30 | 5.16 | 8.56 | 6.08 | 3.73 | 2.56 | 1.29 |
| E056 | Lasso alpha = 0 (least squares), 3 years | 4.20 | 7.65 | 4.93 | 2.69 | 2.01 | 1.40 |
| E012 | Lasso, 3 years, alpha 1.70e-4 | 4.06 | 7.48 | 4.77 | 2.60 | 1.81 | 1.39 |
| E018 | Lasso, 5 years, alpha 1.70e-4 | 4.00 | 7.49 | 4.31 | 2.92 | 1.54 | 1.35 |
| E024 | Lasso, expanding, alpha 1.70e-4 | 3.09 | 6.75 | 2.31 | 2.26 | 1.32 | 1.05 |
| E066 | MLP E059, predictions smoothed (half-life 5) | 2.53 | 4.02 | 3.46 | 1.71 | 1.43 | 5.56 |
| E005 | Baseline: 5-day reversal | 2.18 | 4.09 | 2.59 | 1.39 | 1.25 | 1.56 |
| E006 | Baseline: 12-1 momentum | 0.52 | 0.85 | 1.22 | -0.01 | 0.23 | 10.91 |

Grids tried: Lasso, 6 penalties (1.66e-2 down to 1.70e-4) x 3 training windows; random forest, depth {4, 6, 8} x leaf {200, 1000}; MLP, {16, 32} units x alpha {1e-4, 1e-2}. Then five extension runs requested by the group (E056 to E060).

### 5.2 Controls

| Row | Control | Sharpe 1990-2010 | Reading |
|---|---|---|---|
| E037 | Features shifted one day into the future (deliberate leak) | 68.28 | the pipeline reacts strongly to leakage; 17 times the real Lasso |
| E036 | Training targets shuffled among stocks within each day | -0.02 | nothing is learned when the link is broken |

### 5.3 Post-processing

| Variant | Lasso E012 | RF E029 | MLP E033 |
|---|---|---|---|
| As selected: Sharpe / holding period | 4.06 / 1.39 | 5.50 / 1.14 | 5.16 / 1.29 |
| Scored without daily demeaning | 4.06 | 5.12 | 3.73 |
| Predictions smoothed, half-life 2 | 2.86 / 3.55 | 3.79 / 2.95 | 3.55 / 3.30 |
| Half-life 5 | 2.08 / 6.27 | 2.53 / 5.23 | 2.45 / 5.73 |
| Half-life 10 | 1.61 / 9.92 | 1.76 / 8.29 | 1.75 / 8.87 |

Training the Lasso on the raw target instead of the demeaned one gives the same Sharpe (4.06).

### 5.4 Sentiment (2010 only, one year)

| Row | Model | Features | Sharpe 2010 |
|---|---|---|---|
| E038 / E040 | Lasso, two penalties | return only | 0.43 / 0.37 |
| E039 / E041 | Lasso, two penalties | return + sentiment | 0.32 / 0.37 |
| E042 | Lasso | sentiment only | -0.90 |
| E043 | RF, depth 4 | return + sentiment | 1.32 |

The standard error of a one-year Sharpe is about 1.0. Adding sentiment did not help, and no sentiment model became a finalist. The models with a long return history score 1.66 (Lasso), 3.67 (RF) and 2.43 (MLP) on 2010.

### 5.5 Delayed execution on the development period (diagnostic)

| Finalist | On time | One day late | Share kept |
|---|---|---|---|
| Lasso E012 | 4.06 | 2.70 | 67% |
| RF E058 | 5.78 | 3.81 | 66% |
| MLP E059 | 5.39 | 3.55 | 66% |
| 5-day reversal | 2.18 | 2.29 | 105% |

## 6. Finalists and how they were chosen

| Finalist | Row | Configuration | Refit | Smoothing |
|---|---|---|---|---|
| `lasso` | E012 | Lasso, alpha 1.697e-4, 3-year window | 63 days | none |
| `rf` | E058 | Random forest, 100 trees, depth 12, leaf 200, `max_features` 1/3, 200,000 rows per fit, 3-year window | 252 days | none |
| `mlp` | E059 | MLP, 32 units, alpha 1e-4, batch 1024, epochs chosen in 1..60 on the most recent 20% of training days, 3-year window | 252 days | none |
| `mlp_slow` | E066 | the predictions of `mlp`, smoothed over time | 252 days | half-life 5 |

- The selection metric was the development Sharpe of demeaned predictions, with a preference for consistency across the four windows.
- The 3-year window was chosen from the Lasso grid (4.06, against 4.00 for 5 years and 3.09 for expanding).
- The group fixed a replacement rule before the extension runs: replace a finalist only if an extension beats it by more than 0.22 (one standard error of a 21-year Sharpe). This kept the Lasso (+0.135 for least squares) and replaced the forest (+0.284) and the MLP (+0.235).
- The slow finalist was fixed by the group as "the MLP finalist smoothed with half-life 5", whatever its development Sharpe.

## 7. Test results (Phase 3, evaluated once)

Period: 2011-01-01 to 2013-12-31, 753 days. Standard error of a 3-year Sharpe: about 0.58. File: `results/test_results.csv`.

| Finalist | Test Sharpe | 2011 | 2012 | 2013 | Mean IC | IC t-stat | Holding (days) | Market corr. | Expected range | Inside |
|---|---|---|---|---|---|---|---|---|---|---|
| Lasso | 1.07 | 1.31 | 0.64 | 1.22 | 0.014 | 2.67 | 1.48 | 0.38 | 0.6 to 2.4 | yes |
| Random forest | 3.65 | 3.20 | 3.69 | 4.15 | 0.018 | 5.55 | 0.94 | 0.39 | 1.2 to 3.3 | no, above |
| MLP | 3.01 | 2.02 | 3.22 | 4.09 | 0.018 | 5.65 | 1.04 | 0.31 | 1.2 to 3.3 | yes |
| MLP, smoothed | 1.40 | 0.11 | 1.54 | 2.85 | 0.007 | 1.84 | 4.51 | 0.40 | 0.35 to 2.0 | yes |
| 5-day reversal | 0.73 | 1.55 | -0.32 | 0.85 | 0.008 | 1.87 | 1.61 | 0.15 | 0.2 to 1.8 | yes |
| 12-1 momentum | 0.79 | 0.75 | 0.48 | 1.15 | 0.019 | 3.21 | 11.44 | 0.12 | -0.4 to 0.8 | yes |

Every row has at least 491 predictions per day.

The expected ranges were written in `AGENT_LOG.md` before the freeze. Method: from "decay continues" (0.65 x the 2005-2010 Sharpe) to "no decay" (the 2005-2010 Sharpe), widened by one standard error on each side. One disclosure applies: the group's notebook already showed the reversal baseline's yearly Sharpe for 2011-2013, so the expectation was not blind for that baseline.

Development and test side by side:

| Finalist | Dev 1990-2010 | Dev 2005-10 | Test 2011-13 | Dev holding | Test holding |
|---|---|---|---|---|---|
| Lasso | 4.06 | 1.81 | 1.07 | 1.39 | 1.48 |
| Random forest | 5.78 | 2.70 | 3.65 | 1.09 | 0.94 |
| MLP | 5.39 | 2.69 | 3.01 | 1.27 | 1.04 |
| MLP, smoothed | 2.53 | 1.43 | 1.40 | 5.56 | 4.51 |
| 5-day reversal | 2.18 | 1.25 | 0.73 | 1.56 | 1.61 |
| 12-1 momentum | 0.52 | 0.23 | 0.79 | 10.91 | 11.44 |

### 7.1 The forest above its expected range

Checks made after seeing 3.65, none of which changes a configuration:

| Check | Result |
|---|---|
| Development Sharpes recomputed from the Phase 3 predictions | all six equal to the frozen values to four decimals |
| Reversal baseline by year against the group's notebook | identical |
| 147 refit records of the 1990-2013 run | last training day always before the refit date |
| All raw data after 2012-06-29 removed, everything recomputed, predictions compared (Lasso and forest) | identical, largest difference 0 over 5,672 days |
| Frozen harness and `final_configs.json` | unchanged |

Likely reason the range was too low (found after the fact): the 2005-2010 Sharpe of 2.70 is pulled down by 2008. The forest's yearly Sharpes for 2005 to 2010 are 3.14, 5.17, 2.13, 1.10, 2.69, 3.92 (mean 3.02), and 2009-2010 pooled is 3.18. Its rank IC did not rise (0.020 in 2005-2010, 0.018 on test); its daily P&L was steadier (standard deviation 1.00 on test, 1.40 in 2005-2010, 1.86 in 2008). The result is 1.6 standard errors above the 2.70 reference.

## 8. Post-hoc analyses of the frozen test predictions

Both were requested by the group after the test evaluation and are logged as their own rows (E073 to E084). They reuse the frozen predictions; nothing was refitted or reselected. Because they were run after the results were known, they describe the test result and are not additional out-of-sample tests.

### 8.1 Delayed execution, 2011-2013

Every prediction is used one day late (`P.shift(1)`). File: `results/posthoc_test_delay.csv`.

| Finalist | On time | One day late | Share kept | 2011 | 2012 | 2013 | IC t-stat, late |
|---|---|---|---|---|---|---|---|
| Lasso | 1.07 | 0.72 | 68% | 0.77 | 0.01 | 1.34 | 1.56 |
| Random forest | 3.65 | 1.24 | 34% | 0.73 | 0.04 | 2.99 | 1.45 |
| MLP | 3.01 | 1.02 | 34% | 0.26 | 0.49 | 2.77 | 1.50 |
| MLP, smoothed | 1.40 | 0.93 | 67% | -0.36 | 1.11 | 2.37 | 0.97 |
| 5-day reversal | 0.73 | 0.69 | 96% | 1.12 | -0.64 | 1.63 | 0.80 |
| 12-1 momentum | 0.79 | 0.82 | 104% | 0.74 | 0.51 | 1.23 | 3.28 |

- On the test period the forest and the MLP keep one third of their Sharpe when traded a day late. On the development period they kept two thirds.
- A day late, the forest (1.24) is 0.55 above the reversal rule (0.69), about one standard error, and almost all of it comes from 2013.
- The slow and rule-based strategies are much less sensitive, as expected for signals that change slowly.

### 8.2 Market-hedged Sharpe, 2011-2013

Regression of each daily P&L on the equal-weight market return (the mean over members of the next-day vol-adjusted return, the same series `evaluate()` uses). Hedged P&L = P&L minus beta x market. File: `results/posthoc_test_hedge.csv`.

| Finalist | Market correlation | Beta t-stat | R squared | Sharpe unhedged | Sharpe hedged | 2011 | 2012 | 2013 |
|---|---|---|---|---|---|---|---|---|
| Lasso | 0.38 | 11.3 | 0.15 | 1.07 | 0.70 | 1.36 | 0.22 | 0.51 |
| Random forest | 0.39 | 11.7 | 0.16 | 3.65 | 3.50 | 3.43 | 3.50 | 3.61 |
| MLP | 0.31 | 8.9 | 0.10 | 3.01 | 2.81 | 2.12 | 2.99 | 3.52 |
| MLP, smoothed | 0.40 | 12.0 | 0.16 | 1.40 | 1.04 | 0.09 | 1.19 | 2.17 |
| 5-day reversal | 0.15 | 4.3 | 0.02 | 0.73 | 0.56 | 1.61 | -0.49 | 0.49 |
| 12-1 momentum | 0.12 | 3.3 | 0.01 | 0.79 | 0.66 | 0.73 | 0.36 | 0.94 |

- The equal-weight market itself had a Sharpe of 1.10 over the period.
- Every strategy has a positive and statistically clear market exposure although its predictions are demeaned each day. Demeaning makes the sum of positions zero in volatility-adjusted units; it does not make the beta zero.
- The beta values are in each strategy's own P&L units and cannot be compared across strategies (0.59, 0.61, 0.55, 0.45, 3.73, 129.7); the correlation and the t-statistic can.
- The market explains 10% to 16% of the P&L variance of the fitted models. Hedging costs the forest 0.15 and the MLP 0.20, so their result is not a market effect. It costs the Lasso a third of its Sharpe.
- Beta is estimated on the test period itself. With one regressor and 753 days, the optimism from fitting in-sample is negligible.

## 9. Comparison with the assignment's benchmarks

The assignment calls a Sharpe of about 2 interesting for one-day holding periods and 0.6 or more interesting for long holding periods.

| Finalist | Holding period on test | Test Sharpe | Hedged | One day late | Benchmark | Reading |
|---|---|---|---|---|---|---|
| Random forest | 0.9 days | 3.65 | 3.50 | 1.24 | about 2 | above as measured; below when traded a day late |
| MLP | 1.0 days | 3.01 | 2.81 | 1.02 | about 2 | same |
| Lasso | 1.5 days | 1.07 | 0.70 | 0.72 | about 2 | below |
| MLP, smoothed | 4.5 days | 1.40 | 1.04 | 0.93 | 0.6 or more | above in all three versions, but carried by 2012 and 2013 |
| 5-day reversal | 1.6 days | 0.73 | 0.56 | 0.69 | about 2 | below |
| 12-1 momentum | 11.4 days | 0.79 | 0.66 | 0.82 | 0.6 or more | just above the benchmark; about 1.4 standard errors from zero |

## 10. Interpretation

1. **Non-linear models found something the linear ones did not.** From 2005-2010 to the test period the Lasso and the reversal rule lost about 40% of their Sharpe (1.81 to 1.07, 1.25 to 0.73), continuing a 20-year decline. The forest and the MLP did not (2.70 to 3.65, 2.69 to 3.01).
2. **That advantage lives at the one-day horizon.** A day late, the forest and the MLP fall to 1.24 and 1.02, close to the Lasso (0.72) and the reversal rule (0.69) given a standard error of 0.58. The features with the largest forest importance are `rev_1`, `vol_level` and `rev_1_hl5`.
3. **It is not a market effect.** Hedging the market barely changes the forest and MLP results.
4. **The Lasso barely beats reversal** (1.07 against 0.73, 0.6 standard errors; 0.70 against 0.56 hedged; 0.72 against 0.69 a day late). Its coefficients shrank over time: the sum of their absolute values averaged 0.24 over the 1990-1994 refits, 0.18 over 2005-2010 and 0.08 over the test-period refits. The linear relationships it relies on have largely faded.
5. **Forest against MLP** (3.65 against 3.01) differ by about one standard error. The test does not rank them.
6. **Decay over time is the dominant pattern.** Every model lost about three quarters of its development Sharpe between 1990-1994 and 2005-2010.

## 11. Limitations

1. **No transaction costs.** A holding period near one day means the whole book is traded daily. Costs would reduce the fast strategies the most.
2. **Execution assumption.** The harness assumes a position can be taken at the close using that same close. The delayed-execution analysis shows the test result of the two best models depends heavily on that assumption.
3. **One test period of three years.** Standard error about 0.58 per Sharpe.
4. **Multiple testing.** 62 development configurations and diagnostics were scored before the freeze. Two finalists were picked at the edge of an extended grid by a margin of about one standard error. The grids were never exhausted: deeper forests and longer-trained networks kept improving slightly.
5. **Post-hoc analyses.** Section 8 was run after the test results were known.
6. **Not market neutral.** Market correlation 0.31 to 0.40 on the test period.
7. **The forest result is outside the pre-registered range.** No bug was found; the explanation offered was found afterwards.
8. **Static sector codes.** One GICS code per security for 1980-2013, although GICS did not exist before 1999.
9. **Sentiment.** Only 1.5 years before the test period and 12% daily coverage; it could not be evaluated properly and is not used by any finalist.
10. **One unexplained test failure.** A leakage test failed once and could not be reproduced (Problem 7 in `AGENT_LOG.md`).
11. **Survivorship in the universe.** The data only contains stocks that were index members at some point, so non-member days are used for volatility and features only, never as targets or predictions.

## 12. Counts for the report

| Quantity | Value |
|---|---|
| Rows in `results/experiment_log.csv` | 84 |
| Phase 0 harness checks | 4 |
| Development rows (Phase 2), all before the freeze | 62 |
| of which controls and delay diagnostics (not eligible) | 7 |
| of which eligible configurations | 55 |
| Test evaluations of frozen configurations (Phase 3) | 6 |
| Post-hoc rows on the test period | 12 |
| Failed runs | 0 |
| Slowest single configuration | 492 seconds |
| Automated tests | 30 |

## 13. Where things are

| File | Content |
|---|---|
| `results/experiment_log.csv` | one row per evaluation, append-only |
| `results/phase2_results.csv`, `results/dev_windows.csv` | development results with the four windows |
| `results/final_configs.json` | the frozen finalists |
| `results/test_results.csv`, `results/test_pnl.csv`, `results/test_ic.csv` | the single test evaluation |
| `results/posthoc_test_delay.csv`, `results/posthoc_test_hedge.csv` | the two post-hoc analyses |
| `results/feature_table.csv`, `results/sentiment_column_map.json` | features and the frozen sentiment mapping |
| `results/lasso_coefs_18.csv`, `results/rf_importances_18.csv`, `results/mlp_epochs.csv` | what the models learned at each development refit |
| `results/phase3_lasso_coefs.csv`, `results/phase3_rf_importances.csv`, `results/phase3_mlp_epochs.csv` | the same for the 1990-2013 run of the finalists |
| `results/test_runs/` | complete output of five full test-suite runs |
| `results/figures/` | 13 figures, PDF, fonts 14 pt or larger |
| `notebooks/task3_agent.ipynb` | reproduces the final tables and figures |

Figures: `phase2_lasso_grid`, `phase2_sharpe_by_window`, `phase2_sharpe_vs_holding`, `phase2_smoothing_tradeoff`, `phase2_controls`, `phase2_cumulative_pnl`, `phase3_test_cumulative_pnl`, `phase3_test_cumulative_ic`, `phase3_sharpe_by_window`, `phase3_sharpe_vs_holding`, `phase3_lasso_nonzero_frequency`, `phase3_rf_feature_importance`, `phase3_full_cumulative_pnl`.
