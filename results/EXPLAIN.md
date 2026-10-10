# Task 3: code walkthrough and demo questions

Part 1 shows how the modules fit together. Part 2 explains every function in `src/task3_*.py`. Part 3 has 26 likely demo questions with short answers. All numbers are also in `results/SUMMARY.md`.

Notation used throughout: "row t" or "day t" is a trading day. A prediction made at the close of day t earns the return of day t+1.

## Part 1. How the pieces fit together

| Step | Module | What it produces |
|---|---|---|
| 1. Load and clean | `task3_data.py` (frozen) | returns, membership, volatility, vol-adjusted returns, targets |
| 2. Score predictions | `task3_eval.py` (frozen) | Sharpe, rank IC, holding period, market correlation |
| 3. Build features | `task3_features.py` | 18 return features, 13 sentiment features |
| 4. Guard the training window | `task3_train.py` | the `assert` that stops any fit on future rows |
| 5. Fit one model | `task3_models.py` | a fitted Lasso, random forest or MLP |
| 6. Walk forward in time | `task3_walkforward.py` | a prediction table (days x stocks) |
| 7. Score and record | `task3_experiments.py`, `task3_log.py` | one row per configuration in the experiment log |
| 8. Run the phases | `task3_phase0.py` to `task3_phase3.py`, `task3_finalists.py`, `task3_posthoc.py` | all results files |
| 9. Draw figures | `task3_phase2_report.py`, `task3_phase3_report.py` | the PDF figures |

The four tables that everything else uses, all with one row per trading day and one column per stock:

| Table | Meaning | Used for |
|---|---|---|
| `vadj_c` | vol-adjusted return of day t, clipped to [-5, 5] | features and training targets |
| `Y` | UNclipped vol-adjusted return of day t+1 | evaluation only |
| `mem` | True if the stock is an index member on day t | choosing which cells count |
| `P` | prediction made at the close of day t | what a model outputs |

## Part 2. Function by function

### `task3_data.py` (frozen after Phase 0)

| Function | What it does | Why |
|---|---|---|
| `load_data(data_dir)` | Reads returns, membership and metadata. Renames the second copy of a duplicated permno from `X.1` to `X_b`, gives the metadata the same ids, and drops the 21 securities with no data. | Every later step needs the three tables to share the same 1314 stock ids in the same order. Ids are strings because `12431_b` is not a number. |
| `ewma_vol(R, beta1, beta2, var0, burn_in)` | Loops over days. For each stock it updates a running mean of the return (`beta2 = 0.99`), then a running variance of the return minus that mean (`beta1 = 0.97`), and outputs the square root. A stock gets no estimate until it has 100 observations. On a day with no return the state is carried forward. | Volatility changes over time and across stocks. `sig[t]` uses returns up to day t only, because the loop goes forward. |
| `build_targets(data_dir)` | Calls the two functions above, then computes `vadj = ret / sig.shift(1)`, `vadj_c = vadj.clip(-5, 5)`, `Y = vadj.shift(-1)` and `mem`. | One call gives every table the experiments need. The `shift(1)` is the critical one: today's return is divided by yesterday's volatility. |

### `task3_eval.py` (frozen after Phase 0)

| Function | What it does | Why |
|---|---|---|
| `sharpe(x)` | Mean of the daily P&L divided by its standard deviation, times the square root of 252. | Annualized reward per unit of risk. |
| `evaluate(P, Y, mem, start, end, windows, demean, label, plot)` | Keeps predictions of members between `start` and `end`. Optionally subtracts each day's mean prediction. Daily P&L = sum over stocks of `P[t, i] * Y[t, i]`. Also computes the daily Spearman rank correlation between predictions and returns (the IC), the holding period (average absolute prediction divided by average absolute daily change, per stock with at least 20 predictions), the correlation of the P&L with the equal-weight market, and the smallest number of predictions on any day. | One fixed scoring function for every model, so nobody can change the metric after seeing results. |

### `task3_log.py`

| Function | What it does | Why |
|---|---|---|
| `append_log(row, path)` | Adds one line to `results/experiment_log.csv`, filling in the next `exp_id` and the time. Refuses unknown column names. Opens the file in append mode only. | The log is the honest count of everything tried. Rows cannot be edited or lost. |

### `task3_features.py`

| Function | What it does | Why |
|---|---|---|
| `rank_cs(x, mem)` | For each day, ranks the values of members from smallest to largest and rescales to [-0.5, 0.5]. Non-members and missing values stay missing. | Puts every feature on the same scale every day and removes the effect of extreme values. |
| `sector_relative(x, mem, sector)` | For each day and each GICS sector, subtracts the sector's mean among members. | Removes moves shared by a whole sector, leaving the stock-specific part. |
| `ewma_smooth(x, halflife)` | Per stock: `s[t] = a * x[t] + (1 - a) * s[t-1]`. | A smoother feature changes more slowly. It only looks backwards. |
| `rank_fill_smooth(raw, mem, halflives, smoothed)` | Ranks each raw feature, replaces a missing rank by 0, and for the features listed in `smoothed` adds one EWMA version per half-life. Also returns how many members had a real rank each day. | One place applies the rule "rank, fill with neutral, then smooth" to all features. |
| `build_return_features(vadj_c, sig, mem, meta)` | Builds `rev_1` (today), `rev_5` (mean of days t-4..t), `rev_21` (t-20..t), `mom_12_1` (sum of t-251..t-21), `vol_level` (log of `sig[t]`), three sector-relative versions, and ten smoothed versions: 18 features. | These are the standard short-term reversal, momentum and volatility signals. |
| `load_sentiment(data_dir)` | Reads the bullish and bearish files. Rows are calendar days. | Kept apart from the equity loader, which is frozen. |
| `sentiment_column_map(sent_columns, ret, sent_start, sent_end)` | Decides which equity id each sentiment column belongs to. For a duplicated permno: if exactly one of the two securities has returns between 2009-07-10 and 2010-12-31, use it; if both do, use neither. | The sentiment file does not say which of two securities with the same permno it describes. The decision uses development data only. |
| `align_sentiment(sent, trading_days)` | Moves the sentiment of calendar day d to the first trading day strictly after d, and averages when several calendar days land on the same trading day. | Sentiment dated d may include posts after that day's close, so it is only safe to use one trading day later. |
| `build_sentiment_features(bull, bear, col_map, mem)` | Aligns both files, relabels columns as equity ids, builds bullish, bearish, net and activity, ranks them, adds smoothed versions and a 0/1 flag: 13 features. | Same treatment as the return features. |

### `task3_train.py`

| Function | What it does | Why |
|---|---|---|
| `check_training_window(train_dates, predict_date)` | Raises an `AssertionError` unless the latest training date is strictly before the prediction date. | The target of row s is the return of day s+1. For a model used on day t, row s is only usable if s <= t-1. |
| `fit_model(model, X_train, y_train, train_dates, predict_date)` | Runs the check, then `model.fit`. | Every fit goes through the check, so a lookahead mistake stops the run. |

### `task3_models.py`

| Function | What it does | Why |
|---|---|---|
| `new_mlp(spec)` | Creates an untrained `MLPRegressor` with one hidden layer, the given size and penalty, batch size 1024, seed 42. | One place defines the network. `early_stopping` stays off. |
| `fit_mlp(spec, X, y, dates, predict_date)` | Checks the window. Trains a probe network on the oldest 80% of the training days, one epoch at a time, and records the squared error on the newest 20%. Then trains a fresh network on all training days for the number of epochs with the smallest error. | scikit-learn's own early stopping splits rows at random, which would mix future and past days inside the window. |
| `fit_config(spec, X, y, dates, predict_date)` | Reads `spec["model"]` and fits a Lasso, a least-squares regression, a random forest or an MLP. Returns the model and what to record: coefficients, importances or epochs. | One entry point for all model families keeps the walk-forward loop simple. A Lasso that does not converge raises an error. |
| `lasso_alpha_grid(X, y, n)` | Computes `alpha_max`, the smallest penalty that sets every coefficient to zero, and returns six penalties going down from `alpha_max / 2` by a factor of 2.5. | The grid is derived from the first training window only, not from results. |

### `task3_walkforward.py`

| Function | What it does | Why |
|---|---|---|
| `make_target(vadj_c, mem, demean)` | `vadj_c.shift(-1)` for members, minus the same-day mean if `demean`. | Row t holds the return of day t+1. Demeaning matches what a market-neutral strategy is paid for. |
| `build_panel(feats, names, target, mem, start, end)` | Stacks all member cells between `start` and `end` into one matrix `X` (one row per stock-day, sorted by day), a target vector `y`, and for each row its day and stock. `row_start[k]` is the first row of day k. | scikit-learn needs a flat table. `row_start` lets the code cut "all rows of days a to b-1" exactly. |
| `permute_target_within_day(panel, seed)` | Shuffles the targets among the stocks of each day, in place. | Negative control: keeps every day's returns but destroys the link to the features. |
| `training_rows(panel, k_lo, k)` | Returns the rows of days k_lo to k-1 that have a target. | The single place where a training window is cut: it always ends at day k-1. |
| `walk_forward(panel, specs, first_pred, last_pred, refit_every, train_window, subsample, seed)` | Every `refit_every` days: take the training rows before the refit date, optionally subsample them, standardize features with the training mean and standard deviation, fit every spec, and predict all days until the next refit. | This imitates real use: a model only ever sees the past, and it is renewed as time passes. |
| `smooth_predictions(P, halflife)` | EWMA over time of each stock's predictions; half-life 0 returns the table unchanged. | Trades Sharpe for a longer holding period. Looks backwards only. |

### `task3_experiments.py`

| Function | What it does | Why |
|---|---|---|
| `append_csv(path, row)` | Appends one row to a CSV file, writing the header the first time. | Small helper for side files. |
| `add_column(path, name, series)` | Adds one daily series as a column of a wide CSV. | Keeps the daily P&L and IC of every experiment. |
| `score_and_log(name, P, Y, mem, row, start, end, demean, cache)` | Calls `evaluate()` on the development period, appends the log row, saves the four window Sharpes, the daily P&L and IC, and the prediction table. Refuses any `end` after 2010-12-31. | The single door through which a configuration is scored, with the test-set lockout built in. |
| `load_cached(exp_id)` | Reads the saved prediction table of an earlier experiment. | Post-processing without refitting. |
| `save_records(exp_id, name, records, feature_names)` | Saves per-refit coefficients, importances or chosen epochs. | To report what the models learned. |
| `run_group(panel, specs, rows, Y, mem, refit_every, train_window, start, end, subsample)` | Runs `walk_forward` for several specs that share training data, then scores and logs each. If the run crashes, logs every spec as FAILED before re-raising. | No attempt can disappear from the record. |

### `task3_phase0.py`, `task3_phase1.py`

| Function | What it does |
|---|---|
| `log_control(...)` | Writes one Phase 0 control to the log, marked as not a development result. |
| `run_controls(data_dir)` | Runs perfect foresight, random predictions, always-long, and the 5-day reversal baseline on 1985-2010. |
| `describe_feature(name, halflives)` | Returns the formula and lookback text of a feature for the feature table. |
| `feature_rows(feats, n_valid, mem, halflives, dev_start)` | Computes first available date, coverage and fill fraction of each feature, up to 2010-12-31 only. |
| `run_phase1(data_dir)` | Builds all features, writes `feature_table.csv` and the frozen sentiment mapping. |

### `task3_phase2.py`, `task3_phase2b.py`

| Function | What it does |
|---|---|
| `load_world(data_dir)` | Puts the pipeline outputs in one dict. |
| `return_panel(world, demean_target)` | Builds the 18 return features and stacks them from 1981-05-22 to 2010-12-31. The panel contains no test-period row. |
| `model_row(...)` | Fills the descriptive log columns of a configuration. |
| `stage_baselines` | Scores the 5-day reversal and 12-1 momentum rules. |
| `stage_lasso` | Derives the penalty grid, then runs 6 penalties x 3 training windows. |
| `alpha_grid()` | Reads the saved penalty grid. |
| `stage_rf` | Runs depth {4, 6, 8} x leaf {200, 1000}, 200,000 rows per fit. |
| `stage_mlp` | Runs {16, 32} units x alpha {1e-4, 1e-2}. |
| `stage_variants` | Chosen Lasso with a raw target; negative control (shuffled targets); positive control (features shifted one day into the future). |
| `stage_sentiment` | Six configurations on 2010 only, training from 2009-07-13. |
| `stage_post` | Demean off, and prediction smoothing with half-lives 2, 5, 10, for the best model of each family. |
| `stage_extension` | The five runs beyond the grid edges asked by the group. |
| `stage_delay` | Development diagnostic: predictions used one day late. |
| `stage_slow` | Scores the slow finalist: MLP E059 smoothed with half-life 5. |

### `task3_phase2_report.py`, `task3_phase3_report.py`

| Function | What it does |
|---|---|
| `load_results()` | Joins the Phase 2 log rows with their window Sharpes. |
| `save(fig, name)` | Writes a figure as PDF. |
| `fig_lasso_grid`, `fig_sharpe_by_window`, `fig_sharpe_vs_holding`, `fig_smoothing_tradeoff`, `fig_controls`, `fig_cumulative_pnl` | The six development figures. |
| `read_daily(name)` | Reads a daily P&L or IC file. |
| `line_plot(ax, table, scale)` | One cumulative line per finalist; with `scale` each series is divided by its own daily standard deviation so slopes are comparable. |
| `fig_test_cumulative`, `fig_full_cumulative`, `fig_sharpe_by_window`, `fig_sharpe_vs_holding` | Test-period P&L and IC, the 1990-2013 curve with the test period shaded, Sharpe by window including the test, and development-to-test movement of each finalist. |
| `horizontal_bars`, `fig_model_internals` | Lasso nonzero frequency and forest feature importance. |
| `run_report()` | Builds all figures of the phase. |

### `task3_finalists.py`, `task3_phase3.py`

| Function | What it does | Why |
|---|---|---|
| `return_feature_names()` | Lists the 18 return features in panel order. | The frozen file names the features explicitly. |
| `write_final_configs()` | Writes `results/final_configs.json`: periods, walk-forward rules, features, four finalists, two baselines. | The file is committed before any test-period run. |
| `load_config()` | Reads the frozen file. | Phase 3 takes every setting from it. |
| `save_refit_records(name, records, feature_names)` | Saves what each finalist learned at each refit, 1990-2013. | For the coefficient and importance figures. |
| `compute_predictions(config, vadj_c, sig, mem, meta)` | Walks the three fitted finalists forward from 1990 to 2013, derives the slow finalist by smoothing, and builds the two baselines. Scores nothing. | Kept separate from scoring so it can be reused by the notebook. |
| `stage_predict()` | Calls `compute_predictions`, saves the tables, and checks that the development Sharpe of each equals the frozen value. | If development does not reproduce, the test period is never touched. |
| `stage_test()` | One `evaluate()` call per finalist and baseline on 2011-2013, writes `test_results.csv` and six log rows. Refuses to run if the file exists. | The test period is evaluated exactly once. |

### `task3_posthoc.py`, `task3_truncation_check.py`

| Function | What it does | Why |
|---|---|---|
| `frozen_names_and_period()` | Reads the finalist names and the test period from the frozen file. | No setting is typed by hand. |
| `log_posthoc(...)` | Logs one post-hoc number, labelled as not a selection result. | Every evaluation has a row. |
| `hedge_stats(y, x)` | Least-squares regression of a daily P&L on the market return; returns beta, its t-statistic, the correlation and the Sharpe of `y - beta * x`. | Measures how much of a result is market exposure. |
| `stage_delay()` | Scores `P.shift(1)` on the test period. | Measures dependence on trading at the same close. |
| `stage_hedge()` | Applies `hedge_stats` to each saved test P&L. | Same. |
| `check(name, cut)` | Removes all raw data after `cut`, recomputes everything for one finalist, and compares predictions with the saved ones. | An end-to-end proof on real data that predictions up to `cut` do not depend on later data. |

## Part 3. Demo questions

### Pipeline and targets

**1. Why divide the return of day t by the volatility of day t-1, not day t?**
`sigma_t` is computed from the return of day t itself. Dividing by it would use information that is not available when the position is taken, and it would shrink exactly the large returns. In the code this is `ret / sig.shift(1)`.

**2. Why clip to [-5, 5] for features and training targets, but evaluate on unclipped returns?**
Vol-adjusted returns have very fat tails (excess kurtosis 22.75; 0.2% of values exceed 5). A few extreme values would dominate a squared-error fit. The evaluation uses unclipped returns because that is what a strategy would actually earn.

**3. Why may a model that predicts on day t only train on rows up to t-1?**
The training target of row s is the return of day s+1, which is known only at the close of s+1. At the close of day t, the most recent row with a known target is s = t-1. Row t itself has features but no target yet. `check_training_window` enforces this with an `assert`.

**4. Why demean the target and the predictions across stocks each day?**
The P&L is the sum of prediction times return. If predictions sum to zero each day, the average market move cancels, so the strategy is paid only for ranking stocks correctly. Training on a demeaned target asks the model for exactly that. For the Lasso the two targets gave the same Sharpe (4.06).

**5. Why rank features across members each day, and why fill missing ranks with 0?**
Ranks are on the same scale every day and for every feature, and they are not affected by extreme values. After scaling to [-0.5, 0.5], 0 is the middle rank, so a stock with no information is treated as average.

**6. Why are returns on non-member days used for features but never as targets or predictions?**
The file only contains stocks that were index members at some time. A row before a stock joined the index exists in the file only because the stock joined later. Training on it would leak future index membership. Using it to estimate volatility or a 12-month return for a current member does not.

**7. What is the sentiment timing rule?**
Sentiment dated calendar day d may include posts made after that day's market close. It is therefore usable only at the close of the first trading day strictly after d. Friday, Saturday and Sunday are averaged into Monday's row; Monday's sentiment goes to Tuesday's row. A test checks this example.

### Models and protocol

**8. What does the walk-forward do, and why a 3-year window?**
At each refit date the model is fitted on the 756 trading days before it and then predicts until the next refit (63 days later for the Lasso, 252 for the forest and the MLP). On the Lasso grid, a 3-year window gave 4.06, a 5-year window 4.00 and an expanding window 3.09. The expanding window being clearly worse suggests the relationship drifts, so old data hurts. The difference between 3 and 5 years is within noise.

**9. Why standardize features with training-window statistics only?**
Smoothed features have a much smaller spread than raw ranks, and both a Lasso penalty and an MLP treat small-scale features differently. Computing the mean and standard deviation on the training rows only avoids using information from the days being predicted.

**10. Why not `early_stopping=True` in the MLP? How is the number of epochs chosen?**
scikit-learn's early stopping holds out a random 10% of rows, which mixes days from the future of other training rows. Instead the newest 20% of training days are held out, the number of epochs with the smallest error there is chosen, and a fresh network is trained on all training days for that many epochs. A weakness: 41% of the refits still chose the maximum of 60 epochs, so the networks may be under-trained.

**11. Why `max_features = 1/3` and a 200,000-row subsample for the forest?**
The scikit-learn default for regression uses all features at every split, which is bagging, not a random forest; one third is the textbook value. The subsample was set by the protocol to keep each fit within the time budget.

**12. How was the Lasso penalty grid chosen, and what does it mean that the smallest penalty won?**
`alpha_max` is the smallest penalty that zeroes all coefficients on the first training window (0.033). The grid goes down from half of it by a factor of 2.5, six values. The smallest penalty was best in every window, and no penalty at all (least squares) added only 0.135. With 18 features and several hundred thousand rows there is little overfitting for the penalty to prevent.

**13. What does the Lasso use?**
Over the development refits the largest coefficient is always on `rev_1_hl5`, with a negative sign: a stock that rose over the last week or so is predicted to fall. `rev_1` (yesterday) had a positive coefficient in the 1990s: yesterday's move partly continued. `vol_level` is nonzero in 99% of refits, mostly negative. About 12 of 18 coefficients are nonzero on average.

### Leakage controls

**14. What do the positive and negative controls show?**
Positive: shifting every feature one day into the future gives a Sharpe of 68, against 4.06 for the real Lasso. So if there were leakage, the numbers would show it loudly. Negative: shuffling the training targets among stocks within each day gives -0.02. So the pipeline produces nothing when there is nothing to learn.

**15. The negative control has a Sharpe of 1.22 in 1990-1994. Is that a problem?**
No. A Lasso fitted on shuffled targets still outputs a random mix of real features, and features like reversal have a strong true effect. Whichever random sign the fit lands on persists for many refits, because consecutive 3-year windows overlap by 92%. Over the whole period the signs cancel (-0.02 overall; the four windows are 1.22, -0.05, 0.06, -0.67).

**16. What does the truncation test prove, and what does it not prove?**
It recomputes every feature from data cut at day t and checks that row t is unchanged, on 20 dates plus 30 sentiment dates. It proves the feature code does not look ahead. It cannot detect lookahead that is already in the input data, such as the static sector codes. For the full pipeline there is a second check: with all raw data after 2012-06-29 removed, the Lasso and forest predictions up to that date are bit-for-bit identical.

**17. What guarantees the test period was not used for selection?**
The development panel ends on 2010-12-31, and the scoring function refuses any end date after it. The finalists were written to `final_configs.json` and committed by a group member (`f817d04`) before any test-period run. The test stage refuses to run twice. A test checks that the file still equals the committed version.

### Results

**18. How many configurations were tried, and why does it matter?**
62 development rows before the freeze, of which 55 were eligible configurations and 7 were controls or diagnostics. The best of many tries is biased upward. The standard error of a 21-year Sharpe is about 0.22, so differences inside a family (5.50 against 5.19) mean little. That is why the test period was held back and used once.

**19. How should a Sharpe ratio be read? What is its standard error?**
Roughly 1 divided by the square root of the number of years: 0.22 for 21 years, 0.58 for the 3-year test period, 1.0 for one year. A test Sharpe of 3.65 is about 6 standard errors from zero; the difference between 3.65 and 3.01 is about one.

**20. Why did the forest beat its expected range, and how was that checked?**
The range (1.2 to 3.3) was written before the test, from the forest's 2005-2010 Sharpe of 2.70 plus or minus one standard error, with a lower end for continued decay. The result was 3.65. Five checks followed: the development Sharpes reproduced exactly from the Phase 3 predictions; the reversal baseline matched the group's notebook year by year; in all 147 refits the last training day precedes the refit date; with all data after 2012-06-29 removed, the forest's predictions up to that date were identical; the frozen files were unchanged. No bug was found. The likely reason is that 2.70 was a pessimistic reference: it is pulled down by 2008, when the P&L was very volatile. The forest's yearly Sharpes for 2005-2010 average 3.02, and 2009-2010 gives 3.18. Its IC did not rise on the test period (0.018 against 0.020); the P&L was simply steadier. This explanation was found after seeing the result, so it is weaker than a prediction. The result is 1.6 standard errors above 2.70.

**21. What does the delayed-execution result mean?**
The harness assumes a position is taken at the close of day t using that same close. If instead every prediction is used one day late, the test Sharpe of the forest falls from 3.65 to 1.24 and the MLP from 3.01 to 1.02; they keep one third. The Lasso keeps two thirds (0.72) and the reversal rule almost everything (0.69). So most of what the forest and the MLP add over the simple rule on 2011-2013 comes from the most recent day's information and disappears within a day. Whether that is tradable depends on being able to trade at the close, and on costs. On the development period the same delay cost only one third. This analysis was run after the test results were known.

**22. What does the market-hedged result mean? Why is there market exposure when predictions are demeaned?**
Each daily P&L was regressed on the equal-weight market return. All four finalists have a clearly positive beta (t-statistics 9 to 12; correlation 0.31 to 0.40). Demeaning makes the positions sum to zero in volatility-adjusted units, but that is not the same as zero beta: if the strategy tends to be long stocks that move more with the market, it still gains when the market rises. Removing the market part lowers the forest from 3.65 to 3.50 and the MLP from 3.01 to 2.81, so their results are not a market effect. It lowers the Lasso from 1.07 to 0.70. The market itself had a Sharpe of 1.10 over the period.

**23. Why does the Lasso barely beat the reversal rule on the test period?**
1.07 against 0.73 is a difference of 0.6 standard errors; hedged it is 0.70 against 0.56, and a day late 0.72 against 0.69. Three reasons fit the evidence. First, the Lasso is a linear mix of reversal-type features, and its largest coefficient is on a smoothed reversal, so it is close to the rule by construction. Second, the linear relationships have faded: the sum of the absolute coefficients averaged 0.24 over the 1990-1994 refits, 0.18 over 2005-2010 and 0.08 over the test-period refits. Third, what still worked in 2011-2013 was non-linear, since the forest and the MLP did much better with the same features. The Lasso and the rule both lost about 40% of their 2005-2010 Sharpe, as the "decay continues" scenario predicted.

**24. Why is sentiment in no finalist?**
It starts in 2009-07, so only 2010 was available to judge it, with a standard error of about 1.0. Only 12% of members have a score on a given day. Adding sentiment to the Lasso changed the 2010 Sharpe from 0.43 to 0.32 and from 0.37 to 0.37 at the two penalties; sentiment alone gave -0.90. Nothing there supports using it, and nothing rules it out with more data.

**25. What is the holding period, and why does smoothing predictions lower the Sharpe?**
Per stock, the average absolute prediction divided by the average absolute daily change of the prediction; about 1 means the position is replaced every day. The signal is strongest for the next day, so averaging predictions over several days mixes in stale information. On development data a half-life of 5 days took the MLP from 5.39 at 1.3 days to 2.53 at 5.6 days. The smoothed MLP still scored 1.40 on the test period, above the 0.6 benchmark for long holding periods, but 2011 contributed 0.11.

**26. What are the main limitations, and what about the test that failed once?**
No transaction costs, although the best models trade the whole book daily. A same-close execution assumption that the best results depend on. One 3-year test period. Many configurations tried, with grids that were never exhausted. Static sector codes. The delayed and hedged analyses were run after the test. Separately, one leakage test failed in one run and passed in every other. No test was found to depend on shared or mutable state, the feature code was recomputed many times with bit-identical results, and five further full runs with complete output are recorded in `AGENT_LOG.md` (Problem 7). The cause is not known.
