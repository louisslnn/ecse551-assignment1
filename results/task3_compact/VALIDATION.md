# Compact notebook validation

Validated on 22 October 2026, based on remote main `c975847`, on
`feature/task3-notebook`. The earlier PR was not merged.

## Executed deliverable

- Combined notebook: 68 cells, all 41 code cells executed, zero error outputs.
- Task 3: 28 cells, including 16 code cells (432 code lines) and five inline figures.
- All 52 figures across the combined notebook are saved inline.
- The full notebook executed from a fresh kernel in an isolated directory containing
  only the notebook and supplied CSV files. A second, independent run executed just
  Task 3 in another isolated directory. Neither used repository imports or old results.
- The independent runs reproduced the displayed development, selected-configuration
  and test tables. Task 3 took 8.2 minutes in the combined run; total notebook time
  was about 9.9 minutes on this machine while both runs were active.

## Correctness and scope checks

- 27 behavioral/source-preservation tests passed. The tests load the function
  definitions directly from the submitted notebook.
- 319 recorded refits passed the strict `last_training_outcome < refit_date` assertion.
- All 754 test outcome days were scored, with at least 491 stocks every day.
- Real-data truncation at 2010-06-30 and 2013-12-30 produced exactly unchanged
  features and Lasso predictions: maximum absolute differences were zero.
- Synthetic tests cover year/holiday boundaries, current-day label exclusion,
  missing-day/low-coverage failures, independent P&L/IC/Sharpe/market calculations,
  holding-period gaps and zero turnover, constant forecasts and delayed sentiment.
- Task 1/2 source and experiment behavior match main after removing only local
  import/path bootstrap code. All existing source modules, the old report and
  historical result files remain unchanged.
- Every Task 3 grading category is mapped to notebook evidence in its appendix.

## Reading the accompanying files

`validation_summary.json` records counts, source/notebook/data hashes and truncation
checks. The three CSV files reproduce the **displayed** tables, at notebook display
precision; they are not notebook dependencies. Figures are also saved here as PNG
copies for inspection.

## Deliberately preserved limitations

The compact study follows already viewed test results, so its evaluation is not a
fresh blind test. Returns assume same-close execution and omit trading costs; the
notebook includes a one-day delay diagnostic.

Task 1/2 were packaged, not corrected or certified. Two observed existing issues
remain: classification uses `test_size=0.7` instead of the specified 0.3, and the
Lasso update subtracts its weight gradient twice before soft thresholding.

The assignment PDF separately requests a report PDF. The notebook-only delivery
follows the user's instruction and does not waive that stated requirement.
