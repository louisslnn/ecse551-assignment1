# ECSE 551 Assignment 1

The submission notebook is
`notebooks/Poon_Robatche-Claive_Salanon_Xu_Assignment1.ipynb`.
It includes the existing Diabetes regression and Digits classification code
for Tasks 1/2, followed by one self-contained, compact Task 3 study.

## Run the notebook

1. Use Python 3.12 or later. The existing Task 1/2 notebook uses Python 3.12 syntax.
2. Install `requirements-notebook.txt` into the notebook's Python environment.
3. Put the supplied CSVs in `instructions-data/`, or edit the single `DATA_DIR`
   setting in Task 3 to their directory. Opening from the repository root or
   `notebooks/` is supported. The raw data are not committed.
4. Restart the kernel and run all cells in order. Outputs are saved in the
   submitted notebook. Task 3 requires no project modules, caches, configuration
   files or previously saved results.

The portable submission consists of the notebook, the supplied CSVs and the
installed packages. The repository's older scripts and experiments are historical
evidence and are not imported by the submission notebook.

## Compact Task 3 protocol

- Five ranked return features; Lasso, random forest and one-hidden-layer MLP.
- Eleven tuning configurations, two matched sentiment comparisons, two baselines
  and one fixed smoothed MLP: 16 development rows in total.
- Outcomes in 1990–2010 for development and 2011–2013 for evaluation.
- Only labels realized strictly before each prediction may enter a fit.
- Rolling 756-trading-day training history, refits every 252 trading days, a
  deterministic 100,000-stock-day training cap and seed 42.
- Membership on the prediction date; at least 100 predictions and scored stocks
  every expected evaluation day.
- Assignment P&L, Sharpe, daily/cumulative rank IC, holding period, raw-market
  correlation, yearly stability and a one-day execution-delay diagnostic.

The notebook explains losses, tuning, rationale, metrics, interpretation and LLM
usage, and includes a mapping to every Task 3 rubric category. This redesign follows
previously viewed test results: it is **not** a fresh blind test. Original study
outputs remain unchanged. Task 1/2 code and experiment settings are included without
behavioral changes and are outside this Task 3 correctness review.

Two observed Task 1/2 issues are preserved under that boundary: the classification
split uses `test_size=0.7` although the assignment specifies 0.3, and the existing
Lasso weight update subtracts the same gradient twice before soft thresholding.
The compact-notebook checks establish source preservation, not correctness of
those existing implementations.

The assignment PDF also requests a separate report PDF; including report material
in the notebook does not waive that requirement.

## Validation

Run the compact notebook checks with:

```sh
python -m pytest tests/test_compact_notebook.py -q
```

These checks exercise the notebook's actual function definitions, including time
boundaries, future-data truncation, member coverage, hand-calculated metrics,
sentiment timing and source preservation for Tasks 1/2. They do not rely on the
historical tests' frozen experiment numbers.

Execution evidence and published compact-study tables are recorded separately
under `results/task3_compact/`; they are outputs, not notebook inputs.
