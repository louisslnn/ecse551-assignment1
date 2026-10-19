# Additional ChatGPT assistance for Task 3

This record supplements the original Claude prompt and experiment log; it does not replace the group's disclosure of other AI tools.

## Context and requests

The user supplied `551_Assignment_1.pdf` and the Task 3 repository, including its source, notebooks, saved daily outputs, experiment log, frozen configuration, and draft report.

Initial request, quoted verbatim:

> Read the instructions for assignment 1 carefully in the pdf. Then tell me if task 3 was correctly implemented by looking at the repo and if some things need to be added or replaced

The user next requested an implementation plan for notebook reproducibility, the development/test boundary, market benchmarking, holding-period calculation, and training timing, while deferring report completion. The user also requested LaTeX disclosure wording for Overleaf. In the clarification exchange, the user chose to include the date-boundary correction and the stricter cutoff requiring training targets to precede the prediction day.

The user then explicitly requested implementation of the approved plan. Its main constraints were to retain all model families, hyperparameters, features, scheduled refits and seed; preserve historical results; replay all 62 fixed development definitions; generate an independent notebook; validate dates, coverage and numerical summaries; and leave the report unchanged. The user later supplied the CSVs in `instructions-data` and explicitly required that folder to stay out of GitHub.

## ChatGPT's actual contribution

ChatGPT, through Codex, reviewed the assignment and code, ran targeted checks, verified saved summary statistics against daily outputs, proposed the corrections, drafted the implementation plan, and generated disclosure wording. After explicit implementation authorization, it edited the timing, fitting, evaluation, replay and plotting code; created a fixed experiment manifest, portable-notebook generator, isolated execution runner, tests and usage documentation; and added ignore rules for input data and prediction caches.

The original review lacked the raw CSVs. They were supplied later for the corrected run. An isolated execution starts with only the generated notebook and the five actual CSVs; run outputs and caches are created by the notebook. The notebook's execution metadata, corrected run manifest, test logs and saved-output validation establish which checks completed. Neither an unexecuted template nor a failed run establishes numerical completion.

The changes were made after the original test results had been observed. The same finalist identities and settings are retained, so corrected results must be labeled protocol corrections rather than a newly untouched test. No report edits, Git commits, GitHub pushes, new model-selection rounds or extra datasets are part of this work. The group remains responsible for reviewing, understanding and defending the final submission.

## Completed execution and checks

The isolated numerical run completed all 62 development definitions and the original four finalists plus two baselines, with 754 test outcome dates. The notebook saved 12 inline figures and all code cells executed without error. Independent checks verified the saved daily summaries, corrected finalist development reproduction, 2,388 refit records, and unchanged historical artifacts. The 49-test suite and an additional raw-file truncation check passed; the latter compared 1,393 Lasso prediction days with a maximum difference of 1.1641532182693481e-10 after future-only empty securities were removed. Table formatting and the MLP epoch audit were refreshed from those completed outputs without refitting models.
