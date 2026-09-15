# Reviewer-control results

This directory contains the completed batch-matched Top-1 output-scale factorial and equal-storage control. Every cell uses the same 8-layer, width-512, all-MoE setup, 8 experts, Top-1 routing, 6,500 optimizer steps, three paired seeds (`1337`, `2027`, `3407`), effective batch 128, and 212,992,000 training token positions.

The five model cells are:

- Standard-1024 at output scales `1` and `1/sqrt(2)`;
- Split-25 (`256 shared + 768 private`) at output scales `1` and `1/sqrt(2)`;
- Standard-800 at output scale `1`, which exactly matches Split-25's total storage.

Artifacts:

- `final_metrics.csv`: one final record per model and seed;
- `validation_trajectories.csv`: all 26 fixed validation checkpoints per run;
- `summary.json`: means, paired differences, confidence intervals, domain losses, systems summaries, and provenance;
- `output_scale_factorial.png`: the batch-matched architecture-by-scale comparison;
- `equal_storage_paired.png`: paired Standard-800 versus Split-25 results.

Confidence intervals are two-sided 95% Student's $t$ intervals over the three training seeds. The first experiment launch exposed an effective-batch mismatch between one-T4 Modal and two-T4 Kaggle jobs. Those incompatible runs are not included here. The corrective configurations were committed before their runs in commit [`fa9e55b`](https://github.com/Priyanshu-5257/SplitMoE/commit/fa9e55b), following the original preregistration in [`60f2f68`](https://github.com/Priyanshu-5257/SplitMoE/commit/60f2f68).

`scripts/export_reviewer_controls.py` documents the W&B run IDs and Kaggle artifact URLs used to construct these files. Two cells were logged in offline W&B mode; reproducing the export from raw histories requires downloading those six Kaggle outputs and passing their parent directory with `--offline-root`. The committed CSV and JSON files preserve the complete paper-facing measurements without requiring those large outputs.
