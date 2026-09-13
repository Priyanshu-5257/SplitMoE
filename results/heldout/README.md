# Independent held-out evaluation

This directory contains the three-seed English LAMBADA evaluation used in the paper.

- `dataset_manifest.json`: immutable dataset/tokenizer revisions, preprocessing, counts, license, and checksums.
- `metrics.csv`: one row per architecture and training seed.
- `summary.json`: marginal means and paired Student's t intervals.
- `raw/`: complete per-checkpoint metrics and routing diagnostics.
- `top1_lambada.png` and `top4_lambada.png`: generated paired figures.

All reported records score 5,153 documents through the CPU FP32 path. Full-token autoregressive LM loss is primary. Target-word loss and exact argmax token-sequence match use teacher forcing and are diagnostics, not free-running LAMBADA accuracy.

From the repository root, regenerate the aggregate files with:

```bash
python scripts/summarize_heldout_results.py
```
