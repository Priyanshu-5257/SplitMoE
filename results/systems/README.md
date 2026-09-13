# Standardized systems benchmark

The raw records benchmark deterministic randomly initialized Top-4 architectures on one Tesla T4 using FP16 autocast, batch 4, context 256, ten warmup forwards, and fifty measured forwards. Logging, validation, checkpointing, and backward computation are absent. This isolates reference-implementation forward cost but does not reproduce trained routing distributions.

`metrics.csv` and `standardized_t4.png` are generated from `raw/` with:

```bash
python scripts/summarize_systems_results.py
```

Report these measurements alongside the observed training throughput in `results/paper_scaling`, not as a replacement for it.
