# SplitMoE

SplitMoE is a compact research codebase for testing whether an MoE layer benefits from explicitly splitting active FFN capacity into a shared path and a routed private path:

\[
y = x + \frac{1}{\sqrt{2}}\left(S(x) + P_{i^*}(x)\right).
\]

It includes compute-matched Dense, standard Top-1 MoE, and SplitMoE configurations; streaming pretokenization into memory-mapped blocks; single-GPU and PyTorch DDP training; Weights & Biases logging; domain-conditioned routing statistics; shared/private norm tracking; and post-training expert-similarity analysis.

## Kaggle: pull and run

Enable both T4 GPUs and internet in the Kaggle notebook settings, then run:

```bash
git clone https://github.com/Priyanshu-5257/SplitMoE.git
cd SplitMoE
pip install -e .
wandb login
```

Pretokenize the four-domain corpus once. This streams documents instead of downloading the complete source datasets and writes fixed `(context + 1)` token blocks. Training then reads them through `numpy.memmap`, so tokenization is entirely outside the hot training loop.

```bash
python -m splitmoe.prepare_data \
  --sources data_sources.example.json \
  --output-dir data \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --block-size 256 \
  --validation-fraction 0.01
```

The example caps each domain at 100,000 blocks (about 25.6M input tokens), preventing corpus size from becoming a routing confound. Reduce each `max_blocks` for a short Kaggle run or raise it for the full experiment. Check `data/train/metadata.json` after preprocessing: its `vocab_size` must match `model.vocab_size` in every experiment config.

The code portion uses the script-free, auto-converted Parquet view of `codeparrot/codeparrot-clean`, which is compatible with current versions of Hugging Face `datasets` used by Kaggle.

Launch the compute-matched experiments on both GPUs:

```bash
torchrun --standalone --nproc_per_node=2 -m splitmoe.train --config configs/dense.json
torchrun --standalone --nproc_per_node=2 -m splitmoe.train --config configs/standard.json
torchrun --standalone --nproc_per_node=2 -m splitmoe.train --config configs/split.json
```

Each GPU holds the complete model and receives different batches. There is no expert-parallel all-to-all communication, keeping this an architecture experiment rather than a distributed-systems comparison.

If T4 memory is tight, lower `micro_batch_size` and raise `gradient_accumulation_steps` by the same factor. T4 should use FP16, not BF16.

## Local verification

No dataset download or W&B account is needed for tests:

```bash
pip install -e '.[dev]'
pytest -q
python scripts/smoke_test.py
python scripts/smoke_test.py --ddp
```

The smoke test creates a temporary memory-mapped dataset, performs optimizer steps, evaluates, and saves a checkpoint.

## Experiment controls

The supplied Standard and Split configurations match active SwiGLU width:

- Standard: one selected expert of width 1024.
- Split: one always-active shared FFN of width 512 plus one selected private FFN of width 512.

This is a compute-matched comparison, not a total-parameter-matched comparison. To parameter-match a 4-expert Standard MoE with shared width 512, use private width 896, since

\[
512 + 4(896) = 4(1024).
\]

Run both comparisons before making claims about architectural quality versus parameter efficiency.

The default router mode is `straight_through`. Its selected gate is divided by a detached copy, producing a forward scale of exactly one while retaining task gradients for the router. Available modes are:

- `straight_through`: unattenuated expert output with router task gradients; recommended first run.
- `probability`: multiply the selected expert by its softmax probability.
- `none`: hard unweighted routing; the router then learns only from auxiliary router losses.

## Logged measurements

W&B receives language-model and total loss, perplexity, learning rate, gradient norm, tokens/second, router entropy, load per expert, shared/private activation norms and their ratio, and `P(expert | domain)` during validation.

Measure centered expert-output similarity after training:

```bash
python -m splitmoe.analyze --checkpoint checkpoints/split/final.pt --batches 8
```

The output is one expert-by-expert similarity matrix per MoE layer. Treat similarity as a diagnostic rather than standalone proof of specialization; loss ablations and wrong-expert substitution are stronger follow-up tests.

## Reproducibility notes

- All variants use the same default seed and pretokenized blocks.
- Auxiliary load balancing and router z-loss are included in the reported total loss; `lm_loss` is logged separately.
- Data are packed within each domain, so a block has one domain label and domain-routing measurements are not contaminated by cross-domain packing.
- No capacity-based token dropping is used in this single-node implementation.
- Checkpoints are written atomically and include model, optimizer, scaler, step, and full configuration.

The referenced datasets retain their own licenses. Review their dataset cards before redistributing either raw or pretokenized data.
