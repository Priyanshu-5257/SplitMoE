# SplitMoE paper plan

This document separates evidence already collected from work still needed for a defensible resource-conscious paper. It is a planning document, not a list of claims that the current experiments have already established.

## Candidate contribution

Sparse MoE layers may waste storage when independent experts relearn transformations that are useful to every token. SplitMoE decomposes the active FFN capacity into one shared component and routed private components:

$$
F_{\mathrm{Split}}(x)
=
S(x)+\sum_{i\in\operatorname{TopK}(x)}p_i(x)P_i(x).
$$

The proposed paper should test the following narrow claim:

> A partial-width shared FFN combined with smaller routed private FFNs can improve the validation-quality versus stored-parameter tradeoff of a conventional sparse MoE, while retaining meaningful expert specialization.

The paper should not claim that shared experts are new, that the shared branch semantically contains all “common knowledge,” or that results at sub-billion-parameter scale automatically extend to frontier models.

## Evidence already completed

### Five-seed architecture and parameter frontier

All models below were trained from scratch for 10,000 optimizer steps with the same data, tokenizer, effective batch, and five paired seeds.

| Evidence | Main result | Status |
| --- | --- | --- |
| Dense vs Standard-1024 vs Split-50 | Split-50 removes 9.44M stored parameters relative to Standard at matched activated parameters, with a paired loss difference of `+0.00317` and 95% CI `[−0.00283, +0.00917]` | Complete |
| Shared/private width sweep | Split-25 has the best observed loss/storage tradeoff among the tested ratios | Complete |
| Split-25 vs Standard-1024 | 18.75% less MoE-FFN storage; paired loss difference `−0.00138`, 95% CI `[−0.00604, +0.00328]` | Complete; no detected degradation, not proof of equivalence |
| Equal total storage | Split-50 beats parameter-matched Standard-640 in 5/5 seeds by `−0.01437`, 95% CI `[−0.01982, −0.00893]` | Complete |
| Domain-stratified validation | Stories, Wikipedia, code, and mathematics reported separately | Complete |
| Training efficiency | Throughput, total parameters, activated parameters, and convergence curves reported | Complete for the small models |

The raw metrics and plots are stored under `results/five_seed` and `results/frontier`.

### Mechanism evidence

| Evidence | Main result | Status |
| --- | --- | --- |
| Shared-only/private-only ablations | Both branches contribute to the trained model | Complete |
| All-wrong-private intervention | Every alternative private expert increases loss; a wrong private path is worse than omitting the private path | Complete |
| Layerwise wrong-expert intervention | Correct private routing matters independently in every tested MoE layer | Complete |
| Same-width redundancy control | Width-512 Split private experts have lower centered cosine and linear CKA than width-512 conventional experts | Complete |
| Routing/load diagnostics | No dead experts in the reported small-model runs | Complete |

The causal and representation results are stored under `results/mechanism`.

### Initial scaling evidence

The completed single-seed 10-layer, width-640, 16-expert Top-4 experiment found:

| Model | Total params | Activated params/token | Validation LM loss | Median throughput |
| --- | ---: | ---: | ---: | ---: |
| Standard-16E Top-4 | 256.97M | 109.51M | **2.91064** | **12,754 tok/s** |
| Split-25-16E Top-4 | **210.89M** | **100.29M** | 2.91835 | 11,666 tok/s |

Split used 17.9% fewer stored parameters and 8.4% fewer activated parameters, with `+0.00771` validation loss. Standard was better at all 40 evaluation checkpoints. Split was also 8.5% slower in measured median throughput, showing that activated parameter count is not a substitute for hardware measurement.

This is useful exploratory evidence but has only one seed. Its raw W&B export and figures are committed to the repository.

## Experiment currently prepared

The all-MoE scaling suite uses 12 Transformer layers, an MoE in all 12 layers, width 640, 16 experts, and Top-4 routing.

| Model | Total params | Activated params/token |
| --- | ---: | ---: |
| XLarge all-MoE Standard | 523.28M | 169.39M |
| XLarge all-MoE Split-25 | 412.69M | 147.27M |

The suite is defined in `configs/xlarge_allmoe_16e_top4.json`. It is a single-seed exploratory gate, not yet confirmatory evidence.

### Preregistered resource-conscious replacement

The interrupted 12-layer XLarge gate is superseded for the first preprint by an 8-layer, width-512, all-MoE experiment with 16 experts and Top-4 routing. This replacement was completed for paired seeds 1337, 2027, and 3407. Standard had mean final validation LM loss `3.11376`; the practical Split-25 configuration had `3.12738`, used 20.1% fewer total parameters and 11.2% fewer activated parameters, and had a paired difference of `+0.01362` with 95% CI `[−0.00725, +0.03448]`. The practical comparison therefore shows an efficiency tradeoff, not a quality win.

Before running the next control, the equal-activated-capacity Split configuration is frozen as shared width 256 and private width 960:

$$
256+4(960)=4(1024)=4096.
$$

It will be trained for the same 6,500 steps and paired seeds 1337, 2027, and 3407, using the same data order, effective batch size, validation sampler, optimizer, and Top-4 probability routing as the completed comparison. Final balanced validation LM loss is the primary outcome. The preregistered non-inferiority margin is `0.01` LM loss, approximately a 1% perplexity increase; non-inferiority requires the upper endpoint of the paired 95% confidence interval for Split minus Standard to be below `0.01`.

The equal-active control is complete. Its mean difference from Standard was `+0.00053`, with paired 95% CI `[−0.01815, +0.01921]`. It recovered approximately 96% of practical Split's observed deficit, but the upper confidence endpoint exceeded the preregistered `+0.01` margin, so formal non-inferiority was not established.

## Confirmatory work completed for the first preprint

### 1. Complete the all-MoE scaling gate

Standard and Split-25 were completed with three paired seeds. Final and best validation LM loss, domain losses, complete validation trajectories, throughput, runtime, peak VRAM, router entropy, expert load, and shared/private norm ratios are reported under `results/paper_scaling`.

Decision rule:

- If Split remains close while saving substantial storage, repeat both models with at least two additional paired seeds.
- If Split is clearly worse throughout training, do not spend compute repeating the same result. Report it as a scaling limitation and focus the paper on the demonstrated smaller-scale frontier.
- Define what “close” means before examining additional seeds. A non-inferiority margin must be justified rather than chosen after observing results.

### 2. Add an equal-activated-compute control

The current Top-4 comparison intentionally gives Split less active FFN capacity:

$$
A_{\mathrm{Standard}}=4(1280)=5120,
$$

$$
A_{\mathrm{Split25}}=320+4(960)=4160.
$$

To match Standard's active width while retaining a width-320 shared path, use private width 1200:

$$
320+4(1200)=5120.
$$

This control answers whether sharing helps at equal activated capacity. The existing Split-25 comparison answers the different practical question of how much quality remains when both storage and activated capacity are reduced. Both comparisons belong in the paper.

This control can be run at the cheaper 10-layer scale if the all-MoE version is too expensive.

### 3. Add a shared-expert baseline

The broad idea of always-active shared experts has precedent. Compare SplitMoE against a conventional shared-expert-plus-routed-experts design under a clearly specified matching rule. At least one comparison should match activated FFN width; ideally another should match total stored parameters.

The exact primary baseline is frozen before training as one always-active width-1024 expert plus 15 width-1024 routed experts, with Top-3 routing. It therefore stores 16 full-width FFNs per layer and activates four full-width FFNs per token, matching Standard-16E Top-4 at the FFN level:

$$
1(1024)+15(1024)=16(1024),
$$

$$
1(1024)+3(1024)=4(1024).
$$

The router has 15 rather than 16 outputs, leaving the full model 4,096 parameters smaller than Standard; this negligible difference will be disclosed rather than hidden. The baseline uses the same three paired seeds, 6,500 steps, data order, effective batch, optimizer, probability routing, and branch-output scale as the corresponding experiments. This baseline distinguishes the contribution of explicitly dividing active capacity from merely making one full expert always active.

The shared-expert baseline is complete. It beat Standard in all three paired seeds by mean `−0.01393`, with 95% CI `[−0.02463, −0.00323]`, and had the better mean at all 26 validation checkpoints. It is the strongest tested Top-4 model. This rules out claiming universal superiority for partial-width SplitMoE and narrows the contribution to the measured quality/storage frontier and Top-1 specialization evidence.

### 4. Evaluate on an independent held-out source

The primary Top-1 comparison and strongest matched Top-4 allocation comparison are evaluated on the 5,153-example English LAMBADA test split, which was not used to construct the four-source training mixture. The exact dataset and tokenizer revisions, preprocessing, checksums, raw per-seed results, and evaluation script are published under `results/heldout` and `src/splitmoe`.

This evaluation reuses existing model-only checkpoints and introduces no additional training.

Split-25 improved Top-1 full-token loss in all three paired seeds by mean `−0.06372`, 95% CI `[−0.10240, −0.02504]`. Under Top-4, the full shared expert was numerically better in every seed; the equal-active-Split-minus-shared mean was `+0.01019`, with 95% CI `[−0.00762, +0.02801]`, so the held-out difference was not statistically resolved.

### 5. Complete the novelty and related-work review

Search primary sources for:

- shared experts and shared-expert isolation;
- fine-grained and segmented MoE experts;
- shared-base plus expert-residual parameterizations;
- low-rank or parameter-efficient experts;
- expert merging, compression, and redundancy analysis;
- MoE routing and load-balancing methods.

The primary-source audit is complete in `paper/RELATED_WORK.md`. The manuscript limits the contribution to the controlled partial-width factorization and its empirical parameter/specialization evidence, not the general existence of shared experts.

## Strongly recommended, but not blocking the first preprint

- Evaluate expert similarity on a common probe bank so Standard and Split experts receive exactly the same input vectors.
- Measure tokens per second and peak VRAM using a short standardized benchmark independent of W&B logging and validation time.
- Report approximate forward FLOPs in addition to activated parameter counts.
- Repeat the most important mechanism analysis at the larger scale.
- Release model-only checkpoints when storage permits; optimizer states are not necessary for evaluation releases.
- Test one additional expert count only if the all-MoE result suggests a clear scaling trend.

## Reviewer-control experiment: output scaling and matched storage

This experiment was frozen before training in response to the main reviewer concern. It uses the existing 8-layer, width-512, all-MoE, 8-expert Top-1 setting; 6,500 optimizer steps; seeds 1337, 2027, and 3407; and the identical tokenizer, pretokenized data, effective batch, optimizer, validation sampler, and routing implementation used by the completed scaling comparison.

The primary 2-by-2 control varies architecture and FFN output scale while holding the Standard and Split active FFN width at 1,024:

| Architecture | Output scale $1$ | Output scale $1/\sqrt{2}$ |
| --- | --- | --- |
| Standard-1024 | `review_standard1024_scale1_batch128.json` | `review_standard1024_scale07071.json` |
| Split-25 (256 shared + 768 private) | `review_split25_scale1.json` | `review_split25_scale07071_batch128.json` |

The first Kaggle launch exposed a hardware-dependent batch-size mismatch that was recorded before the missing cells were trained: the earlier Modal checkpoints used one T4 and effective batch 64, whereas the new two-T4 Kaggle jobs used effective batch 128. Those checkpoints are not pooled into one factorial analysis. The remaining two cells above complete the factorial entirely at effective batch 128 (212,992,000 training-token positions). The earlier effective-batch-64 results remain a separate training-budget experiment.

The two completion configs log W&B runs in offline mode on Kaggle. Their complete local histories are downloaded and synced after training; no W&B credential is embedded in the Kaggle notebooks.

The primary outcome is final domain-balanced validation LM loss at step 6,500. Analyses use paired seed differences and two-sided 95% Student-$t$ confidence intervals over the three seeds. The predeclared comparisons are:

1. Split versus Standard at output scale 1;
2. Split versus Standard at output scale $1/\sqrt{2}$;
3. the within-architecture output-scale effect;
4. the architecture-by-scale interaction (difference of the two architecture contrasts).

No confidence interval crossing zero will be described as equivalence. This factorial control separates the effect of sharing from the effect of scaling the residual FFN output. A two-private-branch control is unnecessary because two parallel partial-width SwiGLU branches routed together are algebraically equivalent to one wider conventional expert under the implementation used here.

An additional equal-storage baseline, `review_standard800_scale1.json`, compares width-800 Standard MoE with Split-25 at output scale 1. Both contain exactly 112,370,176 total parameters. Split activates more parameters per token, so this is explicitly a storage/quality comparison rather than an equal-compute comparison.

All nine new runs use separate W&B namespaces:

- project `splitmoe-paper-review-scale` for the six factorial-control runs;
- project `splitmoe-paper-review-storage` for the three equal-storage runs.

The training configs support `--seed` so each Kaggle kernel has an independent seed-specific run name and checkpoint directory without changing the notebook launch arguments otherwise.

## Not necessary for an individual-researcher paper

- Training a frontier-scale or multi-billion-parameter model.
- Exhaustively sweeping every shared fraction, expert count, Top-K value, and router loss.
- Reproducing every benchmark used by industrial foundation models.
- Claiming universal superiority over all MoE implementations.

A transparent sub-billion-parameter study with paired seeds, public code, raw metrics, controls, and carefully bounded claims is a valid research contribution.

## Statistical protocol

- Treat training seeds—not validation checkpoints—as independent samples.
- Use paired seed differences whenever two architectures share the same seeds and data order.
- Report individual seeds, means, effect sizes, and 95% confidence intervals.
- Do not interpret a confidence interval crossing zero as proof of equivalence.
- Predeclare a non-inferiority margin before confirmatory scaling runs if non-inferiority is the intended claim.
- Clearly label single-seed scaling results as exploratory.
- Keep LM loss as the primary quality outcome; auxiliary router loss must be reported separately rather than folded into the headline metric.

## Paper figures and tables

### Main paper

1. Architecture diagram: Standard MoE versus shared/private SplitMoE.
2. Quality versus stored-parameter Pareto frontier.
3. Quality versus activated parameters and measured throughput.
4. Scaling plot showing model size against paired Split-minus-Standard loss.
5. Width allocation sweep.
6. Same-width expert redundancy comparison.
7. Wrong-private and layerwise causal interventions.
8. Main table with parameter count, activated parameters, FLOPs, throughput, VRAM, and domain losses.

### Appendix

- Per-seed learning curves and final values.
- Router loads and entropy by layer.
- Shared/private norm histories.
- Dataset composition and preprocessing.
- Hyperparameters and complete configuration files.
- Additional domain and ablation tables.

## Suggested paper structure

1. Abstract
2. Introduction and hypothesis
3. Related work
4. SplitMoE formulation and parameter analysis
5. Experimental design and matching rules
6. Parameter-efficiency results
7. Scaling results
8. Causal and representation analysis
9. Systems performance
10. Limitations
11. Conclusion

## Reproducibility release checklist

- [x] Public training implementation
- [x] Pretokenized-data preparation pipeline
- [x] Exact small-model configs
- [x] Five-seed raw metrics and plots
- [x] Causal and similarity analysis code
- [x] W&B logging
- [x] Export and commit the completed 16E Top-4 results
- [x] Complete and export the all-MoE scaling runs
- [x] Record peak VRAM and standardized throughput
- [x] Add equal-active-compute control
- [x] Add shared-expert baseline
- [x] Add independent held-out evaluation
- [x] Publish dataset revisions, checksums, and preprocessing manifest
- [x] Decide checkpoint policy: publish exact configs and metrics; do not bundle large model weights in the repository
- [x] Complete primary-source related-work review
- [x] Add licenses and citation metadata appropriate for a paper release

## Resource-conscious stopping point

For a first preprint, stop training after all of the following are available:

1. the existing five-seed small-model frontier and mechanism evidence;
2. one completed larger-scale comparison, with additional paired seeds only if the exploratory result justifies them;
3. one equal-active-compute control;
4. one conventional shared-expert baseline;
5. independent held-out evaluation and standardized systems measurements.

Anything beyond this list should answer a specific reviewer question rather than simply making the experiment grid larger.
