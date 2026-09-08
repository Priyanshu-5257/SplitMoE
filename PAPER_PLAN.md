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

This is useful exploratory evidence but has only one seed. Its raw W&B export and figures still need to be committed to the repository.

## Experiment currently prepared

The all-MoE scaling suite uses 12 Transformer layers, an MoE in all 12 layers, width 640, 16 experts, and Top-4 routing.

| Model | Total params | Activated params/token |
| --- | ---: | ---: |
| XLarge all-MoE Standard | 523.28M | 169.39M |
| XLarge all-MoE Split-25 | 412.69M | 147.27M |

The suite is defined in `configs/xlarge_allmoe_16e_top4.json`. It is a single-seed exploratory gate, not yet confirmatory evidence.

## Required before a paper submission

### 1. Complete the all-MoE scaling gate

Train Standard and Split-25 with seed 1337. Report final and best validation LM loss, domain losses, the complete validation trajectory, throughput, runtime, peak VRAM, router entropy, expert load, and shared/private norm ratios.

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

The exact widths and primary comparison must be frozen before training. This baseline distinguishes the contribution of explicitly dividing active capacity from merely adding an always-on expert.

### 4. Evaluate on an independent held-out source

Keep the current balanced four-domain validation, but add at least one recognized held-out text corpus that was not used to construct the training set. Publish the exact dataset revision, split, preprocessing, tokenizer, sample count, and evaluation script.

This does not require another large training run. Existing checkpoints can be evaluated on the additional corpus.

### 5. Complete the novelty and related-work review

Search primary sources for:

- shared experts and shared-expert isolation;
- fine-grained and segmented MoE experts;
- shared-base plus expert-residual parameterizations;
- low-rank or parameter-efficient experts;
- expert merging, compression, and redundancy analysis;
- MoE routing and load-balancing methods.

The final novelty statement must be written only after this review. The likely contribution is the controlled partial-width factorization and its empirical parameter/specialization evidence, not the general existence of shared experts.

## Strongly recommended, but not blocking the first preprint

- Evaluate expert similarity on a common probe bank so Standard and Split experts receive exactly the same input vectors.
- Measure tokens per second and peak VRAM using a short standardized benchmark independent of W&B logging and validation time.
- Report approximate forward FLOPs in addition to activated parameter counts.
- Repeat the most important mechanism analysis at the larger scale.
- Release model-only checkpoints when storage permits; optimizer states are not necessary for evaluation releases.
- Test one additional expert count only if the all-MoE result suggests a clear scaling trend.

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
- [ ] Export and commit the completed 16E Top-4 results
- [ ] Complete and export the all-MoE scaling runs
- [ ] Record peak VRAM and standardized throughput
- [ ] Add equal-active-compute control
- [ ] Add shared-expert baseline
- [ ] Add independent held-out evaluation
- [ ] Publish dataset revisions, checksums, and preprocessing manifest
- [ ] Decide whether model-only checkpoints can be hosted
- [ ] Complete primary-source related-work review
- [ ] Add licenses and citation metadata appropriate for a paper release

## Resource-conscious stopping point

For a first preprint, stop training after all of the following are available:

1. the existing five-seed small-model frontier and mechanism evidence;
2. one completed larger-scale comparison, with additional paired seeds only if the exploratory result justifies them;
3. one equal-active-compute control;
4. one conventional shared-expert baseline;
5. independent held-out evaluation and standardized systems measurements.

Anything beyond this list should answer a specific reviewer question rather than simply making the experiment grid larger.
