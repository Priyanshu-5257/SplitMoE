# Suggested arXiv metadata

## Title

SplitMoE: Partial-Width Shared Capacity for Parameter-Efficient Sparse Mixture-of-Experts Language Models

## Authors

Priyanshu Maurya

## Current affiliation

Independent Researcher

## Corresponding email

mauryac198@gmail.com

## Categories

- Primary: `cs.LG` (Machine Learning)
- Cross-list: `cs.CL` (Computation and Language), if arXiv permits it

## Comments

9 pages, 7 figures, 4 tables. Code and reproducibility artifacts: https://github.com/Priyanshu-5257/SplitMoE

## Abstract

Conventional sparse mixture-of-experts (MoE) layers store every expert as a complete feed-forward network, even though some transformations may be useful across routes. SplitMoE instead gives every token a small shared branch plus routed private branches: common capacity is stored once, while private capacity remains conditional. In an eight-layer, eight-expert Top-1 language model trained for 213M token positions, a 25/75 shared/private split uses 16.4% fewer total parameters than Standard MoE at equal activated capacity. Across three paired seeds, SplitMoE lowers validation loss by 0.05611 at output scale 1 and 0.03569 at scale 1/sqrt(2); both 95% confidence intervals exclude zero. At equal total storage, it lowers loss by 0.06261 while activating 6.3% more parameters. The benefit is not universal: with 16 experts and Top-4 routing, a conventional full shared expert performs best. Route interventions and width-matched similarity measurements indicate that SplitMoE's private branches are route-dependent and less output-redundant. These results support SplitMoE as a regime-dependent parameter-allocation strategy, not a universally superior MoE.

## License decision

Recommended conservative default: arXiv's perpetual, non-exclusive license to distribute. This keeps copyright with the author while allowing arXiv to host the work. Confirm the choice personally on the submission form because an arXiv license selection is irrevocable. If a later venue or funder requires a Creative Commons license, check that policy before submission.

## Submission notes

- The manuscript is a venue-neutral preprint and has not been assigned a journal reference or DOI.
- The training corpus was not deduplicated against LAMBADA; retain the paper's contamination caveat.
- The repository is public at https://github.com/Priyanshu-5257/SplitMoE.
- Do not add IITR as a current affiliation after graduation unless the institute has explicitly authorized that attribution.
