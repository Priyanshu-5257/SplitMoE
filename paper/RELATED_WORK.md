# Related-work audit and novelty boundary

This audit records the primary sources used to position SplitMoE. It is deliberately narrower than a general MoE survey.

| Area | Primary source | What it establishes | Boundary relative to SplitMoE |
| --- | --- | --- | --- |
| Sparse routed FFNs | Shazeer et al. (2017), *Outrageously Large Neural Networks* | Sparsely gated expert layers provide large conditional capacity | SplitMoE modifies the parameter allocation inside this standard pattern |
| Large-scale Top-K routing | Lepikhin et al. (2020), *GShard* | Scalable Top-2 routing and auxiliary balancing | Used as the conventional independent-expert reference family |
| Top-1 routing and balancing | Fedus et al. (2022), *Switch Transformers* | Simplified Top-1 routing at scale | Motivates the Top-1 setting and load-balancing control |
| Router stability | Zoph et al. (2022), *ST-MoE* | Router z-loss and stability practices | SplitMoE uses auxiliary balance and router z-loss; these are not contributions |
| Fine-grained and shared experts | Dai et al. (2024), *DeepSeekMoE* | Fine segmentation and always-active shared-expert isolation reduce routed-expert redundancy | Closest architectural precedent. Its shared units are complete experts; SplitMoE studies a fixed partial-width shared/private decomposition and controlled allocation frontier |
| Deployed shared experts | Qwen Team (2024), *Qwen1.5-MoE* | Four shared plus sixty routed experts demonstrate practical adoption | Confirms that always-active shared capacity is established, not novel here |
| Low-rank routed capacity | Sarwar et al. (2024), *StructMoE* | Low-rank secondary experts add dynamic capacity | Different structured/secondary routing parameterization |
| Shared base plus deltas in upcycling | Huang et al. (2025), *DeRS* | A shared expert base plus sparse/low-rank expert deltas can parameter-efficiently upcycle dense models | Very close decomposition principle, but aimed at upcycling/compression and lightweight weight deltas rather than from-scratch parallel partial-width FFNs |
| Post-hoc shared-base compression | Gu et al. (2025), *D2-MoE* | Fisher-merged shared bases and low-rank deltas compress pretrained MoEs | Post-training compression rather than an end-to-end architectural allocation study |
| Residual sparsification | Jung et al. (2026), *PARSER* | Output-aware compression improves shared-base/residual sparsification | Reinforces expert redundancy but postdates and differs from from-scratch SplitMoE training |

## Defensible novelty statement

SplitMoE does **not** introduce the general idea of shared experts or shared-base expert residuals. Its contribution is an empirical study of a simple from-scratch architecture in which an MoE layer's active FFN width is explicitly divided between one always-active partial-width FFN and routed partial-width FFNs. The study contributes matched-active and matched-storage controls, a shared-width allocation sweep, paired-seed scaling across Top-1 and Top-4 routing, causal wrong-private interventions, and a same-width expert-output redundancy control.

The evidence supports a conditional statement: partial-width sharing improves the stored-parameter frontier in the tested Top-1 regimes, while under the tested Top-4 regime it becomes a storage/quality tradeoff and a conventional full shared expert gives the best quality at matched FFN storage and activation.

## Sources

- Shazeer et al. (2017): <https://arxiv.org/abs/1701.06538>
- Lepikhin et al. (2020): <https://arxiv.org/abs/2006.16668>
- Fedus et al. (2022): <https://jmlr.org/papers/v23/21-0998.html>
- Zoph et al. (2022): <https://arxiv.org/abs/2202.08906>
- Dai et al. (2024): <https://arxiv.org/abs/2401.06066>
- Qwen Team (2024): <https://qwenlm.github.io/blog/qwen-moe/>
- Sarwar et al. (2024): <https://proceedings.mlr.press/v262/sarwar24a.html>
- Huang et al. (2025): <https://arxiv.org/abs/2503.01359>
- Gu et al. (2025): <https://arxiv.org/abs/2502.17298>
- Jung et al. (2026): <https://arxiv.org/abs/2609.00575>
