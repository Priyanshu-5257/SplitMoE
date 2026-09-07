import pytest
import torch

from splitmoe.config import ModelConfig
from splitmoe.model import DecoderLM, SplitMoE, TopKRouter


@pytest.mark.parametrize("moe_type", ["dense", "standard", "split"])
def test_forward_backward(moe_type):
    cfg = ModelConfig(
        vocab_size=128, max_seq_len=16, n_layers=2, d_model=32, n_heads=4,
        dense_ffn_width=64, moe_every=1, moe_type=moe_type, n_experts=4,
        standard_expert_width=64, shared_width=32, private_width=32, dropout=0.0,
    )
    model = DecoderLM(cfg)
    inputs = torch.randint(0, cfg.vocab_size, (3, cfg.max_seq_len))
    output = model(inputs, inputs, collect_assignments=True)
    assert output.logits.shape == (3, cfg.max_seq_len, cfg.vocab_size)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    if moe_type != "dense":
        assert len(output.router_stats) == 2
        assert output.router_stats[0].assignments.shape == inputs.shape
        assert model.blocks[0].ffn.router.proj.weight.grad is not None
        assert model.blocks[0].ffn.router.proj.weight.grad.abs().sum() > 0


def test_split_norm_metrics():
    cfg = ModelConfig(
        vocab_size=64, max_seq_len=8, n_layers=1, d_model=16, n_heads=2,
        moe_every=1, moe_type="split", n_experts=2, shared_width=16, private_width=16,
    )
    output = DecoderLM(cfg)(torch.randint(0, 64, (2, 8)))
    stats = output.router_stats[0]
    assert stats.shared_norm is not None
    assert stats.private_norm is not None


def test_straight_through_has_unit_forward_scale():
    cfg = ModelConfig(d_model=16, n_heads=2, n_layers=1, n_experts=2, private_width=8)
    module = SplitMoE(cfg)
    assert module.routed.weight_mode == "straight_through"


def test_split_dispatch_under_autocast():
    cfg = ModelConfig(
        vocab_size=64, max_seq_len=8, n_layers=1, d_model=16, n_heads=2,
        moe_every=1, moe_type="split", n_experts=2, shared_width=16, private_width=16,
    )
    model = DecoderLM(cfg)
    inputs = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(inputs, inputs)
    assert torch.isfinite(output.loss)
    output.loss.backward()


def test_topk_router_normalizes_selected_probabilities():
    router = TopKRouter(dim=16, n_experts=8, top_k=4)
    inputs = torch.randn(2, 5, 16)
    indices, selected, stats = router(inputs, collect_assignments=True)

    assert indices.shape == selected.shape == (2, 5, 4)
    assert stats.assignments.shape == (2, 5, 4)
    assert torch.allclose(selected.sum(dim=-1), torch.ones(2, 5))
    assert torch.allclose(stats.expert_fraction.sum(), torch.tensor(1.0))


def test_top1_selected_gate_retains_task_gradient():
    router = TopKRouter(dim=16, n_experts=4, top_k=1)
    inputs = torch.randn(2, 5, 16)
    _, selected, _ = router(inputs)
    selected.sum().backward()

    assert router.proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("moe_type", ["standard", "split"])
def test_topk_forward_backward(moe_type):
    cfg = ModelConfig(
        vocab_size=128, max_seq_len=16, n_layers=2, d_model=32, n_heads=4,
        dense_ffn_width=64, moe_every=1, moe_type=moe_type, n_experts=8,
        top_k=4, standard_expert_width=64, shared_width=16, private_width=48,
        router_weight_mode="probability",
    )
    model = DecoderLM(cfg)
    inputs = torch.randint(0, cfg.vocab_size, (3, cfg.max_seq_len))
    output = model(inputs, inputs, collect_assignments=True)

    assert output.router_stats[0].assignments.shape == (3, cfg.max_seq_len, 4)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.blocks[0].ffn.router.proj.weight.grad.abs().sum() > 0


def test_topk_parameter_summary_counts_every_selected_expert():
    cfg = ModelConfig(
        vocab_size=128, max_seq_len=16, n_layers=2, d_model=32, n_heads=4,
        dense_ffn_width=64, moe_every=1, moe_type="standard", n_experts=8,
        top_k=4, standard_expert_width=64,
    )
    model = DecoderLM(cfg)
    summary = model.parameter_summary()
    one_expert_per_layer = 3 * cfg.d_model * cfg.standard_expert_width

    assert summary["activated_per_token"] == (
        summary["total"] - summary["routed_experts"]
        + cfg.n_layers * cfg.top_k * one_expert_per_layer
    )


def test_compute_matched_variants_have_matching_activated_parameters():
    common = dict(
        vocab_size=128, max_seq_len=16, n_layers=2, d_model=32, n_heads=4,
        dense_ffn_width=64, moe_every=1, n_experts=4, standard_expert_width=64,
        shared_width=32, private_width=32,
    )
    summaries = {
        kind: DecoderLM(ModelConfig(**common, moe_type=kind)).parameter_summary()
        for kind in ("dense", "standard", "split")
    }
    assert summaries["dense"]["activated_per_token"] == summaries["dense"]["total"]
    assert summaries["standard"]["activated_per_token"] == summaries["split"]["activated_per_token"]
    router_delta = summaries["standard"]["routers"]
    assert summaries["standard"]["activated_per_token"] == summaries["dense"]["total"] + router_delta
