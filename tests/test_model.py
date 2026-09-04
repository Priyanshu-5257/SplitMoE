import pytest
import torch

from splitmoe.config import ModelConfig
from splitmoe.model import DecoderLM, SplitMoE


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

