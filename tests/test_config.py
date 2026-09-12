from pathlib import Path

import pytest

from splitmoe.config import ExperimentConfig, ModelConfig
from splitmoe.model import DecoderLM
from splitmoe.train import load_experiment_configs


def test_experiment_configs_use_preregistered_seed_suite():
    root = Path(__file__).parents[1]
    expected_seeds = [1337, 2027, 3407, 4517, 5651]
    for name in ("dense", "standard_1024", "split_50"):
        config = ExperimentConfig.from_json(root / "configs" / f"{name}.json")
        assert config.train.seeds == expected_seeds
        assert config.train.wandb_project == "splitmoe-seeds"


def test_standard_640_is_parameter_matched_to_split():
    root = Path(__file__).parents[1]
    standard = ExperimentConfig.from_json(root / "configs" / "standard.json")
    split = ExperimentConfig.from_json(root / "configs" / "split_50.json")
    assert standard.train.seeds == split.train.seeds
    assert standard.train.wandb_project == "splitmoe-param-matched"
    assert standard.train.wandb_run_name == "standard-640"
    assert standard.model.standard_expert_width == 640
    assert (
        standard.model.n_experts * standard.model.standard_expert_width
        == split.model.shared_width + split.model.n_experts * split.model.private_width
    )
    standard_summary = DecoderLM(standard.model).parameter_summary()
    split_summary = DecoderLM(split.model).parameter_summary()
    assert standard_summary["total"] == split_summary["total"] == 55_722_496
    assert standard_summary["activated_per_token"] == 43_926_016
    assert split_summary["activated_per_token"] == 46_285_312


def test_standard_512_is_same_width_mechanism_control():
    root = Path(__file__).parents[1]
    standard = ExperimentConfig.from_json(root / "configs" / "standard_512.json")

    assert standard.model.moe_type == "standard"
    assert standard.model.n_experts == 4
    assert standard.model.standard_expert_width == 512
    assert standard.train.seeds == [1337, 2027, 3407, 4517, 5651]
    assert standard.train.wandb_project == "splitmoe-mechanism-control"
    assert standard.train.wandb_run_name == "standard-512"
    assert standard.train.output_dir == "checkpoints/standard-512"

    summary = DecoderLM(standard.model).parameter_summary()
    assert summary["total"] == 52_576_768
    assert summary["activated_per_token"] == 43_139_584


def test_split_width_sweep_suite():
    root = Path(__file__).parents[1]
    configs = load_experiment_configs(root / "configs" / "split.json")
    assert [(cfg.model.shared_width, cfg.model.private_width) for cfg in configs] == [
        (256, 768),
        (768, 256),
    ]
    assert all(cfg.model.shared_width + cfg.model.private_width == 1024 for cfg in configs)
    assert all(cfg.train.wandb_project == "splitmoe-width-sweep" for cfg in configs)
    assert [cfg.train.wandb_run_name for cfg in configs] == ["split-25", "split-75"]
    assert all(cfg.train.seeds == [1337, 2027, 3407, 4517, 5651] for cfg in configs)


def test_large_top4_configs_are_paired():
    root = Path(__file__).parents[1]
    standard = ExperimentConfig.from_json(root / "configs" / "large_standard_16e_top4.json")
    split = ExperimentConfig.from_json(root / "configs" / "large_split25_16e_top4.json")

    for config in (standard, split):
        assert config.model.n_layers == 10
        assert config.model.d_model == 640
        assert config.model.n_experts == 16
        assert config.model.top_k == 4
        assert config.train.seeds == [1337]
        assert config.train.wandb_project == "splitmoe-large-16e-top4"
        assert config.train.micro_batch_size * config.train.gradient_accumulation_steps == 32
    assert split.model.shared_width == 320
    assert split.model.private_width == 960


def test_paper_allmoe_configs_define_clean_controls():
    root = Path(__file__).parents[1]
    standard = ExperimentConfig.from_json(root / "configs" / "paper_allmoe_standard_1024.json")
    split = ExperimentConfig.from_json(root / "configs" / "paper_allmoe_split25.json")
    storage_control = ExperimentConfig.from_json(
        root / "configs" / "paper_allmoe_standard_800.json"
    )

    for config in (standard, split, storage_control):
        assert config.model.n_layers == 8
        assert config.model.moe_every == 1
        assert config.model.n_experts == 8
        assert config.model.top_k == 1
        assert config.train.seeds == [1337, 2027, 3407]
        assert config.train.max_steps == 6500
        assert config.train.micro_batch_size * config.train.gradient_accumulation_steps == 64
        assert config.train.wandb_project == "splitmoe-paper-allmoe-8e-top1"

    summaries = {
        "standard": DecoderLM(standard.model).parameter_summary(),
        "split": DecoderLM(split.model).parameter_summary(),
        "storage_control": DecoderLM(storage_control.model).parameter_summary(),
    }
    assert summaries["standard"]["activated_per_token"] == summaries["split"][
        "activated_per_token"
    ] == 46_309_888
    assert summaries["split"]["total"] == summaries["storage_control"][
        "total"
    ] == 112_370_176

    primary = load_experiment_configs(root / "configs" / "paper_allmoe_primary.json")
    all_variants = load_experiment_configs(root / "configs" / "paper_allmoe_all.json")
    assert len(primary) == 2
    assert len(all_variants) == 3


def test_kaggle_allmoe_16e_top4_configs_are_paired_and_isolated():
    root = Path(__file__).parents[1]
    seeds = [1337, 2027, 3407]
    configs = []
    for architecture in ("standard", "split25"):
        for seed in seeds:
            path = root / "configs" / f"kaggle_allmoe_16e_top4_{architecture}_seed{seed}.json"
            config = ExperimentConfig.from_json(path)
            configs.append(config)
            assert config.model.n_layers == 8
            assert config.model.moe_every == 1
            assert config.model.n_experts == 16
            assert config.model.top_k == 4
            assert config.model.router_weight_mode == "probability"
            assert config.train.seeds == [seed]
            assert config.train.micro_batch_size == 4
            assert config.train.gradient_accumulation_steps == 8
            assert config.train.micro_batch_size * config.train.gradient_accumulation_steps * 2 == 64
            assert config.train.max_steps == 6500
            assert config.train.wandb_project == "splitmoe-paper-allmoe-16e-top4-kaggle"

    split_configs = [config for config in configs if config.model.moe_type == "split"]
    assert all(config.model.shared_width == 256 for config in split_configs)
    assert all(config.model.private_width == 768 for config in split_configs)
    assert len({config.train.output_dir for config in configs}) == 6
    assert len({config.train.wandb_run_name for config in configs}) == 6


def test_top_k_cannot_exceed_expert_count():
    with pytest.raises(ValueError, match="top_k"):
        ModelConfig(n_experts=4, top_k=5).validate()
