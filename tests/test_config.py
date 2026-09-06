from pathlib import Path

from splitmoe.config import ExperimentConfig
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
