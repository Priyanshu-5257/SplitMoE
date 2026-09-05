from pathlib import Path

from splitmoe.config import ExperimentConfig
from splitmoe.model import DecoderLM


def test_experiment_configs_use_preregistered_seed_suite():
    root = Path(__file__).parents[1]
    expected_seeds = [1337, 2027, 3407, 4517, 5651]
    for name in ("dense", "standard_1024", "split"):
        config = ExperimentConfig.from_json(root / "configs" / f"{name}.json")
        assert config.train.seeds == expected_seeds
        assert config.train.wandb_project == "splitmoe-seeds"


def test_standard_640_is_parameter_matched_to_split():
    root = Path(__file__).parents[1]
    standard = ExperimentConfig.from_json(root / "configs" / "standard.json")
    split = ExperimentConfig.from_json(root / "configs" / "split.json")
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
