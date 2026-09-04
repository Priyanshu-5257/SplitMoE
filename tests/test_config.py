from pathlib import Path

from splitmoe.config import ExperimentConfig


def test_experiment_configs_use_preregistered_seed_suite():
    root = Path(__file__).parents[1]
    expected_seeds = [1337, 2027, 3407, 4517, 5651]
    for name in ("dense", "standard", "split"):
        config = ExperimentConfig.from_json(root / "configs" / f"{name}.json")
        assert config.train.seeds == expected_seeds
        assert config.train.wandb_project == "splitmoe-seeds"
