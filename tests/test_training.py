import torch
from torch.utils.data import RandomSampler

from splitmoe.train import FastForwardSampler, capture_rng_state, restore_rng_state


class FakeMappedState:
    def __init__(self):
        self.cpu_called = False

    def cpu(self):
        self.cpu_called = True
        return torch.zeros(1, dtype=torch.uint8)


def test_restore_rng_state_moves_checkpoint_tensors_to_cpu(monkeypatch):
    state = capture_rng_state()
    cpu_state = FakeMappedState()
    cuda_state = FakeMappedState()
    state["torch"] = cpu_state
    state["cuda"] = [cuda_state]
    restored = {}

    monkeypatch.setattr(torch, "set_rng_state", lambda value: restored.setdefault("cpu", value))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda value: restored.setdefault("cuda", value),
    )

    restore_rng_state(state)

    assert cpu_state.cpu_called
    assert cuda_state.cpu_called
    assert restored["cpu"].device.type == "cpu"
    assert restored["cuda"][0].device.type == "cpu"


def test_fast_forward_sampler_matches_uninterrupted_random_sampler():
    dataset = list(range(11))
    batch_size = 2
    batches_per_epoch = len(dataset) // batch_size

    reference = RandomSampler(dataset, generator=torch.Generator().manual_seed(17))
    reference_epochs = [list(reference), list(reference), list(reference)]

    consumed_batches = batches_per_epoch + 2
    resumed_base = RandomSampler(dataset, generator=torch.Generator().manual_seed(17))
    resumed = FastForwardSampler(
        resumed_base,
        batches_consumed=consumed_batches,
        batch_size=batch_size,
    )

    expected_offset = 2 * batch_size
    assert list(resumed) == reference_epochs[1][expected_offset:]
    assert list(resumed) == reference_epochs[2]
