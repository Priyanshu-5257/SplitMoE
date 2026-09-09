import torch

from splitmoe.train import capture_rng_state, restore_rng_state


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
