from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class TokenBlockDataset(Dataset):
    """Memory-mapped, fixed-length token blocks produced by prepare_data.py."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        metadata_path = self.directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing pretokenized dataset metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text())
        self.block_size = int(self.metadata["block_size"])
        self.num_blocks = int(self.metadata["num_blocks"])
        self.tokens = np.memmap(
            self.directory / "tokens.bin",
            dtype=np.dtype(self.metadata["token_dtype"]),
            mode="r",
            shape=(self.num_blocks, self.block_size + 1),
        )
        self.domains = np.memmap(
            self.directory / "domains.bin", dtype=np.uint16, mode="r", shape=(self.num_blocks,)
        )

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int):
        tokens = np.asarray(self.tokens[index], dtype=np.int64)
        return (
            torch.from_numpy(tokens[:-1].copy()),
            torch.from_numpy(tokens[1:].copy()),
            torch.tensor(int(self.domains[index]), dtype=torch.long),
        )


class DomainBalancedSampler(Sampler[int]):
    """Fixed, evenly distributed validation sample, sharded across DDP ranks."""

    def __init__(
        self,
        dataset: TokenBlockDataset,
        *,
        samples_per_replica: int,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        domain_count = len(dataset.metadata.get("domains", []))
        if domain_count == 0:
            raise ValueError("Validation metadata must contain domain names")
        base, remainder = divmod(samples_per_replica, domain_count)
        selected_by_domain: list[list[int]] = []
        domain_array = np.asarray(dataset.domains)
        for domain_id in range(domain_count):
            candidates = np.flatnonzero(domain_array == domain_id)
            if candidates.size == 0:
                raise ValueError(f"Validation dataset has no blocks for domain {domain_id}")
            requested_per_replica = base + int(domain_id < remainder)
            requested_global = requested_per_replica * num_replicas
            # Spread the fixed sample across the domain rather than taking only
            # its first blocks. Sampling with replacement is allowed for tiny tests.
            positions = np.linspace(0, candidates.size - 1, requested_global, dtype=np.int64)
            selected_by_domain.append(candidates[positions].tolist()[rank::num_replicas])
        self.indices = [
            indices[position]
            for position in range(max(map(len, selected_by_domain)))
            for indices in selected_by_domain
            if position < len(indices)
        ]
        if len(self.indices) != samples_per_replica:
            raise RuntimeError("Balanced sampler produced an invalid DDP shard size")

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def write_synthetic_dataset(
    directory: str | Path, *, vocab_size: int, block_size: int, num_blocks: int, seed: int = 0
) -> None:
    """Create deterministic data for smoke tests; not intended for real training."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32
    tokens = rng.integers(0, vocab_size, (num_blocks, block_size + 1), dtype=dtype)
    domains = np.arange(num_blocks, dtype=np.uint16) % 4
    tokens.tofile(directory / "tokens.bin")
    domains.tofile(directory / "domains.bin")
    metadata = {
        "version": 1,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "token_dtype": np.dtype(dtype).name,
        "tokenizer": "synthetic",
        "domains": ["stories", "wiki", "code", "math"],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
