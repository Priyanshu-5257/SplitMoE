from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


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

