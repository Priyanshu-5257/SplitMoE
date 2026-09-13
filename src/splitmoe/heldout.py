from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch


LAMBADA_DATASET = "EleutherAI/lambada_openai"
LAMBADA_CONFIG = "en"
LAMBADA_SPLIT = "test"
LAMBADA_REVISION = "900124bf3b8235c6daf21033af9948b3f07346c4"
TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def target_character_start(text: str) -> int:
    """Return the start of LAMBADA's final word, including its leading whitespace."""
    stripped_end = len(text.rstrip())
    if stripped_end == 0:
        raise ValueError("Cannot score an empty example")
    start = stripped_end
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while start > 0 and text[start - 1].isspace():
        start -= 1
    return start


def tokenize_lambada_example(tokenizer, text: str) -> tuple[list[int], int]:
    """Tokenize one document and locate the token suffix representing its final word."""
    text = text.rstrip()
    boundary = target_character_start(text)
    full_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(text[:boundary], add_special_tokens=False)["input_ids"]
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Tokenizer did not preserve the prefix at the target-word boundary")
    if not 0 < len(prefix_ids) < len(full_ids):
        raise ValueError("Example must contain context and at least one target token")
    return list(full_ids), len(prefix_ids)


class HeldoutTokenDataset:
    """Memory-mapped variable-length documents with a final-word target boundary."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        self.offsets = np.load(self.directory / "offsets.npy", mmap_mode="r")
        self.target_starts = np.load(self.directory / "target_starts.npy", mmap_mode="r")
        if len(self.offsets) != int(self.metadata["num_examples"]) + 1:
            raise ValueError("Held-out offsets do not match metadata")
        if len(self.target_starts) != int(self.metadata["num_examples"]):
            raise ValueError("Held-out target starts do not match metadata")
        self.tokens = np.memmap(
            self.directory / "tokens.bin",
            dtype=np.dtype(self.metadata["token_dtype"]),
            mode="r",
            shape=(int(self.offsets[-1]),),
        )

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        tokens = np.asarray(self.tokens[start:end], dtype=np.int64)
        return torch.from_numpy(tokens.copy()), int(self.target_starts[index])

    def length(self, index: int) -> int:
        return int(self.offsets[index + 1] - self.offsets[index])


def length_grouped_batches(
    dataset: HeldoutTokenDataset,
    batch_size: int,
    max_seq_len: int,
    indices: Sequence[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield equal-length batches, avoiding padding in a model without pad masking."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    selected = list(range(len(dataset))) if indices is None else list(indices)
    selected.sort(key=lambda index: (min(dataset.length(index), max_seq_len + 1), index))
    cursor = 0
    while cursor < len(selected):
        truncated_length = min(dataset.length(selected[cursor]), max_seq_len + 1)
        group_end = cursor
        while (
            group_end < len(selected)
            and min(dataset.length(selected[group_end]), max_seq_len + 1) == truncated_length
        ):
            group_end += 1
        for chunk_start in range(cursor, group_end, batch_size):
            chunk = selected[chunk_start : min(chunk_start + batch_size, group_end)]
            documents: list[torch.Tensor] = []
            target_starts: list[int] = []
            for index in chunk:
                tokens, target_start = dataset[index]
                removed = max(0, len(tokens) - (max_seq_len + 1))
                tokens = tokens[removed:]
                target_start -= removed
                if target_start < 1:
                    raise ValueError("Target word exceeds the available model context")
                documents.append(tokens)
                target_starts.append(target_start)
            yield torch.stack(documents), torch.tensor(target_starts, dtype=torch.long)
        cursor = group_end


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
