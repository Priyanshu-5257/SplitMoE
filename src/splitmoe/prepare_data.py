from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def stable_validation_assignment(domain: str, document_index: int, fraction: float) -> bool:
    digest = hashlib.blake2b(f"{domain}:{document_index}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "little") / 2**64
    return value < fraction


def documents(source: dict[str, Any]) -> Iterator[str]:
    kwargs: dict[str, Any] = {
        "path": source["dataset"],
        "split": source.get("split", "train"),
        "streaming": True,
    }
    if source.get("config"):
        kwargs["name"] = source["config"]
    if source.get("data_files"):
        kwargs["data_files"] = source["data_files"]
    dataset = load_dataset(**kwargs)
    field = source.get("text_field", "text")
    maximum = source.get("max_documents")
    for index, row in enumerate(dataset):
        if maximum is not None and index >= int(maximum):
            break
        value = row.get(field)
        if value is not None and str(value).strip():
            yield str(value)


class BlockWriter:
    def __init__(self, directory: Path, block_size: int, dtype: np.dtype):
        directory.mkdir(parents=True, exist_ok=True)
        self.token_file = (directory / "tokens.bin").open("wb")
        self.domain_file = (directory / "domains.bin").open("wb")
        self.block_size = block_size
        self.dtype = dtype
        self.num_blocks = 0
        self.domain_counts: dict[int, int] = {}

    def write(self, token_ids: list[int], domain_id: int) -> None:
        array = np.asarray(token_ids, dtype=self.dtype)
        if array.size != self.block_size + 1:
            raise ValueError("Invalid token block length")
        array.tofile(self.token_file)
        np.asarray([domain_id], dtype=np.uint16).tofile(self.domain_file)
        self.num_blocks += 1
        self.domain_counts[domain_id] = self.domain_counts.get(domain_id, 0) + 1

    def close(self) -> None:
        self.token_file.close()
        self.domain_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream, tokenize, and pack fixed-size training blocks")
    parser.add_argument("--sources", required=True, help="JSON source manifest")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--batch-documents", type=int, default=128)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between zero and one")

    sources = json.loads(Path(args.sources).read_text())["sources"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token for document packing")
    dtype = np.dtype(np.uint16 if len(tokenizer) <= np.iinfo(np.uint16).max else np.uint32)
    root = Path(args.output_dir)
    writers = {
        "train": BlockWriter(root / "train", args.block_size, dtype),
        "validation": BlockWriter(root / "validation", args.block_size, dtype),
    }
    domains = [source["domain"] for source in sources]
    buffers: dict[tuple[str, int], list[int]] = {}

    try:
        for domain_id, source in enumerate(sources):
            batch: list[str] = []
            batch_indices: list[int] = []
            for document_index, text in enumerate(documents(source)):
                maximum_blocks = source.get("max_blocks")
                existing = sum(writer.domain_counts.get(domain_id, 0) for writer in writers.values())
                if maximum_blocks is not None and existing >= int(maximum_blocks):
                    break
                batch.append(text)
                batch_indices.append(document_index)
                if len(batch) < args.batch_documents:
                    continue
                _consume_batch(
                    batch, batch_indices, source["domain"], domain_id, tokenizer, writers, buffers,
                    args, source.get("max_blocks"),
                )
                batch, batch_indices = [], []
            if batch:
                _consume_batch(
                    batch, batch_indices, source["domain"], domain_id, tokenizer, writers, buffers,
                    args, source.get("max_blocks"),
                )
    finally:
        for writer in writers.values():
            writer.close()

    for split, writer in writers.items():
        if writer.num_blocks == 0:
            raise RuntimeError(f"No blocks produced for {split}; increase data or validation fraction")
        metadata = {
            "version": 1,
            "block_size": args.block_size,
            "num_blocks": writer.num_blocks,
            "token_dtype": dtype.name,
            "tokenizer": args.tokenizer,
            "vocab_size": len(tokenizer),
            "domains": domains,
        }
        (root / split / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{split}: {writer.num_blocks:,} blocks")


def _consume_batch(batch, indices, domain, domain_id, tokenizer, writers, buffers, args, max_blocks) -> None:
    encoded = tokenizer(batch, add_special_tokens=False, truncation=False)["input_ids"]
    for document_index, ids in zip(indices, encoded, strict=True):
        split = "validation" if stable_validation_assignment(domain, document_index, args.validation_fraction) else "train"
        key = (split, domain_id)
        buffer = buffers.setdefault(key, [])
        buffer.extend(ids)
        buffer.append(tokenizer.eos_token_id)
        length = args.block_size + 1
        while len(buffer) >= length:
            existing = sum(writer.domain_counts.get(domain_id, 0) for writer in writers.values())
            if max_blocks is not None and existing >= int(max_blocks):
                break
            writers[split].write(buffer[:length], domain_id)
            del buffer[:length]


if __name__ == "__main__":
    main()
