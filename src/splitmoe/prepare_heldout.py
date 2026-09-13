from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from .heldout import (
    LAMBADA_CONFIG,
    LAMBADA_DATASET,
    LAMBADA_REVISION,
    LAMBADA_SPLIT,
    TOKENIZER,
    TOKENIZER_REVISION,
    sha256_file,
    tokenize_lambada_example,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretokenize the pinned LAMBADA held-out set")
    parser.add_argument("--output-dir", default="data/heldout/lambada_openai_en")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        LAMBADA_DATASET,
        LAMBADA_CONFIG,
        split=LAMBADA_SPLIT,
        revision=LAMBADA_REVISION,
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION, use_fast=True)
    dtype = np.dtype(np.uint16 if len(tokenizer) <= np.iinfo(np.uint16).max else np.uint32)
    offsets = [0]
    target_starts: list[int] = []
    text_digest = hashlib.sha256()

    with (output / "tokens.bin").open("wb") as token_file:
        for row in dataset:
            text = str(row["text"])
            encoded, target_start = tokenize_lambada_example(tokenizer, text)
            np.asarray(encoded, dtype=dtype).tofile(token_file)
            offsets.append(offsets[-1] + len(encoded))
            target_starts.append(target_start)
            raw = text.encode("utf-8")
            text_digest.update(len(raw).to_bytes(8, "little"))
            text_digest.update(raw)

    np.save(output / "offsets.npy", np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    np.save(
        output / "target_starts.npy",
        np.asarray(target_starts, dtype=np.uint32),
        allow_pickle=False,
    )
    metadata = {
        "version": 1,
        "dataset": LAMBADA_DATASET,
        "config": LAMBADA_CONFIG,
        "split": LAMBADA_SPLIT,
        "dataset_revision": LAMBADA_REVISION,
        "dataset_license": "MIT",
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "num_examples": len(target_starts),
        "num_tokens": offsets[-1],
        "token_dtype": dtype.name,
        "preprocessing": (
            "Each document is tokenized independently without special tokens. The final "
            "whitespace-delimited word, including its leading whitespace, is the target suffix."
        ),
        "source_text_sha256": text_digest.hexdigest(),
        "tokens_sha256": sha256_file(output / "tokens.bin"),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
