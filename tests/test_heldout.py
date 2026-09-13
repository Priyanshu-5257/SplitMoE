import json

import numpy as np

from splitmoe.heldout import HeldoutTokenDataset, length_grouped_batches, target_character_start


def write_dataset(path, documents, targets):
    path.mkdir(exist_ok=True)
    flattened = np.concatenate([np.asarray(document, dtype=np.uint16) for document in documents])
    flattened.tofile(path / "tokens.bin")
    offsets = np.cumsum([0, *map(len, documents)], dtype=np.int64)
    np.save(path / "offsets.npy", offsets, allow_pickle=False)
    np.save(path / "target_starts.npy", np.asarray(targets, dtype=np.uint32), allow_pickle=False)
    (path / "metadata.json").write_text(json.dumps({
        "num_examples": len(documents), "token_dtype": "uint16"
    }))


def test_target_character_start_includes_leading_whitespace():
    assert target_character_start("context final") == len("context")
    assert target_character_start("context  final  ") == len("context")


def test_length_grouped_batches_do_not_pad_and_rebase_truncated_target(tmp_path):
    write_dataset(tmp_path, [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11]], [2, 4, 2])
    dataset = HeldoutTokenDataset(tmp_path)
    batches = list(length_grouped_batches(dataset, batch_size=2, max_seq_len=3))

    assert batches[0][0].shape == (2, 3)
    assert batches[0][1].tolist() == [2, 2]
    assert batches[1][0].tolist() == [[5, 6, 7, 8]]
    assert batches[1][1].tolist() == [3]
