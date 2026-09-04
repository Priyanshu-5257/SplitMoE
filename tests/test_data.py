from splitmoe.data import TokenBlockDataset, write_synthetic_dataset


def test_memmap_dataset(tmp_path):
    write_synthetic_dataset(tmp_path, vocab_size=100, block_size=8, num_blocks=5)
    dataset = TokenBlockDataset(tmp_path)
    inputs, labels, domain = dataset[0]
    assert len(dataset) == 5
    assert inputs.shape == labels.shape == (8,)
    assert (inputs[1:] == labels[:-1]).all()
    assert domain.ndim == 0
