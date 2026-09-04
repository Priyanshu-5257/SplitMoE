import numpy as np

from splitmoe.data import DomainBalancedSampler, TokenBlockDataset, write_synthetic_dataset


def test_memmap_dataset(tmp_path):
    write_synthetic_dataset(tmp_path, vocab_size=100, block_size=8, num_blocks=5)
    dataset = TokenBlockDataset(tmp_path)
    inputs, labels, domain = dataset[0]
    assert len(dataset) == 5
    assert inputs.shape == labels.shape == (8,)
    assert (inputs[1:] == labels[:-1]).all()
    assert domain.ndim == 0


def test_domain_balanced_sampler_is_balanced_on_every_rank(tmp_path):
    write_synthetic_dataset(tmp_path, vocab_size=100, block_size=8, num_blocks=40)
    dataset = TokenBlockDataset(tmp_path)
    samplers = [
        DomainBalancedSampler(dataset, samples_per_replica=8, num_replicas=2, rank=rank)
        for rank in range(2)
    ]
    for sampler in samplers:
        domains = np.asarray(dataset.domains)[list(sampler)]
        assert np.bincount(domains, minlength=4).tolist() == [2, 2, 2, 2]
