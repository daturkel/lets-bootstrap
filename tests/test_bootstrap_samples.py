"""Tests for bootstrap sampling functionality."""

import numpy as np

import bootstrap


def test_materialize(random_data: np.ndarray):
    n_resamples = 50
    bs = bootstrap.BootstrapSamples(data=random_data, n_resamples=n_resamples)
    assert bs._samples is None
    bs = bs.materialize()
    assert bs._samples.shape == (50, len(bs.data))  # type: ignore


def test_samples(random_data: np.ndarray):
    """Test samples property returns properly shaped samples."""
    n_resamples = 50
    bs = bootstrap.BootstrapSamples(data=random_data, n_resamples=n_resamples)
    assert bs._samples is None
    samples = bs.samples
    assert samples.shape == (n_resamples, len(random_data))


def test_samples_materialized(random_data: np.ndarray):
    """Test samples property with materialized bootstrap samples."""
    n_resamples = 50
    bs = bootstrap.BootstrapSamples(
        data=random_data, n_resamples=n_resamples
    ).materialize()
    assert bs._samples is not None
    samples = bs.samples
    assert samples.shape == (n_resamples, len(random_data))
    assert (samples == bs._samples).all()


def test_samples_batch(random_data: np.ndarray):
    """Test batch sampling functionality."""
    n_resamples = 50
    seed = 5
    bs = bootstrap.BootstrapSamples(
        data=random_data, n_resamples=n_resamples, seed=seed
    )
    i = 0
    assert bs._samples is None
    samples_list = []
    for samples in bs.batch_samples(batch_size=9):
        samples_list.append(samples)
        i += 1
        if i == 6:
            assert samples.shape == (5, len(random_data))
        else:
            assert samples.shape == (9, len(random_data))
    assert bs._samples is None
    assert np.concat(samples_list).shape == (n_resamples, len(random_data))

    # exact same results as non-batched
    bs2 = bootstrap.BootstrapSamples(
        data=random_data, n_resamples=n_resamples, seed=seed
    ).materialize()
    assert np.all(bs2.samples == np.concat(samples_list))


def test_resamples_batch_materialized(random_data: np.ndarray):
    """Test batch sampling with materialized bootstrap samples."""
    n_resamples = 50
    bs = bootstrap.BootstrapSamples(
        data=random_data, n_resamples=n_resamples
    ).materialize()
    i = 0
    assert bs._samples is not None
    resamples_list = []
    for resamples in bs.batch_samples(batch_size=9):
        resamples_list.append(resamples)
        i += 1
        if i == 6:
            assert resamples.shape == (5, len(random_data))
        else:
            assert resamples.shape == (9, len(random_data))
    assert (np.concat(resamples_list) == bs._samples).all()


def test_bootstrap_fixture(bootstrap_samples):
    """Test the bootstrap_samples fixture functionality."""
    bs = bootstrap_samples()
    assert bs.data.shape == (100,)
    assert bs.n_resamples == 50

    # Test with custom seed
    bs_with_seed = bootstrap_samples(seed=42)
    assert bs_with_seed.seed == 42

    # Check that different seeds produce different samples
    bs1 = bootstrap_samples(seed=1).materialize()
    bs2 = bootstrap_samples(seed=2).materialize()
    assert not np.array_equal(bs1.samples, bs2.samples)

    # Check that same seed produces same samples
    bs3 = bootstrap_samples(seed=3).materialize()
    bs3_again = bootstrap_samples(seed=3).materialize()
    assert np.array_equal(bs3.samples, bs3_again.samples)
