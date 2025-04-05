"""Tests for bootstrap distribution functionality."""

import numpy as np

import bootstrap


def test_materialize(random_data: np.ndarray):
    """Test materialization of bootstrap distribution."""
    n_resamples = 50
    bs = bootstrap.BootstrapSamples(
        data=random_data, n_resamples=n_resamples
    ).materialize()
    bd = bs.apply(np.mean)
    assert bd._distribution is None
    bd = bd.materialize()
    assert bd._distribution is not None
    assert bd == bd.materialize()


def test_distribution_from_fixture(bootstrap_samples):
    """Test creating a bootstrap distribution from the fixture."""
    bs = bootstrap_samples(seed=42).materialize()

    # Apply a statistic function to create distribution
    bd = bs.apply(np.mean)
    materialized_bd = bd.materialize()

    # Test that distribution has the correct shape
    assert materialized_bd._distribution is not None
    assert materialized_bd._distribution.shape == (bs.n_resamples,)

    # Test with different statistic
    bd_median = bs.apply(np.median).materialize()
    assert bd_median._distribution.shape == (bs.n_resamples,)

    # Same seed should create same distribution
    bs2 = bootstrap_samples(seed=42).materialize()
    bd2 = bs2.apply(np.mean).materialize()
    assert np.array_equal(materialized_bd._distribution, bd2._distribution)
