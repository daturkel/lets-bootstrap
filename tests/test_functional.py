"""Unit tests for bootstrap.functional."""

from typing import Callable

import numpy as np
import pytest

from bootstrap.functional import (
    bc_ci,
    bca_ci,
    bootstrap_distribution,
    jackknife_distribution,
    percentile_ci,
    reverse_percentile_ci,
    t_ci,
)


@pytest.fixture
def normal_data() -> np.ndarray:
    """Create sample data for testing."""
    np.random.seed(42)
    return np.random.normal(loc=10, scale=2, size=100)


@pytest.fixture
def exponential_data() -> np.ndarray:
    """Create skewed (exponential) data for testing."""
    np.random.seed(42)
    return np.random.exponential(scale=2.0, size=100)


def test_bootstrap_distribution(normal_data: np.ndarray):
    """Test bootstrap_distribution function."""
    n_resamples = 1000
    seed = 42

    result = bootstrap_distribution(normal_data, n_resamples, np.mean, seed)

    assert isinstance(result, np.ndarray)
    assert len(result) == n_resamples
    assert (
        np.abs(np.mean(result) - np.mean(normal_data)) < 0.5
    )  # Should be close to original mean


def test_jackknife_distribution(normal_data: np.ndarray):
    """Test jackknife_distribution function."""
    result = jackknife_distribution(normal_data, np.mean)

    assert isinstance(result, np.ndarray)
    assert len(result) == len(normal_data)  # One resample for each data point


def test_percentile_ci(normal_data: np.ndarray):
    """Test percentile_ci function."""
    n_resamples = 1000
    alpha = 0.05
    seed = 42

    lo, hi = percentile_ci(normal_data, n_resamples, np.mean, alpha, seed)

    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo < hi
    assert np.mean(normal_data) > lo and np.mean(normal_data) < hi


def test_reverse_percentile_ci(normal_data: np.ndarray):
    """Test reverse_percentile_ci function."""
    n_resamples = 1000
    alpha = 0.05
    seed = 42

    lo, hi = reverse_percentile_ci(normal_data, n_resamples, np.mean, alpha, seed)

    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo < hi
    assert np.mean(normal_data) > lo and np.mean(normal_data) < hi


def test_t_ci(normal_data: np.ndarray):
    """Test t_ci function."""
    n_resamples = 1000
    alpha = 0.05
    seed = 42

    def se_fn(x: np.ndarray) -> float:
        return float(np.std(x, ddof=1) / np.sqrt(len(x)))

    lo, hi = t_ci(normal_data, n_resamples, se_fn, np.mean, alpha, seed)

    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo < hi
    assert np.mean(normal_data) > lo and np.mean(normal_data) < hi


def test_bc_ci(normal_data: np.ndarray):
    """Test bc_ci function."""
    n_resamples = 1000
    alpha = 0.05
    seed = 42

    lo, hi = bc_ci(normal_data, n_resamples, np.mean, alpha, seed)

    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo < hi
    assert np.mean(normal_data) > lo and np.mean(normal_data) < hi


def test_bca_ci(normal_data: np.ndarray):
    """Test bca_ci function."""
    n_resamples = 1000
    alpha = 0.05
    seed = 42

    lo, hi = bca_ci(normal_data, n_resamples, np.mean, alpha, seed)

    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo < hi
    assert np.mean(normal_data) > lo and np.mean(normal_data) < hi


def test_expanded_intervals(normal_data: np.ndarray):
    """Test that expanded intervals are wider than non-expanded intervals."""
    n_resamples = 1000
    alpha = 0.05
    seed = 42
    normal_data = normal_data[:10]

    # Test percentile CI
    lo1, hi1 = percentile_ci(
        normal_data, n_resamples, np.mean, alpha, seed, expand=False
    )
    lo2, hi2 = percentile_ci(
        normal_data, n_resamples, np.mean, alpha, seed, expand=True
    )
    assert hi2 > hi1
    assert lo2 < lo1

    # Test reverse percentile CI
    lo1, hi1 = reverse_percentile_ci(
        normal_data, n_resamples, np.mean, alpha, seed, expand=False
    )
    lo2, hi2 = reverse_percentile_ci(
        normal_data, n_resamples, np.mean, alpha, seed, expand=True
    )
    assert hi2 > hi1
    assert lo2 < lo1

    # Test BC CI
    lo1, hi1 = bc_ci(normal_data, n_resamples, np.mean, alpha, seed, expand=False)
    lo2, hi2 = bc_ci(normal_data, n_resamples, np.mean, alpha, seed, expand=True)
    assert hi2 > hi1
    assert lo2 < lo1

    # Test BCa CI
    lo1, hi1 = bca_ci(normal_data, n_resamples, np.mean, alpha, seed, expand=False)
    lo2, hi2 = bca_ci(normal_data, n_resamples, np.mean, alpha, seed, expand=True)
    assert hi2 > hi1
    assert lo2 < lo1


@pytest.mark.parametrize(
    "ci_function",
    [
        percentile_ci,
        reverse_percentile_ci,
        bc_ci,
        bca_ci,
    ],
)
def test_monotone_transformation_invariance(
    exponential_data: np.ndarray, ci_function: Callable
):
    """Test invariance of confidence intervals under monotone transformations."""
    n_resamples = 1000
    alpha = 0.05
    seed = 42

    # Original intervals
    lo1, hi1 = ci_function(exponential_data, n_resamples, np.mean, alpha, seed)

    # Log-transformed data intervals
    lo2, hi2 = ci_function(
        exponential_data, n_resamples, lambda x: np.log(np.mean(x)), alpha, seed
    )

    # Transform back
    lo2, hi2 = np.exp(lo2), np.exp(hi2)

    # Confidence intervals should be approximately transformation-invariant
    np.testing.assert_allclose([lo1, hi1], [lo2, hi2], rtol=0.1)
