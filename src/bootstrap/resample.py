from typing import Callable

import numpy as np

from bootstrap.types import RNGSeed


def _get_rng(seed: RNGSeed = None) -> np.random.Generator:
    """Get a numpy random number generator based on the provided seed.

    Args:
        seed: Seed for random number generation. Can be an integer,
            a numpy random Generator, or None. Defaults to None.

    Returns:
        A numpy random.Generator.

    Raises:
        TypeError: If seed is not an int, np.random.Generator, or None.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, int):
        rng = np.random.default_rng(seed)
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        raise TypeError(
            f"seed must be an int, np.random.Generator, or None, got {type(seed)}"
        )
    return rng


def bootstrap_resample(
    data: np.ndarray | list, n_resamples: int = 10000, seed: RNGSeed = None
) -> np.ndarray:
    """Create bootstrap resamples from the original data.

    Performs bootstrap resampling by randomly sampling with replacement from the original
    data to create multiple new samples of the same size.

    Args:
        data: Original data array or list to resample from.
        n_resamples: Number of bootstrap resamples to generate. Defaults to 10000.
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        A numpy array of shape (n_resamples, len(data)) containing the resampled data.

    Reference:
        Bradley Efron. 1979. Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics 7, 1 (January 1979), 1–26. https://doi.org/10.1214/aos/1176344552
    """
    rng = _get_rng(seed)
    size = (n_resamples, len(data))
    resampled_data = rng.choice(data, size=size, replace=True)
    return resampled_data


def jackknife_resample(data: np.ndarray | list) -> np.ndarray:
    """Create jackknife resamples from the original data.

    Performs jackknife resampling by creating n new samples, each with one observation removed
    from the original dataset, where n is the length of the data.

    Args:
        data: Original data array or list to resample from.

    Returns:
        A numpy array of shape (n, n-1) where n is the length of the data. Each row contains
        a sample with one observation removed.

    Notes:
        This implementation follows that of the AstroPy team [here](https://docs.astropy.org/en/stable/_modules/astropy/stats/jackknife.html).
        Copyright (c) 2011-2024, Astropy Developers.
    """
    n = len(data)
    # adapted from https://docs.astropy.org/en/stable/_modules/astropy/stats/jackknife.html
    samples = np.empty((n, n - 1))
    for i in range(n):
        samples[i] = np.delete(data, i)
    return samples


def bootstrap_distribution(
    data: np.ndarray,
    n_resamples: int,
    statistic: Callable,
    seed: RNGSeed = None,
) -> np.ndarray:
    """Generate bootstrap distribution for a given statistic.

    Args:
        data: Original data array to resample from.
        n_resamples: Number of bootstrap resamples to generate.
        statistic: Function that computes the statistic of interest on the data.
        seed: Random seed for reproducibility. Defaults to None for no seed.

    Returns:
        Array of statistic values calculated on each bootstrap resample.

    Reference:
        Bradley Efron. 1979. Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics 7, 1 (January 1979), 1–26. https://doi.org/10.1214/aos/1176344552

    """
    resamples = bootstrap_resample(data, n_resamples, seed)
    return np.apply_along_axis(statistic, axis=1, arr=resamples)


def jackknife_distribution(data: np.ndarray, statistic: Callable) -> np.ndarray:
    """Generate jackknife distribution for a given statistic.

    Args:
        data: Original data array to resample from.
        statistic: Function that computes the statistic of interest on the data.

    Returns:
        Array of statistic values calculated on each jackknife resample.
    """
    resamples = jackknife_resample(data)
    return np.apply_along_axis(statistic, axis=1, arr=resamples)


__all__ = [
    "bootstrap_resample",
    "jackknife_resample",
    "bootstrap_distribution",
    "jackknife_distribution",
]
