import numpy as np

from bootstrap.types import RNGSeed


def _get_rng(seed: RNGSeed = None) -> np.random.Generator:
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
    rng = _get_rng(seed)
    size = (n_resamples, len(data))
    resampled_data = rng.choice(data, size=size, replace=True)
    return resampled_data


def jackknife_resample(data: np.ndarray | list) -> np.ndarray:
    n = len(data)
    # adapted from https://docs.astropy.org/en/stable/_modules/astropy/stats/jackknife.html
    samples = np.empty((n, n - 1))
    for i in range(n):
        samples[i] = np.delete(data, i)
    return samples


__all__ = ["bootstrap_resample", "jackknife_resample"]
