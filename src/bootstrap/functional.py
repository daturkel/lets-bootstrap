"""Functions for bootstrap resampling and confidence interval calculation.

This module contains functions for bootstrap resampling to estimate the sampling
distribution of statistics and to calculate confidence intervals.
"""

from typing import Callable

import numpy as np
from scipy import stats

from bootstrap.resample import bootstrap_resample, jackknife_resample
from bootstrap.types import RNGSeed


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


def _expand_alpha(alpha: float, n: int) -> float:
    expansion = np.sqrt(n / (n - 1))
    t_alpha = stats.t(n - 1).ppf(alpha)
    alpha_prime = float(stats.norm.cdf(expansion * t_alpha))
    return alpha_prime


def _percentile_ci(
    theta_star: np.ndarray,
    alpha: float = 0.05,
    expand: bool = False,
    n_samples: int | None = None,
) -> tuple[float, float]:
    """Calculate percentile confidence interval from a bootstrap distribution.

    Args:
        theta_star: Bootstrap distribution of the statistic.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.
        expand: If True, use the expanded percentile interval to adjust for narrowness bias.
            Defaults to False.
        n_samples: Number of samples in original dataset, required if expand=True.
            Defaults to None.

    Raises:
        ValueError: If expand is True and n_samples is None.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.
    """
    alpha_lo = alpha / 2
    if expand:
        if n_samples is None:
            raise ValueError("n_samples must be provided when expand=True")
        else:
            alpha_lo = _expand_alpha(alpha_lo, n_samples)

    lo = float(np.quantile(theta_star, alpha_lo))
    hi = float(np.quantile(theta_star, 1 - alpha_lo))
    return lo, hi


def percentile_ci(
    data: np.ndarray,
    n_resamples: int,
    statistic: Callable,
    alpha: float = 0.05,
    seed: RNGSeed = None,
    expand: bool = False,
) -> tuple[float, float]:
    """Calculate percentile bootstrap confidence interval.

    Args:
        data: Original data array to resample from.
        n_resamples: Number of bootstrap resamples to generate.
        statistic: Function that computes the statistic of interest on the data.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.
        seed: Random seed for reproducibility. Defaults to None for no seed.
        expand: If True, use the expanded percentile interval to adjust for narrowness bias.
            Defaults to False.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.
    """
    theta_star = bootstrap_distribution(data, n_resamples, statistic, seed=seed)
    return _percentile_ci(theta_star, alpha, expand, len(data) if expand else None)


def _reverse_percentile_ci(
    theta_star: np.ndarray,
    theta_hat: float,
    alpha: float = 0.05,
    expand: bool = False,
    n_samples: int | None = None,
) -> tuple[float, float]:
    """Calculate reverse percentile confidence interval from a bootstrap distribution.

    Args:
        theta_star: Bootstrap distribution of the statistic.
        theta_hat: Value of statistic computed on original sample.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.
        expand: If True, use the expanded reverse percentile interval to adjust for narrowness
            bias. Defaults to False.
        n_samples: Number of samples in original dataset, required if expand=True.
            Defaults to None.

    Raises:
        ValueError: If expand is True and n_samples is None.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.
    """
    alpha_lo = alpha / 2
    if expand:
        if n_samples is None:
            raise ValueError("n_samples must be provided when expand=True")
        else:
            alpha_lo = _expand_alpha(alpha_lo, n_samples)

    lo = float(2 * theta_hat - np.quantile(theta_star, 1 - alpha_lo))
    hi = float(2 * theta_hat - np.quantile(theta_star, alpha_lo))
    return lo, hi


def reverse_percentile_ci(
    data: np.ndarray,
    n_resamples: int,
    statistic: Callable,
    alpha: float = 0.05,
    seed: RNGSeed = None,
    expand: bool = False,
) -> tuple[float, float]:
    """Calculate reverse percentile (basic) bootstrap confidence interval.

    This method reflects the bootstrap distribution around the original sample statistic.
    It can perform better than the standard percentile method when the bootstrap
    distribution is skewed.

    Args:
        data: Original data array to resample from.
        n_resamples: Number of bootstrap resamples to generate.
        statistic: Function that computes the statistic of interest on the data.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.
        seed: Random seed for reproducibility. Defaults to None for no seed.
        expand: If True, use the expanded reverse percentile interval to adjust for narrowness
            bias. Defaults to False.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.
    """
    theta_star = bootstrap_distribution(data, n_resamples, statistic, seed=seed)
    theta_hat = statistic(data)
    return _reverse_percentile_ci(
        theta_star, theta_hat, alpha, expand, len(data) if expand else None
    )


def _t_ci(
    theta_star: np.ndarray,
    theta_hat: float,
    se_fn: Callable,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Calculate t-distribution bootstrap confidence interval from a bootstrap distribution.

    Args:
        theta_star: Bootstrap distribution of the statistic.
        theta_hat: Value of statistic computed on original sample.
        se_fn: A function to calculate the standard error of the statistic on the sample.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.
    """
    se_hat = se_fn(theta_star)
    t_star = (theta_star - theta_hat) / se_hat
    lo = float(theta_hat - np.quantile(t_star, 1 - alpha / 2) * se_hat)
    hi = float(theta_hat - np.quantile(t_star, alpha / 2) * se_hat)
    return lo, hi


def t_ci(
    data: np.ndarray,
    n_resamples: int,
    se_fn: Callable,
    statistic: Callable,
    alpha: float = 0.05,
    seed: RNGSeed = None,
) -> tuple[float, float]:
    """Calculate t-distribution bootstrap confidence interval.

    Args:
        data: Array of data to compute the confidence interval for.
        n_resamples: Number of bootstrap resamples to generate.
        se_fn: A function to calculate the standard error of the statistic on the sample.
        statistic: Function that computes the statistic of interest on the data.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.
        seed: Random seed for reproducibility. Defaults to None for no seed.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.
    """
    theta_hat = statistic(data)
    theta_star = bootstrap_distribution(data, n_resamples, statistic, seed=seed)
    return _t_ci(theta_star, theta_hat, se_fn, alpha)


def _bc_ci(
    theta_star: np.ndarray,
    theta_hat: float,
    alpha: float = 0.05,
    expand: bool = False,
    n_samples: int | None = None,
):
    """Calculate bias-corrected bootstrap confidence interval from a bootstrap distribution.

    Args:
        theta_star: Bootstrap distribution of the statistic.
        theta_hat: Value of statistic computed on original sample.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.
        expand: If True, use the expanded bias-corrected interval to adjust for narrowness
            bias. Defaults to False.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.
    """
    p0 = (theta_star <= theta_hat).mean()
    z0 = stats.norm.ppf(p0)
    alpha_lo = alpha / 2
    if expand:
        if n_samples is None:
            raise ValueError("n_samples must be provided when expand=True")
        else:
            alpha_lo = _expand_alpha(alpha_lo, n_samples)
    alpha_lo_prime = stats.norm.cdf(2 * z0 + stats.norm.ppf(alpha_lo))
    alpha_hi_prime = stats.norm.cdf(2 * z0 + stats.norm.ppf(1 - alpha_lo))
    lo = float(np.quantile(theta_star, alpha_lo_prime))
    hi = float(np.quantile(theta_star, alpha_hi_prime))
    return lo, hi


def bc_ci(
    data: np.ndarray,
    n_resamples: int,
    statistic: Callable,
    alpha: float = 0.05,
    seed: RNGSeed = None,
    expand: bool = False,
):
    """Calculate bias-corrected bootstrap confidence interval.

    Args:
        data: Array of data to compute the confidence interval for.
        n_resamples: Number of bootstrap resamples to generate.
        statistic: Function that computes the statistic of interest on the data.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.
        seed: Random seed for reproducibility. Defaults to None for no seed.
        expand: If True, use the expanded bias-corrected interval to adjust for narrowness
            bias. Defaults to False.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.
    """
    theta_hat = statistic(data)
    theta_star = bootstrap_distribution(data, n_resamples, statistic, seed=seed)
    return _bc_ci(theta_star, theta_hat, alpha, expand, len(data) if expand else None)


def _bca_ci(
    theta_star: np.ndarray,
    theta_jack: np.ndarray,
    theta_hat: float,
    alpha: float = 0.05,
    expand: bool = False,
    n_samples: int | None = None,
):
    """Calculate BCa bootstrap confidence interval from a bootstrap distribution.

    Args:
        theta_star: Bootstrap distribution of the statistic.
        theta_star: Jackknife distribution of the statistic.
        theta_hat: Value of statistic computed on original sample.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.
        expand: If True, use the expanded BCa interval to adjust for narrowness bias. Defaults
            to False.
        n_samples: Number of samples in original dataset, required if expand=True.
            Defaults to None.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.
    """
    p0 = (theta_star <= theta_hat).mean()
    z0 = stats.norm.ppf(p0)
    alpha_lo = alpha / 2
    if expand:
        if n_samples is None:
            raise ValueError("n_samples must be provided when expand=True")
        else:
            alpha_lo = _expand_alpha(alpha_lo, n_samples)
    theta_jack_bar = np.mean(theta_jack)
    empirical_influence = theta_jack - theta_jack_bar
    a_num = (empirical_influence**3).sum()
    a_denom = (6 * (empirical_influence**2).sum()) ** (3 / 2)
    a = a_num / a_denom
    z_alpha_lo = stats.norm.ppf(alpha_lo)
    alpha_lo_prime = stats.norm.cdf(
        z0 + (z0 + z_alpha_lo) / (1 - a * (z0 + z_alpha_lo))
    )
    z_alpha_hi = stats.norm.ppf(1 - alpha_lo)
    alpha_hi_prime = stats.norm.cdf(
        z0 + (z0 + z_alpha_hi) / (1 - a * (z0 + z_alpha_hi))
    )
    lo = float(np.quantile(theta_star, alpha_lo_prime))
    hi = float(np.quantile(theta_star, alpha_hi_prime))
    return lo, hi


def bca_ci(
    data: np.ndarray,
    n_resamples: int,
    statistic: Callable,
    alpha: float = 0.05,
    seed: RNGSeed = None,
    expand: bool = False,
):
    """Calculate BCa bootstrap confidence interval.

    Args:
        data: Array of data to compute the confidence interval for.
        n_resamples: Number of bootstrap resamples to generate.
        statistic: Function that computes the statistic of interest on the data.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.
        seed: Random seed for reproducibility. Defaults to None for no seed.
        expand: If True, use the expanded BCa interval to adjust for narrowness bias. Defaults
            to False.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.
    """
    theta_hat = statistic(data)
    theta_star = bootstrap_distribution(data, n_resamples, statistic, seed=seed)
    theta_jack = jackknife_distribution(data, statistic)
    return _bca_ci(
        theta_star, theta_jack, theta_hat, alpha, expand, len(data) if expand else None
    )
