"""Functions for bootstrap resampling and confidence interval calculation.

This module contains functions for bootstrap resampling to estimate the sampling
distribution of statistics and to calculate confidence intervals.
"""

from typing import Callable

import numpy as np
from scipy import stats

from bootstrap.resample import (
    bootstrap_distribution,
    bootstrap_resample,
    jackknife_distribution,
)
from bootstrap.types import RNGSeed


def percentile_ci_from_bootstrap_distribution(
    theta_star: np.ndarray,
    alpha: float = 0.05,
    expand: bool = False,
    n_samples: int | None = None,
) -> tuple[float, float]:
    """Calculate percentile confidence interval from a bootstrap distribution.

    For algorithmic details, see `percentile_ci`.

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

    Reference:
        Bradley Efron. 1981. Nonparametric standard errors and confidence intervals. Can J Statistics 9, 2 (January 1981), 139–158. https://doi.org/10.2307/3314608
    """
    alpha_lo = alpha / 2
    if expand:
        if n_samples is None:
            raise ValueError("n_samples must be provided when expand=True")
        else:
            alpha_lo = expand_alpha(alpha_lo, n_samples)

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
    """Calculate percentile bootstrap confidence interval from sample data.

    $$(\\hat\\theta^ * _{\\alpha/2},\\hat\\theta^ * _{1-\\alpha/2})$$

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

    Reference:
        Bradley Efron. 1981. Nonparametric standard errors and confidence intervals. Can J Statistics 9, 2 (January 1981), 139–158. https://doi.org/10.2307/3314608
    """
    theta_star = bootstrap_distribution(data, n_resamples, statistic, seed=seed)
    return percentile_ci_from_bootstrap_distribution(
        theta_star, alpha, expand, len(data) if expand else None
    )


def reverse_percentile_ci_from_bootstrap_distribution(
    theta_star: np.ndarray,
    theta_hat: float,
    alpha: float = 0.05,
    expand: bool = False,
    n_samples: int | None = None,
) -> tuple[float, float]:
    """Calculate reverse percentile confidence interval from a bootstrap distribution.

    For algorithmic details, see `reverse_percentile_ci`.

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

    Reference:
        Anthony Davison and David Hinkley. 1997. Bootstrap Methods and Their Application. Journal of the American Statistical Association 94, (January 1997). DOI:https://doi.org/10.2307/1271471
    """
    alpha_lo = alpha / 2
    if expand:
        if n_samples is None:
            raise ValueError("n_samples must be provided when expand=True")
        else:
            alpha_lo = expand_alpha(alpha_lo, n_samples)

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
    """Calculate reverse percentile (basic) bootstrap confidence interval  from sample data.

    $$(2\\hat\\theta - \\hat\\theta^ * _{1 - \\alpha/2}, 2\\hat\\theta-\\hat\\theta^ * _ {\\alpha})$$

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

    Reference:
        Anthony Davison and David Hinkley. 1997. Bootstrap Methods and Their Application. Journal of the American Statistical Association 94, (January 1997). DOI:https://doi.org/10.2307/1271471
    """
    theta_star = bootstrap_distribution(data, n_resamples, statistic, seed=seed)
    theta_hat = statistic(data)
    return reverse_percentile_ci_from_bootstrap_distribution(
        theta_star, theta_hat, alpha, expand, len(data) if expand else None
    )


def t_ci_from_bootstrap_distribution(
    theta_star: np.ndarray,
    theta_hat: float,
    se_star: np.ndarray,
    se_hat: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Calculate t-distribution bootstrap confidence interval from a bootstrap distribution.

    For algorithmic details, see `t_ci`.

    Args:
        theta_star: Bootstrap distribution of the statistic.
        theta_hat: Value of statistic computed on original sample.
        se_star: Bootstrap distribution of the standard error.
        se_hat: Standard error of the statistic computed on original sample.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.
    """
    t_star = (theta_star - theta_hat) / se_star
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
    """Calculate t-distribution bootstrap confidence interval from sample data.

    $$(\\hat\\theta-t^ * _{1-\\alpha/2}\\widehat{se},\\hat\\theta-t^ * _{\\alpha/2}\\widehat{se})$$

    where:
    - $t^ * _{q}$ is the $q$th percentile of the bootstrap distribution of $(\\hat\\theta^ * - \\hat\\theta)/\\widehat{se}^ *$
    - $\\widehat{se}^ *$ is the standard error of each bootstrap resample, calculated with the user-provided `se_fn`
    - $\\widehat{se}$ is the standard error of the original sample, calculated with the user-provided `se_fn`

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
    resamples = bootstrap_resample(data, n_resamples, seed)
    theta_hat = statistic(data)
    theta_star = np.apply_along_axis(statistic, 1, resamples)
    se_hat = se_fn(data)
    se_star = np.apply_along_axis(se_fn, 1, resamples)
    return t_ci_from_bootstrap_distribution(
        theta_star, theta_hat, se_star, se_hat, alpha
    )


def bc_ci_from_bootstrap_distribution(
    theta_star: np.ndarray,
    theta_hat: float,
    alpha: float = 0.05,
    expand: bool = False,
    n_samples: int | None = None,
):
    """Calculate bias-corrected bootstrap confidence interval from a bootstrap distribution.

    For algorithmic details, see `bc_ci`.

    Args:
        theta_star: Bootstrap distribution of the statistic.
        theta_hat: Value of statistic computed on original sample.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.
        expand: If True, use the expanded bias-corrected interval to adjust for narrowness
            bias. Defaults to False.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.

    Reference:
        Bradley Efron. 1981. Nonparametric standard errors and confidence intervals. Can J Statistics 9, 2 (January 1981), 139–158. https://doi.org/10.2307/3314608
    """
    p0 = (theta_star <= theta_hat).mean()
    z0 = stats.norm.ppf(p0)
    alpha_lo = alpha / 2
    if expand:
        if n_samples is None:
            raise ValueError("n_samples must be provided when expand=True")
        else:
            alpha_lo = expand_alpha(alpha_lo, n_samples)
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
    """Calculate bias-corrected bootstrap confidence interval from sample data.

    $$(\\hat\\theta^ * _{(\\alpha/2)_{\\text{BC}}}, \\hat\\theta^ * _{(1-\\alpha/2)_{\\text{BC}}})$$

    where
    - $q_{\\text{BC}} = \\Phi(2z_0 + \\Phi^{-1}(q))$
    - $z_0 = \\Phi^{-1}(P(\\hat\\theta ^* \\leq \\hat\\theta))$, where $P(\\hat\\theta ^* \\leq \\hat\\theta)$ denotes the proportion of the bootstrap distribution less than or equal to the original estimate

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

    Reference:
        Bradley Efron. 1981. Nonparametric standard errors and confidence intervals. Can J Statistics 9, 2 (January 1981), 139–158. https://doi.org/10.2307/3314608

    """
    theta_hat = statistic(data)
    theta_star = bootstrap_distribution(data, n_resamples, statistic, seed=seed)
    return bc_ci_from_bootstrap_distribution(
        theta_star, theta_hat, alpha, expand, len(data) if expand else None
    )


def bca_ci_from_bootstrap_distribution(
    theta_star: np.ndarray,
    theta_jack: np.ndarray,
    theta_hat: float,
    alpha: float = 0.05,
    expand: bool = False,
    n_samples: int | None = None,
):
    """Calculate BCa bootstrap confidence interval from a bootstrap distribution.

    For algorithmic details, see `bca_ci`.

    Args:
        theta_star: Bootstrap distribution of the statistic.
        theta_jack: Jackknife distribution of the statistic.
        theta_hat: Value of statistic computed on original sample.
        alpha: Significance level. Defaults to 0.05 for a 95% confidence interval.
        expand: If True, use the expanded BCa interval to adjust for narrowness bias. Defaults
            to False.
        n_samples: Number of samples in original dataset, required if expand=True.
            Defaults to None.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval.

    Reference:
        Bradley Efron. 1987. Better Bootstrap Confidence Intervals. Journal of the American Statistical Association 82, 397 (1987), 171–185. https://doi.org/10.2307/2289144
    """
    p0 = (theta_star <= theta_hat).mean()
    z0 = stats.norm.ppf(p0)
    alpha_lo = alpha / 2
    if expand:
        if n_samples is None:
            raise ValueError("n_samples must be provided when expand=True")
        else:
            alpha_lo = expand_alpha(alpha_lo, n_samples)
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
    """Calculate BCa bootstrap confidence interval from sample data.

    $$(\\hat\\theta^ * _{(\\alpha/2)_{\\text{BCa}}}, \\hat\\theta^ * _{(1-\\alpha/2)_{\\text{BCa}}})$$

    where
    - $q_{\\text{BCa}} = \\Phi(z_0+(z_0+\\Phi^{-1}(q))/(1-a(z_0+\\Phi^{-1}(q))))$
    - $z_0 = \\Phi^{-1}(P(\\hat\\theta ^* \\leq \\hat\\theta))$, where $P(\\hat\\theta ^* \\leq \\hat\\theta)$ denotes the proportion of the bootstrap distribution less than or equal to the original estimate
    - $a=(\\hat\\theta^ * - \\hat\\theta)^3 / (6\\sum(\\hat\\theta ^ * - \\hat\\theta )^2)) ^ {3 / 2}$

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

    Reference:
        Bradley Efron. 1987. Better Bootstrap Confidence Intervals. Journal of the American Statistical Association 82, 397 (1987), 171–185. https://doi.org/10.2307/2289144
    """
    theta_hat = statistic(data)
    theta_star = bootstrap_distribution(data, n_resamples, statistic, seed=seed)
    theta_jack = jackknife_distribution(data, statistic)
    return bca_ci_from_bootstrap_distribution(
        theta_star, theta_jack, theta_hat, alpha, expand, len(data) if expand else None
    )


def expand_alpha(alpha: float, n: int) -> float:
    """Calculate expanded percentiles for expanded confidence intervals.

    $$\\alpha' = \\Phi\\left(\\sqrt{n/(n-1)}t_{\\alpha,n}\\right)$$

    Args:
        alpha: The nominal percentile to be adjusted.
        n: The number of observations in the sample.

    Returns:
        An adjusted alpha to counter narrowness bias of intervals with small n.

    Reference:
        Tim Hesterberg. 1999. Bootstrap Tilting Confidence Intervals. Mathsoft, Inc. Retrieved April 7, 2025 from [https://www.researchgate.net/publication/2269406_Bootstrap_Tilting_Confidence_Intervals](https://www.researchgate.net/publication/2269406_Bootstrap_Tilting_Confidence_Intervals)

    """
    expansion = np.sqrt(n / (n - 1))
    t_alpha = stats.t(n - 1).ppf(alpha)
    alpha_prime = float(stats.norm.cdf(expansion * t_alpha))
    return alpha_prime
