import inspect
from typing import Callable

import numpy as np
from scipy.stats._distn_infrastructure import rv_frozen, rv_generic

Distribution = rv_generic | rv_frozen


def simulate(
    population: Distribution,
    theta: float,
    sample_size: int,
    statistic: Callable,
    interval: Callable,
    n_sims: int = 1000,
    n_resamples: int = 5000,
    alpha: float = 0.05,
    **kwargs,
) -> tuple[int, int, int, list[float], list[tuple[float, float]]]:
    """
    Simulates the coverage of a confidence interval for a parameter of a distribution.

    Args:
        population: The distribution to sample from.
        theta: The true value of the parameter.
        sample_size: The size of each sample.
        statistic: A function that computes the statistic from a sample.
        interval: A function that computes a confidence interval from a sample.
        n_sims: The number of simulations to run.
        n_resamples: The number of resamples to use for bootstrap methods.
        alpha: The significance level of the confidence interval.
        **kwargs: Additional arguments to pass to the interval function.

    Returns:
        A tuple (n_covered, n_low, n_high, sampling_distribution, intervals):
          n_covered: number of simulations where the confidence interval contains the true
            statistic
          n_low: number of simulations where the confidence interval is below the true statistic
          n_high: number of simulations where the confidence interval is above the true statistic
          sampling_distribution: the empirical distribution of the statistic across every
            run of the simulation
          intervals: list of intervals generated during the simulation
    """
    n_covered = 0
    n_low = 0
    n_high = 0
    parameters = inspect.signature(interval).parameters
    sampling_distribution = []
    intervals = []
    for i in range(n_sims):
        rng = np.random.default_rng(i)
        sample = population.rvs(size=sample_size, random_state=rng)
        kwargs_ = {
            "data": sample,
            "n_resamples": n_resamples,
            "statistic": statistic,
            "alpha": alpha,
            "seed": i,
            **kwargs,
        }
        kwargs_ = {k: v for k, v in kwargs_.items() if k in parameters}
        lo, hi = interval(**kwargs_)
        if theta < lo:
            n_high += 1
        elif theta > hi:
            n_low += 1
        else:
            n_covered += 1
        sampling_distribution += statistic(sample)
        intervals += (lo, hi)
    return (
        n_covered,
        n_low,
        n_high,
        sampling_distribution,
        intervals,
    )
