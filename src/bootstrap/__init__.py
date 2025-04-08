"""
.. include:: ../../README.md
   :start-line: 3
   :end-before: Documentation

## Documentation
"""

from __future__ import annotations

from typing import Callable, Generator, Literal, Self

import numpy as np

from bootstrap import ci, resample, simulation, types
from bootstrap.ci import (
    bc_ci_from_bootstrap_distribution,
    bca_ci_from_bootstrap_distribution,
    percentile_ci_from_bootstrap_distribution,
    reverse_percentile_ci_from_bootstrap_distribution,
    t_ci_from_bootstrap_distribution,
)
from bootstrap.resample import _get_rng, bootstrap_resample, jackknife_distribution
from bootstrap.types import RNGSeed


def bootstrap(
    data: np.ndarray | list,
    n_resamples: int = 10000,
    seed: RNGSeed = None,
    lazy: bool = True,
) -> BootstrapSamples:
    """Create bootstrap samples from the given data.

    This function creates bootstrap samples by randomly sampling with replacement
    from the provided data.

    Args:
        data: The dataset to resample from. Can be a numpy array or a list.
        n_resamples: Number of bootstrap resamples to generate. Default is 10000.
        seed: Random number generator seed for reproducibility. Can be an int,
            numpy.random.Generator, numpy.random.RandomState, or None.
        lazy: If True, resampling is deferred until the samples are actually
            needed. If False, resampling is done immediately. Default is True.

    Returns:
        A `BootstrapSamples` object containing the resamples or the configuration to generate
        them.

    Examples:
        >>> import numpy as np
        >>> from bootstrap import bootstrap
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> bs = bootstrap(data, n_resamples=1000, seed=42)
        >>> bs_samples = bs.samples  # This triggers the resampling
    """
    bs = BootstrapSamples(data=np.asarray(data), n_resamples=n_resamples, seed=seed)
    if not lazy:
        bs = bs.materialize()
    return bs


class BootstrapSamples:
    """A container for bootstrap resamples.

    This class manages bootstrap resamples created from an original dataset. It supports
    lazy evaluation where resamples are only generated when needed and batched processing
    for memory efficiency.

    Attributes:
        data: The original dataset from which resamples are created.
        n_resamples: Number of bootstrap resamples to generate.
        seed: Random number generator seed for reproducibility.
        samples: The resampled samples from data. Generated on first access or with a call
            to `materialize`.
    """

    def __init__(
        self, data: np.ndarray, n_resamples: int = 10000, seed: RNGSeed = None
    ):
        """Initialize a `BootstrapSamples` object.

        Args:
            data: The original dataset to be resampled.
            n_resamples: Number of bootstrap resamples to generate. Default is 10000.
            seed: Random number generator seed for reproducibility. Can be an int,
                numpy.random.Generator, numpy.random.RandomState, or None.
        """
        self.data = data
        self.n_resamples = n_resamples
        self.seed = seed
        self._samples: np.ndarray | None = None

    @property
    def samples(self) -> np.ndarray:
        """Generate and return bootstrap resamples.

        Returns:
            A numpy array of bootstrap resamples with shape (n_resamples, len(data)).

        Note:
            The samples are generated on first access and cached for subsequent calls.
        """
        if self._samples is None:
            self._samples = bootstrap_resample(
                data=self.data, n_resamples=self.n_resamples, seed=self.seed
            )
        return self._samples

    def batch_samples(self, batch_size: int) -> Generator[np.ndarray]:
        """Generate bootstrap resamples in batches.

        This method is useful for processing large numbers of resamples without
        exhausting memory. The final result will be the same as a call to `samples`.

        Args:
            batch_size: Number of resamples to generate or return in each batch.

        Yields:
            Batches of bootstrap resamples as numpy arrays.

        Note:
            If samples have already been generated, this method returns batches
            of the existing samples. Otherwise, it generates new batches on demand.
        """
        rng = _get_rng(self.seed)
        for i in range(0, self.n_resamples, batch_size):
            if self._samples is None:
                if i + batch_size > self.n_resamples:
                    batch_size = self.n_resamples - i
                yield bootstrap_resample(
                    data=self.data, n_resamples=batch_size, seed=rng
                )
            else:
                yield self._samples[i : i + batch_size]

    def materialize(self) -> Self:
        """Generate all bootstrap resamples immediately.

        Returns:
            This object, with all samples generated and available in `samples`.
        """
        self.samples

        return self

    def apply(
        self,
        stat_fn: Callable,
        lazy: bool = True,
        batch_size: int | None = None,
        how: Literal["vectorized", "loop"] = "vectorized",
        se_fn: Callable | None = None,
        se_how: Literal["vectorized", "loop"] = "vectorized",
    ):
        """Apply a statistical function to the bootstrap resamples.

        This method creates a `BootstrapDistribution` by applying the given function to each
        bootstrap sample.

        Args:
            stat_fn: The statistical function to apply to each resample.
            lazy: If True, computation is deferred until the distribution is actually needed.
                If False, computation is done immediately. Useful with `batch_size` for
                memory efficiency with large datasets. Default is True.
            batch_size: Number of resamples to process at once. Useful in combination with
                `lazy` for memory efficiency with large datasets. Default is None (all at
                once).
            how: Method for applying the statistical function. Options are "vectorized",
                which calls `stat_fn(batch, axis=1)`, and "loop", which calls `np.apply_along_axis(stat_fn, 1, batch)`.
                Default is "vectorized".
            se_fn: Optional function to compute standard error, required for bootstrap t confidence
                intervals.
            se_how: Same as `how`, but for `se_fn`.

        Returns:
            A `BootstrapDistribution` object containing the results of applying the statistical
                function to the resamples.
        """
        bd = BootstrapDistribution(
            stat_fn=stat_fn,
            sample=self,
            batch_size=batch_size,
            how=how,
            se_fn=se_fn,
            se_how=se_how,
        )
        if not lazy:
            bd = bd.materialize()
        return bd


class BootstrapDistribution:
    """A container for bootstrap distributions created by applying a statistic to resamples.

    This class manages the computation and storage of statistical values derived from bootstrap
    resamples. It supports lazy evaluation, batched processing for memory efficiency, and
    various confidence interval methods.

    Attributes:
        stat_fn: The statistical function to apply to each bootstrap resample.
        sample: The `BootstrapSamples` object containing the resamples.
        theta_hat: The statistic computed on the original dataset.
        batch_size: Number of resamples to process at once, or `None` to process at once.
        how: Method for applying the stat_fn to resamples ("vectorized" or "loop").
        se_fn: Optional function to compute standard error, required for bootstrap t confidence
            intervals.
    """

    def __init__(
        self,
        stat_fn: Callable,
        sample: BootstrapSamples,
        batch_size: int | None = None,
        how: Literal["vectorized", "loop"] = "vectorized",
        se_fn: Callable | None = None,
        se_how: Literal["vectorized", "loop"] = "vectorized",
    ):
        """Initialize a `BootstrapDistribution` object.

        Args:
            stat_fn: The statistical function to apply to each bootstrap resample.
            sample: The `BootstrapSamples` object containing the resamples.
            batch_size: Number of resamples to process at once. Default is None (all at once).
            how: Method for applying the `stat_fn`. Options are "vectorized" (calls
                stat_fn(batch, axis=1)) or "loop" (calls np.apply_along_axis(stat_fn, 1, batch)).
                Default is "vectorized".
            se_fn: Function to compute standard error, required for t-type confidence
                intervals. Default is None.
            se_how: Same as `how`, but for `se_fn`.
        """
        self.stat_fn = stat_fn
        self.sample = sample
        self.theta_hat = self.stat_fn(self.sample.data)
        self.batch_size = batch_size
        self.how: Literal["vectorized", "loop"] = how
        self.se_fn = se_fn
        self._distribution: np.ndarray | None = None
        self._se: np.ndarray | None = None
        self._se_hat: float | None = None
        self.se_how: Literal["vectorized", "loop"] = se_how

    @property
    def distribution(self) -> np.ndarray:
        """Generate and return the bootstrap distribution.

        Computes the statistic on all bootstrap resamples, either all at once or in batches
        depending on batch_size.

        Returns:
            A numpy array containing the statistic computed on each bootstrap resample.

        Note:
            The distribution is computed on first access and cached for subsequent calls.
        """
        if self._distribution is None:
            if self.batch_size is None:
                self._distribution = self._compute(
                    self.stat_fn, self.sample.samples, self.how
                )
            else:
                values_list = []
                for batch in self.sample.batch_samples(self.batch_size):
                    values_list.append(self._compute(self.stat_fn, batch, self.how))
                self._distribution = np.concat(values_list)

        return self._distribution

    def _compute(
        self, fn: Callable, batch: np.ndarray, how: Literal["vectorized", "loop"]
    ) -> np.ndarray:
        if how == "vectorized":
            return fn(batch, axis=1)
        elif how == "loop":
            return np.apply_along_axis(fn, 1, batch)
        else:
            raise ValueError(f"Got unknown value for `how`: {self.how}.")

    @property
    def se(self) -> np.ndarray:
        if self.se_fn is None:
            raise ValueError(
                "The se_fn is set to None. Set the se_fn of this object in order to calculate stndard errors."
            )
        if self._se is None:
            if self.batch_size is None:
                self._se = self._compute(self.se_fn, self.sample.samples, self.se_how)
            else:
                values_list = []
                for batch in self.sample.batch_samples(self.batch_size):
                    values_list.append(self._compute(self.se_fn, batch, self.se_how))
                self._se = np.concat(values_list)

        return self._se

    @property
    def se_hat(self) -> float:
        if self.se_fn is None:
            raise ValueError(
                "The se_fn is set to None. Set the se_fn of this object in order to calculate stndard errors."
            )
        if self._se_hat is None:
            self._se_hat = self.se_fn(self.sample.data)

        return self._se_hat  # type: ignore

    def materialize(self) -> Self:
        """Generate the bootstrap distribution immediately.

        Returns:
            This object, with the distribution computed and available in `distribution`.
        """
        self.distribution

        return self

    def ci(
        self,
        alpha: float = 0.5,
        interval: Literal[
            "percentile", "reverse_percentile", "t", "bc", "bca"
        ] = "percentile",
        expand: bool = True,
    ) -> tuple[float, float]:
        """Compute confidence intervals for the bootstrap distribution.

        Args:
            alpha: The significance level (0 to 1). Default is 0.5 (50% confidence).
            how: Method for computing confidence intervals. Options are:
                - "percentile": Percentile method.
                - "reverse_percentile": Reverse percentile or "basic" method.
                - "t": Bootstrap t interval (requires se_fn to be set).
                - "bc": Bias-corrected interval.
                - "bca": Bias-corrected and accelerated interval.
                Default is "percentile".
            expand: Whether to expand the confidence interval to account for narrowness
                bias when sample size is small. Default is True.

        Returns:
            A tuple representing the confidence interval bounds.

        Raises:
            ValueError: If an unknown method is specified or if se_fn is None when
                using the bootstrap t interval.
        """
        if interval == "percentile":
            return percentile_ci_from_bootstrap_distribution(
                self.distribution,
                alpha,
                expand,
                len(self.sample.data) if expand else None,
            )
        elif interval == "reverse_percentile":
            return reverse_percentile_ci_from_bootstrap_distribution(
                self.distribution,
                self.theta_hat,
                alpha,
                expand,
                len(self.sample.data) if expand else None,
            )
        elif interval == "t":
            if self.se_fn is None:
                raise ValueError(
                    "The se_fn is set to None. Set the se_fn of this object in order to use the bootstrap t interval."
                )
            se_hat = self.se_fn(self.sample.data)
            return t_ci_from_bootstrap_distribution(
                self.distribution, self.theta_hat, self.se, se_hat, alpha
            )
        elif interval == "bc":
            return bc_ci_from_bootstrap_distribution(
                self.distribution,
                self.theta_hat,
                alpha,
                expand,
                len(self.sample.data) if expand else None,
            )
        elif interval == "bca":
            return bca_ci_from_bootstrap_distribution(
                self.distribution,
                jackknife_distribution(self.sample.data, self.stat_fn),
                self.theta_hat,
                alpha,
                expand,
                len(self.sample.data) if expand else None,
            )
        else:
            raise ValueError(f"Got unknown value for `how`: {self.how}.")


__all__ = [
    "bootstrap",
    "BootstrapSamples",
    "BootstrapDistribution",
    "ci",
    "resample",
    "simulation",
    "types",
]
