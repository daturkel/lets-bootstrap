from typing import Callable, Generator, Literal, Self

import numpy as np

from bootstrap.functional import (
    _bc_ci,
    _bca_ci,
    _percentile_ci,
    _reverse_percentile_ci,
    bootstrap_resample,
    jackknife_distribution,
)
from bootstrap.resample import _get_rng
from bootstrap.types import RNGSeed


class BootstrapSamples:
    def __init__(
        self, data: np.ndarray, n_resamples: int = 10000, seed: RNGSeed = None
    ):
        self.data = data
        self.n_resamples = n_resamples
        self.seed = seed
        self._samples: np.ndarray | None = None

    @property
    def samples(self) -> np.ndarray:
        if self._samples is None:
            self._samples = bootstrap_resample(
                data=self.data, n_resamples=self.n_resamples, seed=self.seed
            )
        return self._samples

    def batch_samples(self, batch_size: int) -> Generator[np.ndarray]:
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
        if self._samples is not None:
            return self

        self.samples

        return self

    def apply(
        self,
        stat_fn: Callable,
        lazy: bool = True,
        batch_size: int | None = None,
        how: Literal["vectorized", "loop"] = "vectorized",
    ):
        bd = BootstrapDistribution(
            stat_fn=stat_fn,
            sample=self,
            batch_size=batch_size,
            how=how,
        )
        if not lazy:
            bd = bd.materialize()
        return bd


class BootstrapDistribution:
    def __init__(
        self,
        stat_fn: Callable,
        sample: BootstrapSamples,
        batch_size: int | None = None,
        how: Literal["vectorized", "loop"] = "vectorized",
        se_fn: Callable | None = None,
    ):
        self.stat_fn = stat_fn
        self.sample = sample
        self.theta_hat = self.stat_fn(self.sample.data)
        self.batch_size = batch_size
        self.how = how
        self.se_fn = se_fn
        self._distribution: np.ndarray | None = None

    @property
    def distribution(self) -> np.ndarray:
        if self._distribution is None:
            if self.batch_size is None:
                self._distribution = self._compute(self.sample.samples)
            else:
                values_list = []
                for batch in self.sample.batch_samples(self.batch_size):
                    values_list.append(self._compute(batch))
                self._distribution = np.concat(values_list)

        return self._distribution

    def _compute(self, batch: np.ndarray) -> np.ndarray:
        if self.how == "vectorized":
            return self.stat_fn(batch, axis=1)
        elif self.how == "loop":
            return np.apply_along_axis(self.stat_fn, 1, batch)
        else:
            raise ValueError(f"Got unknown value for `how`: {self.how}.")

    def materialize(self) -> Self:
        if self._distribution is not None:
            return self

        self.distribution

        return self

    def ci(
        self,
        alpha: float = 0.5,
        how: Literal[
            "percentile", "reverse_percentile", "t", "bc", "bca"
        ] = "percentile",
        expand: bool = False,
    ) -> tuple[float, float]:
        if how == "percentile":
            return _percentile_ci(
                self.distribution,
                alpha,
                expand,
                len(self.sample.data) if expand else None,
            )
        elif how == "reverse_percentile":
            return _reverse_percentile_ci(
                self.distribution,
                self.theta_hat,
                alpha,
                expand,
                len(self.sample.data) if expand else None,
            )
        elif how == "t":
            if self.se_fn is None:
                raise ValueError(
                    "The se_fn is set to None. Set the se_fn of this object in order to use the bootstrap t interval."
                )
            theta_hat = self.stat_fn(self.sample.data)
            if self.batch_size is None:
                t_star = (self.distribution - theta_hat) / np.apply_along_axis(
                    self.se_fn, 1, self.sample.samples
                )
            else:
                t_star_list = []
                for sample in self.sample.batch_samples(self.batch_size):
                    t_star_list = t_star_list + list(
                        (sample - theta_hat) / self.se_fn(sample)
                    )
                t_star = np.array(t_star_list)
            pct_lo = alpha / 2
            pct_hi = 1 - pct_lo
            se_hat = self.se_fn(self.sample.data)
            lo = theta_hat - se_hat * np.quantile(t_star, pct_hi)
            hi = theta_hat - se_hat * np.quantile(t_star, pct_lo)
            return float(lo), float(hi)
        elif how == "bc":
            return _bc_ci(
                self.distribution,
                self.theta_hat,
                alpha,
                expand,
                len(self.sample.data) if expand else None,
            )
        elif how == "bca":
            return _bca_ci(
                self.distribution,
                jackknife_distribution(self.sample.data, self.stat_fn),
                self.theta_hat,
                alpha,
                expand,
                len(self.sample.data) if expand else None,
            )
        else:
            raise ValueError(f"Got unknown value for `how`: {self.how}.")


def bootstrap(
    data: np.ndarray | list,
    n_resamples: int = 10000,
    seed: RNGSeed = None,
    lazy: bool = True,
) -> BootstrapSamples:
    bs = BootstrapSamples(data=np.asarray(data), n_resamples=n_resamples, seed=seed)
    if not lazy:
        bs = bs.materialize()
    return bs


__all__ = ["BootstrapSamples", "BootstrapDistribution", "bootstrap"]
