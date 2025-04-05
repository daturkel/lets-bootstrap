"""Pytest fixtures for bootstrap tests."""

from typing import Callable

import numpy as np
import pytest

from bootstrap import BootstrapSamples
from bootstrap.types import RNGSeed


@pytest.fixture
def random_data() -> np.ndarray:
    """Generate random data array for tests."""
    return np.random.random(size=(100))


@pytest.fixture
def bootstrap_samples(
    random_data: np.ndarray,
) -> Callable[[RNGSeed], BootstrapSamples]:
    """Create a BootstrapSamples factory."""

    def _bs_factory(seed: RNGSeed = None) -> BootstrapSamples:
        return BootstrapSamples(data=random_data, n_resamples=50, seed=seed)

    return _bs_factory
