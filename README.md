# Let's Bootstrap 🥾

A readable, user-friendly Python library for bootstrap statistics and confidence intervals.

## Overview

Let's Bootstrap is designed to be an educational tool for learning about bootstrapping and different bootstrap confidence intervals. It provides an intuitive API for performing bootstrap analyses with a focus on:

- **Ease of use**: Simple, readable syntax for common bootstrap operations
- **Performance**: Efficient implementation using NumPy for fast resampling
- **Memory efficiency**: Support for lazy evaluation and batch processing
- **Flexibility**: Multiple confidence interval methods and customization options

> [!NOTE]
> Let's Bootstrap includes [SciPy](https://scipy.org/) as a dependency, though it doesn't use SciPy's own [excellent bootstrap implementations](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
>
> SciPy is the workhorse of scientific computing in Python and Let's Bootstrap doesn't intend to compete with it, even for bootstrapping. I was motivated to create this library by a desire to learn more about bootstrapping methods—there are a lot and I actually implemented a few not available in SciPy—and I hope the readable implementations will be just as educational for anyone else curious to learn more about computational statistics.

## Features

- Bootstrap resampling with reproducible results (seed support)
- Lazy evaluation for memory-efficient processing
- Batch processing for large datasets
- Multiple confidence interval methods:
  - Percentile
  - Reverse percentile (basic)
  - Bootstrap-t
  - Bias-corrected (BC)
  - Bias-corrected and accelerated (BCa)
- Support for both vectorized and loop-based statistic calculation

## Installation

```bash
pip install lets-bootstrap
```

## Quick Start

```python
import numpy as np
from bootstrap import bootstrap

# Create some example data
data = np.random.normal(loc=5, scale=2, size=100)

# Create bootstrap samples
bs = bootstrap(data, n_resamples=10000, seed=42)

# Calculate the mean of each bootstrap sample
bs_means = bs.apply(np.mean)

# Get 95% confidence interval for the mean
lower, upper = bs_means.ci(alpha=0.05, interval="percentile")
print(f"95% CI for the mean: ({lower:.2f}, {upper:.2f})")
```

## Advanced Usage

### Lazy Evaluation

By default, bootstrap samples are only generated when needed:

```python
# Creating the bootstrap object doesn't perform any resampling yet
bs = bootstrap(data, n_resamples=10000)

# Resampling happens here when accessing .samples
first_few_samples = bs.samples[:5]
```

### Memory-Efficient Batch Processing

For large datasets, process bootstrap samples in batches:

```python
# Apply a statistic in batches to avoid memory issues
bs_stats = bs.apply(
    stat_fn=my_complex_statistic, 
    batch_size=1000,
    lazy=False, # compute immediately
)
```

Combine batches with lazy evaluation to defer batched computation until you need it.

```python
# Apply a statistic in batches to avoid memory issues
bs_stats = bs.apply(
    stat_fn=my_complex_statistic, 
    batch_size=1000,
    lazy=True,
)

# Computation happens, in batches, when accessing the distribution
results = bs_stats.distribution
```

### Different Confidence Interval Methods

```python
# Basic percentile method
ci_percentile = bs_stats.ci(alpha=0.05, interval="percentile")

# Bias-corrected and accelerated method
ci_bca = bs_stats.ci(alpha=0.05, interval="bca")
```

### Fluent Interface

```python
data = np.random.exponential(scale=5, size=100)

lo, hi = bootstrap(data).apply(np.median).ci(method="bca")
```

## Documentation

Full documentation is available [here](https://daturkel.github.io/lets-bootstrap/bootstrap.html).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.