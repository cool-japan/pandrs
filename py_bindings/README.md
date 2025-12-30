# PandRS Python Bindings

Python bindings for PandRS (Rust-powered pandas-like DataFrame library).

## Overview

PandRS is a high-performance DataFrame library written in Rust and exposed to Python through PyO3 bindings. It aims to provide a familiar pandas-like API while leveraging Rust's performance benefits.

## Installation

### From PyPI (not yet available)

```bash
pip install pandrs
```

### From Source

```bash
# Clone the repository
git clone https://github.com/cool-japan/pandrs.git
cd pandrs/py_bindings

# Install in development mode
pip install -e .
```

## Features

- Rust-powered core for high performance
- Pandas-like API for easy adoption
- Seamless interoperability with pandas and NumPy
- Jupyter Notebook integration
- Type-safe operations

## Quick Start

```python
import pandrs as pr
import numpy as np

# Create a DataFrame
df = pr.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e'],
    'C': [1.1, 2.2, 3.3, 4.4, 5.5]
})

# Display the DataFrame
print(df)

# Convert to pandas DataFrame if needed
pd_df = df.to_pandas()

# Read from CSV
df = pr.DataFrame.read_csv('data.csv')

# Write to CSV
df.to_csv('output.csv')
```

## Interoperability with pandas

PandRS provides seamless interoperability with pandas:

```python
import pandas as pd
import pandrs as pr

# Convert pandas DataFrame to PandRS
pd_df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['x', 'y', 'z']
})
pr_df = pr.DataFrame.from_pandas(pd_df)

# Convert PandRS DataFrame to pandas
pd_df_again = pr_df.to_pandas()
```

## Jupyter Notebook Integration

PandRS has built-in Jupyter Notebook integration:

```python
import pandrs as pr
from pandrs.jupyter import display_dataframe

# Create a DataFrame
df = pr.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c']
})

# Display the DataFrame with custom options
display_dataframe(df, max_rows=10, max_cols=5)
```

## Why PandRS?

- **Performance**: Rust's memory safety and performance characteristics make PandRS faster for many operations
- **Memory Efficiency**: Better memory usage patterns compared to pandas
- **Type Safety**: Rust's type system helps prevent many common errors
- **Familiar API**: Similar API to pandas, making it easy to adopt

## Performance Benchmarks

PandRS offers excellent performance in its native Rust implementation, though Python bindings add some overhead due to data conversion.

### Native Rust Performance

The native Rust implementation is very fast:

```
DataFrame creation (3 columns x 100,000 rows): ~50ms
```

### Python Binding Performance

When using PandRS through Python bindings, there's some overhead from Python-Rust data conversion:

```
# Run benchmarks
python -m pandrs.benchmark
```

Key observations:
- Native Rust implementation is significantly faster
- Python binding overhead is mainly due to data conversion between languages
- For larger datasets, the performance difference becomes less significant

## Requirements

- Python 3.8 or later
- NumPy 1.20 or later
- pandas 1.3 or later (for interoperability features)

## Development Notes

### Git Configuration

Unlike the main PandRS library, we intentionally track `Cargo.lock` in this Python bindings subproject. This is because:

1. The py_bindings subproject produces a binary package (Python extension module) rather than a library
2. Tracking `Cargo.lock` ensures reproducible builds for binary packages
3. This follows Cargo's recommendation to track `Cargo.lock` for binary crates and not track it for library crates

This explains why the root `.gitignore` excludes `/Cargo.lock` (for the main library) but not `py_bindings/Cargo.lock` (for this binary extension).

## License

Apache License 2.0