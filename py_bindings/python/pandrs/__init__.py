"""
PandRS: Rust-powered DataFrame library for Python with pandas-like API
===================================================================

PandRS is a high-performance DataFrame library written in Rust and
exposed to Python through PyO3 bindings. It aims to provide a familiar
pandas-like API while leveraging Rust's performance benefits.

Main Classes
-----------
DataFrame : 2-dimensional labeled data structure
Series : 1-dimensional labeled array
NASeries : Series with explicit handling of missing values

Examples
--------
>>> import pandrs as pr
>>> import numpy as np
>>> 
>>> # Create a DataFrame
>>> df = pr.DataFrame({
...     'A': [1, 2, 3],
...     'B': ['a', 'b', 'c'],
...     'C': [np.nan, 4.5, 6.0]
... })
>>> 
>>> # Convert to pandas DataFrame if needed
>>> pd_df = df.to_pandas()
"""

# Import Rust extensions
from .pandrs import (
    DataFrame,
    Series,
    NASeries,
    __version__,
)

# Public API
__all__ = [
    'DataFrame',
    'Series',
    'NASeries',
    '__version__',
]

# benchmark module - can be run as `python -m pandrs.benchmark`
from . import benchmark