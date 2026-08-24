# PandRS Python Bindings

Python bindings for [PandRS](https://github.com/cool-japan/pandrs), a
Rust-powered DataFrame library. Built with [PyO3](https://pyo3.rs) and
[maturin](https://www.maturin.rs).

> **Status (2026-08, pandrs 0.4.x).** These bindings expose a *subset* of the
> Rust crate. They are **not** a drop-in pandas replacement: only the methods
> listed in the coverage tables below exist, and unsupported options raise a
> Python exception rather than being silently ignored. The coverage table is
> maintained by hand against the source and dated so drift is visible.

## Installation

### From source (maturin)

```bash
git clone https://github.com/cool-japan/pandrs.git
cd pandrs/py_bindings

# Build & install the extension into the active environment
pip install maturin
maturin develop --release
```

`maturin` is the build backend (`pyproject.toml`); there is no `setup.py`.

## What is actually exposed

The compiled extension registers several classes, but the pure-Python shim
package (`python/pandrs/__init__.py`) re-exports only a few of them at the top
level. The distinction matters:

| Python name | Import path | Kind |
|---|---|---|
| `pandrs.DataFrame` | top level | string-typed general DataFrame |
| `pandrs.Series` | top level | 1-D string series |
| `pandrs.NASeries` | top level | series with explicit NA handling |
| `pandrs.__version__` | top level | sourced from `Cargo.toml` at build time |
| `PandasDataFrame` | `pandrs.pandrs` | pandas-shaped compatibility frame |
| `PandasSeries`, `GroupBy`, `ILocIndexer`, `LocIndexer` | `pandrs.pandrs` | helpers for `PandasDataFrame` |
| `OptimizedDataFrame`, `LazyFrame`, `StringPool` | `pandrs.pandrs` | typed columnar engine |
| `read_csv`, `concat` | `pandrs.pandrs` | pandas-style free functions |

```python
import pandrs as pr

# Top-level (forwarded by the shim):
df = pr.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

# Compatibility / optimized classes live on the compiled module for now:
from pandrs.pandrs import PandasDataFrame, OptimizedDataFrame, read_csv, concat
```

> **Note.** Forwarding `PandasDataFrame` / `OptimizedDataFrame` / `read_csv` /
> `concat` to the top-level `pandrs` namespace requires a change to the
> pure-Python shim (`python/pandrs/__init__.py`), which is tracked separately.

## The three DataFrame classes

There are three distinct DataFrame types with different trade-offs. They are
**not** interchangeable and are not merged into one.

### `DataFrame` (top-level, string-typed)

Every column is stored as text. Backed by the real pandrs `DataFrame`, so I/O
and structural operations are genuine, but numeric work goes through string
parsing. Good for interop and CSV/JSON round-trips.

| Method | Notes |
|---|---|
| `DataFrame(dict)` | build from `{col: list}` |
| `columns` (get/set), `shape` | real |
| `__getitem__("col")` | returns a `Series` |
| `rename_columns(dict)` | real |
| `iloc([i, j, ...])` | positional row selection (reindexed `0..k`) |
| `to_csv(path)` / `read_csv(path)` | real CSV I/O |
| `to_json()` / `read_json(str)` | real JSON I/O |
| `to_pandas()` / `from_pandas(df)` | requires pandas; columns become object dtype |

### `PandasDataFrame` (pandas-shaped compatibility layer)

A pandas-API-shaped wrapper over the string-typed `DataFrame`. Every method
drives real pandrs code; unsupported options raise. Numeric operations parse
strings on the fly (a `"NA"` or empty cell counts as missing; a column is
treated as numeric only when all its non-missing cells parse as floats).

| Method | Backed by | Notes |
|---|---|---|
| `head(n)` / `tail(n)` | `DataFrame::head/tail` | real slice |
| `describe()` | `pandas_compat::describe` | numeric columns only; result is a `statistic` column + one column per numeric column (a transpose of pandas' layout); empty frame if no numeric columns |
| `query(expr)` | `QueryExt::query` | real expression engine; `inplace=True` raises |
| `groupby(by).{size,count,sum,mean,min,max,std,var,agg}()` | `pandas_compat::DataFrameGroupBy` | real aggregation |
| `drop(columns=…, index=…, labels=…, axis=…)` | `select_columns` + row filter | `inplace=True` raises; row-drop reindexes |
| `merge(right, how, on, …)` | `pandas_compat::merge` | single shared key; `left_index`/`right_index`/`sort`/differing `left_on`/`right_on` raise |
| `to_csv(path=…, sep, na_rep, index, header)` | in-crate serializer | honors all four options; writes file or returns string |
| `PandasDataFrame.read_csv(path)` | `DataFrame::from_csv` | `sep != ","`, `names`, `index_col` raise |
| `__getitem__` | `get_column` / `select` | `df["c"]` → Series, `df[["a","b"]]` → frame |
| `iloc()[key]` / `loc()[key]` | positional selection | int / list / slice; `loc` raises on a non-default index |
| `info()`, `columns`, `shape`, `index` | real | `index` is the default positional `0..n` |

Constructor `index=` / `columns=` / `dtype=` are rejected (this frame always
uses a default positional index and string columns). A genuine string value
equal to `"NA"` is indistinguishable from missing in this model.

### `OptimizedDataFrame` (typed columnar engine)

Real typed columns (int64 / float64 / bool / string-pooled) with the best
performance. This is the class to prefer for heavy numeric work.

| Method | Notes |
|---|---|
| `add_int_column` / `add_float_column` / `add_string_column` / `add_boolean_column` | typed inserts |
| `add_string_column_from_pylist` | string-pool optimized |
| `column_names`, `shape`, `set_column_names`, `rename_columns` | real |
| `filter(bool_column)` | real row filter |
| `to_pandas()` / `from_pandas(df)` | typed conversion |
| `to_parquet(path, compression=…)` / `from_parquet(path)` | requires the `parquet` feature; unknown compression names raise |
| `LazyFrame(df).filter().select().aggregate().execute()` | lazy pipeline |

## Interoperability with pandas

```python
import pandas as pd
import pandrs as pr

pr_df = pr.DataFrame.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
pd_df = pr_df.to_pandas()
```

`to_pandas()` on the string-typed `DataFrame` yields object-dtype columns; use
`OptimizedDataFrame` for typed round-trips.

## Jupyter integration

```python
from pandrs.jupyter import display_dataframe
display_dataframe(df, max_rows=10, max_cols=5)
```

## Not yet exposed

A large majority of the Rust crate's surface (advanced statistics, time series,
window functions, most string/rolling helpers, ML, distributed/GPU) is **not**
reachable from these bindings. Use the Rust crate directly for that
functionality. Contributions that widen coverage are welcome — please update the
tables above and their date when you do.

## Requirements

- Python 3.8+
- NumPy 1.20+
- pandas 1.3+ (only for the `to_pandas` / `from_pandas` interop paths)

## Development notes

`Cargo.lock` is intentionally tracked in this subproject: it produces a binary
package (a Python extension module), and Cargo recommends committing the lockfile
for binaries. The root library's `.gitignore` excludes `/Cargo.lock`; this
subproject's lockfile is kept.

## License

Apache License 2.0
