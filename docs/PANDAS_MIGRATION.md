# Migrating from pandas

This is a concise, source-verified comparison for pandas users learning
PandRS's real API — not an exhaustive reference, and not a claim of
pandas-equivalence. PandRS is a statically-typed Rust library; several
things that are trivial in pandas (dynamic column indexing, mixed-type
columns, `df['col']` syntax) don't have a direct Rust equivalent, for
language reasons, not because they're unimplemented.

**If you specifically want pandas-*syntax*, not just pandas-*inspired*
Rust**, look at `py_bindings/` instead of this guide — it's a real,
separate PyO3 crate with `PandasDataFrame`/`PandasSeries`/`ILocIndexer`
classes built to be a closer syntactic match, callable from Python. This
guide is about the native Rust API.

## The Big Structural Difference: Two DataFrame Types

Unlike pandas' single `DataFrame`, PandRS has two, with different APIs —
see [USER_GUIDE.md](USER_GUIDE.md#two-dataframe-types--pick-one) for the
full comparison table. Examples below use `OptimizedDataFrame` (the
columnar, performance-oriented one) unless noted.

## Creating a DataFrame

```python
# pandas
df = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "age": [30, 25],
})
```

```rust
// pandrs (OptimizedDataFrame)
use pandrs::OptimizedDataFrame;

let mut df = OptimizedDataFrame::new();
df.add_string_column("name", vec!["Alice".to_string(), "Bob".to_string()])?;
df.add_int_column("age", vec![30, 25])?;
```

There is no dict-literal constructor and no `df['name']` column-indexing
syntax — Rust has neither Python's `__getitem__` overloading nor dynamic
typing for heterogeneous dict values. Build columns one at a time.

## Reading / Writing Files

```python
# pandas
df = pd.read_csv("data.csv")
df.to_csv("out.csv", index=False)
df2 = pd.read_parquet("data.parquet")
```

```rust
// pandrs
use pandrs::OptimizedDataFrame;

let df = OptimizedDataFrame::from_csv("data.csv", true)?;   // true = has header
df.to_csv("out.csv", true)?;                                 // true = write header

// Parquet needs the `parquet` feature:
let df2 = OptimizedDataFrame::from_parquet("data.parquet")?;

// Excel needs the `excel` feature; XLSX (OOXML) only, no legacy .xls:
let df3 = OptimizedDataFrame::from_excel("data.xlsx", None, true, 0, None)?;
```

*(There is a separate free function, `pandrs::io::read_csv(path, has_header)
-> Result<DataFrame>`, that returns the **other**, traditional `DataFrame`
type — don't mix it up with `OptimizedDataFrame::from_csv`.)*

**No SQL.** `pd.read_sql(...)` has no equivalent — PandRS has no database
connectivity of any kind (see
[ECOSYSTEM_INTEGRATION_GUIDE.md](ECOSYSTEM_INTEGRATION_GUIDE.md)).

## Inspecting a DataFrame

```python
# pandas
df.head(5)
df.tail(5)
df.shape
df.dtypes
df.describe()
```

```rust
// pandrs
let first5 = df.head(5)?;
let last5 = df.tail(5)?;
let (rows, cols) = (df.row_count(), df.column_count());
let dtype = df.column_type("age")?;       // per-column, not a whole-frame summary
// There is no `.describe()` — compute the statistics you need directly:
let mean = df.mean("age")?;
let (min, max) = (df.min("age")?, df.max("age")?);
```

## GroupBy

```python
# pandas
result = df.groupby("department").agg(
    salary_mean=("salary", "mean"),
    salary_sum=("salary", "sum"),
)
```

```rust
// pandrs
use pandrs::dataframe::{AggFunc, GroupByExt, NamedAgg};

let result = GroupByExt::groupby(&df, &["department"])?.agg(vec![
    NamedAgg::new("salary".to_string(), AggFunc::Mean, "salary_mean".to_string()),
    NamedAgg::new("salary".to_string(), AggFunc::Sum, "salary_sum".to_string()),
])?;
```

Grouping columns are always a slice (`&["department"]`), even for one
column — there's a same-named inherent `groupby(&str)` from a different
module that shadows the trait method for single-string calls, so the
explicit `GroupByExt::groupby(&df, &[...])` form avoids ambiguity. See
[USER_GUIDE.md](USER_GUIDE.md#groupby) for the (unrelated) legacy
`Series`-level `GroupBy` struct too — don't confuse the two.

## Missing Data

```python
# pandas — works on any Series/DataFrame
df["col"].fillna(0)
df["col"].dropna()
df["col"].isna()
```

```rust
// pandrs — fillna/dropna/is_na exist ONLY on NASeries<T>, not on plain
// Series<T> or on DataFrame/OptimizedDataFrame directly.
use pandrs::NASeries;
use pandrs::na::NA;

let s: NASeries<i64> = NASeries::new(vec![NA::Value(1), NA::NA, NA::Value(3)], None)?;
let filled = s.fillna(0)?;
let dropped = s.dropna()?;
let mask: Vec<bool> = s.is_na();
```

There is no `Option<T>`-based NA on plain `Series<T>` — if you need NA
tracking, use `NASeries<T>` from the start.

## Sorting

```python
# pandas
df.sort_values("age")
df.sort_values(["dept", "age"], ascending=[True, False])
```

```rust
// pandrs
let sorted = df.sort_by("age", true)?;                                       // single column
let sorted = df.sort_by_columns(&["dept", "age"], Some(&[true, false]))?;    // multi-column
```

## Joining / Merging

```python
# pandas
pd.merge(left, right, left_on="lkey", right_on="rkey", how="inner")
pd.merge(left, right, on="key", how="left")
```

```rust
// pandrs
let joined = left.inner_join(&right, "lkey", "rkey")?;
let joined = left.left_join(&right, "key", "key")?; // same name on both sides — pass it twice
```

`inner_join`/`left_join` always take **two** explicit key-column names
(`left_on`, `right_on`) — there's no single shared `on` shorthand.

## Not (Yet) Available

These pandas staples have **no** PandRS equivalent on the stable API today
— don't assume they exist because the name "sounds standard":

- **`get_dummies`** — no one-hot-encoding free function on `DataFrame` (there is one-hot-encoding machinery in `src/ml/preprocessing`, but it's a different, ML-pipeline-shaped API, not a `pd.get_dummies`-style function)
- **`.resample()` / `.ewm()`** on the stable `DataFrame` — real resampling/EWM logic exists elsewhere in the codebase (`src/time_series`), but isn't exposed as a `DataFrame` method the way pandas exposes it
- **`.loc[...]` / `.iloc[...]`** indexers on the Rust `DataFrame`/`OptimizedDataFrame` — this exists on the **Python** side only (`py_bindings`' `ILocIndexer`), not in Rust
- **`.describe()`** — compute individual statistics (`mean`/`min`/`max`/etc.) yourself
- **SQL** (`pd.read_sql`) — no database connectivity at all

If you need one of these today, either implement it against the lower-level
building blocks that do exist (many of the pieces are real, just not
assembled into that exact pandas-shaped method), or use `py_bindings` from
Python where more of the pandas-compatibility layer has already been built.

## Error Handling

pandas raises exceptions; PandRS returns `Result<T, pandrs::error::Error>`
— use `?` to propagate, or `match`/`if let` to handle specific variants
(`ColumnNotFound`, `IndexOutOfBounds`, `InconsistentRowCount`,
`InvalidOperation`, `ParseError`, ...). See
[API_GUIDE.md](API_GUIDE.md#error-handling) for the full pattern.

## Performance Notes for pandas Users

There is no dated, reproducible pandas-vs-PandRS benchmark table published
in this repository — a previous one was unverifiable and has been removed
(see the top-level [README.md](../README.md#performance)). Run
`benches/pandas_benchmark.py` (real pandas) and PandRS's own `cargo bench`
suites yourself if you need a number for your specific workload; don't
carry over marketing-style speedup claims from older versions of this
document.
