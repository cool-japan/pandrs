# PandRS User Guide

This is a concise, source-verified guide to PandRS's real, current API — not
an exhaustive reference. Every code block below is checked against the
current source (paths noted inline); anything not yet wired up on the
stable API is called out explicitly rather than shown as if it worked. For
the full API surface, use `cargo doc --open` or browse `src/`.

## Installation

```toml
[dependencies]
pandrs = { version = "0.4.1", features = ["all-safe"] } # or pick individual features
```

`all-safe` bundles everything except CUDA/WASM/distributed (which need
external toolchains). See the top-level [README.md](../README.md#feature-flags)
for the full feature list.

## Two DataFrame Types — Pick One

PandRS ships **two** DataFrame implementations with different APIs. This is
a real, current architectural fact, not a stepping-stone you can ignore:

| | `pandrs::DataFrame` (traditional) | `pandrs::OptimizedDataFrame` (recommended) |
|---|---|---|
| Storage | `Series<T>`-backed, generic over any `T` | Columnar, 4 fixed types: `Int64`/`Float64`/`String`/`Boolean` |
| Best for | Compatibility, arbitrary types (chrono, custom structs) | Performance, string pooling |
| Add column | `add_column(name: String, series: Series<T>)` | `add_int_column`/`add_float_column`/`add_string_column`/`add_boolean_column` |
| CSV read | `pandrs::io::read_csv(path, has_header) -> Result<DataFrame>` | `OptimizedDataFrame::from_csv(path, has_header) -> Result<Self>` |

**They don't automatically convert into each other except via**
`OptimizedDataFrame::from_dataframe(&DataFrame) -> Result<Self>`. Decide
which one you're using up front; mixing free functions that return one type
with methods on the other is a common source of confusion (see the note in
[docs/API_GUIDE.md](API_GUIDE.md#io-operations) about `to_csv`/`read_csv`).

## Building a DataFrame

```rust
use pandrs::{DataFrame, Series};

let mut df = DataFrame::new();
df.add_column(
    "name".to_string(),
    Series::new(vec!["Alice".to_string(), "Bob".to_string()], Some("name".to_string()))?,
)?;
df.add_column(
    "age".to_string(),
    Series::new(vec![30i64, 25], Some("age".to_string()))?,
)?;

println!("{} rows, {} columns", df.row_count(), df.column_count());
```

Or, for the columnar/performance path:

```rust
use pandrs::OptimizedDataFrame;

let mut df = OptimizedDataFrame::new();
df.add_string_column("name", vec!["Alice".to_string(), "Bob".to_string()])?;
df.add_int_column("age", vec![30, 25])?;

let mean_age = df.mean("age")?; // -> f64 (sum/mean/min/max on OptimizedDataFrame all return f64)
```

## Series and Missing Values

`Series<T>` (generic, any type) has **no** `fillna`/`dropna`/`is_na` —
those live on the separate `NASeries<T>` type:

```rust
use pandrs::{Series, NASeries};
use pandrs::na::NA; // pandrs' own NA<T> enum: NA::Value(T) or NA::NA — not std::Option<T>

// Plain Series: no NA-handling methods.
let s = Series::new(vec![1, 2, 3], Some("x".to_string()))?;
let total = s.sum();     // widened return type (e.g. i32 -> i64); see API_GUIDE.md
let mean = s.mean()?;    // Result<f64>

// NASeries: this is where fillna/dropna/is_na actually live.
// `NASeries::new` takes `Vec<NA<T>>`, not `Vec<Option<T>>` — but `NA<T>`
// implements `From<Option<T>>`, so `.into()` from an Option works too.
let na_series: NASeries<i64> = NASeries::new(
    vec![NA::Value(1), NA::NA, NA::Value(3)],
    Some("x".to_string()),
)?;
let filled = na_series.fillna(0)?;
let dropped = na_series.dropna()?;
let mask: Vec<bool> = na_series.is_na();
```

## GroupBy

Two, unrelated `GroupBy` APIs exist. Use `GroupByExt` (trait, on
`DataFrame`) unless you have a specific reason to reach for the older
`Series`-level `GroupBy` struct:

```rust
use pandrs::{DataFrame, Series};
use pandrs::dataframe::{AggFunc, GroupByExt, NamedAgg};

// ... build `df` as above, with a "department" and a "salary" column ...

let grouped = GroupByExt::groupby(&df, &["department"])?.agg(vec![
    NamedAgg::new("salary".to_string(), AggFunc::Mean, "salary_mean".to_string()),
    NamedAgg::new("salary".to_string(), AggFunc::Sum, "salary_sum".to_string()),
])?;
```

`DataFrame` also has an *inherent* `groupby(&str)` from the pivot module
that shadows the trait method for a single string argument — use the
explicit `GroupByExt::groupby(&df, &[...])` form shown above to avoid
ambiguity, as the top-level README's Quick Start does.

(The legacy `pandrs::GroupBy` struct — `GroupBy::new(keys, &series, name)`
— operates directly on a single `Series`, is unrelated to the above, and is
demonstrated in `examples/groupby_example.rs`.)

## Joins

```rust
// On OptimizedDataFrame:
let joined = left_df.inner_join(&right_df, "left_key_col", "right_key_col")?;
let joined = left_df.left_join(&right_df, "left_key_col", "right_key_col")?;
```

Note the two join-key arguments are separate (`left_on`, `right_on`) — there
is no single shared `on` parameter for same-named columns.

## Sorting

```rust
let sorted = df.sort_by("age", true)?;                       // ascending
let sorted = df.sort_by_columns(&["dept", "age"], None)?;     // multi-column, default ascending
```

## Type-Safe Column Access

```rust
let col_view = df.column("age")?;
if let Some(int_col) = col_view.as_int64() {
    let value = int_col.get(0)?;  // Result<Option<i64>> — owned i64, not a reference
    let sum = int_col.sum();      // i64
}
```

See [API_GUIDE.md](API_GUIDE.md) for the full type-safe access pattern,
including `get_int_column`/`get_float_column`/`get_string_column` (real,
but copy the whole column into a `Vec` — prefer `column()` + `as_int64()`
above for large data).

## I/O

```rust
// CSV
df.to_csv("out.csv", true)?;                              // OptimizedDataFrame -> CSV
let df2 = OptimizedDataFrame::from_csv("out.csv", true)?;  // symmetric read

// Parquet (needs the `parquet` feature)
use pandrs::io::{write_parquet, ParquetCompression};
write_parquet(&df, "out.parquet", Some(ParquetCompression::Snappy))?;
let df3 = OptimizedDataFrame::from_parquet("out.parquet")?;

// Excel — XLSX only, Pure Rust (needs the `excel` feature); no legacy .xls
let df4 = OptimizedDataFrame::from_excel("in.xlsx", None, true, 0, None)?;
// (sheet_name: Option<&str>, header: bool, skip_rows: usize, use_cols: Option<&[&str]>)

// JSON (needs no extra feature)
let df5 = OptimizedDataFrame::from_json("in.json")?;
```

## Error Handling

```rust
use pandrs::error::{Error, Result}; // stable, non-deprecated re-export of core::error

fn process(df: &OptimizedDataFrame) -> Result<f64> {
    if !df.contains_column("sales") {
        return Err(Error::ColumnNotFound("sales".to_string()));
    }
    df.sum("sales")
}
```

Common variants: `ColumnNotFound(String)`, `IndexOutOfBounds { index, size }`,
`InconsistentRowCount { expected, found }`, `InvalidOperation(String)`,
`IoError(String)` (wraps a `String`, not `std::io::Error`), `ParseError { value, target_type, error_msg }`.

## What's Not Wired Up Yet on the Stable `DataFrame`

The following appear in illustrative/target-API snippets elsewhere in this
repo (marked `rust,ignore` in the top-level README) but are **not** callable
methods on the stable `DataFrame`/`OptimizedDataFrame` today: `fillna`
(plain `DataFrame`-level — use `NASeries::fillna` above instead),
`resample`, `ewm`, `get_dummies`, `apply_columns`, `DataFrame::read_parquet`
(associated function — use the free function `pandrs::io::read_csv` or
`OptimizedDataFrame::from_parquet` instead). Don't be surprised if you see
these names in older blog posts or comments; they describe a target API,
not the current one.

## Where to Go Next

- [API_GUIDE.md](API_GUIDE.md) — deeper dive on `OptimizedDataFrame` patterns, error handling, `ColumnView`
- [PERFORMANCE_PLAN.md](PERFORMANCE_PLAN.md) — SIMD, JIT, GPU, parallel, distributed
- [PANDAS_MIGRATION.md](PANDAS_MIGRATION.md) — coming from pandas
- `examples/` — 100+ compiling example programs, organized by topic in the top-level README
