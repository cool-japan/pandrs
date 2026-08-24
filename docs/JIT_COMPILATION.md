# JIT Compilation in pandrs

This document describes the `jit`-feature-gated custom-aggregation system in
pandrs (`pandrs::optimized::jit`) and, importantly, what it honestly does
and does not do today. It is *not* a Numba-style dynamic compiler yet —
see "What 'JIT' Currently Means Here" below before you build on this
expecting Numba-equivalent behavior.

## Enabling It

```bash
cargo build --features jit
```

This pulls in Cranelift (`cranelift`, `cranelift-jit`, `cranelift-module`,
`cranelift-frontend`, `cranelift-native`) as dependencies.

## What "JIT" Currently Means Here

`pandrs::optimized::jit::jit_core::jit()` — the decorator-like entry point —
does **not** compile your closure via Cranelift at call time. Its own source
comment says so plainly:

```rust
// In a real implementation, we'd compile right away
// For now, just return the function
```

What you get is a named wrapper around an ordinary Rust closure, executed as
an ordinary Rust closure (still fast — Rust closures are already compiled —
just not dynamically specialized the way Numba specializes Python
functions per-call-site). The `cranelift-*` dependencies are present in the
dependency graph but not wired into this code path. Don't design around a
"first call compiles, later calls are faster" assumption; that isn't
implemented.

## Two Different Constructor Families — Don't Mix Their Closure Shapes

There are two, separately-typed ways to build a JIT-wrapped function, and
they are **not interchangeable**:

| Constructor | Closure shape | Pairs with |
|---|---|---|
| `jit_core::jit(name, f)` | `Fn(Vec<f64>) -> f64` | Nothing else in this crate today — it's a standalone wrapper |
| `core::jit_f64(name, f)` | `Fn(&[f64]) -> f64` | `GroupByJitExt::aggregate_jit` |
| `core::jit_i64(name, f)` | `Fn(&[i64]) -> i64` | (same family) |

Only `f64` and `i64` have type-specific constructors — **there is no
`jit_f32` or `jit_i32`**, despite what you may see suggested elsewhere.

Passing a `jit()`-built function where `aggregate_jit` expects a
`jit_f64()`-built one is a type error (`Vec<f64>` closures don't satisfy the
`Fn(&[f64]) -> f64` bound `aggregate_jit` requires). If you want to use
`GroupByJitExt`, start from `jit_f64`/`jit_i64`, not `jit`.

## Basic Usage: Custom Aggregation via `aggregate_jit`

```rust
use pandrs::optimized::jit::core::jit_f64;
use pandrs::optimized::jit::groupby::GroupByJitExt;

let weighted_mean = jit_f64("weighted_mean", |values: &[f64]| -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for (i, val) in values.iter().enumerate() {
        let weight = (i + 1) as f64;
        weighted_sum += val * weight;
        weight_sum += weight;
    }
    weighted_sum / weight_sum
});

// `aggregate_jit` (and the rest of `GroupByJitExt`) is implemented on
// `optimized::split_dataframe::group::GroupBy`, obtained via that module's
// own `group_by(...)`, not the public `pandrs::optimized::OptimizedDataFrame`
// you build DataFrames with. Getting from a DataFrame to the right `GroupBy`
// type is easy to get wrong copying a doc snippet — use
// `examples/jit_parallel_example.rs` as the verified, complete,
// compiling reference for the full chain.
// let result = grouped.aggregate_jit("value", weighted_mean, "weighted_mean")?;
```

## Built-in `GroupByJitExt` Methods

These are real, pre-built (not user-supplied-closure) methods on the
`GroupByJitExt` trait — no `array_ops` module involved (see below):

- `sum_jit(column, alias)`, `mean_jit(column, alias)`, `std_jit(column, alias)`, `min_jit(column, alias)`, `max_jit(column, alias)`
- `aggregate_jit(column, jit_fn, alias)` — your own `jit_f64`/`jit_i64` function
- `parallel_sum_jit(column, alias, config)`, `parallel_mean_jit(column, alias, config)`, `parallel_std_jit(column, alias, config)` — `config: Option<ParallelConfig>`

There is **no** `var_jit`, `median_jit`, `aggregate_multi_jit`,
`parallel_min_jit`, or `parallel_max_jit` on this trait — those names
appeared in earlier drafts of this document but don't exist in the current
API. If you need variance/median per group, compute it inside your own
`jit_f64` closure.

## There Is No `array_ops` Module

Earlier versions of this document described a
`pandrs::optimized::jit::array_ops` module with pre-built `sum()`/`mean()`/
`std(ddof)`/`quantile(q)`/etc. factory functions. **That module was
deleted** (it existed on disk but was never wired into `mod.rs`, and has
since been removed entirely) — don't `use` it. For the same functionality,
either call the `GroupByJitExt` methods listed above (`sum_jit`, `mean_jit`,
`std_jit`, ...) or call the direct SIMD statistics functions in
[PERFORMANCE_PLAN.md](PERFORMANCE_PLAN.md#direct-simd-statistics-no-jit-machinery-needed)
(`simd_variance_f64`, `simd_skewness_f64`, `simd_kurtosis_f64`, etc., in
`pandrs::optimized::jit::simd_stats`) from inside your own closure.

## SIMD Functions Are Not Zero-Argument Factories

`pandrs::optimized::jit::simd::simd_sum_f64` and `simd_mean_f64` take a
`&[f64]` slice directly and return the computed value — they are **not**
factories you call with no arguments to get back a JIT-pluggable function:

```rust
use pandrs::optimized::jit::simd::simd_sum_f64;

let data = vec![1.0_f64, 2.0, 3.0, 4.0];
let total = simd_sum_f64(&data); // not simd_sum_f64()
```

There is no `simd_sum_f32`/`simd_mean_f32` — the SIMD kernels here are
`f64`-only (see `src/optimized/jit/simd.rs`'s module docs, summarized in the
top-level README's SIMD section, for exactly which operations are AVX2
hand-vectorized vs. scalar-with-auto-vectorization).

## Parallel Execution

Prefer the `GroupByJitExt` parallel methods (`parallel_sum_jit`,
`parallel_mean_jit`, `parallel_std_jit`) — they take an
`Option<ParallelConfig>` directly and don't require you to reason about
`ParallelJitFunction` vs. `JitFunction` types:

```rust
use pandrs::optimized::jit::config::ParallelConfig;

let config = ParallelConfig::new()
    .with_min_chunk_size(10_000)
    .with_max_threads(4);
// (There is no `.with_thread_local(...)` — that method doesn't exist.)

// let parallel_result = grouped.parallel_sum_jit("value", "parallel_sum", Some(config))?;
```

Lower-level primitives also exist (`pandrs::optimized::jit::parallel::{parallel_sum_f64, parallel_mean_f64, parallel_std_f64, parallel_min_f64, parallel_max_f64, parallel_custom}`),
each returning its own `ParallelJitFunction` wrapper type — see
`examples/jit_parallel_example.rs` for a verified working example of that
lower layer; it's easy to get the wrapper-type plumbing subtly wrong
copying a hand-written snippet.

## Type System

Real building blocks, for reference:

- Constructors: `jit_f64`, `jit_i64` (not `jit_f32`/`jit_i32`)
- Traits: `JitNumeric`, `JitType` (`pandrs::optimized::jit::types`)
- `TypedVector` / `NumericValue`: type-erased value plumbing used internally by the generic JIT machinery

## Examples

Verified example files (see the top-level README's Examples section for the
current full list):

- `examples/jit_parallel_example.rs` — the most complete, verified reference for `jit_f64` + `aggregate_jit` + parallel methods together
- `examples/jit_window_operations_example.rs`
- `examples/jit_compilation_demo.rs`
- `examples/integrated_jit_performance_showcase.rs`

## Limitations (Honest)

- `jit()`/`jit_f64()`/`jit_i64()` wrap closures as named Rust closures —
  they do not perform Cranelift-based runtime code generation for your
  closure body today, despite the Cranelift dependencies being present.
- Only `f64` and `i64` have type-specific constructors and SIMD kernels; no
  `f32`/`i32`/bool/string JIT support.
- `array_ops`, `var_jit`, `median_jit`, `aggregate_multi_jit`,
  `parallel_min_jit`/`parallel_max_jit`, and an `auto_vectorize` helper do
  not exist, despite appearing in earlier drafts of this document.
- No GPU integration in this module (see [GPU_ACCELERATION_GUIDE.md](GPU_ACCELERATION_GUIDE.md) for the separate, real `cuda`-feature GPU path).
