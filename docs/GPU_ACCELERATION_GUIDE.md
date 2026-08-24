# GPU Acceleration Guide

This guide is a concise, verified-against-source description of the `cuda`
feature. It is deliberately shorter than an exhaustive API reference — for
anything beyond the basics below, use the real, compiling example files
listed at the end rather than a hand-copied snippet, since GPU code can't be
compile-checked on every machine (it needs the CUDA toolkit and CUDA
hardware) and this guide would otherwise risk repeating the same kind of
unverifiable claims it's trying to fix.

## Enabling It

```toml
[dependencies]
pandrs = { version = "0.4.2", features = ["cuda"] }
```

Building with `cuda` links against `cudarc`, which needs the **CUDA
toolkit** installed. Without a CUDA-capable GPU at runtime, code using this
feature is designed to fall back to CPU (see "CPU Fallback" below) — but
the *build* still needs the toolkit present.

## Core Types

`pandrs::gpu`:

- **`GpuConfig`** — a plain struct with public fields and a `Default` impl.
  **There is no builder** (`GpuConfig::new().with_x(...)` does not exist):
  ```rust
  use pandrs::gpu::GpuConfig;

  let config = GpuConfig {
      memory_limit: 1_000_000_000, // 1GB, in bytes
      min_size_threshold: 25_000,  // below this element count, stay on CPU
      device_id: 0,
      fallback_to_cpu: true,
      ..Default::default()
  };
  ```
  Defaults: `enabled: true`, `memory_limit: 1GB`, `device_id: 0`,
  `fallback_to_cpu: true`, `use_pinned_memory: true`,
  `min_size_threshold: 10_000`.

- **`GpuContext::new(config: GpuConfig) -> Self`** — `is_available()`,
  `get_device_status() -> GpuDeviceStatus`, `should_use_gpu(size: usize) -> bool`,
  `config() -> &GpuConfig`.

- **`GpuDeviceStatus`** — `available: bool`, `cuda_version: Option<String>`,
  `device_name: Option<String>`, `total_memory: Option<usize>`,
  `free_memory: Option<usize>`, `core_count: Option<usize>`.

- **Free functions**: `init_gpu() -> Result<GpuDeviceStatus>`,
  `init_gpu_with_config(config: GpuConfig) -> Result<GpuDeviceStatus>`,
  `get_gpu_manager() -> Result<GpuManager>`.

```rust
use pandrs::gpu::{init_gpu_with_config, GpuConfig};

let status = init_gpu_with_config(GpuConfig::default())?;
if status.available {
    println!("GPU: {:?}, {} cores", status.device_name, status.core_count.unwrap_or(0));
} else {
    println!("No CUDA GPU detected — code should fall back to CPU");
}
```

This part is verified against `src/gpu/mod.rs` directly and should compile
with the `cuda` feature enabled and the CUDA toolkit installed.

## CPU Fallback

`fallback_to_cpu` (default `true`) is the intended behavior when no GPU is
present or a GPU op fails — this repository's CHANGELOG for 0.4.1 lists
"real CPU fallbacks for non-CUDA GPU paths" as a fix that landed, so treat
this as a real, exercised code path rather than aspirational. If you build
without the `cuda` feature at all, GPU-accelerated entry points are not
compiled in — check each module for its own `#[cfg(feature = "cuda")]`
gating before assuming a function is always present.

## Where GPU Acceleration Actually Lives

Real, `cuda`-gated GPU code exists in several places — this guide doesn't
attempt to enumerate every method signature (they change; the example files
below are kept in sync with the real API by virtue of being compiled):

- **Window operations**: `src/dataframe/gpu_window.rs` (`GpuWindowContext`, rolling/expanding/EWM helpers)
- **ML algorithms**: `src/ml/gpu.rs` — real `linear_regression`, `kmeans`, `pca` (and others) with GPU/CPU paths
- **Stats**: `src/stats/gpu.rs`
- **Matrix ops**: `src/gpu/multi_gpu.rs`, `src/gpu/cuda.rs`
- **Python bindings**: `py_bindings/src/py_gpu.rs` — real pyo3 classes (`GpuConfig`, `GpuDeviceStatus`, `GpuMatrix`) exposing GPU-accelerated PCA, k-means, linear regression, and correlation to Python

> **Known issue, not a doc problem:** as of this writing, chaining
> `GpuWindowContext`-based rolling windows into `.mean()` on a DataFrame
> (`df.gpu_rolling(window, &ctx).mean()`) returns a DataFrame of **string**
> data rather than the numeric result you'd expect
> (`src/dataframe/gpu_window.rs`, the `.mean()`/`.sum()` adapter methods
> around the `WindowOperationResult` conversion). This is a source bug, not
> something this doc can paper over — until it's fixed, verify any
> DataFrame-level `gpu_rolling(...).<agg>()` result's column type before
> trusting it in production code. The lower-level `GpuWindowContext` methods
> that take `&[f64]` directly (e.g. computing on a column's raw slice) are
> not affected by this specific bug.

## Performance Thresholds

`GpuContext::should_use_gpu(size)` and `GpuConfig::min_size_threshold`
(default 10,000 elements) gate whether an operation goes to the GPU.
`GpuWindowContext` has its own, separate default `gpu_threshold_size` (also
50,000 in the struct's `Default` impl) for **window** operations
specifically — these are two different thresholds, not one shared setting;
check `src/gpu/mod.rs` and `src/dataframe/gpu_window.rs` respectively if you
need the exact current numbers, since they're plain struct fields that can
change between releases without this doc being updated in lockstep.

## Verified Examples

Run these directly (`cargo run --example <name> --features cuda`, requires
a CUDA-capable GPU and toolkit at runtime, not just build time):

- `examples/gpu_dataframe_api_example.rs`, `examples/gpu_dataframe_example.rs`
- `examples/gpu_window_operations_example.rs`
- `examples/gpu_matrix_example.rs`
- `examples/gpu_ml_example.rs`, `examples/gpu_stats_example.rs`
- `examples/gpu_benchmark_example.rs`

These are part of the example suite built under `--features all-safe`
(which includes everything except CUDA/WASM/distributed) as
stub/no-GPU-path binaries; the actual GPU code paths inside them are only
exercised on CUDA-capable CI/hardware, which this guide cannot verify from
a docs-only pass — treat "compiles under `all-safe`" and "the GPU path
inside actually runs correctly on real hardware" as two different claims.
