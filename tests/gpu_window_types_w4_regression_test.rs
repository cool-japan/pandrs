//! Regression tests for the `gpu-window-types` Wave-4 task (WAVE4_BACKLOG.md
//! item A7): `gpu_rolling(window_size, &context).{mean,sum,min,max,std,var}()`
//! must return NUMERIC result columns on the CPU-fallback path, not
//! stringified text.
//!
//! ## Historical bug
//! `GpuDataFrameRolling::execute_rolling_operation` (`src/dataframe/gpu_window.rs`)
//! used to build the result column via
//! `processed_data.into_iter().map(|v| v.to_string()).collect()`, turning
//! every numeric window result -- including `.mean()`, whose entire job is
//! to produce a number -- into a `Series<String>`. That silently broke any
//! further numeric use (arithmetic, plotting, more window ops) of the
//! result, and made `get_column::<f64>(..)` fail on the very column the
//! operation was supposed to compute. Fixed (commit `5293024`) to build the
//! result directly from the already-`Vec<f64>` `processed_data` instead of
//! stringifying it; `f64::NAN` remains the warmup/insufficient-data
//! sentinel, matching this crate's documented convention for a plain
//! (non-NA-wrapped) `Series<f64>` (see the per-type comment in
//! `src/series/base.rs`).
//!
//! ## Relationship to existing coverage
//! `tests/gpu_honesty_w2_regression_test.rs` already regression-tests this
//! same bug class for `.mean()` (`gpu_window_mean_returns_numeric_column`)
//! and `.var()`'s `ddof` handling (`gpu_window_var_honors_ddof`). This file
//! re-covers `.mean()` directly (it is this task's explicitly named
//! deliverable) and adds the operations that file does not touch --
//! `.sum()`, `.min()`, `.max()`, `.std()` -- plus one consolidated
//! type-only check across all six ops as defense-in-depth against a future
//! regression reintroducing stringification at the single shared
//! `execute_rolling_operation` result-construction site.
//!
//! ## Why this whole file is gated on `cuda_available`
//! `pandrs::dataframe::gpu_window` is declared `#[cfg(cuda_available)]` at
//! its `pub mod gpu_window;` site (`src/dataframe/mod.rs:110-111`), and its
//! `GpuWindowContext` struct holds a `crate::gpu::GpuManager` field --
//! `crate::gpu` is itself `#[cfg(cuda_available)]`-gated (`src/lib.rs:451`).
//! `build.rs` only ever emits `cuda_available` for a non-macOS target with
//! the `cuda` Cargo feature enabled (never for `all-safe`, which does not
//! include `cuda`, and never on macOS regardless of feature flags), so
//! `pandrs::dataframe::gpu_window` cannot be named at all in an `all-safe`
//! (or any macOS) build -- confirmed empirically: referencing it under
//! `--features all-safe` fails with
//! `error[E0432]: unresolved import ... found an item that was configured out`.
//! This file must therefore be gated the same way `tests/gpu_test.rs` and
//! `tests/gpu_honesty_w2_regression_test.rs` already gate their GPU-window
//! coverage, or `cargo build --features all-safe` would fail outright
//! instead of merely skipping these tests. Both `src/dataframe/mod.rs` and
//! `src/lib.rs` are outside this task's file ownership (`src/dataframe/gpu_window.rs`
//! only), so that gate could not be (and is not) changed here; it is
//! reported separately as a cross-cutting finding.
//!
//! Run for real on a non-macOS host with the CUDA toolkit installed:
//! ```text
//! cargo test --test gpu_window_types_w4_regression_test --features cuda
//! ```
//! No actual GPU device is required -- every code path exercised here
//! computes on the CPU regardless of device availability (window_size = 3
//! over 10 elements is far below `GpuWindowContext`'s default 50,000-element
//! GPU-dispatch threshold, so every op below runs through
//! `GpuDataFrameRolling::fallback_to_jit_operation` -> `cpu_rolling_*`, the
//! CPU-fallback path this task targets); see this module's own honesty note
//! in `src/dataframe/gpu_window.rs`.
#[cfg(cuda_available)]
mod tests {
    use pandrs::dataframe::gpu_window::{GpuDataFrameWindowExt, GpuWindowContext};
    use pandrs::{DataFrame, Series};

    /// 1.0..=10.0, one numeric column named "value".
    fn test_df() -> DataFrame {
        let mut df = DataFrame::new();
        let data: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        df.add_column(
            "value".to_string(),
            Series::new(data, Some("value".to_string())).expect("valid series"),
        )
        .expect("add_column should succeed");
        df
    }

    /// Asserts `column_name` in `df` downcasts to a real `Series<f64>` and
    /// is specifically NOT reachable as a `Series<String>` -- the two halves
    /// of "numeric column, not stringified text".
    fn assert_numeric_not_string(df: &DataFrame, column_name: &str, op: &str) {
        assert!(
            df.get_column::<f64>(column_name).is_ok(),
            "{op}: result column must downcast to Series<f64>"
        );
        assert!(
            df.get_column::<String>(column_name).is_err(),
            "{op}: result column must NOT be reachable as Series<String>"
        );
    }

    // -------------------------------------------------------------------
    // This task's explicitly named deliverable: gpu_rolling(window).mean()
    // -------------------------------------------------------------------

    #[test]
    fn gpu_rolling_mean_returns_numeric_column_with_na_at_warmup() {
        let df = test_df();
        let context = GpuWindowContext::new().expect("context should construct");

        let result = df
            .gpu_rolling(3, &context)
            .mean()
            .expect("gpu_rolling(..).mean() should succeed");
        assert_numeric_not_string(&result, "value", "mean");

        let values = result.get_column::<f64>("value").unwrap().values();
        assert_eq!(values.len(), 10);

        // window_size = 3, min_periods defaults to window_size: the first
        // `window_size - 1 = 2` positions have too few observations and
        // must be NA (f64::NAN, this crate's sentinel), never a silent 0.0.
        assert!(values[0].is_nan(), "expected NA at warmup position 0");
        assert!(values[1].is_nan(), "expected NA at warmup position 1");

        let expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        for (offset, &expected_mean) in expected.iter().enumerate() {
            let i = offset + 2;
            assert!(
                (values[i] - expected_mean).abs() < 1e-9,
                "position {i}: expected {expected_mean}, got {}",
                values[i]
            );
        }
    }

    // -------------------------------------------------------------------
    // New coverage: .sum(), .min()/.max(), .std() -- not exercised by
    // gpu_honesty_w2_regression_test.rs.
    // -------------------------------------------------------------------

    #[test]
    fn gpu_rolling_sum_returns_numeric_column_with_na_at_warmup() {
        let df = test_df();
        let context = GpuWindowContext::new().expect("context should construct");

        let result = df
            .gpu_rolling(3, &context)
            .sum()
            .expect("gpu_rolling(..).sum() should succeed");
        assert_numeric_not_string(&result, "value", "sum");

        let values = result.get_column::<f64>("value").unwrap().values();
        assert_eq!(values.len(), 10);
        assert!(values[0].is_nan(), "expected NA at warmup position 0");
        assert!(values[1].is_nan(), "expected NA at warmup position 1");

        let expected = [6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0];
        for (offset, &expected_sum) in expected.iter().enumerate() {
            let i = offset + 2;
            assert!(
                (values[i] - expected_sum).abs() < 1e-9,
                "position {i}: expected {expected_sum}, got {}",
                values[i]
            );
        }
    }

    #[test]
    fn gpu_rolling_min_and_max_return_numeric_columns() {
        let df = test_df();
        let context = GpuWindowContext::new().expect("context should construct");

        let min_result = df
            .gpu_rolling(3, &context)
            .min()
            .expect("gpu_rolling(..).min() should succeed");
        assert_numeric_not_string(&min_result, "value", "min");
        let min_values = min_result.get_column::<f64>("value").unwrap().values();

        let max_result = df
            .gpu_rolling(3, &context)
            .max()
            .expect("gpu_rolling(..).max() should succeed");
        assert_numeric_not_string(&max_result, "value", "max");
        let max_values = max_result.get_column::<f64>("value").unwrap().values();

        assert!(
            min_values[0].is_nan(),
            "min: expected NA at warmup position 0"
        );
        assert!(
            min_values[1].is_nan(),
            "min: expected NA at warmup position 1"
        );
        assert!(
            max_values[0].is_nan(),
            "max: expected NA at warmup position 0"
        );
        assert!(
            max_values[1].is_nan(),
            "max: expected NA at warmup position 1"
        );

        // Strictly increasing input: the min of each 3-wide window is its
        // leftmost element, the max is its rightmost.
        let expected_min = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected_max = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        for (offset, (&emin, &emax)) in expected_min.iter().zip(expected_max.iter()).enumerate() {
            let i = offset + 2;
            assert!(
                (min_values[i] - emin).abs() < 1e-9,
                "min[{i}]: expected {emin}, got {}",
                min_values[i]
            );
            assert!(
                (max_values[i] - emax).abs() < 1e-9,
                "max[{i}]: expected {emax}, got {}",
                max_values[i]
            );
        }
    }

    #[test]
    fn gpu_rolling_std_returns_numeric_column() {
        let df = test_df();
        let context = GpuWindowContext::new().expect("context should construct");

        // ddof = 1 (sample std). Every 3-wide window over a run of
        // consecutive integers has deviations {-1, 0, 1} from its own mean,
        // so sample variance = 2 / (3 - 1) = 1.0 and std = 1.0 at every
        // valid position -- a convenient invariant for this fixture.
        let result = df
            .gpu_rolling(3, &context)
            .std(1)
            .expect("gpu_rolling(..).std(1) should succeed");
        assert_numeric_not_string(&result, "value", "std");

        let values = result.get_column::<f64>("value").unwrap().values();
        assert!(values[0].is_nan(), "expected NA at warmup position 0");
        assert!(values[1].is_nan(), "expected NA at warmup position 1");
        for (i, &v) in values.iter().enumerate().skip(2) {
            assert!((v - 1.0).abs() < 1e-9, "std[{i}]: expected 1.0, got {v}");
        }
    }

    // -------------------------------------------------------------------
    // Defense-in-depth: all six chains (including .mean()/.var(), already
    // covered in detail above and in gpu_honesty_w2_regression_test.rs)
    // share the single `execute_rolling_operation` result-construction
    // site -- one consolidated type check across all of them guards
    // against a future regression reintroducing stringification for a
    // subset of ops that a per-op test elsewhere might not happen to name.
    // -------------------------------------------------------------------

    #[test]
    fn gpu_rolling_all_ops_produce_numeric_columns() {
        let df = test_df();
        let context = GpuWindowContext::new().expect("context should construct");

        assert_numeric_not_string(
            &df.gpu_rolling(3, &context).mean().expect("mean"),
            "value",
            "mean",
        );
        assert_numeric_not_string(
            &df.gpu_rolling(3, &context).sum().expect("sum"),
            "value",
            "sum",
        );
        assert_numeric_not_string(
            &df.gpu_rolling(3, &context).std(1).expect("std"),
            "value",
            "std",
        );
        assert_numeric_not_string(
            &df.gpu_rolling(3, &context).var(1).expect("var"),
            "value",
            "var",
        );
        assert_numeric_not_string(
            &df.gpu_rolling(3, &context).min().expect("min"),
            "value",
            "min",
        );
        assert_numeric_not_string(
            &df.gpu_rolling(3, &context).max().expect("max"),
            "value",
            "max",
        );
    }
}
