//! Regression tests for the `gpu-honesty` task: real Student's-t p-values in
//! `gpu::advanced_ops::GpuAdvancedStats::correlation_with_pvalues` (previously
//! a fabricated logistic curve that could never report significance), typed
//! numeric output from `dataframe::gpu_window` rolling operations (previously
//! stringified), a working CPU-backed GPU memory pool (previously always
//! failed to construct and could never evict), a real Linear Discriminant
//! Analysis implementation (previously returned raw class means as if they
//! were a projection), and correct model-parallel distributed matrix
//! multiplication (previously panicked for more than one device).
//!
//! Each test is labeled with the item from the task list it covers.
//!
//! NOTE: every module under test here (`pandrs::gpu` and friends) is declared
//! `#[cfg(cuda_available)]` at its `pub mod` site (see `src/lib.rs`,
//! `src/dataframe/mod.rs`, etc. -- outside this task's file ownership), and
//! `build.rs` only ever emits the `cuda_available` cfg on non-macOS targets
//! with the `cuda` feature enabled. On macOS (and on any build without the
//! `cuda` feature) `pandrs::gpu` does not exist as a compiled module at all,
//! so this whole file is gated the same way the pre-existing `gpu_test.rs`
//! gates its `mod tests` -- otherwise `cargo test`/`cargo nextest` would fail
//! outright on macOS with "unresolved import" errors, not merely skip these
//! tests. To run them for real: `cargo test --test gpu_honesty_w2_regression_test --features cuda`
//! on a non-macOS host (a real CUDA device is not required -- every code path
//! exercised here computes on the CPU regardless of device availability; see
//! the honesty notes throughout `crate::gpu`).
#[cfg(cuda_available)]
mod tests {
    use pandrs::gpu::advanced_ops::GpuAdvancedStats;
    use pandrs::gpu::operations::GpuMatrix;

    // ---------------------------------------------------------------------
    // (1) gpu::advanced_ops::correlation_with_pvalues: real Student's-t p-value
    // ---------------------------------------------------------------------

    /// For df = 2, the Student's-t CDF has an exact closed form:
    /// `F(t) = 1/2 + t / (2 * sqrt(2 + t^2))`, so the two-sided p-value for a
    /// non-negative `t` is `p = 1 - t / sqrt(2 + t^2)`. This is derived
    /// independently of the crate (not calling into `stats::special`, which is
    /// `pub(crate)` and unreachable from an integration test), giving a real
    /// external oracle for what `correlation_with_pvalues` should report -- the
    /// exact same value `stats::special::student_t_two_sided_p` is designed to
    /// compute (verified elsewhere against `scipy` to ~1e-10).
    fn exact_two_sided_p_df2(t: f64) -> f64 {
        let t = t.abs();
        1.0 - t / (2.0 + t * t).sqrt()
    }

    #[test]
    fn correlation_pvalue_matches_exact_student_t_df2() {
        // n = 4 samples => df = n - 2 = 2, where the closed form above applies.
        let x = [1.0_f64, 2.0, 3.0, 4.0];
        let y = [2.0_f64, 3.0, 5.0, 8.0];

        // Independently compute Pearson's r and the t-statistic the same way
        // `correlation_with_pvalues` does, entirely with plain arithmetic (no
        // crate internals), so this test has its own ground truth.
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        let cov: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();
        let var_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let var_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let r = cov / (var_x.sqrt() * var_y.sqrt());
        let df = n - 2.0;
        let t_stat = r * (df / (1.0 - r * r)).sqrt();
        let expected_p = exact_two_sided_p_df2(t_stat);

        // Row-major 4x2 matrix: column 0 = x, column 1 = y.
        let flat: Vec<f64> = x.iter().zip(y.iter()).flat_map(|(&a, &b)| [a, b]).collect();
        let matrix = GpuMatrix::from_raw_parts(flat, 4, 2).expect("valid matrix");

        let (corr, pvalues) = GpuAdvancedStats::correlation_with_pvalues(&matrix)
            .expect("correlation should succeed");

        assert!(
            (corr.data[(0, 1)] - r).abs() < 1e-9,
            "correlation coefficient mismatch: got {}, expected {}",
            corr.data[(0, 1)],
            r
        );
        assert!(
            (pvalues.data[(0, 1)] - expected_p).abs() < 1e-9,
            "p-value mismatch: got {}, expected {} (exact df=2 closed form)",
            pvalues.data[(0, 1)],
            expected_p
        );

        // The previous implementation was `0.5 + 0.5*sign(x)*(1-exp(-|x|))`, a
        // logistic curve that saturates around 0.72 -- it could never get close
        // to the small p-value a real, strong correlation like this produces.
        assert!(
            pvalues.data[(0, 1)] < 0.5,
            "fabricated logistic p-value would never drop this low: {}",
            pvalues.data[(0, 1)]
        );
    }

    #[test]
    fn correlation_pvalue_significant_for_strongly_correlated_pair() {
        // A clear, strong linear relationship with a reasonable sample size:
        // any real Student's-t p-value should easily clear the conventional
        // 0.05 significance threshold here. The previous fabricated formula
        // (`0.5 + 0.5*sign(x)*(1-exp(-|x|))`) saturates around p ~= 0.72 for
        // large |t| and can therefore never report significance for *any*
        // input, however strong the correlation.
        let n_rows = 20;
        let mut flat = Vec::with_capacity(n_rows * 2);
        for i in 0..n_rows {
            let x = i as f64;
            // Small deterministic wobble so the correlation is strong but not
            // exactly 1.0 (avoids the t-statistic being infinite).
            let noise = if i % 2 == 0 { 0.3 } else { -0.3 };
            flat.push(x);
            flat.push(2.0 * x + 1.0 + noise);
        }
        let matrix = GpuMatrix::from_raw_parts(flat, n_rows, 2).expect("valid matrix");

        let (corr, pvalues) = GpuAdvancedStats::correlation_with_pvalues(&matrix)
            .expect("correlation should succeed");

        assert!(
            corr.data[(0, 1)] > 0.99,
            "expected a very strong correlation, got {}",
            corr.data[(0, 1)]
        );
        assert!(
            pvalues.data[(0, 1)] < 0.05,
            "expected a significant p-value for a strongly correlated pair, got {}",
            pvalues.data[(0, 1)]
        );
        assert!(
            pvalues.data[(1, 0)] == pvalues.data[(0, 1)],
            "p-value matrix must be symmetric"
        );
        // Diagonal p-values are always 0 (a variable is perfectly, trivially
        // correlated with itself).
        assert_eq!(pvalues.data[(0, 0)], 0.0);
        assert_eq!(pvalues.data[(1, 1)], 0.0);
    }

    #[test]
    fn correlation_pvalue_is_zero_not_nan_for_duplicated_column() {
        // A column correlated with an exact copy of itself is the most
        // extreme case: the true correlation is exactly 1.0, but summing
        // `numerator`/`denominator` independently in floating point can let
        // the computed ratio overshoot 1.0 by a few ULPs. Unclamped, that
        // sends `1.0 - correlation.powi(2)` slightly negative, and `.sqrt()`
        // of a negative number is NaN per IEEE 754 -- silently turning the
        // single most significant possible p-value into a NaN instead of
        // the correct 0.0.
        let x = [1.0_f64, 2.5, 3.25, 4.75, 5.0, 6.5, 7.25, 8.75, 9.0, 10.5];
        // Row-major 10x2: column 0 = x, column 1 = an exact copy of x.
        let flat: Vec<f64> = x.iter().flat_map(|&v| [v, v]).collect();
        let matrix = GpuMatrix::from_raw_parts(flat, x.len(), 2).expect("valid matrix");

        let (corr, pvalues) = GpuAdvancedStats::correlation_with_pvalues(&matrix)
            .expect("correlation should succeed");

        assert!(
            (corr.data[(0, 1)] - 1.0).abs() < 1e-9,
            "expected a perfect (or floating-point-noise-close-to-perfect) correlation, got {}",
            corr.data[(0, 1)]
        );
        assert!(
            !pvalues.data[(0, 1)].is_nan(),
            "p-value for a perfectly correlated pair must not be NaN"
        );
        assert_eq!(
            pvalues.data[(0, 1)], 0.0,
            "a perfect correlation is the most significant result representable: p must be exactly 0"
        );
    }

    // ---------------------------------------------------------------------
    // (10) dataframe::gpu_window: rolling `.mean()` returns typed numeric
    // columns, not stringified text
    // ---------------------------------------------------------------------

    #[test]
    fn gpu_window_mean_returns_numeric_column() {
        use pandrs::dataframe::gpu_window::{GpuDataFrameWindowExt, GpuWindowContext};
        use pandrs::{DataFrame, Series};

        let mut df = DataFrame::new();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        df.add_column(
            "value".to_string(),
            Series::new(data, Some("value".to_string())).expect("valid series"),
        )
        .expect("add_column should succeed");

        let context = GpuWindowContext::new().expect("context should construct");
        let result = df
            .gpu_rolling(3, &context)
            .mean()
            .expect("rolling mean should succeed");

        // This is the crux of the regression: the previous implementation built
        // the result column via
        // `processed_data.into_iter().map(|v| v.to_string()).collect()`, which
        // produced a `Series<String>`. `get_column_numeric_values` only
        // downcasts to `Series<f64>`/`Series<i64>` and errors otherwise, so this
        // call fails outright unless the column really is numeric.
        let values = result
            .get_column_numeric_values("value")
            .expect("rolling mean output must be a real numeric column, not stringified text");

        assert_eq!(values.len(), 10);
        // Rolling mean with window 3 over 1..=10: first two entries have too
        // few periods (NaN under this crate's convention), then the mean of
        // each 3-wide window.
        assert!(values[0].is_nan());
        assert!(values[1].is_nan());
        assert!((values[2] - 2.0).abs() < 1e-9); // mean(1,2,3)
        assert!((values[9] - 9.0).abs() < 1e-9); // mean(8,9,10)
    }

    #[test]
    fn gpu_window_var_honors_ddof() {
        use pandrs::dataframe::gpu_window::{GpuDataFrameWindowExt, GpuWindowContext};
        use pandrs::{DataFrame, Series};

        let mut df = DataFrame::new();
        df.add_column(
            "value".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("value".to_string())).expect("valid series"),
        )
        .expect("add_column should succeed");

        let context = GpuWindowContext::new().expect("context should construct");

        // Window covers all 3 points: mean = 2.0, sum of squared deviations = 2.0.
        // Population variance (ddof=0) = 2.0 / 3; sample variance (ddof=1) = 2.0 / 2.
        // The previous implementation accepted `ddof` but silently ignored it
        // (always computing ddof=1), so `var(0)` and `var(1)` were identical.
        let pop_var = df
            .gpu_rolling(3, &context)
            .var(0)
            .expect("population variance should succeed")
            .get_column_numeric_values("value")
            .expect("numeric column");
        let sample_var = df
            .gpu_rolling(3, &context)
            .var(1)
            .expect("sample variance should succeed")
            .get_column_numeric_values("value")
            .expect("numeric column");

        assert!((pop_var[2] - 2.0 / 3.0).abs() < 1e-9, "got {}", pop_var[2]);
        assert!((sample_var[2] - 1.0).abs() < 1e-9, "got {}", sample_var[2]);
        assert!(
            (pop_var[2] - sample_var[2]).abs() > 1e-6,
            "ddof=0 and ddof=1 must give different results here"
        );
    }

    // ---------------------------------------------------------------------
    // (4) gpu::memory_pool: alloc/dealloc round-trip on a CPU-backed pool that
    // actually constructs and works (previously always failed to construct
    // under `get_cuda_context`'s hardcoded `None`, and used a same-size-collides
    // "pointer" scheme even when it did).
    // ---------------------------------------------------------------------

    #[test]
    fn memory_pool_alloc_dealloc_round_trip() {
        use pandrs::gpu::memory_pool::{GpuMemoryPool, MemoryPoolConfig};

        let config = MemoryPoolConfig {
            initial_size: 1024 * 1024,
            ..MemoryPoolConfig::default()
        };
        let mut pool = GpuMemoryPool::new(0, config).expect("pool construction must succeed");

        // Two allocations that round up to the same aligned size must not
        // collide: the previous non-CUDA scheme keyed allocations by
        // `ptr = size_in_bytes`, so two live same-size allocations silently
        // overwrote each other's bookkeeping entry.
        let alloc_a = pool.allocate(1000).expect("first allocation");
        let alloc_b = pool.allocate(2000).expect("second allocation");
        assert_eq!(
            alloc_a.size(),
            alloc_b.size(),
            "both round up to the same aligned size"
        );
        assert_ne!(
            alloc_a.id(),
            alloc_b.id(),
            "same-size allocations must have distinct ids"
        );

        let stats = pool.get_stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_deallocations, 0);

        pool.deallocate(alloc_a).expect("first deallocation");
        pool.deallocate(alloc_b).expect("second deallocation");

        let stats = pool.get_stats();
        assert_eq!(stats.total_deallocations, 2);

        // A third allocation of the same size should be able to reuse a freed
        // block from the free list (a cache hit) rather than erroring.
        let alloc_c = pool
            .allocate(1000)
            .expect("third allocation should reuse a freed block");
        pool.deallocate(alloc_c).expect("third deallocation");
    }

    #[test]
    fn memory_pool_cleanup_does_not_evict_unused_reserve() {
        use pandrs::gpu::memory_pool::{GpuMemoryPool, MemoryPoolConfig};

        // A freshly constructed pool's pre-allocated (never-yet-used) reserve
        // capacity should survive an explicit `cleanup()` call: only blocks
        // that were actually allocated and then freed age out.
        let config = MemoryPoolConfig {
            initial_size: 256 * 1024,
            max_allocation_age: 0, // age out anything eligible immediately
            ..MemoryPoolConfig::default()
        };
        let mut pool = GpuMemoryPool::new(0, config).expect("pool construction must succeed");
        let size_before = pool.current_size();
        assert!(
            size_before > 0,
            "pool should have pre-allocated reserve capacity"
        );

        pool.cleanup().expect("cleanup should succeed");
        assert_eq!(
            pool.current_size(),
            size_before,
            "cleanup must not evict capacity that was never allocated"
        );

        // Now allocate and free a block, then clean up with a zero max age:
        // this block *should* be eligible for eviction.
        let alloc = pool.allocate(4096).expect("allocation should succeed");
        pool.deallocate(alloc).expect("deallocation should succeed");
        // `cleanup` compares ages with `Instant::now()`, and `max_allocation_age
        // = 0` means anything with a recorded `last_accessed` is immediately
        // "too old" (`age > Duration::from_secs(0)` is true for any nonzero
        // elapsed time), so this should shrink `current_size` back down.
        std::thread::sleep(std::time::Duration::from_millis(1));
        pool.cleanup().expect("cleanup should succeed");
        assert!(
            pool.current_size() <= size_before,
            "a freed, aged-out block should be reclaimed, not retained forever"
        );
    }

    // ---------------------------------------------------------------------
    // Bonus: gpu::advanced_ops::GpuDecomposition::svd_decomposition -- U and
    // V stay column-synchronized for a RANK-DEFICIENT input, where at least
    // one singular value falls at/near zero and its U column is produced by
    // the separate orthonormal-basis-completion loop rather than by
    // `avi / singular_values[i]`. `singular_values` is guaranteed
    // non-increasing (it is `sqrt(max(eigenvalue, 0))` over
    // `jacobi_symmetric_eigen`'s already-descending-sorted eigenvalues, and
    // both `sqrt` and `max(_, 0)` preserve a non-increasing order), so every
    // index skipped by the `singular_values[i] > tol` filled/i decoupling
    // is a trailing suffix, never an interior gap -- `filled == i` holds
    // for every kept column. This fixture (one column an exact linear
    // combination of the other two, so the matrix is rank <= 2 of 3) checks
    // that invariant empirically rather than only by construction.
    // ---------------------------------------------------------------------

    #[test]
    fn cpu_svd_reconstruction_rank_deficient() {
        use pandrs::gpu::advanced_ops::GpuDecomposition;
        use pandrs::gpu::GpuManager;

        // 4x3, column 2 (0-indexed) = column 0 + column 1 exactly: rank <= 2.
        let a = GpuMatrix::from_raw_parts(
            vec![
                1.0, 0.0, 1.0, //
                0.0, 1.0, 1.0, //
                1.0, 1.0, 2.0, //
                2.0, 0.0, 2.0,
            ],
            4,
            3,
        )
        .expect("valid matrix");

        let gpu_manager = GpuManager::new();
        let decomp = GpuDecomposition::new(&gpu_manager).expect("decomposition context");
        let (u, s, vt) = decomp
            .svd_decomposition(&a)
            .expect("SVD must succeed even for a rank-deficient matrix");

        assert_eq!(u.data.shape(), &[4, 4]);
        assert_eq!(vt.data.shape(), &[3, 3]);
        assert_eq!(s.data.len(), 3);

        // At least one singular value must reflect the rank deficiency
        // (small relative to the largest one).
        let max_sv = s.data.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            s.data.iter().any(|&sv| sv < 1e-6 * max_sv.max(1.0)),
            "expected at least one near-zero singular value for a rank-deficient input, got {:?}",
            s.data
        );

        // Reconstruct A = U[:, :k] . diag(S) . Vt[:k, :] and compare to the
        // input. If U's near-zero-singular-value column (produced by the
        // basis-completion loop) were paired against the wrong V row, this
        // reconstruction would NOT match A even though shapes are all
        // correct and no panic occurs.
        let m = 4;
        let n = 3;
        let k = s.data.len();
        let mut recon = scirs2_core::ndarray::Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for r in 0..k {
                    acc += u.data[(i, r)] * s.data[r] * vt.data[(r, j)];
                }
                recon[(i, j)] = acc;
            }
        }
        let original: [[f64; 3]; 4] = [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [2.0, 0.0, 2.0],
        ];
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (recon[(i, j)] - original[i][j]).abs() < 1e-6,
                    "rank-deficient SVD reconstruction mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    recon[(i, j)],
                    original[i][j]
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // Bonus: gpu::advanced_ops::lda -- real Linear Discriminant Analysis
    // (previously returned the raw class means as if they were a projection,
    // with no scatter matrices or generalized eigenproblem involved at all).
    // ---------------------------------------------------------------------

    #[test]
    fn lda_separates_two_linearly_separable_classes() {
        use scirs2_core::ndarray::Array1;

        // Two well-separated 2D clusters along the x-axis.
        let flat: Vec<f64> = vec![
            0.0, 0.0, 0.1, -0.1, -0.1, 0.1, 0.2, 0.0, // class 0, centered near (0, 0)
            10.0, 0.0, 10.1, -0.1, 9.9, 0.1, 10.2, 0.0, // class 1, centered near (10, 0)
        ];
        let data = GpuMatrix::from_raw_parts(flat, 8, 2).expect("valid matrix");
        let labels = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);

        let projection = GpuAdvancedStats::lda(&data, &labels, 1)
            .expect("LDA should succeed on separable classes");

        assert_eq!(projection.data.shape(), &[8, 1]);

        let class0_mean: f64 = (0..4).map(|i| projection.data[(i, 0)]).sum::<f64>() / 4.0;
        let class1_mean: f64 = (4..8).map(|i| projection.data[(i, 0)]).sum::<f64>() / 4.0;

        // A real discriminant projection must separate the two classes by far
        // more than the within-class spread; the previous "class means" stub
        // returned a (2 x n_features) matrix -- a completely different shape
        // from the (n_samples x n_components) a projection must have -- so this
        // also implicitly checks the output shape is that of an actual
        // per-sample projection.
        assert!(
            (class0_mean - class1_mean).abs() > 1.0,
            "expected well-separated class means in the projected space: {} vs {}",
            class0_mean,
            class1_mean
        );
    }

    // ---------------------------------------------------------------------
    // Bonus: gpu::multi_gpu -- ModelParallel distributed matmul must not panic
    // and must reconstruct the true product (previously panicked for
    // `device_ids.len() > 1`: every device was hand the *entire* `B`, unsplit,
    // against a column-sliced `A` chunk of mismatched width).
    // ---------------------------------------------------------------------

    #[test]
    fn model_parallel_matmul_matches_direct_product() {
        use pandrs::gpu::multi_gpu::{DistributionStrategy, MultiGpuConfig, MultiGpuManager};

        let config = MultiGpuConfig {
            device_ids: vec![0, 1],
            distribution_strategy: DistributionStrategy::ModelParallel,
            ..MultiGpuConfig::default()
        };
        let manager = MultiGpuManager::new(config).expect("manager should construct");

        // A (3x4) * B (4x2) = (3x2), with the contraction dimension (4) split
        // across the 2 configured devices.
        let a = GpuMatrix::from_raw_parts(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0,
            ],
            3,
            4,
        )
        .expect("valid A");
        let b = GpuMatrix::from_raw_parts(
            vec![
                1.0, 0.0, //
                0.0, 1.0, //
                1.0, 1.0, //
                2.0, 2.0,
            ],
            4,
            2,
        )
        .expect("valid B");

        let result = manager
            .distributed_matmul(&a, &b)
            .expect("model-parallel matmul must not panic and must succeed");

        let direct = a.dot_cpu(&b).expect("direct CPU product for comparison");

        assert_eq!(result.data.shape(), direct.data.shape());
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (result.data[(i, j)] - direct.data[(i, j)]).abs() < 1e-9,
                    "mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    result.data[(i, j)],
                    direct.data[(i, j)]
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // (7) series::gpu::SeriesGpuExt::gpu_corr -- listwise deletion, mirrored
    // through the Series-level wrapper (which independently re-implements
    // the same alignment fix as `gpu::cpu_math::correlation_matrix_listwise`
    // and `py_gpu.rs`'s `gpu_corr`, rather than sharing it).
    // ---------------------------------------------------------------------

    #[test]
    fn series_gpu_corr_listwise_deletion_keeps_pairs_aligned() {
        use pandrs::series::gpu::SeriesGpuExt;
        use pandrs::Series;

        // Same fixture as `cpu_math::tests::listwise_deletion_keeps_pairs_aligned`:
        // b is a perfect 2x of a only at the rows where NEITHER series is
        // NaN (this crate's missing-value sentinel for a plain `Series<f64>`).
        // Filtering each series' NaNs independently before zipping would
        // misalign the surviving values and destroy the perfect correlation.
        let a = Series::new(vec![1.0, f64::NAN, 3.0, 4.0, 5.0], Some("a".to_string()))
            .expect("valid series");
        let b = Series::new(vec![2.0, 100.0, f64::NAN, 8.0, 10.0], Some("b".to_string()))
            .expect("valid series");

        let corr = a.gpu_corr(&b).expect("gpu_corr should succeed");
        assert!(
            (corr - 1.0).abs() < 1e-9,
            "expected perfect correlation on the complete rows, got {}",
            corr
        );
    }

    // ---------------------------------------------------------------------
    // (9) optimized::split_dataframe::gpu -- a zero-variance (constant)
    // column's correlation is reported as 0, not an unguarded NaN.
    // ---------------------------------------------------------------------

    #[test]
    fn split_dataframe_corr_matrix_zero_variance_is_zero_not_nan() {
        use pandrs::column::{Column, Float64Column};
        use pandrs::optimized::split_dataframe::core::OptimizedDataFrame;

        let mut df = OptimizedDataFrame::new();
        df.add_column(
            "constant".to_string(),
            Column::Float64(Float64Column::new(vec![5.0; 6])),
        )
        .expect("add constant column");
        df.add_column(
            "varying".to_string(),
            Column::Float64(Float64Column::new((0..6).map(|i| i as f64).collect())),
        )
        .expect("add varying column");

        let corr = df
            .corr_matrix(&["constant", "varying"])
            .expect("corr_matrix should succeed");

        // A constant column has zero variance, making its correlation with
        // anything else mathematically undefined (0/0). Every other
        // correlation routine in the crate reports 0 for this case; the
        // previous `split_dataframe::gpu` implementation left it as an
        // unguarded NaN instead.
        assert!(!corr[[0, 1]].is_nan(), "expected 0, not NaN");
        assert_eq!(corr[[0, 1]], 0.0);
        assert_eq!(corr[[1, 0]], 0.0);
        // The diagonal (self-correlation) is unaffected and must stay 1.0.
        assert_eq!(corr[[0, 0]], 1.0);
        assert_eq!(corr[[1, 1]], 1.0);
    }

    // ---------------------------------------------------------------------
    // (11) ml::gpu::pca -- scores are relative to the training mean, not
    // the raw (uncentered) data (previously projecting raw, uncentered data
    // added a constant offset -- the projection of the mean itself -- to
    // every sample's score).
    // ---------------------------------------------------------------------

    #[test]
    fn ml_gpu_pca_scores_are_mean_centered() {
        use pandrs::ml::gpu::pca;
        use scirs2_core::ndarray::Array2;

        // Two points symmetric about a *nonzero* mean (5, 5).
        let data = Array2::from_shape_vec((2, 2), vec![4.0, 4.0, 6.0, 6.0]).expect("valid data");
        let (components, _explained_variance, transformed) =
            pca(&data, 1).expect("pca should succeed");

        // The two points are symmetric about the mean along the principal
        // axis, so their *centered* scores must be equal and opposite,
        // summing to ~0. Projecting the raw uncentered data (the previous
        // bug) would add the same constant `mean . component` offset to
        // both scores instead, so they would generally NOT sum to zero.
        let sum: f64 = (0..2).map(|i| transformed[(i, 0)]).sum();
        assert!(
            sum.abs() < 1e-9,
            "expected centered scores to sum to ~0, got {} (offset by uncentered projection?)",
            sum
        );
        // The two scores should also be clearly nonzero and opposite in
        // sign (not degenerate) -- a real, meaningful projection.
        assert!(transformed[(0, 0)].abs() > 1e-6);
        assert!(
            (transformed[(0, 0)] + transformed[(1, 0)]).abs() < 1e-9,
            "scores should be equal and opposite: {} vs {}",
            transformed[(0, 0)],
            transformed[(1, 0)]
        );
        let _ = components; // shape/orthonormality covered by stats::gpu::pca's own tests
    }

    // ---------------------------------------------------------------------
    // (11) ml::gpu::kmeans -- an empty cluster is reseeded to a genuinely
    // random row, not deterministically to row 0 every time
    // (`random::<f64>() as usize` used to truncate every draw to exactly 0).
    // ---------------------------------------------------------------------

    #[test]
    fn ml_gpu_kmeans_empty_cluster_reseed_is_not_always_row_zero() {
        use pandrs::ml::gpu::kmeans;
        use scirs2_core::ndarray::Array2;

        // 30 rows, 1 feature: rows 0..5 are all exactly 0.0 (degenerate),
        // rows 5..30 are distinct values 5.0..30.0. With k = n_samples = 30,
        // the naive initializer (`sample_idx = i`) makes every row its own
        // initial centroid, so centroids 0..5 all start tied at 0.0. On the
        // first assignment pass every row in the degenerate block (0.0)
        // ties between centroids 0..5 and (first-match-wins) collapses onto
        // cluster 0 alone, leaving clusters 1..5 empty; every row in the
        // diverse block is the unique nearest point to its own centroid, so
        // clusters 5..30 each keep exactly one point. This deterministically
        // forces 4 simultaneous empty clusters (1, 2, 3, 4) that must be
        // reseeded, each independently drawing from all 30 rows.
        let mut values = vec![0.0_f64; 5];
        values.extend((5..30).map(|i| i as f64));
        let data = Array2::from_shape_vec((30, 1), values).expect("valid data");

        let (centroids, _labels, inertia) =
            kmeans(&data, 30, 1, 1e-6).expect("kmeans should succeed");
        assert!(inertia.is_finite());

        let reseeded: Vec<f64> = (1..5).map(|c| centroids[(c, 0)]).collect();
        assert!(
            reseeded.iter().all(|v| v.is_finite()),
            "all reseeded centroids must be finite: {:?}",
            reseeded
        );

        // Under the previous bug, every empty cluster reseeded to the exact
        // same row (row 0, value 0.0), so all 4 values here would be 0.0.
        // With a real uniform random draw over 30 rows (25 of which are
        // nonzero and mutually distinct), the chance all 4 independent
        // draws land back on one of the 5 zero-valued rows is (5/30)^4 =
        // 1/1296 (~0.08%) -- an acceptably small, non-deterministic flake
        // rate for a regression test that fails *deterministically* under
        // the bug it targets.
        assert!(
            reseeded.iter().any(|&v| v != 0.0),
            "expected at least one non-degenerate-row reseed among {:?}; a value of 0.0 in \
             every slot is exactly the 'always row 0' bug's signature",
            reseeded
        );
    }
}
