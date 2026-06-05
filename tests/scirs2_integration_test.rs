#![allow(clippy::result_large_err)]
//! Integration tests for SciRS2 ↔ PandRS bridge.
//!
//! All tests are gated behind the `scirs2` feature flag. Run with:
//! ```bash
//! cargo test --test scirs2_integration_test --features scirs2
//! ```

#[cfg(feature = "scirs2")]
mod tests {
    use pandrs::scirs2_integration::conversion::{
        array1_to_series, array2_to_dataframe, dataframe_to_array2, series_to_array1,
    };
    use pandrs::scirs2_integration::dataframe_ext::SciRS2Ext;
    use pandrs::scirs2_integration::linalg::SciRS2LinAlg;
    use pandrs::scirs2_integration::stats::SciRS2Stats;
    use pandrs::{DataFrame, Series};

    // -----------------------------------------------------------------------
    // Conversion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_series_to_array1() {
        let series =
            Series::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0], Some("vals".to_string())).expect("ok");
        let arr = series_to_array1(&series).expect("series_to_array1 ok");
        assert_eq!(arr.len(), 5);
        assert!((arr[0] - 1.0).abs() < 1e-12);
        assert!((arr[4] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_array1_to_series() {
        use scirs2_core::ndarray::array;
        let arr = array![10.0f64, 20.0, 30.0];
        let series = array1_to_series(&arr, Some("result".to_string())).expect("ok");
        assert_eq!(series.len(), 3);
        assert_eq!(series.name(), Some(&"result".to_string()));
        let values = series.values();
        assert!((values[0] - 10.0).abs() < 1e-12);
        assert!((values[2] - 30.0).abs() < 1e-12);
    }

    #[test]
    fn test_series_roundtrip() {
        let original = vec![1.5f64, 2.5, 3.5, 4.5];
        let series = Series::new(original.clone(), Some("data".to_string())).expect("ok");
        let arr = series_to_array1(&series).expect("to array ok");
        let series2 = array1_to_series(&arr, Some("data".to_string())).expect("to series ok");
        for (a, b) in original.iter().zip(series2.values().iter()) {
            assert!((a - b).abs() < 1e-12, "roundtrip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_dataframe_to_array2() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(vec![1.0f64, 4.0], Some("x".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "y".to_string(),
            Series::new(vec![2.0f64, 5.0], Some("y".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "z".to_string(),
            Series::new(vec![3.0f64, 6.0], Some("z".to_string())).expect("ok"),
        )
        .expect("ok");

        let arr = dataframe_to_array2(&df, &["x", "y", "z"]).expect("to array2 ok");
        assert_eq!(arr.shape(), &[2, 3]);
        assert!((arr[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((arr[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((arr[[0, 2]] - 3.0).abs() < 1e-12);
        assert!((arr[[1, 0]] - 4.0).abs() < 1e-12);
        assert!((arr[[1, 2]] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_array2_to_dataframe() {
        use scirs2_core::ndarray::array;
        let arr = array![[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let df =
            array2_to_dataframe(&arr, vec!["a".to_string(), "b".to_string()]).expect("to df ok");
        assert_eq!(df.row_count(), 3);
        assert_eq!(df.column_count(), 2);

        let a_vals = df.get_column_numeric_values("a").expect("a ok");
        assert!((a_vals[0] - 1.0).abs() < 1e-12);
        assert!((a_vals[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_dataframe_array2_roundtrip() {
        let mut df = DataFrame::new();
        df.add_column(
            "p".to_string(),
            Series::new(vec![10.0f64, 20.0, 30.0], Some("p".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "q".to_string(),
            Series::new(vec![40.0f64, 50.0, 60.0], Some("q".to_string())).expect("ok"),
        )
        .expect("ok");

        let arr = dataframe_to_array2(&df, &["p", "q"]).expect("ok");
        let df2 = array2_to_dataframe(&arr, vec!["p".to_string(), "q".to_string()]).expect("ok");

        let p_orig = df.get_column_numeric_values("p").expect("ok");
        let p_rt = df2.get_column_numeric_values("p").expect("ok");
        for (a, b) in p_orig.iter().zip(p_rt.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    // -----------------------------------------------------------------------
    // SciRS2Ext trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_ndarray_ext() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0f64, 2.0, 3.0], Some("a".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "b".to_string(),
            Series::new(vec![4.0f64, 5.0, 6.0], Some("b".to_string())).expect("ok"),
        )
        .expect("ok");

        let arr = df.to_ndarray(&["a", "b"]).expect("ext ok");
        assert_eq!(arr.shape(), &[3, 2]);
        assert!((arr[[1, 1]] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_from_ndarray_ext() {
        use scirs2_core::ndarray::array;
        let arr = array![[1.0f64, 2.0], [3.0, 4.0]];
        let df = DataFrame::from_ndarray(&arr, vec!["x".to_string(), "y".to_string()]).expect("ok");
        assert_eq!(df.row_count(), 2);
        assert_eq!(df.column_count(), 2);
    }

    // -----------------------------------------------------------------------
    // SciRS2Stats tests
    // -----------------------------------------------------------------------

    fn make_numeric_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "b".to_string(),
            Series::new(vec![2.0f64, 4.0, 6.0, 8.0, 10.0], Some("b".to_string())).expect("ok"),
        )
        .expect("ok");
        df
    }

    #[test]
    fn test_describe() {
        let df = make_numeric_df();
        let desc = SciRS2Stats::describe(&df, &["a", "b"]).expect("describe ok");
        assert!(desc.contains_column("statistic"));
        assert!(desc.contains_column("a"));
        assert!(desc.contains_column("b"));
        assert_eq!(desc.row_count(), 8); // count, mean, std, min, 25%, 50%, 75%, max

        let mean_vals = desc.get_column_numeric_values("a").expect("ok");
        // row index 1 = mean; mean of [1,2,3,4,5] = 3.0
        assert!(
            (mean_vals[1] - 3.0).abs() < 1e-6,
            "mean mismatch: {}",
            mean_vals[1]
        );
    }

    #[test]
    fn test_scirs2_describe_ext() {
        let df = make_numeric_df();
        let desc = df.scirs2_describe().expect("ext describe ok");
        assert!(desc.contains_column("a"));
        assert!(desc.contains_column("b"));
    }

    #[test]
    fn test_correlation_matrix() {
        let df = make_numeric_df();
        let corr = SciRS2Stats::correlation_matrix(&df, &["a", "b"]).expect("corr ok");

        // Should have "column", "a", "b" columns
        assert!(corr.contains_column("column"));
        assert!(corr.contains_column("a"));
        assert!(corr.contains_column("b"));
        assert_eq!(corr.row_count(), 2);

        // Diagonal should be ~1.0 (correlation of a variable with itself)
        let a_col = corr.get_column_numeric_values("a").expect("ok");
        assert!(
            (a_col[0] - 1.0).abs() < 1e-6,
            "diagonal corr(a,a) should be 1.0, got {}",
            a_col[0]
        );

        // corr(a, b) should be 1.0 since b = 2*a (perfect linear relationship)
        let b_col = corr.get_column_numeric_values("b").expect("ok");
        assert!(
            (b_col[0] - 1.0).abs() < 1e-6,
            "corr(a,b) should be 1.0, got {}",
            b_col[0]
        );
    }

    #[test]
    fn test_scirs2_corr_ext() {
        let df = make_numeric_df();
        let corr = df.scirs2_corr().expect("ext corr ok");
        assert!(corr.contains_column("column"));
        assert_eq!(corr.row_count(), 2);
    }

    #[test]
    fn test_ttest_1samp() {
        let data = vec![5.0f64, 5.1, 4.9, 5.2, 4.8, 5.0, 5.05, 4.95];
        let result = SciRS2Stats::ttest_1samp(&data, 5.0).expect("ttest ok");
        // With mean ≈ 5.0 and H0: μ=5.0, p-value should be high (fail to reject)
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.df > 0.0);
    }

    #[test]
    fn test_ttest_ind() {
        let group1 = vec![10.0f64, 11.0, 12.0, 13.0, 14.0];
        let group2 = vec![5.0f64, 6.0, 7.0, 8.0, 9.0];
        let result = SciRS2Stats::ttest_ind(&group1, &group2).expect("ttest_ind ok");
        // Groups are clearly different; t-statistic should be large, p-value small
        assert!(result.statistic.abs() > 1.0);
        assert!(
            result.p_value < 0.05,
            "p-value should be significant: {}",
            result.p_value
        );
    }

    #[test]
    fn test_f_oneway() {
        let g1: &[f64] = &[1.0, 2.0, 3.0];
        let g2: &[f64] = &[4.0, 5.0, 6.0];
        let g3: &[f64] = &[7.0, 8.0, 9.0];
        let result = SciRS2Stats::f_oneway(&[g1, g2, g3]).expect("anova ok");
        // Groups have different means; F-statistic should be significant
        assert!(result.f_statistic > 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    // -----------------------------------------------------------------------
    // SciRS2LinAlg tests
    // -----------------------------------------------------------------------

    fn make_2x2_df(a00: f64, a01: f64, a10: f64, a11: f64) -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "c0".to_string(),
            Series::new(vec![a00, a10], Some("c0".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "c1".to_string(),
            Series::new(vec![a01, a11], Some("c1".to_string())).expect("ok"),
        )
        .expect("ok");
        df
    }

    #[test]
    fn test_matmul() {
        // Identity × identity = identity
        let eye = make_2x2_df(1.0, 0.0, 0.0, 1.0);
        let result = SciRS2LinAlg::matmul(&eye, &eye).expect("matmul ok");
        let c0 = result.get_column_numeric_values("c0").expect("c0 ok");
        let c1 = result.get_column_numeric_values("c1").expect("c1 ok");
        assert!((c0[0] - 1.0).abs() < 1e-10);
        assert!((c0[1] - 0.0).abs() < 1e-10);
        assert!((c1[0] - 0.0).abs() < 1e-10);
        assert!((c1[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_det() {
        // [[2, 1], [1, 2]] has det = 3
        let df = make_2x2_df(2.0, 1.0, 1.0, 2.0);
        let d = SciRS2LinAlg::det(&df).expect("det ok");
        assert!((d - 3.0).abs() < 1e-8, "det mismatch: {}", d);
    }

    #[test]
    fn test_inv() {
        // Inverse of [[2, 1], [1, 2]] = (1/3) * [[2, -1], [-1, 2]]
        let df = make_2x2_df(2.0, 1.0, 1.0, 2.0);
        let inv = SciRS2LinAlg::inv(&df).expect("inv ok");
        let c0 = inv.get_column_numeric_values("c0").expect("ok");
        let c1 = inv.get_column_numeric_values("c1").expect("ok");
        assert!((c0[0] - 2.0 / 3.0).abs() < 1e-8, "inv[0][0] mismatch");
        assert!((c0[1] - (-1.0 / 3.0)).abs() < 1e-8, "inv[1][0] mismatch");
        assert!((c1[0] - (-1.0 / 3.0)).abs() < 1e-8, "inv[0][1] mismatch");
        assert!((c1[1] - 2.0 / 3.0).abs() < 1e-8, "inv[1][1] mismatch");
    }

    #[test]
    fn test_svd() {
        let df = make_2x2_df(1.0, 0.0, 0.0, 2.0);
        let result = SciRS2LinAlg::svd(&df).expect("svd ok");
        // Singular values of diag(1, 2) = [2, 1] (sorted descending)
        assert_eq!(result.s.len(), 2);
        let mut sv = result.s.clone();
        sv.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert!((sv[0] - 2.0).abs() < 1e-8, "largest sv mismatch: {}", sv[0]);
        assert!(
            (sv[1] - 1.0).abs() < 1e-8,
            "smallest sv mismatch: {}",
            sv[1]
        );
    }

    #[test]
    fn test_eig_symmetric() {
        // Symmetric matrix [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let df = make_2x2_df(2.0, 1.0, 1.0, 2.0);
        let result = SciRS2LinAlg::eig(&df).expect("eig ok");
        assert_eq!(result.values.len(), 2);
        let mut eigenvalues = result.values.clone();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!(
            (eigenvalues[0] - 1.0).abs() < 1e-6,
            "eigenvalue[0] mismatch: {}",
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 3.0).abs() < 1e-6,
            "eigenvalue[1] mismatch: {}",
            eigenvalues[1]
        );
    }

    #[test]
    fn test_solve() {
        // Solve [[2, 1], [1, 2]] * x = [[3], [3]] => x = [[1], [1]]
        let a = make_2x2_df(2.0, 1.0, 1.0, 2.0);
        let mut b = DataFrame::new();
        b.add_column(
            "rhs".to_string(),
            Series::new(vec![3.0f64, 3.0], Some("rhs".to_string())).expect("ok"),
        )
        .expect("ok");

        let x = SciRS2LinAlg::solve(&a, &b).expect("solve ok");
        let x_vals = x.get_column_numeric_values("x0").expect("ok");
        assert!(
            (x_vals[0] - 1.0).abs() < 1e-8,
            "x[0] mismatch: {}",
            x_vals[0]
        );
        assert!(
            (x_vals[1] - 1.0).abs() < 1e-8,
            "x[1] mismatch: {}",
            x_vals[1]
        );
    }

    // -----------------------------------------------------------------------
    // PCA tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pca_basic() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0], Some("x".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "y".to_string(),
            Series::new(vec![2.0f64, 4.0, 6.0, 8.0, 10.0], Some("y".to_string())).expect("ok"),
        )
        .expect("ok");

        let pca = SciRS2Stats::pca(&df, &["x", "y"], 1).expect("pca ok");
        assert_eq!(pca.explained_variance.len(), 1);
        assert_eq!(pca.explained_variance_ratio.len(), 1);
        // First PC should explain most of the variance
        assert!(pca.explained_variance_ratio[0] > 0.0);
    }

    #[test]
    fn test_pca_ext() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0f64, 2.0, 3.0], Some("a".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "b".to_string(),
            Series::new(vec![4.0f64, 5.0, 6.0], Some("b".to_string())).expect("ok"),
        )
        .expect("ok");

        let pca = df.scirs2_pca(1).expect("pca ext ok");
        assert_eq!(pca.explained_variance.len(), 1);
    }

    // -----------------------------------------------------------------------
    // New stats tests: Spearman, covariance, paired ttest, chi2, shapiro, KS
    // -----------------------------------------------------------------------

    fn make_three_col_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "b".to_string(),
            Series::new(vec![2.0f64, 4.0, 6.0, 8.0, 10.0], Some("b".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "c".to_string(),
            Series::new(vec![5.0f64, 3.0, 1.0, 7.0, 9.0], Some("c".to_string())).expect("ok"),
        )
        .expect("ok");
        df
    }

    #[test]
    fn test_spearman_correlation_matrix() {
        let df = make_three_col_df();
        let corr = SciRS2Stats::spearman_correlation_matrix(&df, &["a", "b", "c"])
            .expect("spearman corr ok");
        assert!(corr.contains_column("column"));
        assert!(corr.contains_column("a"));
        assert!(corr.contains_column("b"));
        assert!(corr.contains_column("c"));
        assert_eq!(corr.row_count(), 3);

        // Diagonal elements must be 1.0
        let a_col = corr.get_column_numeric_values("a").expect("ok");
        assert!(
            (a_col[0] - 1.0).abs() < 1e-6,
            "spearman corr(a,a) should be 1.0, got {}",
            a_col[0]
        );

        // a and b have perfect monotone relationship → rho = 1
        let b_col = corr.get_column_numeric_values("b").expect("ok");
        assert!(
            (b_col[0] - 1.0).abs() < 1e-6,
            "spearman corr(a,b) should be 1.0, got {}",
            b_col[0]
        );
    }

    #[test]
    fn test_spearman_corr_ext() {
        let df = make_three_col_df();
        let corr = df.scirs2_spearman_corr().expect("spearman ext ok");
        assert!(corr.contains_column("column"));
        assert_eq!(corr.row_count(), 3);
    }

    #[test]
    fn test_covariance_matrix() {
        let df = make_three_col_df();
        let cov = SciRS2Stats::covariance_matrix(&df, &["a", "b"]).expect("covariance_matrix ok");
        assert!(cov.contains_column("column"));
        assert!(cov.contains_column("a"));
        assert!(cov.contains_column("b"));
        assert_eq!(cov.row_count(), 2);

        // cov(a, a) = var(a) = 2.5 for [1,2,3,4,5] with ddof=1
        let a_col = cov.get_column_numeric_values("a").expect("ok");
        assert!(
            (a_col[0] - 2.5).abs() < 1e-6,
            "var(a) should be 2.5, got {}",
            a_col[0]
        );
    }

    #[test]
    fn test_cov_ext() {
        let df = make_three_col_df();
        let cov = df.scirs2_cov().expect("cov ext ok");
        assert!(cov.contains_column("column"));
    }

    #[test]
    fn test_ttest_paired() {
        // Groups with a clear paired difference
        let before = vec![100.0f64, 102.0, 98.0, 105.0, 99.0];
        let after = vec![90.0f64, 95.0, 89.0, 96.0, 91.0];
        let result = SciRS2Stats::ttest_paired(&before, &after).expect("ttest_paired ok");
        // Mean difference is 9-10 points; should be highly significant
        assert!(
            result.statistic > 0.0,
            "t-stat should be positive (before > after)"
        );
        assert!(
            result.p_value < 0.05,
            "paired ttest should be significant, p={}",
            result.p_value
        );
        assert!(result.df > 0.0);
    }

    #[test]
    fn test_chi2_goodness_of_fit() {
        // Uniform expected: observed roughly uniform → high p-value
        let observed = vec![10.0f64, 11.0, 9.0, 10.0, 10.0];
        let expected = vec![10.0f64, 10.0, 10.0, 10.0, 10.0];
        let result = SciRS2Stats::chi2_goodness_of_fit(&observed, &expected).expect("chi2_gof ok");
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.dof, 4); // k-1
    }

    #[test]
    fn test_shapiro_wilk_on_normal_data() {
        // Near-normal data: should have high p-value (fail to reject normality)
        let data = vec![
            -0.20f64, 0.10, -0.10, 0.20, -0.30, 0.30, 0.00, -0.15, 0.15, 0.05,
        ];
        let result = SciRS2Stats::shapiro_wilk_test(&data).expect("shapiro_wilk ok");
        assert!(
            result.statistic > 0.0 && result.statistic <= 1.0,
            "W should be in (0, 1], got {}",
            result.statistic
        );
        assert!(
            result.p_value >= 0.0 && result.p_value <= 1.0,
            "p_value out of range: {}",
            result.p_value
        );
        // Normal-looking data should not be rejected at 0.05 level
        assert!(
            result.is_normal,
            "Near-normal data should not be rejected by Shapiro-Wilk, p={}",
            result.p_value
        );
    }

    #[test]
    fn test_ks_two_sample() {
        // Two samples from the same linear spacing → should not be rejected
        let data1: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let data2: Vec<f64> = (0..20).map(|i| i as f64 * 0.1 + 0.01).collect();
        let result = SciRS2Stats::ks_two_sample(&data1, &data2).expect("ks_2samp ok");
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    // -----------------------------------------------------------------------
    // New linalg tests: QR, matrix_rank
    // -----------------------------------------------------------------------

    #[test]
    fn test_qr_decomposition() {
        // Q * R should reconstruct A
        let mut df = DataFrame::new();
        df.add_column(
            "c0".to_string(),
            Series::new(vec![1.0f64, 3.0, 2.0], Some("c0".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "c1".to_string(),
            Series::new(vec![2.0f64, 4.0, 1.0], Some("c1".to_string())).expect("ok"),
        )
        .expect("ok");

        let qr_res = SciRS2LinAlg::qr(&df).expect("qr ok");
        assert!(qr_res.q.row_count() > 0);
        assert!(qr_res.r.row_count() > 0);

        // Q * R should approximately equal the original A
        let qr_product = SciRS2LinAlg::matmul(&qr_res.q, &qr_res.r).expect("Q*R matmul ok");
        assert_eq!(qr_product.row_count(), 3);
        assert_eq!(qr_product.column_count(), 2);

        let orig_c0 = df.get_column_numeric_values("c0").expect("ok");
        let prod_c0 = qr_product.get_column_numeric_values("c0").expect("ok");
        for (a, b) in orig_c0.iter().zip(prod_c0.iter()) {
            assert!(
                (a - b).abs() < 1e-8,
                "Q*R ≈ A mismatch in c0: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_qr_ext() {
        let mut df = DataFrame::new();
        df.add_column(
            "c0".to_string(),
            Series::new(vec![1.0f64, 2.0], Some("c0".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "c1".to_string(),
            Series::new(vec![3.0f64, 4.0], Some("c1".to_string())).expect("ok"),
        )
        .expect("ok");

        let qr_res = df.scirs2_qr().expect("qr ext ok");
        assert!(qr_res.q.row_count() > 0);
    }

    #[test]
    fn test_matrix_rank_full() {
        // Identity matrix has full rank = 2
        let mut df = DataFrame::new();
        df.add_column(
            "c0".to_string(),
            Series::new(vec![1.0f64, 0.0], Some("c0".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "c1".to_string(),
            Series::new(vec![0.0f64, 1.0], Some("c1".to_string())).expect("ok"),
        )
        .expect("ok");

        let rank = SciRS2LinAlg::matrix_rank(&df).expect("matrix_rank ok");
        assert_eq!(rank, 2, "Identity 2×2 should have rank 2");
    }

    #[test]
    fn test_matrix_rank_deficient() {
        // Rank-1 matrix: second row = first row
        let mut df = DataFrame::new();
        df.add_column(
            "c0".to_string(),
            Series::new(vec![1.0f64, 2.0], Some("c0".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "c1".to_string(),
            Series::new(vec![2.0f64, 4.0], Some("c1".to_string())).expect("ok"),
        )
        .expect("ok");

        let rank = SciRS2LinAlg::matrix_rank(&df).expect("matrix_rank rank-deficient ok");
        assert_eq!(rank, 1, "Rank-deficient 2×2 should have rank 1");
    }

    #[test]
    fn test_matrix_rank_ext() {
        let mut df = DataFrame::new();
        df.add_column(
            "c0".to_string(),
            Series::new(vec![1.0f64, 0.0], Some("c0".to_string())).expect("ok"),
        )
        .expect("ok");
        df.add_column(
            "c1".to_string(),
            Series::new(vec![0.0f64, 1.0], Some("c1".to_string())).expect("ok"),
        )
        .expect("ok");

        let rank = df.scirs2_matrix_rank().expect("matrix_rank ext ok");
        assert_eq!(rank, 2);
    }
}

// If feature is not enabled, ensure at least one test exists (so the test suite compiles)
#[cfg(not(feature = "scirs2"))]
#[test]
fn scirs2_feature_disabled_placeholder() {
    // This test always passes. The actual SciRS2 integration tests require
    // the `scirs2` feature flag:
    //   cargo test --test scirs2_integration_test --features scirs2
}
