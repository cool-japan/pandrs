use super::*;
use crate::time_series::core::{Frequency, TimeSeriesBuilder};
use chrono::{TimeZone, Utc};

fn create_test_series() -> TimeSeries {
    let mut builder = TimeSeriesBuilder::new();

    for i in 0..100 {
        let timestamp = Utc
            .timestamp_opt(1640995200 + i * 86400, 0)
            .single()
            .expect("operation should succeed");
        let value = 10.0 + i as f64 * 0.1 + (i as f64 % 7.0 - 3.0) * 0.5;
        builder = builder.add_point(timestamp, value);
    }

    builder
        .frequency(Frequency::Daily)
        .build()
        .expect("operation should succeed")
}

#[test]
fn test_time_series_stats_computation() {
    let ts = create_test_series();
    let stats = TimeSeriesStats::compute(&ts).expect("operation should succeed");

    assert!(stats.descriptive.count > 0);
    assert!(stats.descriptive.mean > 0.0);
    assert!(stats.descriptive.std > 0.0);
    assert!(stats.descriptive.min < stats.descriptive.max);
}

#[test]
fn test_adf_test() {
    let values: Vec<f64> = (0..50).map(|i| i as f64 + (i as f64 * 0.1).sin()).collect();
    let result = AugmentedDickeyFullerTest::compute(&values).expect("operation should succeed");

    assert!(result.statistic != 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(result.critical_values.contains_key("5%"));
}

#[test]
fn test_kpss_test() {
    let values: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
    let result = KwiatkowskiPhillipsSchmidtShinTest::compute(&values, "constant")
        .expect("operation should succeed");

    assert!(result.statistic >= 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(result.critical_values.contains_key("5%"));
}

#[test]
fn test_ljung_box_test() {
    let values: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
    let result = LjungBoxTest::compute(&values, 10).expect("operation should succeed");

    assert!(result.statistic >= 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert_eq!(result.n_lags, 10);
}

#[test]
fn test_jarque_bera_test() {
    let values: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let result = JarqueBeraTest::compute(&values).expect("operation should succeed");

    assert!(result.statistic >= 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(result.skewness_stat >= 0.0);
    assert!(result.kurtosis_stat >= 0.0);
}

#[test]
fn test_grubbs_test() {
    let mut values: Vec<f64> = (0..20).map(|i| i as f64).collect();
    values.push(100.0); // Add outlier

    let result = GrubbsTest::compute(&values).expect("operation should succeed");

    assert!(result.statistic > 0.0);
    assert!(result.has_outlier);
    assert_eq!(result.outlier_index, Some(20)); // Should detect the outlier
}

#[test]
fn test_modified_z_score_test() {
    let mut values: Vec<f64> = (0..20).map(|i| i as f64).collect();
    values.push(100.0); // Add outlier

    let result = ModifiedZScoreTest::compute(&values, 3.5).expect("operation should succeed");

    assert_eq!(result.modified_z_scores.len(), values.len());
    assert!(result.has_outliers);
    assert!(!result.outlier_indices.is_empty());
}

#[test]
fn test_iqr_outlier_test() {
    let mut values: Vec<f64> = (0..20).map(|i| i as f64).collect();
    values.push(100.0); // Add outlier

    let result = IQROutlierTest::compute(&values).expect("operation should succeed");

    assert!(result.q3 > result.q1);
    assert!(result.iqr > 0.0);
    assert!(result.upper_fence > result.lower_fence);
    assert!(result.has_outliers);
}
