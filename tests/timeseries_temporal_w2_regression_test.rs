#![allow(clippy::result_large_err)]
//! Wave-2 regression tests for `src/time_series/**` and `src/temporal/**`.
//!
//! Every test here pins a behaviour that was previously wrong in a way that
//! produced a *plausible-looking* answer: a fabricated constant, a statistic
//! computed on the wrong scale, a silent `0.0` where a real number belonged, or
//! a panic on ordinary calendar input.

use std::f64::consts::PI;

use chrono::{NaiveDate, TimeZone, Utc};
use pandrs::na::NA;
use pandrs::temporal::{date_range, Frequency as TemporalFrequency, TimeSeries as TemporalSeries};
use pandrs::time_series::advanced_forecasting::SarimaForecaster;
use pandrs::time_series::analysis::ChangePointDetection as ChangePointAnalysis;
use pandrs::time_series::core::{Frequency, TimeSeriesBuilder};
use pandrs::time_series::decomposition::{DecompositionMethod, SeasonalDecomposition};
use pandrs::time_series::features::TimeSeriesFeatureExtractor;
use pandrs::time_series::forecasting::Forecaster;
use pandrs::time_series::preprocessing::{
    OutlierDetection, SmoothingConfig, SmoothingMethod, TimeSeriesPreprocessor,
};
use pandrs::time_series::stats::{DurbinWatsonTest, FriedmanTest, PhillipsPerronTest};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Build a daily `time_series::TimeSeries` from raw values.
fn build_series(values: &[f64]) -> pandrs::time_series::TimeSeries {
    let mut builder = TimeSeriesBuilder::new();
    for (i, &value) in values.iter().enumerate() {
        let timestamp = Utc
            .timestamp_opt(1_640_995_200 + i as i64 * 86_400, 0)
            .single()
            .expect("test timestamp is representable");
        builder = builder.add_point(timestamp, value);
    }
    builder
        .frequency(Frequency::Daily)
        .build()
        .expect("test series builds")
}

/// Deterministic pseudo-random noise so the tests never flake.
fn pseudo_noise(seed: u64, count: usize) -> Vec<f64> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            // 64-bit xorshift, mapped to (-0.5, 0.5).
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

fn day(year: i32, month: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(year, month, day).expect("valid test date")
}

// ---------------------------------------------------------------------------
// features.rs — permutation entropy
// ---------------------------------------------------------------------------

/// `permutation_entropy` discarded the argsort it had just computed and rebuilt
/// `0, 1, …, order-1` for every window, so the pattern histogram was always a
/// single atom and the function returned exactly `0.0` for *every* input.
#[test]
fn permutation_entropy_is_nonzero_on_irregular_data() {
    let values: Vec<f64> = pseudo_noise(0x5eed_1234, 200)
        .into_iter()
        .map(|n| n * 10.0)
        .collect();
    let ts = build_series(&values);

    let features = TimeSeriesFeatureExtractor::new()
        .with_complexity_features(true)
        .extract_features(&ts)
        .expect("feature extraction succeeds");

    let entropy = features.complexity.permutation_entropy;
    assert!(
        entropy > 1.0,
        "permutation entropy of irregular data should be well above 0, got {entropy}"
    );
    // Order 3 has 3! = 6 patterns, so the entropy is bounded by ln 6.
    assert!(
        entropy <= 6.0_f64.ln() + 1e-9,
        "permutation entropy cannot exceed ln(3!) = {}, got {entropy}",
        6.0_f64.ln()
    );
}

/// The ordinal pattern must actually depend on the *ordering* of the data: a
/// monotone ramp uses exactly one pattern (entropy 0) while shuffled data uses
/// many. Under the old implementation both were 0.0 and indistinguishable.
#[test]
fn permutation_entropy_is_sensitive_to_ordering() {
    let monotone: Vec<f64> = (0..200).map(|i| i as f64).collect();
    let mut shuffled = monotone.clone();
    // Deterministic interleave: strictly not monotone, many ordinal patterns.
    shuffled.sort_by_key(|v| ((*v as i64) * 7919) % 200);

    let monotone_entropy = TimeSeriesFeatureExtractor::new()
        .with_complexity_features(true)
        .extract_features(&build_series(&monotone))
        .expect("feature extraction succeeds")
        .complexity
        .permutation_entropy;

    let shuffled_entropy = TimeSeriesFeatureExtractor::new()
        .with_complexity_features(true)
        .extract_features(&build_series(&shuffled))
        .expect("feature extraction succeeds")
        .complexity
        .permutation_entropy;

    assert!(
        monotone_entropy.abs() < 1e-12,
        "a strictly increasing ramp has a single ordinal pattern, so entropy must be 0, got {monotone_entropy}"
    );
    assert!(
        shuffled_entropy > monotone_entropy + 1.0,
        "shuffled data ({shuffled_entropy}) must score far above a monotone ramp ({monotone_entropy})"
    );
}

// ---------------------------------------------------------------------------
// features.rs / spectral.rs — real periodogram
// ---------------------------------------------------------------------------

/// The old "simplified FFT" summed at most 50 autocorrelation lags against
/// `cos(2π f l)`; its peak did not track the true dominant oscillation. With a
/// real OxiFFT periodogram the peak lands on the injected frequency.
#[test]
fn dominant_frequency_matches_the_injected_frequency() {
    // 16 cycles over 256 samples => f = 0.0625 cycles/sample. No trend, so the
    // mean-removed spectrum has a single dominant line.
    let n = 256usize;
    let values: Vec<f64> = (0..n)
        .map(|i| 100.0 + 5.0 * (2.0 * PI * 16.0 * i as f64 / n as f64).sin())
        .collect();
    let ts = build_series(&values);

    let features = TimeSeriesFeatureExtractor::new()
        .with_frequency_features(true)
        .extract_features(&ts)
        .expect("feature extraction succeeds");

    assert_eq!(
        features.frequency.psd.len(),
        features.frequency.frequencies.len()
    );
    assert_eq!(
        features.frequency.psd.len(),
        n / 2 + 1,
        "a one-sided real periodogram has n/2 + 1 bins"
    );

    let dominant = features.frequency.dominant_frequency;
    assert!(
        (dominant - 0.0625).abs() < 1e-9,
        "dominant frequency should be 16/256 = 0.0625, got {dominant}"
    );

    // Parseval: the one-sided power sums to the mean square of the detrended
    // input, i.e. A²/2 for a pure sinusoid of amplitude A.
    let total_power: f64 = features.frequency.psd.iter().sum();
    assert!(
        (total_power - 12.5).abs() < 1e-6,
        "total power of a 5.0-amplitude sinusoid should be 12.5, got {total_power}"
    );

    // The spectral centroid must sit at the line, not at an artefact.
    assert!(
        (features.frequency.spectral_centroid - 0.0625).abs() < 1e-3,
        "spectral centroid should sit at the single spectral line, got {}",
        features.frequency.spectral_centroid
    );
}

// ---------------------------------------------------------------------------
// advanced_forecasting.rs — ARIMA integration
// ---------------------------------------------------------------------------

/// `ARIMA(1,1,1)` fitted a model on the differenced series and then returned
/// the *differenced-scale* forecasts unchanged — for a series around 100 rising
/// by ~1 per step, the "forecast" was ~1. The integration step maps them back.
#[test]
fn arima_111_forecasts_on_the_level_scale() {
    let noise = pseudo_noise(0x00a1_1ce5, 120);
    let values: Vec<f64> = (0..120)
        .map(|i| 100.0 + i as f64 * 1.5 + noise[i] * 0.5)
        .collect();
    let last_level = values[values.len() - 1];
    let ts = build_series(&values);

    let mut model = SarimaForecaster::arima(1, 1, 1);
    model.fit(&ts).expect("ARIMA(1,1,1) fits");
    let result = model.forecast(5, 0.95).expect("ARIMA(1,1,1) forecasts");

    for h in 0..5 {
        let point = result.forecast.values.get_f64(h).expect("forecast value");
        assert!(
            point > last_level - 20.0,
            "horizon {h} forecast {point} is not on the level scale (last observed level {last_level})"
        );
        assert!(
            (point - (last_level + 1.5 * (h + 1) as f64)).abs() < 15.0,
            "horizon {h} forecast {point} should extrapolate the trend from {last_level}"
        );
    }

    // Interval bounds must be integrated too, and must widen with the horizon.
    let mut previous_width = 0.0;
    for h in 0..5 {
        let lower = result.lower_ci.values.get_f64(h).expect("lower bound");
        let upper = result.upper_ci.values.get_f64(h).expect("upper bound");
        let point = result.forecast.values.get_f64(h).expect("forecast value");

        assert!(
            lower < point && point < upper,
            "bounds must bracket the point forecast"
        );
        let width = upper - lower;
        assert!(
            width > previous_width,
            "prediction interval must widen with the horizon: {width} <= {previous_width} at h={h}"
        );
        previous_width = width;
    }
}

/// The confidence level actually requested must be the one used. The old
/// `match (confidence_level * 100.0) as i32 { 90|95|99 => …, _ => 1.96 }` ladder
/// silently produced a 95% interval for an 80% request while still reporting
/// `confidence_level: 0.80`.
#[test]
fn forecast_confidence_level_is_honoured_off_the_ladder() {
    let noise = pseudo_noise(0x00c0_ffee, 80);
    let values: Vec<f64> = (0..80).map(|i| 50.0 + i as f64 * 0.2 + noise[i]).collect();
    let ts = build_series(&values);

    let mut model = SarimaForecaster::arima(1, 0, 0);
    model.fit(&ts).expect("AR(1) fits");

    let narrow = model.forecast(3, 0.80).expect("80% forecast");
    let wide = model.forecast(3, 0.95).expect("95% forecast");

    let narrow_width = narrow.upper_ci.values.get_f64(0).expect("upper")
        - narrow.lower_ci.values.get_f64(0).expect("lower");
    let wide_width = wide.upper_ci.values.get_f64(0).expect("upper")
        - wide.lower_ci.values.get_f64(0).expect("lower");

    assert!(
        wide_width > narrow_width * 1.1,
        "a 95% interval ({wide_width}) must be materially wider than an 80% one ({narrow_width})"
    );

    // z(0.90) / z(0.975) = 1.2816 / 1.9600
    let ratio = narrow_width / wide_width;
    assert!(
        (ratio - 1.281_551_565_545_0 / 1.959_963_984_540_054).abs() < 1e-6,
        "interval widths should scale with the normal quantiles, ratio was {ratio}"
    );

    // Out-of-range confidence levels are rejected instead of silently defaulting.
    assert!(model.forecast(3, 0.0).is_err());
    assert!(model.forecast(3, 1.0).is_err());
}

// ---------------------------------------------------------------------------
// stats.rs — Durbin-Watson, Friedman, Phillips-Perron
// ---------------------------------------------------------------------------

/// The Durbin-Watson bounds were the literal constants 1.5 / 2.5, which are not
/// critical values of anything. They are now Savin-White table values that
/// depend on the sample size.
#[test]
fn durbin_watson_bounds_track_sample_size() {
    let small: Vec<f64> = (0..20).map(|i| (i as f64 * 0.7).sin()).collect();
    let large: Vec<f64> = (0..100).map(|i| (i as f64 * 0.7).sin()).collect();

    let dw_small = DurbinWatsonTest::compute(&small).expect("DW on 20 points");
    let dw_large = DurbinWatsonTest::compute(&large).expect("DW on 100 points");

    assert!(
        (dw_small.lower_critical - 1.201).abs() < 1e-9,
        "d_L at n=20 is 1.201, got {}",
        dw_small.lower_critical
    );
    assert!(
        (dw_small.upper_critical - 1.411).abs() < 1e-9,
        "d_U at n=20 is 1.411, got {}",
        dw_small.upper_critical
    );
    assert!(
        dw_large.lower_critical > dw_small.lower_critical,
        "the bounds must tighten towards 2 as n grows"
    );
    assert!(dw_small.lower_critical < dw_small.upper_critical);
}

/// The Friedman test summed *raw values* per within-cycle position instead of
/// ranking within blocks. The consequence: it was not scale- or shift-invariant.
/// A rank test must be invariant to any strictly increasing transform.
#[test]
fn friedman_is_rank_based_and_shift_invariant() {
    // 6 cycles of a 4-period pattern with a clear ordering inside each cycle.
    let base = [1.0_f64, 4.0, 2.0, 8.0];
    let values: Vec<f64> = (0..24)
        .map(|i| base[i % 4] + (i / 4) as f64 * 0.01)
        .collect();
    let shifted: Vec<f64> = values.iter().map(|v| v + 1_000.0).collect();
    let scaled: Vec<f64> = values.iter().map(|v| v * 37.0).collect();

    let plain = FriedmanTest::compute(&values, 4).expect("Friedman on 6 complete cycles");
    let shifted = FriedmanTest::compute(&shifted, 4).expect("Friedman on shifted data");
    let scaled = FriedmanTest::compute(&scaled, 4).expect("Friedman on scaled data");

    assert!(
        (plain.statistic - shifted.statistic).abs() < 1e-9,
        "a rank test must be shift invariant: {} vs {}",
        plain.statistic,
        shifted.statistic
    );
    assert!(
        (plain.statistic - scaled.statistic).abs() < 1e-9,
        "a rank test must be scale invariant: {} vs {}",
        plain.statistic,
        scaled.statistic
    );

    // The within-cycle ordering is identical in all 6 blocks, so Q hits its
    // maximum: b(k−1) = 6·3 = 18.
    assert!(
        (plain.statistic - 18.0).abs() < 1e-9,
        "a perfectly consistent ranking over 6 blocks of 4 gives Q = 18, got {}",
        plain.statistic
    );
    assert!((plain.df - 3.0).abs() < 1e-12);
    assert!(
        plain.is_seasonal,
        "p = {} should be significant",
        plain.p_value
    );

    // Too few complete cycles is an error, not a zero-filled "not seasonal".
    assert!(FriedmanTest::compute(&values, 24).is_err());
}

/// Phillips-Perron was literally the ADF test scaled by `√((n−1)/n)` — a factor
/// of at most 1.005. The real `Z_τ` statistic applies a Newey-West correction
/// and, like every t-type statistic, must be invariant to rescaling the series.
#[test]
fn phillips_perron_is_scale_invariant_and_detects_a_unit_root() {
    let noise = pseudo_noise(0x9911_2233, 200);

    // Random walk (unit root): should not reject.
    let mut walk = Vec::with_capacity(200);
    let mut level = 0.0;
    for n in &noise {
        level += n * 4.0;
        walk.push(level);
    }

    // Stationary series around a constant mean: should reject.
    let stationary: Vec<f64> = noise.iter().map(|n| 5.0 + n * 4.0).collect();

    let pp_walk = PhillipsPerronTest::compute(&walk).expect("PP on a random walk");
    let pp_stationary = PhillipsPerronTest::compute(&stationary).expect("PP on white noise");

    assert!(
        !pp_walk.is_stationary,
        "a random walk must not be called stationary (Z_tau = {})",
        pp_walk.statistic
    );
    assert!(
        pp_stationary.is_stationary,
        "mean-reverting noise must be called stationary (Z_tau = {})",
        pp_stationary.statistic
    );
    assert!(
        pp_stationary.statistic < pp_walk.statistic,
        "the stationary series must produce the more negative statistic"
    );

    // Scale invariance.
    let scaled: Vec<f64> = walk.iter().map(|v| v * 1_000.0).collect();
    let pp_scaled = PhillipsPerronTest::compute(&scaled).expect("PP on the rescaled walk");
    assert!(
        (pp_walk.statistic - pp_scaled.statistic).abs() < 1e-6,
        "Z_tau must be scale invariant: {} vs {}",
        pp_walk.statistic,
        pp_scaled.statistic
    );
}

// ---------------------------------------------------------------------------
// analysis.rs — BOCPD
// ---------------------------------------------------------------------------

/// "Bayesian detection" was `|mean(before) − mean(after)| > prior_scale · 10`,
/// an absolute threshold with no prior, no likelihood and no scale invariance.
/// Real BOCPD puts changepoint mass at the level shift and nowhere else.
#[test]
fn bocpd_locates_a_level_shift() {
    let noise = pseudo_noise(0xb0cd_d001_u64, 120);
    let values: Vec<f64> = (0..120)
        .map(|i| if i < 60 { 0.0 } else { 12.0 } + noise[i])
        .collect();
    let ts = build_series(&values);

    let detection = ChangePointAnalysis::bayesian_detection(&ts, None)
        .expect("BOCPD runs on a 120-point series");

    assert!(
        detection.method.starts_with("BOCPD"),
        "method should name the algorithm actually used, got {}",
        detection.method
    );
    assert_eq!(detection.scores.len(), values.len());
    assert!(
        detection.scores.iter().all(|p| (0.0..=1.0).contains(p)),
        "BOCPD scores are posterior probabilities and must lie in [0, 1]"
    );

    assert!(
        detection
            .change_points
            .iter()
            .any(|&i| (58..=62).contains(&i)),
        "the level shift at index 60 should be detected, got {:?}",
        detection.change_points
    );
    assert!(
        detection.change_points.len() <= 4,
        "a single shift should not produce a flood of changepoints, got {:?}",
        detection.change_points
    );

    // A hazard rate outside (0, 1) is rejected instead of quietly used.
    assert!(ChangePointAnalysis::bayesian_detection(&ts, Some(0.0)).is_err());
    assert!(ChangePointAnalysis::bayesian_detection(&ts, Some(1.0)).is_err());
}

// ---------------------------------------------------------------------------
// decomposition.rs — strengths and period detection
// ---------------------------------------------------------------------------

/// The strength formulas were `1 − (Var(R) + Var(T))/Var(Y)`, which routinely
/// goes negative. Hyndman's definitions are bounded in `[0, 1]` by construction.
#[test]
fn decomposition_strengths_stay_in_the_unit_interval() {
    let noise = pseudo_noise(0x00de_c0de, 140);
    let values: Vec<f64> = (0..140)
        .map(|i| 20.0 + i as f64 * 0.3 + (2.0 * PI * i as f64 / 7.0).sin() * 4.0 + noise[i])
        .collect();
    let ts = build_series(&values);

    let result = SeasonalDecomposition::new(DecompositionMethod::Additive)
        .with_period(7)
        .decompose(&ts)
        .expect("additive decomposition succeeds");

    let seasonality = result.metrics.seasonality_strength;
    let trend = result.metrics.trend_strength;

    assert!(
        (0.0..=1.0).contains(&seasonality),
        "seasonality strength must lie in [0, 1], got {seasonality}"
    );
    assert!(
        (0.0..=1.0).contains(&trend),
        "trend strength must lie in [0, 1], got {trend}"
    );
    assert!(
        seasonality > 0.5,
        "a series with a dominant weekly cycle should score high, got {seasonality}"
    );
    assert!(
        trend > 0.5,
        "a series with a clear linear trend should score high, got {trend}"
    );
}

/// Auto-detection seeded `best_period = 12` and returned it whenever nothing
/// beat a correlation of 0.0, so an aperiodic series was silently decomposed at
/// a fabricated period of 12.
#[test]
fn decomposition_refuses_to_invent_a_period() {
    // Pure noise: no lag is significantly autocorrelated. The timestamps are
    // deliberately irregular so `DateTimeIndex` infers no frequency and the
    // decomposer has to fall back on autocorrelation-based period detection.
    let values = pseudo_noise(0x4041_4243, 200);
    let mut builder = TimeSeriesBuilder::new();
    let mut offset = 0i64;
    for (i, &value) in values.iter().enumerate() {
        offset += 86_400 * (1 + (i % 3) as i64);
        let timestamp = Utc
            .timestamp_opt(1_640_995_200 + offset, 0)
            .single()
            .expect("test timestamp is representable");
        builder = builder.add_point(timestamp, value);
    }
    let ts = builder.build().expect("test series builds");
    assert!(
        ts.index.frequency.is_none(),
        "the fixture must have no inferred frequency for this test to exercise detection"
    );

    let err = SeasonalDecomposition::new(DecompositionMethod::Additive)
        .decompose(&ts)
        .expect_err("aperiodic data must not silently decompose at a fabricated period 12");
    let message = err.to_string();
    assert!(
        message.contains("No seasonal period detected"),
        "error should say the period could not be detected, got: {message}"
    );
}

// ---------------------------------------------------------------------------
// temporal/date_range — month-end clamping
// ---------------------------------------------------------------------------

/// Month-end starts used to run `NaiveDate::from_ymd_opt(...).expect(...)` and
/// `T::from_str(...).expect(...)`; the range now clamps to the last day of the
/// destination month (pandas semantics) and reports errors instead of panicking.
#[test]
fn month_end_date_ranges_do_not_panic() {
    let monthly = date_range(
        day(2024, 1, 31),
        day(2024, 12, 31),
        TemporalFrequency::Monthly,
        true,
    )
    .expect("monthly range from a 31st must not panic");

    assert_eq!(monthly[0], day(2024, 1, 31));
    assert_eq!(
        monthly[1],
        day(2024, 2, 29),
        "Jan 31 + 1 month clamps to Feb 29 in 2024"
    );
    assert_eq!(monthly[2], day(2024, 3, 29));

    // Non-leap year: February clamps to the 28th.
    let non_leap = date_range(
        day(2023, 1, 31),
        day(2023, 4, 30),
        TemporalFrequency::Monthly,
        true,
    )
    .expect("monthly range in a non-leap year must not panic");
    assert_eq!(non_leap[1], day(2023, 2, 28));

    // Quarterly steps wrap the month past December without panicking.
    let quarterly = date_range(
        day(2023, 12, 31),
        day(2025, 1, 1),
        TemporalFrequency::Quarterly,
        true,
    )
    .expect("quarterly range across a year boundary must not panic");
    assert_eq!(quarterly[0], day(2023, 12, 31));
    assert_eq!(quarterly[1], day(2024, 3, 31));

    // Leap day + one year maps onto February 28.
    let yearly = date_range(
        day(2024, 2, 29),
        day(2027, 1, 1),
        TemporalFrequency::Yearly,
        true,
    )
    .expect("yearly range from a leap day must not panic");
    assert_eq!(yearly[1], day(2025, 2, 28));
}

// ---------------------------------------------------------------------------
// temporal/window — EWM standard deviation
// ---------------------------------------------------------------------------

/// `ewm_std` computed the variance as `E[X²] − (E[X])²`. With a large mean the
/// two accumulators agree past `f64`'s precision, the difference goes negative,
/// and the `if variance > 0.0 { … } else { 0.0 }` guard reported the deviation
/// as exactly `0.0`. The sound recursion is immune.
#[test]
fn ewm_std_survives_a_large_mean() {
    let offset = 1.0e8_f64;
    let pattern = [0.0_f64, 2.0, -2.0, 4.0, -4.0, 1.0, -1.0, 3.0, -3.0, 0.5];

    let dates = date_range(
        day(2024, 1, 1),
        day(2024, 1, 10),
        TemporalFrequency::Daily,
        true,
    )
    .expect("daily range");

    let shifted: Vec<NA<f64>> = pattern.iter().map(|v| NA::Value(offset + v)).collect();
    let centered: Vec<NA<f64>> = pattern.iter().map(|v| NA::Value(*v)).collect();

    let shifted_series = TemporalSeries::new(shifted, dates.clone(), None).expect("series");
    let centered_series = TemporalSeries::new(centered, dates, None).expect("series");

    let shifted_std = shifted_series
        .ewm(None, Some(0.4), true)
        .expect("ewm window")
        .std(0)
        .expect("ewm std");
    let centered_std = centered_series
        .ewm(None, Some(0.4), true)
        .expect("ewm window")
        .std(0)
        .expect("ewm std");

    for i in 1..pattern.len() {
        let large = match shifted_std.values()[i] {
            NA::Value(v) => v,
            NA::NA => panic!("ewm_std should produce a value at index {i} with ddof = 0"),
        };
        let small = match centered_std.values()[i] {
            NA::Value(v) => v,
            NA::NA => panic!("ewm_std should produce a value at index {i} with ddof = 0"),
        };

        assert!(
            large > 0.0,
            "ewm_std collapsed to {large} at index {i} for a series centred on {offset}"
        );
        // Standard deviation is shift invariant.
        assert!(
            (large - small).abs() < 1e-6 * small.max(1.0),
            "ewm_std must be shift invariant: {large} (offset) vs {small} (centred) at index {i}"
        );
    }
}

/// `ewm_std` validated `ddof` and then never used it.
#[test]
fn ewm_std_honours_ddof() {
    let dates = date_range(
        day(2024, 1, 1),
        day(2024, 1, 10),
        TemporalFrequency::Daily,
        true,
    )
    .expect("daily range");
    let values: Vec<NA<f64>> = (0..10).map(|i| NA::Value((i as f64 * 1.7).sin())).collect();
    let series = TemporalSeries::new(values, dates, None).expect("series");

    let biased = series
        .ewm(None, Some(0.3), true)
        .expect("ewm window")
        .std(0)
        .expect("ddof = 0");
    let unbiased = series
        .ewm(None, Some(0.3), true)
        .expect("ewm window")
        .std(1)
        .expect("ddof = 1");

    // ddof = 1 divides by a smaller effective sample size, so it is larger.
    let mut compared = 0;
    for i in 0..10 {
        if let (NA::Value(b), NA::Value(u)) = (biased.values()[i], unbiased.values()[i]) {
            if b > 0.0 {
                assert!(
                    u > b,
                    "ddof = 1 must exceed ddof = 0 at index {i}: {u} vs {b}"
                );
                compared += 1;
            }
        }
    }
    assert!(
        compared >= 5,
        "expected several comparable positions, only got {compared}"
    );

    // The first observation has no effective degrees of freedom left after a
    // ddof = 1 correction; that is NA, not a fabricated 0.0.
    assert!(unbiased.values()[0].is_na());
    assert!(matches!(biased.values()[0], NA::Value(v) if v == 0.0));
}

// ---------------------------------------------------------------------------
// temporal/resample — empty buckets
// ---------------------------------------------------------------------------

/// Buckets with no observation were dropped while the result was still tagged
/// with the resampling frequency, so the index silently lied about the spacing.
#[test]
fn resample_emits_na_for_empty_buckets() {
    // Days 1, 2 and 5 only: days 3 and 4 are missing entirely.
    let timestamps = vec![day(2024, 3, 1), day(2024, 3, 2), day(2024, 3, 5)];
    let values = vec![NA::Value(10.0), NA::Value(20.0), NA::Value(50.0)];
    let series =
        TemporalSeries::new(values, timestamps, Some("gapped".to_string())).expect("series builds");

    let resampled = series
        .resample(TemporalFrequency::Daily)
        .mean()
        .expect("daily resample");

    assert_eq!(
        resampled.len(),
        5,
        "a daily resample spanning Mar 1..Mar 5 must emit 5 rows, got {}",
        resampled.len()
    );
    assert!(matches!(resampled.values()[0], NA::Value(v) if (v - 10.0).abs() < 1e-12));
    assert!(matches!(resampled.values()[1], NA::Value(v) if (v - 20.0).abs() < 1e-12));
    assert!(
        resampled.values()[2].is_na(),
        "the empty Mar 3 bucket must be NA, not 0.0"
    );
    assert!(
        resampled.values()[3].is_na(),
        "the empty Mar 4 bucket must be NA, not 0.0"
    );
    assert!(matches!(resampled.values()[4], NA::Value(v) if (v - 50.0).abs() < 1e-12));

    assert_eq!(resampled.timestamps()[2], day(2024, 3, 3));
    assert_eq!(resampled.timestamps()[4], day(2024, 3, 5));
}

// ---------------------------------------------------------------------------
// temporal/core — days_in_month is total
// ---------------------------------------------------------------------------

/// `days_in_month` is `pub` and used to `panic!("Invalid month")` for anything
/// outside `1..=12`. It now wraps the month into range, carrying the year.
#[test]
fn days_in_month_wraps_instead_of_panicking() {
    use pandrs::temporal::core::days_in_month;

    assert_eq!(days_in_month(2024, 2), 29, "2024 is a leap year");
    assert_eq!(days_in_month(2023, 2), 28);
    assert_eq!(days_in_month(2024, 12), 31);

    // Month 13 of 2024 is January 2025.
    assert_eq!(days_in_month(2024, 13), 31);
    // Month 14 of 2023 is February 2024 (a leap year): 29 days.
    assert_eq!(days_in_month(2023, 14), 29);
    // Month 0 of 2024 is December 2023.
    assert_eq!(days_in_month(2024, 0), 31);
}

// ---------------------------------------------------------------------------
// preprocessing.rs — smoothers that used to be `NotImplemented`
// ---------------------------------------------------------------------------

/// Build a `TimeSeriesPreprocessor` that applies only the requested smoother:
/// missing-value handling and outlier removal are switched off so the assertion
/// is about the smoother alone.
fn smoothing_only(method: SmoothingMethod) -> TimeSeriesPreprocessor {
    TimeSeriesPreprocessor::new()
        .with_outlier_detection(OutlierDetection::None)
        .with_smoothing(SmoothingConfig {
            method,
            parameters: std::collections::HashMap::new(),
        })
}

fn smoothed_values(method: SmoothingMethod, values: &[f64]) -> Vec<f64> {
    let series = build_series(values);
    let result = smoothing_only(method)
        .preprocess(&series)
        .expect("preprocessing succeeds");
    let processed = result.processed_series;
    (0..processed.len())
        .map(|i| processed.values.get_f64(i).expect("dense output"))
        .collect()
}

/// `SmoothingMethod::Lowess` returned `Error::NotImplemented`. It is now
/// Cleveland's locally-weighted regression: it must reproduce a straight line
/// exactly and pull a noisy series towards its underlying signal.
#[test]
fn lowess_smoothing_is_a_real_local_regression() {
    // A local linear fit is exact on globally linear data at every span,
    // including the asymmetric boundary windows.
    let line: Vec<f64> = (0..60).map(|i| 3.0 + 0.75 * i as f64).collect();
    let fitted = smoothed_values(SmoothingMethod::Lowess { fraction: 0.3 }, &line);
    for (i, (&f, &expected)) in fitted.iter().zip(&line).enumerate() {
        assert!(
            (f - expected).abs() < 1e-6,
            "index {i}: LOWESS of a line gave {f}, expected {expected}"
        );
    }

    // Noisy sine: the smoother must be far closer to the signal than the data.
    let n = 240;
    let noise = pseudo_noise(0xa11ce, n);
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / n as f64).sin())
        .collect();
    let noisy: Vec<f64> = signal
        .iter()
        .zip(&noise)
        .map(|(&s, &e)| s + 0.6 * e)
        .collect();

    let fitted = smoothed_values(SmoothingMethod::Lowess { fraction: 0.25 }, &noisy);
    let sse_fit: f64 = fitted
        .iter()
        .zip(&signal)
        .map(|(&f, &s)| (f - s) * (f - s))
        .sum();
    let sse_raw: f64 = noisy
        .iter()
        .zip(&signal)
        .map(|(&f, &s)| (f - s) * (f - s))
        .sum();
    assert!(
        sse_fit < sse_raw / 3.0,
        "LOWESS SSE {sse_fit} should be far below the raw SSE {sse_raw}"
    );
}

/// A bad span used to be silently accepted (the whole method was a stub); it is
/// now validated.
#[test]
fn lowess_rejects_an_impossible_span() {
    let series = build_series(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let err = smoothing_only(SmoothingMethod::Lowess { fraction: 0.0 }).preprocess(&series);
    assert!(err.is_err(), "a zero span must be rejected");
    let err = smoothing_only(SmoothingMethod::Lowess { fraction: 2.0 }).preprocess(&series);
    assert!(err.is_err(), "a span above 1 must be rejected");
}

/// `SmoothingMethod::KalmanFilter` returned `Error::NotImplemented`. It is now a
/// local-level state-space smoother whose signal-to-noise ratio is estimated by
/// maximum likelihood, so it averages measurement noise away while still
/// following a genuine level change.
#[test]
fn kalman_smoothing_denoises_but_keeps_a_level_shift() {
    // Constant level plus alternating measurement error.
    let values: Vec<f64> = (0..80)
        .map(|i| 7.0 + if i % 2 == 0 { 0.9 } else { -0.9 })
        .collect();
    let fitted = smoothed_values(SmoothingMethod::KalmanFilter, &values);
    let sse_fit: f64 = fitted.iter().map(|&a| (a - 7.0) * (a - 7.0)).sum();
    let sse_raw: f64 = values.iter().map(|&a| (a - 7.0) * (a - 7.0)).sum();
    assert!(
        sse_fit < sse_raw / 5.0,
        "Kalman SSE {sse_fit} should be far below the raw SSE {sse_raw}"
    );

    // A genuine break must survive.
    let stepped: Vec<f64> = (0..80)
        .map(|i| {
            let level = if i < 40 { 2.0 } else { 12.0 };
            level + if i % 2 == 0 { 0.3 } else { -0.3 }
        })
        .collect();
    let fitted = smoothed_values(SmoothingMethod::KalmanFilter, &stepped);
    assert!(
        fitted[70] - fitted[10] > 8.0,
        "the level shift was flattened: {} -> {}",
        fitted[10],
        fitted[70]
    );
}

/// `SmoothingMethod::HodrickPrescott` returned `Error::NotImplemented`, so
/// `lambda` was a decorative parameter. It is now the exact minimizer of the
/// HP objective, and `lambda` genuinely controls smoothness.
#[test]
fn hodrick_prescott_honours_lambda() {
    // The HP trend of a straight line is the line itself (zero second
    // differences pay no penalty), at any lambda.
    let line: Vec<f64> = (0..50).map(|i| -4.0 + 1.25 * i as f64).collect();
    let trend = smoothed_values(SmoothingMethod::HodrickPrescott { lambda: 1600.0 }, &line);
    for (i, (&t, &v)) in trend.iter().zip(&line).enumerate() {
        assert!((t - v).abs() < 1e-6, "index {i}: {t} vs {v}");
    }

    // Increasing lambda must strictly smooth a saw-tooth further.
    let saw: Vec<f64> = (0..90)
        .map(|i| 0.1 * i as f64 + if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let roughness = |v: &[f64]| -> f64 {
        v.windows(3)
            .map(|w| (w[2] - 2.0 * w[1] + w[0]).powi(2))
            .sum()
    };
    let light = smoothed_values(SmoothingMethod::HodrickPrescott { lambda: 10.0 }, &saw);
    let heavy = smoothed_values(SmoothingMethod::HodrickPrescott { lambda: 100_000.0 }, &saw);
    assert!(
        roughness(&heavy) < roughness(&light),
        "lambda must control smoothness: {} vs {}",
        roughness(&heavy),
        roughness(&light)
    );
    assert!(roughness(&light) < roughness(&saw));

    // A negative lambda is not a valid penalty weight.
    let series = build_series(&saw);
    assert!(
        smoothing_only(SmoothingMethod::HodrickPrescott { lambda: -1.0 })
            .preprocess(&series)
            .is_err(),
        "a negative lambda must be rejected"
    );
}

// ---------------------------------------------------------------------------
// decomposition.rs — STL
// ---------------------------------------------------------------------------

/// `DecompositionMethod::STL` returned `Error::NotImplemented`. It is now the
/// real Cleveland et al. (1990) algorithm, so — unlike the classical additive
/// decomposition, whose seasonal index is one period-average repeated — the
/// seasonal component is allowed to *evolve*.
#[test]
fn stl_tracks_an_evolving_seasonal_pattern() {
    let period = 12usize;
    let n = 240usize;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let amplitude = 1.0 + 4.0 * i as f64 / n as f64;
            20.0 + 0.02 * i as f64
                + amplitude * (2.0 * PI * (i % period) as f64 / period as f64).sin()
        })
        .collect();
    let series = build_series(&values);

    let stl = SeasonalDecomposition::new(DecompositionMethod::STL)
        .with_period(period)
        .decompose(&series)
        .expect("STL decomposition is implemented");

    assert_eq!(stl.method, DecompositionMethod::STL);
    assert_eq!(stl.period, period);

    // The components must reconstruct the data exactly.
    for (i, &value) in values.iter().enumerate() {
        let sum = stl.trend.values.get_f64(i).expect("trend")
            + stl.seasonal.values.get_f64(i).expect("seasonal")
            + stl.residual.values.get_f64(i).expect("residual");
        assert!(
            (sum - value).abs() < 1e-9,
            "index {i}: components sum to {sum}, data is {value}"
        );
    }

    let amplitude = |from: usize, to: usize| -> f64 {
        let window: Vec<f64> = (from..to)
            .map(|i| stl.seasonal.values.get_f64(i).expect("seasonal"))
            .collect();
        let max = window.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        let min = window.iter().fold(f64::INFINITY, |m, &v| m.min(v));
        max - min
    };
    let early = amplitude(period, 2 * period);
    let late = amplitude(n - 2 * period, n - period);
    assert!(
        late > 1.8 * early,
        "STL must follow the growing seasonal amplitude: {early} -> {late}"
    );

    // The classical additive decomposition cannot: its seasonal index is the
    // same in every cycle by construction.
    let classical = SeasonalDecomposition::new(DecompositionMethod::Additive)
        .with_period(period)
        .decompose(&series)
        .expect("additive decomposition");
    let classical_early = classical.seasonal.values.get_f64(period).expect("seasonal");
    let classical_late = classical
        .seasonal
        .values
        .get_f64(period * 10)
        .expect("seasonal");
    assert!(
        (classical_early - classical_late).abs() < 1e-9,
        "the classical seasonal index is constant across cycles by construction"
    );
}

/// STL is a global fit, so it must refuse a series it cannot decompose rather
/// than inventing components.
#[test]
fn stl_refuses_impossible_input() {
    let short = build_series(&(0..10).map(|i| i as f64).collect::<Vec<_>>());
    assert!(
        SeasonalDecomposition::new(DecompositionMethod::STL)
            .with_period(12)
            .decompose(&short)
            .is_err(),
        "fewer than two full periods must be an error"
    );
    assert!(
        SeasonalDecomposition::new(DecompositionMethod::STL)
            .with_period(1)
            .decompose(&short)
            .is_err(),
        "a period below 2 must be an error"
    );
}

/// X-13ARIMA-SEATS stays an honest `NotImplemented`: its output is defined by
/// the Census Bureau program, so any approximation shipped under that name would
/// report different numbers as if they were X-13's.
#[test]
fn x13_reports_not_implemented_rather_than_substituting_an_algorithm() {
    let values: Vec<f64> = (0..48)
        .map(|i| 10.0 + (2.0 * PI * (i % 12) as f64 / 12.0).sin())
        .collect();
    let series = build_series(&values);

    let err = SeasonalDecomposition::new(DecompositionMethod::X13)
        .with_period(12)
        .decompose(&series)
        .expect_err("X-13 must not silently return another algorithm's numbers");
    let message = err.to_string();
    assert!(
        message.contains("X-13"),
        "the error must name the missing method, got: {message}"
    );
    assert!(
        message.contains("STL"),
        "the error must point at the implemented alternative, got: {message}"
    );
}

/// The STL outer (robustness) loop is reachable from the public API through
/// `with_robust`, and it does what it says: a gross spike is left in the
/// remainder instead of bending the trend towards it.
#[test]
fn robust_stl_leaves_an_outlier_in_the_remainder() {
    let period = 12usize;
    let n = 168usize;
    let mut values: Vec<f64> = (0..n)
        .map(|i| {
            10.0 + 0.03 * i as f64 + 2.0 * (2.0 * PI * (i % period) as f64 / period as f64).sin()
        })
        .collect();
    values[80] += 50.0;
    let series = build_series(&values);

    let plain = SeasonalDecomposition::new(DecompositionMethod::STL)
        .with_period(period)
        .decompose(&series)
        .expect("non-robust STL");
    let robust = SeasonalDecomposition::new(DecompositionMethod::STL)
        .with_period(period)
        .with_robust(true)
        .decompose(&series)
        .expect("robust STL");

    let plain_residual = plain.residual.values.get_f64(80).expect("residual");
    let robust_residual = robust.residual.values.get_f64(80).expect("residual");
    assert!(
        robust_residual.abs() > plain_residual.abs(),
        "the robust fit must push more of the spike into the remainder: {robust_residual} vs \
         {plain_residual}"
    );

    let plain_trend = plain.trend.values.get_f64(80).expect("trend");
    let robust_trend = robust.trend.values.get_f64(80).expect("trend");
    assert!(
        robust_trend < plain_trend,
        "the robust trend ({robust_trend}) must be pulled towards the spike less than the \
         non-robust trend ({plain_trend})"
    );
}
