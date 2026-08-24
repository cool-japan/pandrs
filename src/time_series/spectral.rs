//! Spectral estimation primitives for the time-series module.
//!
//! Every frequency-domain feature in [`crate::time_series::features`] is built
//! on the single [`periodogram`] entry point below, which computes a genuine
//! discrete Fourier transform through **OxiFFT** (the Pure Rust COOLJAPAN FFT
//! implementation) rather than the `O(n²)` truncated-autocorrelation cosine
//! sum this module replaced. That earlier "simplified FFT" summed at most 50
//! autocorrelation lags against `cos(2π f l)`, which is a *lag-window* spectral
//! estimate with a rectangular window of a fixed, series-length-independent
//! width — it neither converges to the true spectrum nor resolves frequencies
//! finer than `1/50` of the sampling rate, so peak location (the "dominant
//! frequency") was systematically wrong for slowly varying series.

use crate::core::error::{Error, Result};
use oxifft::{Complex, Flags, RealPlan};

/// A one-sided periodogram: the classical non-parametric power spectrum
/// estimate `P(f_k) = |X_k|² / n²` of a real-valued series.
#[derive(Debug, Clone)]
pub(crate) struct Periodogram {
    /// Normalized frequencies in cycles per sample: `k / n` for `k = 0..=n/2`.
    pub frequencies: Vec<f64>,
    /// One-sided power at each frequency. Interior bins carry the power of
    /// both the positive and negative frequency, so they are doubled; the DC
    /// bin (and the Nyquist bin, when `n` is even) are not.
    pub psd: Vec<f64>,
}

impl Periodogram {
    /// Total power across all retained bins.
    pub fn total_power(&self) -> f64 {
        self.psd.iter().sum()
    }

    /// Index of the largest-power bin, ignoring the DC bin (index 0). Returns
    /// `None` when there is no non-DC bin or every candidate is non-finite.
    pub fn dominant_bin(&self) -> Option<usize> {
        self.psd
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, p)| p.is_finite())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
    }
}

/// Detrending applied before the transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Detrend {
    /// Subtract the sample mean (SciPy's `periodogram(detrend='constant')`
    /// default). Without it the DC bin absorbs `n · mean²`, which swamps every
    /// real oscillation for series with a non-zero mean.
    Mean,
}

/// Compute the one-sided periodogram of `values`.
///
/// The transform is unwindowed (rectangular window), matching
/// `scipy.signal.periodogram(x, window='boxcar', scaling='spectrum')`:
/// `P(f_k) = |X_k|² / n²`, doubled on the interior bins, so that `Σ P(f_k)`
/// equals the mean square of the detrended input (Parseval) and a pure
/// sinusoid of amplitude `A` shows a peak of `A² / 2`.
///
/// # Errors
/// Returns [`Error::InvalidInput`] when the series has fewer than two points
/// (a periodogram needs at least one non-DC bin) or contains a non-finite
/// value, and [`Error::InvalidOperation`] if OxiFFT cannot plan the transform.
pub(crate) fn periodogram(values: &[f64], detrend: Detrend) -> Result<Periodogram> {
    let n = values.len();
    if n < 2 {
        return Err(Error::InvalidInput(format!(
            "periodogram requires at least 2 observations, got {n}"
        )));
    }
    if let Some(bad) = values.iter().position(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "periodogram requires finite values; index {bad} is {}",
            values[bad]
        )));
    }

    let input: Vec<f64> = match detrend {
        Detrend::Mean => {
            let mean = values.iter().sum::<f64>() / n as f64;
            values.iter().map(|v| v - mean).collect()
        }
    };

    let plan = RealPlan::<f64>::r2c_1d(n, Flags::ESTIMATE).ok_or_else(|| {
        Error::InvalidOperation(format!("OxiFFT could not plan a real FFT of length {n}"))
    })?;
    let mut spectrum = vec![Complex::<f64>::zero(); plan.complex_size()];
    plan.execute_r2c(&input, &mut spectrum);

    let nf = n as f64;
    let norm = nf * nf;
    let n_even = n % 2 == 0;
    let last = spectrum.len() - 1;

    let mut psd = Vec::with_capacity(spectrum.len());
    let mut frequencies = Vec::with_capacity(spectrum.len());
    for (k, bin) in spectrum.iter().enumerate() {
        // `norm_sqr` is |X_k|²; the one-sided fold doubles every bin whose
        // negative-frequency twin is not itself represented.
        let fold = if k == 0 || (n_even && k == last) {
            1.0
        } else {
            2.0
        };
        psd.push(fold * bin.norm_sqr() / norm);
        frequencies.push(k as f64 / nf);
    }

    Ok(Periodogram { frequencies, psd })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn periodogram_peaks_at_the_injected_frequency() {
        // 8 cycles over 256 samples => f = 8/256 = 0.03125 cycles/sample.
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| 5.0 + 3.0 * (2.0 * PI * 8.0 * i as f64 / n as f64).sin())
            .collect();

        let pg = periodogram(&signal, Detrend::Mean).expect("periodogram");
        let peak = pg.dominant_bin().expect("dominant bin");
        assert_eq!(peak, 8, "peak bin should be k=8, got {peak}");
        assert!((pg.frequencies[peak] - 0.03125).abs() < 1e-12);
    }

    #[test]
    fn periodogram_conserves_power() {
        // Parseval: the summed one-sided power equals the mean square of the
        // mean-removed input.
        let values: Vec<f64> = (0..64)
            .map(|i| (i as f64 * 0.37).sin() * 2.0 + 7.0)
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let mean_square =
            values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64;

        let pg = periodogram(&values, Detrend::Mean).expect("periodogram");
        assert!(
            (pg.total_power() - mean_square).abs() < 1e-9,
            "total power {} != mean square {}",
            pg.total_power(),
            mean_square
        );
    }

    #[test]
    fn periodogram_rejects_short_and_non_finite_input() {
        assert!(periodogram(&[1.0], Detrend::Mean).is_err());
        assert!(periodogram(&[1.0, f64::NAN, 3.0], Detrend::Mean).is_err());
    }
}
