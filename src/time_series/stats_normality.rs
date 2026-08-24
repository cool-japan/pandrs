//! Normality and outlier statistical tests for time series.
//!
//! Split out of `stats.rs` to keep each source file under the 2000-line
//! guideline. These are inherent `impl` blocks for `ShapiroWilkTest`,
//! `AndersonDarlingTest`, `GrubbsTest`, `ModifiedZScoreTest` and
//! `IQROutlierTest`, whose struct definitions live in the parent `stats`
//! module (brought in via `use super::*`). Tail probabilities route through
//! `crate::stats::special` like the rest of the module.

use super::*;
#[allow(unused_imports)]
use crate::core::error::{Error, Result};
#[allow(unused_imports)]
use crate::stats::special::*;
#[allow(unused_imports)]
use std::collections::HashMap;

impl ShapiroWilkTest {
    /// Compute the Shapiro-Wilk test for normality using Royston's (1992)
    /// AS R94 algorithm.
    ///
    /// The order-statistic weights are the Blom normal scores corrected by
    /// Royston's polynomials for the two extreme weights; `W = (Σ aᵢ·x₍ᵢ₎)² /
    /// Σ(xᵢ − x̄)²`, and the p-value comes from Royston's normalizing
    /// transformation. This replaces the previous fabricated statistic
    /// (`1 − range²/(n·var)`) and its 2-bucket p-value. Valid for `3 ≤ n ≤
    /// 5000`; outside that range the test is not defined and a neutral
    /// (non-significant) result is returned.
    pub fn compute(values: &[f64]) -> Result<Self> {
        let n = values.len();
        if !(3..=5000).contains(&n) {
            return Ok(Self {
                statistic: 1.0,
                p_value: 1.0,
                is_normal: true,
            });
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let an = n as f64;
        let mean = sorted.iter().sum::<f64>() / an;
        let ss: f64 = sorted.iter().map(|x| (x - mean).powi(2)).sum();
        if ss <= 0.0 {
            return Ok(Self {
                statistic: 1.0,
                p_value: 1.0,
                is_normal: true,
            });
        }

        // Normal scores m_i for the lower half (these are negative).
        let n2 = n / 2;
        let mut m = vec![0.0_f64; n2 + 1]; // 1-indexed
        let mut summ2 = 0.0;
        for i in 1..=n2 {
            let mi = inv_normal_cdf((i as f64 - 0.375) / (an + 0.25));
            m[i] = mi;
            summ2 += mi * mi;
        }
        summ2 *= 2.0;
        let ssumm2 = summ2.sqrt();
        let rsn = 1.0 / an.sqrt();

        // Royston's polynomial corrections for the extreme weights.
        let c1 = [0.0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056];
        let c2 = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];

        // `a[i]` holds the positive weight magnitudes b_i for i = 1..=n2.
        let mut a = vec![0.0_f64; n2 + 1];
        let a1 = poly(&c1, rsn) - m[1] / ssumm2;
        let (i1, fac);
        if n > 5 {
            let a2 = poly(&c2, rsn) - m[2] / ssumm2;
            a[2] = a2;
            i1 = 3;
            fac = ((summ2 - 2.0 * m[1] * m[1] - 2.0 * m[2] * m[2])
                / (1.0 - 2.0 * a1 * a1 - 2.0 * a2 * a2))
                .sqrt();
        } else {
            i1 = 2;
            fac = ((summ2 - 2.0 * m[1] * m[1]) / (1.0 - 2.0 * a1 * a1)).sqrt();
        }
        a[1] = a1;
        if fac.is_finite() && fac > 0.0 {
            for i in i1..=n2 {
                a[i] = -m[i] / fac;
            }
        }

        // W = (Σ aᵢ (x₍n+1−i₎ − x₍i₎))² / Σ(xᵢ − x̄)², using weight antisymmetry.
        let mut numerator = 0.0;
        for i in 1..=n2 {
            numerator += a[i] * (sorted[n - i] - sorted[i - 1]);
        }
        let w = (numerator * numerator / ss).min(1.0);

        // Royston's p-value transform.
        let p_value = if n == 3 {
            // Exact null distribution for n = 3.
            let pi6 = 6.0 / std::f64::consts::PI;
            let stqr = (0.75_f64).sqrt().asin();
            (pi6 * (w.sqrt().asin() - stqr)).clamp(0.0, 1.0)
        } else {
            let w1 = 1.0 - w;
            let z = if n <= 11 {
                let gamma = -2.273 + 0.459 * an;
                let mu = poly(&[0.5440, -0.39978, 0.025054, -6.714e-4], an);
                let sigma = poly(&[1.3822, -0.77857, 0.062767, -0.0020322], an).exp();
                let y = -(gamma - w1.ln()).ln();
                (y - mu) / sigma
            } else {
                let ln_an = an.ln();
                let mu = poly(&[-1.5861, -0.31082, -0.083751, 0.0038915], ln_an);
                let sigma = poly(&[-0.4803, -0.082676, 0.0030302], ln_an).exp();
                let y = w1.ln();
                (y - mu) / sigma
            };
            normal_sf(z).clamp(0.0, 1.0)
        };

        let is_normal = p_value > 0.05;

        Ok(Self {
            statistic: w,
            p_value,
            is_normal,
        })
    }
}

impl AndersonDarlingTest {
    /// Compute the Anderson-Darling test for normality.
    ///
    /// Uses the standard `A²` statistic against a normal CDF fitted by the
    /// sample mean and (population) standard deviation, the small-sample
    /// correction `A*² = A²(1 + 4/n − 25/n²)`, and the D'Agostino & Stephens
    /// (1986) p-value approximation for the case of estimated parameters. This
    /// replaces the previous 2-bucket p-value ladder; the approximation is
    /// documented (it is not an exact tail probability).
    pub fn compute(values: &[f64]) -> Result<Self> {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let n = sorted.len() as f64;
        let mean = sorted.iter().sum::<f64>() / n;
        let std = (sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();

        // A² statistic against the fitted normal CDF.
        let mut statistic = 0.0;
        if std > 0.0 {
            for (i, &x) in sorted.iter().enumerate() {
                let z = (x - mean) / std;
                let phi = normal_cdf(z).clamp(1e-12, 1.0 - 1e-12);
                statistic += (2.0 * (i + 1) as f64 - 1.0) * (phi.ln() + (1.0 - phi).ln());
            }
            statistic = -n - statistic / n;
        }

        let mut critical_values = HashMap::new();
        critical_values.insert("1%".to_string(), 1.035);
        critical_values.insert("5%".to_string(), 0.752);
        critical_values.insert("10%".to_string(), 0.631);

        // Small-sample correction (parameters estimated from the data).
        let a_star = statistic * (1.0 + 4.0 / n - 25.0 / (n * n));

        // D'Agostino & Stephens (1986) p-value approximation.
        let p_value = if a_star >= 0.6 {
            (1.2937 - 5.709 * a_star + 0.0186 * a_star * a_star).exp()
        } else if a_star >= 0.34 {
            (0.9177 - 4.279 * a_star - 1.38 * a_star * a_star).exp()
        } else if a_star > 0.2 {
            1.0 - (-8.318 + 42.796 * a_star - 59.938 * a_star * a_star).exp()
        } else {
            1.0 - (-13.436 + 101.14 * a_star - 223.73 * a_star * a_star).exp()
        }
        .clamp(0.0, 1.0);

        let is_normal = p_value > 0.05;

        Ok(Self {
            statistic,
            critical_values,
            p_value,
            is_normal,
        })
    }
}

impl GrubbsTest {
    /// Compute Grubbs test for outliers
    pub fn compute(values: &[f64]) -> Result<Self> {
        if values.len() < 3 {
            return Ok(Self {
                statistic: 0.0,
                p_value: 1.0,
                critical_value: 0.0,
                outlier_index: None,
                has_outlier: false,
            });
        }

        let n = values.len();
        let nf = n as f64;
        let mean = values.iter().sum::<f64>() / nf;
        // Grubbs uses the sample standard deviation (n − 1 divisor).
        let std = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();

        // Two-sided Grubbs statistic G = max|xᵢ − x̄| / s.
        let mut max_z = 0.0;
        let mut outlier_index = None;
        for (i, &value) in values.iter().enumerate() {
            let z = if std > 0.0 {
                (value - mean).abs() / std
            } else {
                0.0
            };
            if z > max_z {
                max_z = z;
                outlier_index = Some(i);
            }
        }
        let statistic = max_z;

        // Critical value at α = 0.05 from the exact t relationship:
        //   G_crit = ((n-1)/√n) · √(t² / (n-2 + t²)),  t = t_{α/(2n), n-2}.
        let df = (n - 2) as f64;
        let alpha = 0.05_f64;
        let t_crit = student_t_ppf(1.0 - alpha / (2.0 * nf), df);
        let critical_value =
            ((nf - 1.0) / nf.sqrt()) * (t_crit * t_crit / (df + t_crit * t_crit)).sqrt();

        // Exact two-sided (Bonferroni) p-value from the observed G via its
        // Student-t relationship: t_obs² = n(n-2)·G² / ((n-1)² − n·G²).
        let denom = (nf - 1.0).powi(2) - nf * statistic * statistic;
        let p_value = if df > 0.0 && denom > 0.0 {
            let t_obs = (nf * df * statistic * statistic / denom).sqrt();
            (nf * student_t_two_sided_p(t_obs, df)).min(1.0)
        } else {
            0.0
        };

        let has_outlier = p_value < 0.05;

        Ok(Self {
            statistic,
            p_value,
            critical_value,
            outlier_index,
            has_outlier,
        })
    }
}

impl ModifiedZScoreTest {
    /// Compute modified Z-score test
    pub fn compute(values: &[f64], threshold: f64) -> Result<Self> {
        if values.is_empty() {
            return Ok(Self {
                modified_z_scores: Vec::new(),
                threshold,
                outlier_indices: Vec::new(),
                has_outliers: false,
            });
        }

        // Calculate median
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Calculate MAD (Median Absolute Deviation)
        let deviations: Vec<f64> = values.iter().map(|&x| (x - median).abs()).collect();
        let mut sorted_deviations = deviations.clone();
        sorted_deviations.sort_by(|a, b| a.total_cmp(b));
        let mad = if sorted_deviations.len() % 2 == 0 {
            (sorted_deviations[sorted_deviations.len() / 2 - 1]
                + sorted_deviations[sorted_deviations.len() / 2])
                / 2.0
        } else {
            sorted_deviations[sorted_deviations.len() / 2]
        };

        // Calculate modified Z-scores
        let modified_z_scores: Vec<f64> = values
            .iter()
            .map(|&x| {
                if mad > 0.0 {
                    0.6745 * (x - median) / mad
                } else {
                    0.0
                }
            })
            .collect();

        // Find outliers
        let outlier_indices: Vec<usize> = modified_z_scores
            .iter()
            .enumerate()
            .filter(|(_, &z)| z.abs() > threshold)
            .map(|(i, _)| i)
            .collect();

        let has_outliers = !outlier_indices.is_empty();

        Ok(Self {
            modified_z_scores,
            threshold,
            outlier_indices,
            has_outliers,
        })
    }
}

impl IQROutlierTest {
    /// Compute IQR-based outlier test
    pub fn compute(values: &[f64]) -> Result<Self> {
        if values.len() < 4 {
            return Ok(Self {
                q1: 0.0,
                q3: 0.0,
                iqr: 0.0,
                lower_fence: 0.0,
                upper_fence: 0.0,
                outlier_indices: Vec::new(),
                has_outliers: false,
            });
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let n = sorted.len();
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;

        let lower_fence = q1 - 1.5 * iqr;
        let upper_fence = q3 + 1.5 * iqr;

        let outlier_indices: Vec<usize> = values
            .iter()
            .enumerate()
            .filter(|(_, &x)| x < lower_fence || x > upper_fence)
            .map(|(i, _)| i)
            .collect();

        let has_outliers = !outlier_indices.is_empty();

        Ok(Self {
            q1,
            q3,
            iqr,
            lower_fence,
            upper_fence,
            outlier_indices,
            has_outliers,
        })
    }
}
