//! Statistical probability distributions
//!
//! This module provides implementations of common probability distributions
//! used in statistical analysis, including normal, t, chi-square, F, and others.

use crate::core::error::{Error, Result};
use std::f64::consts::PI;

/// Trait for probability distributions
pub trait Distribution {
    /// Probability density function (PDF)
    fn pdf(&self, x: f64) -> f64;

    /// Cumulative distribution function (CDF)
    fn cdf(&self, x: f64) -> f64;

    /// Inverse CDF (quantile function)
    fn inverse_cdf(&self, p: f64) -> f64;

    /// Mean of the distribution
    fn mean(&self) -> f64;

    /// Variance of the distribution
    fn variance(&self) -> f64;

    /// Standard deviation of the distribution
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Standard normal distribution N(0,1)
#[derive(Debug, Clone)]
pub struct StandardNormal;

impl StandardNormal {
    pub fn new() -> Self {
        StandardNormal
    }
}

impl Distribution for StandardNormal {
    fn pdf(&self, x: f64) -> f64 {
        (1.0 / (2.0 * PI).sqrt()) * (-0.5 * x * x).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        // Delegates to the crate's single-source-of-truth normal CDF
        // (`stats::special`, implemented via the regularized incomplete
        // gamma function). The previous local Abramowitz & Stegun `erf`
        // approximation was one of several independently-maintained normal
        // CDF implementations scattered across the `stats` module; this is
        // both more accurate (~1e-12 vs ~1e-7 relative error) and removes a
        // duplicate.
        crate::stats::special::normal_cdf(x)
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }

        // Beasley-Springer-Moro algorithm approximation
        let a0 = -3.969683028665376e+01;
        let a1 = 2.209460984245205e+02;
        let a2 = -2.759285104469687e+02;
        let a3 = 1.383577518672690e+02;
        let a4 = -3.066479806614716e+01;
        let a5 = 2.506628277459239e+00;

        let b1 = -5.447609879822406e+01;
        let b2 = 1.615858368580409e+02;
        let b3 = -1.556989798598866e+02;
        let b4 = 6.680131188771972e+01;
        let b5 = -1.328068155288572e+01;

        let c0 = -7.784894002430293e-03;
        let c1 = -3.223964580411365e-01;
        let c2 = -2.400758277161838e+00;
        let c3 = -2.549732539343734e+00;
        let c4 = 4.374664141464968e+00;
        let c5 = 2.938163982698783e+00;

        let d1 = 7.784695709041462e-03;
        let d2 = 3.224671290700398e-01;
        let d3 = 2.445134137142996e+00;
        let d4 = 3.754408661907416e+00;

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            // Rational approximation for lower region
            let q = (-2.0 * p.ln()).sqrt();
            (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        } else if p <= p_high {
            // Rational approximation for central region
            let q = p - 0.5;
            let r = q * q;
            (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
                / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        } else {
            // Rational approximation for upper region
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        }
    }

    fn mean(&self) -> f64 {
        0.0
    }

    fn variance(&self) -> f64 {
        1.0
    }
}

/// Normal distribution N(μ, σ²)
#[derive(Debug, Clone)]
pub struct Normal {
    pub mean: f64,
    pub std_dev: f64,
    standard_normal: StandardNormal,
}

impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> Result<Self> {
        if std_dev <= 0.0 {
            return Err(Error::InvalidValue(
                "Standard deviation must be positive".into(),
            ));
        }

        Ok(Normal {
            mean,
            std_dev,
            standard_normal: StandardNormal::new(),
        })
    }
}

impl Distribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        self.standard_normal.pdf(z) / self.std_dev
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        self.standard_normal.cdf(z)
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        let z = self.standard_normal.inverse_cdf(p);
        self.mean + self.std_dev * z
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        self.std_dev * self.std_dev
    }
}

/// Student's t-distribution
#[derive(Debug, Clone)]
pub struct TDistribution {
    pub degrees_of_freedom: f64,
}

impl TDistribution {
    pub fn new(degrees_of_freedom: f64) -> Result<Self> {
        if degrees_of_freedom <= 0.0 {
            return Err(Error::InvalidValue(
                "Degrees of freedom must be positive".into(),
            ));
        }

        Ok(TDistribution { degrees_of_freedom })
    }

    /// Natural log of the gamma function.
    ///
    /// Delegates to the crate's accurate Lanczos implementation in
    /// `stats::special` (the previous one-term Stirling approximation was
    /// inaccurate — e.g. `lnΓ(1) = -0.081` instead of `0` — corrupting every
    /// t / χ² / F PDF built on it).
    fn ln_gamma(x: f64) -> f64 {
        crate::stats::special::ln_gamma(x)
    }

    /// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    fn ln_beta(a: f64, b: f64) -> f64 {
        Self::ln_gamma(a) + Self::ln_gamma(b) - Self::ln_gamma(a + b)
    }
}

impl Distribution for TDistribution {
    fn pdf(&self, x: f64) -> f64 {
        let nu = self.degrees_of_freedom;
        let coeff = (-Self::ln_beta(0.5, nu / 2.0) - 0.5 * (nu * PI).ln()).exp();
        coeff * (1.0 + x * x / nu).powf(-(nu + 1.0) / 2.0)
    }

    fn cdf(&self, x: f64) -> f64 {
        // Exact (incomplete-beta) Student-t CDF. The previous ad-hoc
        // approximation could return values > 1 (an impossible probability).
        crate::stats::special::student_t_cdf(x, self.degrees_of_freedom)
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }
        // Numerically inverted exact CDF (the prior Cornish-Fisher truncation
        // was ~36% low for small df, e.g. t₀.₉₇₅(2) = 2.75 vs 4.30).
        crate::stats::special::student_t_ppf(p, self.degrees_of_freedom)
    }

    fn mean(&self) -> f64 {
        if self.degrees_of_freedom > 1.0 {
            0.0
        } else {
            f64::NAN // Undefined for df <= 1
        }
    }

    fn variance(&self) -> f64 {
        let nu = self.degrees_of_freedom;
        if nu > 2.0 {
            nu / (nu - 2.0)
        } else if nu > 1.0 {
            f64::INFINITY
        } else {
            f64::NAN // Undefined for df <= 1
        }
    }
}

/// Chi-squared distribution
#[derive(Debug, Clone)]
pub struct ChiSquared {
    pub degrees_of_freedom: f64,
}

impl ChiSquared {
    pub fn new(degrees_of_freedom: f64) -> Result<Self> {
        if degrees_of_freedom <= 0.0 {
            return Err(Error::InvalidValue(
                "Degrees of freedom must be positive".into(),
            ));
        }

        Ok(ChiSquared { degrees_of_freedom })
    }
}

impl Distribution for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }

        let k = self.degrees_of_freedom;
        let coeff = 1.0 / (2.0_f64.powf(k / 2.0) * TDistribution::ln_gamma(k / 2.0).exp());
        coeff * x.powf(k / 2.0 - 1.0) * (-x / 2.0).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }

        crate::stats::special::chi2_cdf(x, self.degrees_of_freedom)
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        crate::stats::special::chi2_ppf(p, self.degrees_of_freedom)
    }

    fn mean(&self) -> f64 {
        self.degrees_of_freedom
    }

    fn variance(&self) -> f64 {
        2.0 * self.degrees_of_freedom
    }
}

/// F-distribution
#[derive(Debug, Clone)]
pub struct FDistribution {
    pub df1: f64, // numerator degrees of freedom
    pub df2: f64, // denominator degrees of freedom
}

impl FDistribution {
    pub fn new(df1: f64, df2: f64) -> Result<Self> {
        if df1 <= 0.0 || df2 <= 0.0 {
            return Err(Error::InvalidValue(
                "Both degrees of freedom must be positive".into(),
            ));
        }

        Ok(FDistribution { df1, df2 })
    }

    /// Beta function
    fn beta(a: f64, b: f64) -> f64 {
        TDistribution::ln_beta(a, b).exp()
    }
}

impl Distribution for FDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }

        let d1 = self.df1;
        let d2 = self.df2;

        let coeff = Self::beta(d1 / 2.0, d2 / 2.0);
        let numerator = (d1 / d2).powf(d1 / 2.0) * x.powf(d1 / 2.0 - 1.0);
        let denominator = (1.0 + d1 * x / d2).powf((d1 + d2) / 2.0);

        numerator / (coeff * denominator)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }

        crate::stats::special::f_cdf(x, self.df1, self.df2)
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        // Numerically inverted exact CDF (the prior stub returned a constant
        // `1.0` regardless of `p`).
        crate::stats::special::f_ppf(p, self.df1, self.df2)
    }

    fn mean(&self) -> f64 {
        if self.df2 > 2.0 {
            self.df2 / (self.df2 - 2.0)
        } else {
            f64::NAN // Undefined for df2 <= 2
        }
    }

    fn variance(&self) -> f64 {
        let d2 = self.df2;
        if d2 > 4.0 {
            let d1 = self.df1;
            2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0))
        } else {
            f64::NAN // Undefined for df2 <= 4
        }
    }
}

/// Binomial distribution
#[derive(Debug, Clone)]
pub struct Binomial {
    pub n: usize, // number of trials
    pub p: f64,   // probability of success
}

impl Binomial {
    pub fn new(n: usize, p: f64) -> Result<Self> {
        if p < 0.0 || p > 1.0 {
            return Err(Error::InvalidValue(
                "Probability must be between 0 and 1".into(),
            ));
        }

        Ok(Binomial { n, p })
    }

    /// Binomial coefficient C(n, k) = n! / (k! * (n-k)!)
    fn binomial_coefficient(n: usize, k: usize) -> f64 {
        if k > n {
            return 0.0;
        }
        if k == 0 || k == n {
            return 1.0;
        }

        let k = k.min(n - k); // Use symmetry

        let mut result = 1.0;
        for i in 0..k {
            result *= (n - i) as f64 / (i + 1) as f64;
        }
        result
    }

    /// Probability mass function
    pub fn pmf(&self, k: usize) -> f64 {
        if k > self.n {
            return 0.0;
        }

        let coeff = Self::binomial_coefficient(self.n, k);
        coeff * self.p.powi(k as i32) * (1.0 - self.p).powi((self.n - k) as i32)
    }
}

impl Distribution for Binomial {
    fn pdf(&self, x: f64) -> f64 {
        // For discrete distributions, PDF at non-integer points is 0
        if x.fract() != 0.0 || x < 0.0 {
            return 0.0;
        }
        self.pmf(x as usize)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x >= self.n as f64 {
            return 1.0;
        }

        let k_max = x.floor() as usize;
        let mut sum = 0.0;
        for k in 0..=k_max {
            sum += self.pmf(k);
        }
        sum
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return self.n as f64;
        }

        let mut cumulative = 0.0;
        for k in 0..=self.n {
            cumulative += self.pmf(k);
            if cumulative >= p {
                return k as f64;
            }
        }
        self.n as f64
    }

    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }

    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
}

/// Poisson distribution
#[derive(Debug, Clone)]
pub struct Poisson {
    pub lambda: f64, // rate parameter
}

impl Poisson {
    pub fn new(lambda: f64) -> Result<Self> {
        if lambda <= 0.0 {
            return Err(Error::InvalidValue("Lambda must be positive".into()));
        }

        Ok(Poisson { lambda })
    }

    /// Probability mass function, computed in log-space as
    /// `exp(k·ln(λ) − λ − lnΓ(k+1))` — exact for every `k` (`lnΓ(k+1) =
    /// ln(k!)` via the crate's Lanczos `ln_gamma`), rather than the
    /// previous direct `λᵏ / k!` with a 1-term Stirling approximation for
    /// `k > 20` swapped in for the factorial. That approximation was both
    /// inaccurate (~0.4% relative error) and unstable for large `k` (`k!`
    /// itself overflows `f64` well before `k ≈ 170`, so the direct-division
    /// form breaks down exactly where Stirling was supposed to save it);
    /// staying in log-space until the final `exp` avoids the overflow
    /// entirely.
    pub fn pmf(&self, k: usize) -> f64 {
        let k_f = k as f64;
        (k_f * self.lambda.ln() - self.lambda - crate::stats::special::ln_gamma(k_f + 1.0)).exp()
    }
}

impl Distribution for Poisson {
    fn pdf(&self, x: f64) -> f64 {
        if x.fract() != 0.0 || x < 0.0 {
            return 0.0;
        }
        self.pmf(x as usize)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }

        let k_max = x.floor() as usize;
        let mut sum = 0.0;
        for k in 0..=k_max {
            sum += self.pmf(k);
        }
        sum
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }

        let mut cumulative = 0.0;
        let mut k = 0;

        loop {
            cumulative += self.pmf(k);
            if cumulative >= p {
                return k as f64;
            }
            k += 1;

            // Prevent infinite loop for very large lambda
            if k > (self.lambda * 10.0) as usize {
                break;
            }
        }
        k as f64
    }

    fn mean(&self) -> f64 {
        self.lambda
    }

    fn variance(&self) -> f64 {
        self.lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_normal() {
        let dist = StandardNormal::new();

        // Test PDF at 0
        assert!((dist.pdf(0.0) - 0.3989422804014327).abs() < 1e-10);

        // Test CDF at 0
        assert!((dist.cdf(0.0) - 0.5).abs() < 1e-6);

        // Test inverse CDF
        assert!((dist.inverse_cdf(0.5) - 0.0).abs() < 1e-10);

        // Test mean and variance
        assert_eq!(dist.mean(), 0.0);
        assert_eq!(dist.variance(), 1.0);
    }

    #[test]
    fn test_normal() {
        let dist = Normal::new(10.0, 2.0).expect("operation should succeed");

        assert_eq!(dist.mean(), 10.0);
        assert_eq!(dist.variance(), 4.0);

        // CDF at mean should be 0.5
        assert!((dist.cdf(10.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_chi_squared() {
        let dist = ChiSquared::new(5.0).expect("operation should succeed");

        assert_eq!(dist.mean(), 5.0);
        assert_eq!(dist.variance(), 10.0);

        // PDF should be 0 for negative values
        assert_eq!(dist.pdf(-1.0), 0.0);
    }

    #[test]
    fn test_t_distribution() {
        let dist = TDistribution::new(10.0).expect("operation should succeed");

        assert_eq!(dist.mean(), 0.0);
        assert!(dist.variance() > 1.0); // Should be > 1 for df > 2

        // Should be symmetric around 0
        assert!((dist.cdf(0.0) - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_binomial() {
        let dist = Binomial::new(10, 0.3).expect("operation should succeed");

        assert_eq!(dist.mean(), 3.0);
        assert!((dist.variance() - 2.1).abs() < 1e-10);

        // PMF should sum to 1
        let sum: f64 = (0..=10).map(|k| dist.pmf(k)).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_poisson() {
        let dist = Poisson::new(3.0).expect("operation should succeed");

        assert_eq!(dist.mean(), 3.0);
        assert_eq!(dist.variance(), 3.0);

        // PMF should be positive for non-negative integers
        assert!(dist.pmf(0) > 0.0);
        assert!(dist.pmf(1) > 0.0);
        assert!(dist.pmf(3) > 0.0);
    }

    /// `scipy.stats.poisson.pmf` reference values (scipy 1.17), including
    /// `k = 170` and `k = 480` — magnitudes where the previous
    /// Stirling-for-`k>20` implementation's direct `λᵏ / k!` division would
    /// already have overflowed `f64` (`k!` exceeds `f64::MAX` around
    /// `k ≈ 170`).
    #[test]
    fn test_poisson_pmf_matches_scipy() {
        let close_rel =
            |actual: f64, expected: f64, tol: f64| ((actual - expected) / expected).abs() < tol;

        assert!(close_rel(
            Poisson::new(3.0).expect("valid lambda").pmf(5),
            0.10081881344492458,
            1e-9
        ));
        assert!(close_rel(
            Poisson::new(10.0).expect("valid lambda").pmf(25),
            2.9269109009328616e-05,
            1e-9
        ));
        assert!(close_rel(
            Poisson::new(50.0).expect("valid lambda").pmf(45),
            0.045826241434197924,
            1e-9
        ));
        assert!(close_rel(
            Poisson::new(100.0).expect("valid lambda").pmf(170),
            5.1258962876176525e-11,
            1e-6
        ));
        assert!(close_rel(
            Poisson::new(500.0).expect("valid lambda").pmf(480),
            0.012137592474089706,
            1e-9
        ));
    }
}
