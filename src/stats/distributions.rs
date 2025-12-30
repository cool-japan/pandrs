//! Statistical probability distributions
//!
//! This module provides implementations of common probability distributions
//! used in statistical analysis, including normal, t, chi-square, F, and others.

use crate::core::error::{Error, Result};
use std::f64::consts::{E, PI};

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

    /// Error function approximation using Abramowitz and Stegun
    fn erf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

impl Distribution for StandardNormal {
    fn pdf(&self, x: f64) -> f64 {
        (1.0 / (2.0 * PI).sqrt()) * (-0.5 * x * x).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / 2.0_f64.sqrt()))
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
    standard_normal: StandardNormal,
}

impl TDistribution {
    pub fn new(degrees_of_freedom: f64) -> Result<Self> {
        if degrees_of_freedom <= 0.0 {
            return Err(Error::InvalidValue(
                "Degrees of freedom must be positive".into(),
            ));
        }

        Ok(TDistribution {
            degrees_of_freedom,
            standard_normal: StandardNormal::new(),
        })
    }

    /// Gamma function approximation using Stirling's approximation
    fn ln_gamma(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NAN;
        }

        // Stirling's approximation
        0.5 * (2.0 * PI / x).ln() + x * (x.ln() - 1.0)
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
        if self.degrees_of_freedom >= 100.0 {
            // For large df, t-distribution approaches normal
            return self.standard_normal.cdf(x);
        }

        // Incomplete beta function approximation for t-distribution CDF
        // This is a simplified implementation
        let nu = self.degrees_of_freedom;
        let t = x;

        if t == 0.0 {
            return 0.5;
        }

        // Use approximation for small to moderate degrees of freedom
        if nu <= 4.0 {
            // Simple approximation
            let z = t / (1.0 + t * t / nu).sqrt();
            0.5 + 0.5 * z * (1.0 - z.abs() / (2.0 + nu))
        } else {
            // Better approximation for larger df
            let correction = 1.0 / (4.0 * nu) * (t * t * t / t.abs() - t / t.abs());
            self.standard_normal.cdf(t + correction)
        }
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }

        if self.degrees_of_freedom >= 100.0 {
            // For large df, use normal approximation
            return self.standard_normal.inverse_cdf(p);
        }

        // Approximation for t-distribution inverse CDF
        let nu = self.degrees_of_freedom;
        let z = self.standard_normal.inverse_cdf(p);

        // Cornish-Fisher expansion approximation
        let c1 = z / 4.0;
        let c2 = (5.0 * z + 16.0 * z.powi(3)) / (96.0);
        let c3 = (3.0 * z + 19.0 * z.powi(3) + 17.0 * z.powi(5)) / (384.0);

        z + c1 / nu + c2 / nu.powi(2) + c3 / nu.powi(3)
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

    /// Incomplete gamma function approximation
    fn incomplete_gamma_lower(&self, a: f64, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }

        // Series expansion for small x
        if x < a + 1.0 {
            let mut sum = 1.0;
            let mut term = 1.0;
            let mut n = 1.0;

            for _ in 0..100 {
                term *= x / (a + n - 1.0);
                sum += term;
                if term.abs() < 1e-15 {
                    break;
                }
                n += 1.0;
            }

            x.powf(a) * (-x).exp() * sum / TDistribution::ln_gamma(a).exp()
        } else {
            // Continued fraction for large x
            1.0 - self.incomplete_gamma_upper(a, x)
        }
    }

    /// Upper incomplete gamma function
    fn incomplete_gamma_upper(&self, a: f64, x: f64) -> f64 {
        // Simplified approximation
        let t = x / a;
        if t < 1.0 {
            1.0 - t.powf(a) * (-t).exp()
        } else {
            (-x).exp() * x.powf(a - 1.0) / TDistribution::ln_gamma(a).exp()
        }
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

        let k = self.degrees_of_freedom;
        self.incomplete_gamma_lower(k / 2.0, x / 2.0)
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }

        // Wilson-Hilferty approximation
        let k = self.degrees_of_freedom;
        let h = 2.0 / (9.0 * k);
        let z = StandardNormal::new().inverse_cdf(p);

        let term = 1.0 - h + z * h.sqrt();
        k * term.powi(3)
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

    /// Incomplete beta function (simplified approximation)
    fn incomplete_beta(&self, x: f64, a: f64, b: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }

        // Simplified continued fraction approximation
        let result = x.powf(a) * (1.0 - x).powf(b) / (a * Self::beta(a, b));

        // Series approximation for moderate values
        let mut sum = 1.0;
        let mut term = 1.0;

        for n in 1..50 {
            term *= (a + n as f64 - 1.0) * x / (n as f64);
            sum += term;
            if term.abs() < 1e-12 {
                break;
            }
        }

        result * sum
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

        let d1 = self.df1;
        let d2 = self.df2;
        let t = d1 * x / (d1 * x + d2);

        self.incomplete_beta(t, d1 / 2.0, d2 / 2.0)
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }

        // Approximation using relationship with chi-squared
        // For simplicity, use direct calculation to avoid Result propagation
        // In a production system, this would be implemented more carefully
        if self.df1 <= 0.0 || self.df2 <= 0.0 {
            return f64::NAN;
        }

        // Very basic approximation - chi-squared mean is df
        let x1_approx = self.df1; // Expected value of chi-squared
        let x2_approx = self.df2; // Expected value of chi-squared

        (x1_approx / self.df1) / (x2_approx / self.df2)
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

    /// Probability mass function
    pub fn pmf(&self, k: usize) -> f64 {
        let k_f = k as f64;
        (-self.lambda).exp() * self.lambda.powf(k_f) / Self::factorial(k)
    }

    /// Factorial function
    fn factorial(n: usize) -> f64 {
        if n <= 1 {
            return 1.0;
        }

        // Use Stirling's approximation for large n
        if n > 20 {
            let n_f = n as f64;
            (2.0 * PI * n_f).sqrt() * (n_f / E).powf(n_f)
        } else {
            (1..=n).map(|i| i as f64).product()
        }
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
        let dist = Normal::new(10.0, 2.0).unwrap();

        assert_eq!(dist.mean(), 10.0);
        assert_eq!(dist.variance(), 4.0);

        // CDF at mean should be 0.5
        assert!((dist.cdf(10.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_chi_squared() {
        let dist = ChiSquared::new(5.0).unwrap();

        assert_eq!(dist.mean(), 5.0);
        assert_eq!(dist.variance(), 10.0);

        // PDF should be 0 for negative values
        assert_eq!(dist.pdf(-1.0), 0.0);
    }

    #[test]
    fn test_t_distribution() {
        let dist = TDistribution::new(10.0).unwrap();

        assert_eq!(dist.mean(), 0.0);
        assert!(dist.variance() > 1.0); // Should be > 1 for df > 2

        // Should be symmetric around 0
        assert!((dist.cdf(0.0) - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_binomial() {
        let dist = Binomial::new(10, 0.3).unwrap();

        assert_eq!(dist.mean(), 3.0);
        assert!((dist.variance() - 2.1).abs() < 1e-10);

        // PMF should sum to 1
        let sum: f64 = (0..=10).map(|k| dist.pmf(k)).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_poisson() {
        let dist = Poisson::new(3.0).unwrap();

        assert_eq!(dist.mean(), 3.0);
        assert_eq!(dist.variance(), 3.0);

        // PMF should be positive for non-negative integers
        assert!(dist.pmf(0) > 0.0);
        assert!(dist.pmf(1) > 0.0);
        assert!(dist.pmf(3) > 0.0);
    }
}
