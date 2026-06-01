//! Semicircle law — the "free Gaussian".
//!
//! The Wigner semicircle law describes the eigenvalue distribution of large
//! random symmetric/Hermitian matrices. It is the free probability analog
//! of the Gaussian/normal distribution.

use serde::{Deserialize, Serialize};

/// Wigner semicircle distribution with radius R and center 0.
///
/// Density: f(x) = (2/(π*R²)) * sqrt(R² - x²) for |x| ≤ R
/// where R = 2σ (σ is the standard deviation parameter).
///
/// If X ~ Semicircle(σ²), then:
/// - E[X] = 0
/// - Var(X) = σ²
/// - All odd moments = 0
/// - 2k-th moment = C_k * σ^(2k) where C_k is the k-th Catalan number
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemicircleLaw {
    /// Variance parameter σ². Radius R = 2σ.
    pub sigma_sq: f64,
}

impl SemicircleLaw {
    /// Create a standard semicircle law with σ² = 1.
    pub fn standard() -> Self {
        Self { sigma_sq: 1.0 }
    }

    /// Create with given variance σ².
    pub fn new(sigma_sq: f64) -> Self {
        Self { sigma_sq }
    }

    /// Radius R = 2σ = 2*sqrt(σ²).
    pub fn radius(&self) -> f64 {
        2.0 * self.sigma_sq.sqrt()
    }

    /// Standard deviation σ = sqrt(σ²).
    pub fn sigma(&self) -> f64 {
        self.sigma_sq.sqrt()
    }

    /// Support: [-R, R].
    pub fn support(&self) -> (f64, f64) {
        let r = self.radius();
        (-r, r)
    }

    /// Probability density function at x.
    ///
    /// f(x) = (2/(π*R²)) * sqrt(R² - x²) for |x| ≤ R, else 0.
    pub fn pdf(&self, x: f64) -> f64 {
        let r = self.radius();
        if x.abs() > r {
            0.0
        } else {
            (2.0 / (std::f64::consts::PI * r * r)) * (r * r - x * x).sqrt()
        }
    }

    /// Cumulative distribution function at x.
    ///
    /// F(x) = 0.5 + (1/π) * [x/R * sqrt(1 - (x/R)²) + arcsin(x/R)]
    pub fn cdf(&self, x: f64) -> f64 {
        let r = self.radius();
        if x <= -r {
            0.0
        } else if x >= r {
            1.0
        } else {
            0.5 + (1.0 / std::f64::consts::PI)
                * (x / r * (1.0 - (x / r).powi(2)).sqrt() + (x / r).asin())
        }
    }

    /// n-th moment of the semicircle distribution.
    ///
    /// Odd moments = 0. Even moments: m_{2k} = C_k * σ^{2k}
    /// where C_k is the k-th Catalan number.
    pub fn moment(&self, n: usize) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n % 2 == 1 {
            return 0.0;
        }
        let k = n / 2;
        // Catalan number C_k = binom(2k, k) / (k+1)
        let mut cat: f64 = 1.0;
        if k > 0 {
            let mut binom: u128 = 1;
            for i in 0..k {
                binom = binom * (2 * k - i) as u128 / (i + 1) as u128;
            }
            cat = binom as f64 / (k + 1) as f64;
        }
        cat * self.sigma_sq.powi(k as i32)
    }

    /// Compute moments up to order n.
    pub fn moments(&self, n: usize) -> Vec<f64> {
        (0..=n).map(|i| self.moment(i)).collect()
    }

    /// Mean = 0.
    pub fn mean(&self) -> f64 {
        0.0
    }

    /// Variance = σ².
    pub fn variance(&self) -> f64 {
        self.sigma_sq
    }

    /// Free cumulant κ_n.
    /// κ_1 = 0, κ_2 = σ², κ_n = 0 for n ≥ 3.
    pub fn free_cumulant(&self, n: usize) -> f64 {
        match n {
            0 => 1.0,
            1 => 0.0,
            2 => self.sigma_sq,
            _ => 0.0,
        }
    }

    /// R-transform: R(z) = σ² for all z (it's a constant!).
    /// The R-transform of the semicircle is the free analog of the
    /// cumulant generating function.
    pub fn r_transform(&self, _z: f64) -> f64 {
        self.sigma_sq
    }

    /// Cauchy transform (Stieltjes transform) evaluated at complex z.
    /// G(z) = (z - sqrt(z² - 4σ²)) / (2σ²)
    /// where sqrt picks the branch with |G(z)| < ∞ as z → ∞.
    pub fn cauchy_transform(&self, z_re: f64, z_im: f64) -> (f64, f64) {
        let s2 = self.sigma_sq;
        // z² - 4σ²
        let w_re = z_re * z_re - z_im * z_im - 4.0 * s2;
        let w_im = 2.0 * z_re * z_im;
        // sqrt(w)
        let (sqrt_re, sqrt_im) = complex_sqrt(w_re, w_im);
        // z - sqrt(z² - 4σ²)
        let num_re = z_re - sqrt_re;
        let num_im = z_im - sqrt_im;
        // Divide by 2σ²
        let denom = 2.0 * s2;
        (num_re / denom, num_im / denom)
    }

    /// Quantile function (inverse CDF) via numerical inversion.
    pub fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return -self.radius();
        }
        if p >= 1.0 {
            return self.radius();
        }
        // Binary search
        let r = self.radius();
        let mut lo = -r;
        let mut hi = r;
        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            if self.cdf(mid) < p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0
    }

    /// Sample n points from the semicircle distribution using the quantile function.
    pub fn sample_quantiles(&self, n: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let p = (i as f64 + 0.5) / n as f64;
            result.push(self.quantile(p));
        }
        result
    }
}

/// Complex square root, returns (re, im) for sqrt(re + i*im).
fn complex_sqrt(re: f64, im: f64) -> (f64, f64) {
    let mag = (re * re + im * im).sqrt();
    let r = mag.sqrt();
    let theta = im.atan2(re) / 2.0;
    (r * theta.cos(), r * theta.sin())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_standard_semicircle_radius() {
        let sc = SemicircleLaw::standard();
        assert_relative_eq!(sc.radius(), 2.0);
    }

    #[test]
    fn test_semicircle_support() {
        let sc = SemicircleLaw::new(4.0);
        let (lo, hi) = sc.support();
        assert_relative_eq!(lo, -4.0);
        assert_relative_eq!(hi, 4.0);
    }

    #[test]
    fn test_semicircle_pdf_at_center() {
        let sc = SemicircleLaw::standard();
        let r = sc.radius();
        let expected = 2.0 / (std::f64::consts::PI * r * r) * r;
        assert_relative_eq!(sc.pdf(0.0), expected, epsilon = 1e-10);
        // f(0) = 2/(π*4) * 2 = 1/π
        assert_relative_eq!(sc.pdf(0.0), 1.0 / std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn test_semicircle_pdf_outside_support() {
        let sc = SemicircleLaw::standard();
        assert_relative_eq!(sc.pdf(3.0), 0.0);
        assert_relative_eq!(sc.pdf(-3.0), 0.0);
    }

    #[test]
    fn test_semicircle_pdf_at_boundary() {
        let sc = SemicircleLaw::standard();
        assert_relative_eq!(sc.pdf(2.0), 0.0);
        assert_relative_eq!(sc.pdf(-2.0), 0.0);
    }

    #[test]
    fn test_semicircle_cdf_at_boundaries() {
        let sc = SemicircleLaw::standard();
        assert_relative_eq!(sc.cdf(-2.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(sc.cdf(2.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(sc.cdf(0.0), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_semicircle_moments() {
        let sc = SemicircleLaw::standard();
        assert_relative_eq!(sc.moment(0), 1.0);
        assert_relative_eq!(sc.moment(1), 0.0);
        assert_relative_eq!(sc.moment(2), 1.0);     // C_1 * 1
        assert_relative_eq!(sc.moment(3), 0.0);
        assert_relative_eq!(sc.moment(4), 2.0);     // C_2 * 1
        assert_relative_eq!(sc.moment(5), 0.0);
        assert_relative_eq!(sc.moment(6), 5.0);     // C_3 * 1
        assert_relative_eq!(sc.moment(8), 14.0);    // C_4 * 1
    }

    #[test]
    fn test_semicircle_moments_with_variance() {
        let sc = SemicircleLaw::new(4.0); // σ² = 4, σ = 2
        assert_relative_eq!(sc.moment(0), 1.0);
        assert_relative_eq!(sc.moment(1), 0.0);
        assert_relative_eq!(sc.moment(2), 4.0);     // σ² = 4
        assert_relative_eq!(sc.moment(4), 32.0);    // C_2 * σ⁴ = 2 * 16
    }

    #[test]
    fn test_semicircle_mean_variance() {
        let sc = SemicircleLaw::new(3.0);
        assert_relative_eq!(sc.mean(), 0.0);
        assert_relative_eq!(sc.variance(), 3.0);
    }

    #[test]
    fn test_semicircle_free_cumulants() {
        let sc = SemicircleLaw::new(2.0);
        assert_relative_eq!(sc.free_cumulant(1), 0.0);
        assert_relative_eq!(sc.free_cumulant(2), 2.0);
        assert_relative_eq!(sc.free_cumulant(3), 0.0);
        assert_relative_eq!(sc.free_cumulant(4), 0.0);
    }

    #[test]
    fn test_semicircle_r_transform() {
        let sc = SemicircleLaw::new(5.0);
        assert_relative_eq!(sc.r_transform(0.0), 5.0);
        assert_relative_eq!(sc.r_transform(1.0), 5.0);
        assert_relative_eq!(sc.r_transform(-1.0), 5.0);
    }

    #[test]
    fn test_semicircle_cauchy_transform_large_z() {
        let sc = SemicircleLaw::standard();
        // G(z) ≈ 1/z for large z
        let (re, _im) = sc.cauchy_transform(1000.0, 0.0);
        assert_relative_eq!(re, 0.001, epsilon = 1e-4);
    }

    #[test]
    fn test_semicircle_quantile_median() {
        let sc = SemicircleLaw::standard();
        assert_relative_eq!(sc.quantile(0.5), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_semicircle_quantile_extremes() {
        let sc = SemicircleLaw::standard();
        assert_relative_eq!(sc.quantile(0.0), -2.0, epsilon = 1e-10);
        assert_relative_eq!(sc.quantile(1.0), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_semicircle_cdf_symmetry() {
        let sc = SemicircleLaw::standard();
        for x in [0.5, 1.0, 1.5] {
            assert_relative_eq!(sc.cdf(x) + sc.cdf(-x), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_semicircle_pdf_integrates_to_one() {
        let sc = SemicircleLaw::standard();
        // Numerical integration using trapezoidal rule
        let n = 10000;
        let r = sc.radius();
        let dx = 2.0 * r / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let x = -r + (i as f64 + 0.5) * dx;
            integral += sc.pdf(x) * dx;
        }
        assert_relative_eq!(integral, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_semicircle_sample_quantiles() {
        let sc = SemicircleLaw::standard();
        let samples = sc.sample_quantiles(100);
        assert_eq!(samples.len(), 100);
        for &s in &samples {
            assert!(s >= -2.0 && s <= 2.0, "Sample {s} outside support");
        }
    }
}
