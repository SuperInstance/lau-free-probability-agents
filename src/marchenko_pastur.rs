//! Marchenko-Pastur law — eigenvalue distribution of sample covariance matrices.
//!
//! When X is an n × p matrix with i.i.d. entries (mean 0, variance σ²),
//! the eigenvalues of the sample covariance matrix (1/n) X^T X converge
//! to the Marchenko-Pastur distribution as n, p → ∞ with p/n → c.

use serde::{Deserialize, Serialize};

/// Marchenko-Pastur distribution with aspect ratio c and variance σ².
///
/// Parameter c = p/n (ratio of dimensions).
/// - If c ≤ 1: support is [σ²(1-√c)², σ²(1+√c)²]
/// - If c > 1: same support, plus a point mass at 0 of weight (1 - 1/c)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarchenkoPasturLaw {
    /// Aspect ratio c = p/n.
    pub c: f64,
    /// Variance parameter σ².
    pub sigma_sq: f64,
}

impl MarchenkoPasturLaw {
    /// Create standard MP law with c and σ² = 1.
    pub fn new(c: f64) -> Self {
        Self { c, sigma_sq: 1.0 }
    }

    /// Create with variance.
    pub fn with_variance(c: f64, sigma_sq: f64) -> Self {
        Self { c, sigma_sq }
    }

    /// Lower bound of support: σ²(1 - √c)².
    pub fn lambda_min(&self) -> f64 {
        self.sigma_sq * (1.0 - self.c.sqrt()).powi(2)
    }

    /// Upper bound of support: σ²(1 + √c)².
    pub fn lambda_max(&self) -> f64 {
        self.sigma_sq * (1.0 + self.c.sqrt()).powi(2)
    }

    /// Support interval.
    pub fn support(&self) -> (f64, f64) {
        (self.lambda_min(), self.lambda_max())
    }

    /// Probability density function.
    ///
    /// f(x) = (1/(2πσ²c)) * sqrt((λ_+ - x)(x - λ_-)) / x
    /// for λ_- ≤ x ≤ λ_+, where λ_± = σ²(1 ± √c)²
    pub fn pdf(&self, x: f64) -> f64 {
        if self.c <= 0.0 {
            return 0.0;
        }
        let lam_min = self.lambda_min();
        let lam_max = self.lambda_max();

        if x < lam_min || x > lam_max || x <= 0.0 {
            0.0
        } else {
            let numerator = (lam_max - x) * (x - lam_min);
            (1.0 / (2.0 * std::f64::consts::PI * self.sigma_sq * self.c))
                * numerator.sqrt()
                / x
        }
    }

    /// Cumulative distribution function (numerical integration).
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= self.lambda_min() {
            return 0.0;
        }
        if x >= self.lambda_max() {
            return if self.c <= 1.0 { 1.0 } else { 1.0 / self.c };
        }

        // Numerical integration
        let n = 5000;
        let lam_min = self.lambda_min();
        let dx = (x - lam_min) / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let xi = lam_min + (i as f64 + 0.5) * dx;
            integral += self.pdf(xi) * dx;
        }

        if self.c > 1.0 {
            // Point mass at 0 is (1 - 1/c), continuous part is scaled
            integral / self.c
        } else {
            integral
        }
    }

    /// n-th moment of the Marchenko-Pastur distribution.
    ///
    /// m_k = σ^{2k} * Σ_{j=0}^{k-1} (1/k) * binom(k, j) * binom(k, j+1) * c^j
    /// These are related to Narayana numbers.
    pub fn moment(&self, k: usize) -> f64 {
        if k == 0 {
            return 1.0;
        }

        let mut total = 0.0;
        for j in 0..k {
            // Narayana number N(k, j+1) = (1/k) * binom(k, j) * binom(k, j+1)
            let narayana = Self::narayana(k, j + 1);
            total += narayana * self.c.powi(j as i32);
        }
        total * self.sigma_sq.powi(k as i32)
    }

    /// Narayana number N(n, k) = (1/n) * C(n, k) * C(n, k-1)
    fn narayana(n: usize, k: usize) -> f64 {
        if n == 0 || k == 0 || k > n {
            return 0.0;
        }
        let binom_nk = Self::binomial(n, k) as f64;
        let binom_nk1 = Self::binomial(n, k - 1) as f64;
        binom_nk * binom_nk1 / n as f64
    }

    fn binomial(n: usize, k: usize) -> u128 {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }
        let k = k.min(n - k);
        let mut result: u128 = 1;
        for i in 0..k {
            result = result * (n - i) as u128 / (i + 1) as u128;
        }
        result
    }

    /// Moments up to order n.
    pub fn moments(&self, n: usize) -> Vec<f64> {
        (0..=n).map(|i| self.moment(i)).collect()
    }

    /// Mean = σ².
    pub fn mean(&self) -> f64 {
        self.sigma_sq
    }

    /// Variance = σ⁴ * c.
    pub fn variance(&self) -> f64 {
        self.sigma_sq * self.sigma_sq * self.c
    }

    /// R-transform of the Marchenko-Pastur distribution.
    /// R(z) = σ² / (1 - σ²*c*z)
    pub fn r_transform(&self, z: f64) -> f64 {
        self.sigma_sq / (1.0 - self.sigma_sq * self.c * z)
    }

    /// S-transform of the Marchenko-Pastur distribution.
    /// S(z) = 1 / (σ² * (1 + c*z))
    pub fn s_transform(&self, z: f64) -> f64 {
        1.0 / (self.sigma_sq * (1.0 + self.c * z))
    }

    /// Cauchy transform G(z) evaluated at real z.
    pub fn cauchy_transform(&self, z: f64) -> f64 {
        if self.c <= 0.0 {
            return 1.0 / z;
        }
        // G(z) = (z + σ²(c-1) - sqrt((z - λ_-)(z - λ_+))) / (2σ²c*z)
        let lam_min = self.lambda_min();
        let lam_max = self.lambda_max();

        let term = z + self.sigma_sq * (self.c - 1.0);
        let disc = (z - lam_min) * (z - lam_max);
        let sqrt_disc = if disc >= 0.0 {
            disc.sqrt()
        } else {
            // Complex, just use magnitude
            (-disc).sqrt()
        };

        // Choose the branch so that G(z) → 1/z as z → ∞
        let g1 = (term - sqrt_disc) / (2.0 * self.sigma_sq * self.c * z);
        let g2 = (term + sqrt_disc) / (2.0 * self.sigma_sq * self.c * z);

        // Pick the branch that gives |G(z)| → 1/|z| for large z
        if z.abs() > 100.0 {
            if (g1 - 1.0 / z).abs() < (g2 - 1.0 / z).abs() {
                g1
            } else {
                g2
            }
        } else {
            // For finite z, the minus sign is usually correct
            g1
        }
    }

    /// Point mass at 0 when c > 1: weight = 1 - 1/c.
    pub fn point_mass_at_zero(&self) -> f64 {
        if self.c > 1.0 {
            1.0 - 1.0 / self.c
        } else {
            0.0
        }
    }

    /// Quantile function via numerical inversion.
    pub fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return self.lambda_min();
        }
        if p >= 1.0 {
            return self.lambda_max();
        }

        let mut lo = self.lambda_min();
        let mut hi = self.lambda_max();
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mp_support_c1() {
        let mp = MarchenkoPasturLaw::new(1.0);
        assert_relative_eq!(mp.lambda_min(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(mp.lambda_max(), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mp_support_c_half() {
        let mp = MarchenkoPasturLaw::new(0.5);
        let sqc = 0.5_f64.sqrt();
        assert_relative_eq!(mp.lambda_min(), (1.0 - sqc).powi(2), epsilon = 1e-10);
        assert_relative_eq!(mp.lambda_max(), (1.0 + sqc).powi(2), epsilon = 1e-10);
    }

    #[test]
    fn test_mp_pdf_at_boundary() {
        let mp = MarchenkoPasturLaw::new(1.0);
        assert_relative_eq!(mp.pdf(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(mp.pdf(4.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mp_pdf_positive_in_support() {
        let mp = MarchenkoPasturLaw::new(0.5);
        for x in [0.5, 1.0, 1.5, 2.0] {
            assert!(mp.pdf(x) > 0.0, "PDF should be positive at {x}");
        }
    }

    #[test]
    fn test_mp_pdf_zero_outside_support() {
        let mp = MarchenkoPasturLaw::new(0.5);
        assert_relative_eq!(mp.pdf(0.01), 0.0);
        assert_relative_eq!(mp.pdf(5.0), 0.0);
    }

    #[test]
    fn test_mp_moments_c1() {
        let mp = MarchenkoPasturLaw::new(1.0);
        assert_relative_eq!(mp.moment(0), 1.0);
        assert_relative_eq!(mp.moment(1), 1.0); // mean = σ² = 1
        assert_relative_eq!(mp.moment(2), 2.0, epsilon = 1e-10); // 1 + c = 2
    }

    #[test]
    fn test_mp_mean_variance() {
        let mp = MarchenkoPasturLaw::new(0.5);
        assert_relative_eq!(mp.mean(), 1.0);
        assert_relative_eq!(mp.variance(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mp_mean_variance_with_sigma() {
        let mp = MarchenkoPasturLaw::with_variance(2.0, 3.0);
        assert_relative_eq!(mp.mean(), 3.0);
        assert_relative_eq!(mp.variance(), 18.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mp_r_transform() {
        let mp = MarchenkoPasturLaw::new(1.0);
        assert_relative_eq!(mp.r_transform(0.0), 1.0);
        assert_relative_eq!(mp.r_transform(0.5), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mp_s_transform() {
        let mp = MarchenkoPasturLaw::new(1.0);
        assert_relative_eq!(mp.s_transform(0.0), 1.0);
    }

    #[test]
    fn test_mp_point_mass_c_leq_1() {
        let mp = MarchenkoPasturLaw::new(0.5);
        assert_relative_eq!(mp.point_mass_at_zero(), 0.0);
    }

    #[test]
    fn test_mp_point_mass_c_gt_1() {
        let mp = MarchenkoPasturLaw::new(2.0);
        assert_relative_eq!(mp.point_mass_at_zero(), 0.5);
    }

    #[test]
    fn test_mp_cdf_at_boundaries() {
        let mp = MarchenkoPasturLaw::new(0.5);
        assert_relative_eq!(mp.cdf(mp.lambda_min()), 0.0, epsilon = 1e-6);
        assert_relative_eq!(mp.cdf(mp.lambda_max()), 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_mp_cdf_monotonic() {
        let mp = MarchenkoPasturLaw::new(0.5);
        let (lo, hi) = mp.support();
        let n = 50;
        let mut prev = 0.0;
        for i in 0..=n {
            let x = lo + (hi - lo) * i as f64 / n as f64;
            let cdf = mp.cdf(x);
            assert!(cdf >= prev - 1e-6, "CDF not monotonic at {x}");
            prev = cdf;
        }
    }

    #[test]
    fn test_mp_quantile_median() {
        let mp = MarchenkoPasturLaw::new(1.0);
        let median = mp.quantile(0.5);
        assert!(median > 0.0 && median < 4.0);
        assert_relative_eq!(mp.cdf(median), 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_mp_moments_sum_to_catalan() {
        // For c=1, m_k = C_k (Catalan numbers)
        let mp = MarchenkoPasturLaw::new(1.0);
        let catalans = [1.0, 1.0, 2.0, 5.0, 14.0];
        for (k, &c) in catalans.iter().enumerate() {
            assert_relative_eq!(mp.moment(k), c, epsilon = 1e-10);
        }
    }
}
