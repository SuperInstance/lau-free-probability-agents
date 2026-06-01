//! R-transform — the free analog of the logarithm of the Fourier transform.
//!
//! The R-transform is the fundamental tool for computing free additive convolutions.
//! If X and Y are freely independent, then:
//!   R_{X+Y}(z) = R_X(z) + R_Y(z)
//!
//! The R-transform is related to the Cauchy transform G(z) via:
//!   G(R(z) + 1/z) = z   (i.e., R and G are functional inverses after shifting)

use serde::{Deserialize, Serialize};
use crate::cumulants::FreeCumulants;

/// R-transform of a non-commutative random variable.
///
/// The R-transform encodes the free cumulants:
///   R(z) = Σ_{n≥1} κ_n z^{n-1}
///
/// where κ_n are the free cumulants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RTransform {
    /// Free cumulants κ_1, κ_2, κ_3, ...
    pub cumulants: Vec<f64>,
}

impl RTransform {
    /// Create from free cumulants.
    pub fn from_cumulants(cumulants: Vec<f64>) -> Self {
        Self { cumulants }
    }

    /// Create from moments via free cumulants.
    pub fn from_moments(moments: &[f64]) -> Self {
        let fc = FreeCumulants::from_moments(moments);
        Self {
            cumulants: fc.cumulants,
        }
    }

    /// Evaluate R(z) = Σ κ_n * z^{n-1} = κ_1 + κ_2*z + κ_3*z² + ...
    pub fn evaluate(&self, z: f64) -> f64 {
        let mut result = 0.0;
        for (i, &kappa) in self.cumulants.iter().enumerate() {
            result += kappa * z.powi(i as i32);
        }
        result
    }

    /// Get the n-th free cumulant κ_n (1-indexed).
    pub fn cumulant(&self, n: usize) -> f64 {
        if n == 0 || n > self.cumulants.len() {
            0.0
        } else {
            self.cumulants[n - 1]
        }
    }

    /// Free additive convolution: R_{X+Y}(z) = R_X(z) + R_Y(z).
    ///
    /// Returns the R-transform of X + Y when X and Y are freely independent.
    pub fn add(&self, other: &RTransform) -> RTransform {
        let max_len = self.cumulants.len().max(other.cumulants.len());
        let mut result = vec![0.0; max_len];
        for i in 0..self.cumulants.len() {
            result[i] += self.cumulants[i];
        }
        for i in 0..other.cumulants.len() {
            result[i] += other.cumulants[i];
        }
        RTransform { cumulants: result }
    }

    /// Scalar multiple: R_{cX}(z) = c * R_X(cz).
    /// But for free cumulants: κ_n(cX) = c^n * κ_n(X).
    pub fn scale(&self, c: f64) -> RTransform {
        RTransform {
            cumulants: self
                .cumulants
                .iter()
                .enumerate()
                .map(|(i, &kappa)| c.powi((i + 1) as i32) * kappa)
                .collect(),
        }
    }

    /// Shift: R_{X+a}(z) = R_X(z) + a (shifts κ_1 by a).
    pub fn shift(&self, a: f64) -> RTransform {
        let mut result = self.cumulants.clone();
        if result.is_empty() {
            result.push(a);
        } else {
            result[0] += a;
        }
        RTransform { cumulants: result }
    }

    /// Recover moments from the R-transform.
    ///
    /// This uses the inverse relation: moments from free cumulants via
    /// the moment-cumulant formula.
    pub fn to_moments(&self) -> Vec<f64> {
        let fc = FreeCumulants {
            cumulants: self.cumulants.clone(),
        };
        fc.to_moments()
    }

    /// Approximate the Cauchy transform G(z) from the R-transform.
    ///
    /// G(z) satisfies: R(G(z)) + 1/G(z) = z
    /// Solve iteratively: G_{n+1}(z) = 1/(z - R(G_n(z)))
    pub fn to_cauchy(&self, z: f64, iterations: usize) -> f64 {
        let mut g = 1.0 / z; // initial guess
        for _ in 0..iterations {
            let r = self.evaluate(g);
            let denom = z - r;
            if denom.abs() < 1e-15 {
                break;
            }
            g = 1.0 / denom;
        }
        g
    }

    /// Number of free cumulants stored.
    pub fn order(&self) -> usize {
        self.cumulants.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_r_transform_semicircle() {
        // Semicircle: κ_1=0, κ_2=σ², κ_n=0 for n≥3
        let rt = RTransform::from_cumulants(vec![0.0, 1.0]);
        assert_relative_eq!(rt.evaluate(0.0), 0.0);
        assert_relative_eq!(rt.evaluate(1.0), 1.0); // 0 + 1*1 = 1
        assert_relative_eq!(rt.evaluate(2.0), 2.0); // 0 + 1*2 = 2
    }

    #[test]
    fn test_r_transform_from_moments() {
        // Semicircle moments: 0, 1, 0, 2
        let rt = RTransform::from_moments(&[0.0, 1.0, 0.0, 2.0]);
        assert_relative_eq!(rt.cumulant(1), 0.0, epsilon = 1e-10);
        assert_relative_eq!(rt.cumulant(2), 1.0, epsilon = 1e-10);
        assert_relative_eq!(rt.cumulant(3), 0.0, epsilon = 1e-10);
        assert_relative_eq!(rt.cumulant(4), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_r_transform_additive_convolution() {
        // Two semicircles with σ²=1 each
        let rt1 = RTransform::from_cumulants(vec![0.0, 1.0]);
        let rt2 = RTransform::from_cumulants(vec![0.0, 1.0]);
        let rt_sum = rt1.add(&rt2);
        // Sum should be semicircle with σ²=2
        assert_relative_eq!(rt_sum.cumulant(1), 0.0);
        assert_relative_eq!(rt_sum.cumulant(2), 2.0);
    }

    #[test]
    fn test_r_transform_additive_different() {
        // Semicircle(1) + semicircle(3)
        let rt1 = RTransform::from_cumulants(vec![0.0, 1.0]);
        let rt2 = RTransform::from_cumulants(vec![0.0, 3.0]);
        let rt_sum = rt1.add(&rt2);
        assert_relative_eq!(rt_sum.cumulant(2), 4.0);
    }

    #[test]
    fn test_r_transform_scale() {
        let rt = RTransform::from_cumulants(vec![1.0, 2.0, 3.0]);
        let scaled = rt.scale(2.0);
        assert_relative_eq!(scaled.cumulant(1), 2.0); // 2^1 * 1
        assert_relative_eq!(scaled.cumulant(2), 8.0); // 2^2 * 2
        assert_relative_eq!(scaled.cumulant(3), 24.0); // 2^3 * 3
    }

    #[test]
    fn test_r_transform_shift() {
        let rt = RTransform::from_cumulants(vec![1.0, 2.0]);
        let shifted = rt.shift(5.0);
        assert_relative_eq!(shifted.cumulant(1), 6.0);
        assert_relative_eq!(shifted.cumulant(2), 2.0);
    }

    #[test]
    fn test_r_transform_to_moments_roundtrip() {
        let moments = vec![2.0, 5.0, 14.0];
        let rt = RTransform::from_moments(&moments);
        let recovered = rt.to_moments();
        for (_i, (m, r)) in moments.iter().zip(recovered.iter()).enumerate() {
            assert_relative_eq!(*m, *r, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_r_transform_to_cauchy_large_z() {
        let rt = RTransform::from_cumulants(vec![0.0, 1.0]); // semicircle
        let g = rt.to_cauchy(100.0, 50);
        assert_relative_eq!(g, 0.01, epsilon = 1e-4); // 1/z for large z
    }

    #[test]
    fn test_r_transform_evaluate() {
        let rt = RTransform::from_cumulants(vec![1.0, 2.0, 3.0]);
        // R(z) = 1 + 2z + 3z²
        assert_relative_eq!(rt.evaluate(0.0), 1.0);
        assert_relative_eq!(rt.evaluate(1.0), 6.0);
        assert_relative_eq!(rt.evaluate(2.0), 17.0);
    }

    #[test]
    fn test_r_transform_empty() {
        let rt = RTransform::from_cumulants(vec![]);
        assert_relative_eq!(rt.evaluate(1.0), 0.0);
        assert_eq!(rt.order(), 0);
    }

    #[test]
    fn test_r_transform_moment_to_cauchy_consistency() {
        // Build R from moments of a shifted semicircle
        let rt = RTransform::from_cumulants(vec![2.0, 1.0]); // mean 2, var 1
        let g = rt.to_cauchy(10.0, 100);
        // G(z) ≈ 1/(z - mean) for large z
        assert_relative_eq!(g, 1.0 / 8.0, epsilon = 0.01);
    }
}
