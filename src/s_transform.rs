//! S-transform — the free analog of the moment-generating function.
//!
//! The S-transform is used for free multiplicative convolution.
//! If X and Y are freely independent, then:
//!   S_{XY}(z) = S_X(z) * S_Y(z)

use serde::{Deserialize, Serialize};
use crate::cumulants::FreeCumulants;

/// S-transform of a non-commutative random variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STransform {
    /// Coefficients of the polynomial expansion.
    pub coefficients: Vec<f64>,
    /// Whether this is a Marchenko-Pastur rational S-transform.
    #[serde(default)]
    is_mp: bool,
    #[serde(default)]
    mp_c: f64,
    #[serde(default)]
    mp_sigma_sq: f64,
}

impl STransform {
    /// Create from moments.
    pub fn from_moments(moments: &[f64]) -> Self {
        if moments.is_empty() || moments[0].abs() < 1e-15 {
            return Self::empty();
        }

        let fc = FreeCumulants::from_moments(moments);
        Self::from_free_cumulants(&fc)
    }

    /// Create from free cumulants.
    pub fn from_free_cumulants(cumulants: &FreeCumulants) -> Self {
        if cumulants.cumulants.is_empty() || cumulants.cumulants[0].abs() < 1e-15 {
            return Self::empty();
        }

        let kappa1 = cumulants.cumulants[0];
        let n = cumulants.cumulants.len();

        let mut coeffs = vec![1.0 / kappa1];

        if n >= 2 {
            let kappa2 = cumulants.cumulants[1];
            coeffs.push(-kappa2 / (kappa1 * kappa1 * kappa1));
        }

        if n >= 3 {
            let kappa2 = cumulants.cumulants[1];
            let kappa3 = cumulants.cumulants[2];
            coeffs.push((2.0 * kappa2 * kappa2 / kappa1 - kappa3) / kappa1.powi(4));
        }

        Self {
            coefficients: coeffs,
            is_mp: false,
            mp_c: 0.0,
            mp_sigma_sq: 0.0,
        }
    }

    /// Create the S-transform for the Marchenko-Pastur distribution.
    /// S(z) = 1 / (σ² * (1 + c*z))
    pub fn marchenko_pastur(c: f64, sigma_sq: f64) -> Self {
        Self {
            coefficients: vec![1.0 / sigma_sq, -c / sigma_sq],
            is_mp: true,
            mp_c: c,
            mp_sigma_sq: sigma_sq,
        }
    }

    fn empty() -> Self {
        Self {
            coefficients: vec![],
            is_mp: false,
            mp_c: 0.0,
            mp_sigma_sq: 0.0,
        }
    }

    /// Evaluate the S-transform at z.
    pub fn evaluate(&self, z: f64) -> f64 {
        if self.is_mp {
            1.0 / (self.mp_sigma_sq * (1.0 + self.mp_c * z))
        } else {
            let mut result = 0.0;
            let mut z_power = 1.0;
            for &coeff in &self.coefficients {
                result += coeff * z_power;
                z_power *= z;
            }
            result
        }
    }

    /// S(0) = 1/mean = 1/κ_1.
    pub fn s_at_zero(&self) -> f64 {
        self.coefficients.first().copied().unwrap_or(0.0)
    }

    /// Free multiplicative convolution: S_{XY}(z) = S_X(z) * S_Y(z).
    pub fn multiply(&self, other: &STransform) -> STransform {
        if self.coefficients.is_empty() || other.coefficients.is_empty() {
            return Self::empty();
        }

        // If both are MP, the product is also rational
        if self.is_mp && other.is_mp {
            // S_X * S_Y = 1/(σ1²(1+c1*z)) * 1/(σ2²(1+c2*z))
            // = 1/(σ1²σ2²(1+c1*z)(1+c2*z))
            // Not a simple MP form, so use polynomial
        }

        let n = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![0.0; n];

        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                if i + j < n {
                    result[i + j] += a * b;
                }
            }
        }

        while result.len() > 1 && result.last().copied().unwrap_or(0.0).abs() < 1e-15 {
            result.pop();
        }

        Self {
            coefficients: result,
            is_mp: false,
            mp_c: 0.0,
            mp_sigma_sq: 0.0,
        }
    }

    /// Number of coefficients.
    pub fn order(&self) -> usize {
        self.coefficients.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_s_transform_from_moments_s0() {
        let s = STransform::from_moments(&[3.0, 10.0]);
        assert_relative_eq!(s.s_at_zero(), 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_s_transform_from_moments_zero_mean() {
        let s = STransform::from_moments(&[0.0, 1.0]);
        assert!(s.coefficients.is_empty());
    }

    #[test]
    fn test_s_transform_mp() {
        let s = STransform::marchenko_pastur(1.0, 1.0);
        assert_relative_eq!(s.evaluate(0.0), 1.0);
        assert_relative_eq!(s.evaluate(1.0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(s.evaluate(2.0), 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_s_transform_mp_c_half() {
        let s = STransform::marchenko_pastur(0.5, 1.0);
        assert_relative_eq!(s.evaluate(0.0), 1.0);
        assert_relative_eq!(s.evaluate(1.0), 1.0 / 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_s_transform_multiplicative_convolution() {
        let s1 = STransform::marchenko_pastur(1.0, 1.0);
        let s2 = STransform::marchenko_pastur(1.0, 1.0);
        let product = s1.multiply(&s2);
        assert_relative_eq!(product.evaluate(0.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_s_transform_from_cumulants() {
        let fc = FreeCumulants {
            cumulants: vec![2.0, 1.0],
        };
        let s = STransform::from_free_cumulants(&fc);
        assert_relative_eq!(s.s_at_zero(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_s_transform_evaluate_zero() {
        let s = STransform::from_moments(&[4.0, 20.0, 100.0]);
        assert_relative_eq!(s.evaluate(0.0), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_s_transform_order() {
        let s = STransform::marchenko_pastur(1.0, 1.0);
        assert_eq!(s.order(), 2);
    }

    #[test]
    fn test_s_transform_multiply_preserves_identity() {
        let identity = STransform {
            coefficients: vec![1.0],
            is_mp: false,
            mp_c: 0.0,
            mp_sigma_sq: 0.0,
        };
        let s = STransform::marchenko_pastur(1.0, 1.0);
        let product = identity.multiply(&s);
        assert_relative_eq!(product.evaluate(0.0), s.evaluate(0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_s_transform_from_moments_positive_mean() {
        let moments = vec![5.0, 30.0, 205.0];
        let s = STransform::from_moments(&moments);
        assert_relative_eq!(s.s_at_zero(), 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_s_transform_empty() {
        let s = STransform::from_moments(&[]);
        assert!(s.coefficients.is_empty());
        assert_eq!(s.order(), 0);
    }

    #[test]
    fn test_s_transform_mp_with_variance() {
        let s = STransform::marchenko_pastur(1.0, 2.0);
        assert_relative_eq!(s.evaluate(0.0), 0.5, epsilon = 1e-10);
    }
}
