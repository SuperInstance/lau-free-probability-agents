//! Free convolution — combining free random variables.
//!
//! Free additive convolution (⊕) and free multiplicative convolution (⊗).
//! These are the free probability analogs of classical convolution for
//! independent random variables.

use serde::{Deserialize, Serialize};
use crate::r_transform::RTransform;
use crate::s_transform::STransform;
use crate::cumulants::FreeCumulants;

/// Free convolution operations for combining freely independent random variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeConvolution;

impl FreeConvolution {
    /// Free additive convolution of two distributions given by moments.
    ///
    /// If X and Y are freely independent with given moments, compute
    /// the moments of X + Y.
    ///
    /// This uses: R_{X+Y}(z) = R_X(z) + R_Y(z)
    pub fn additive_convolution_from_moments(
        moments_x: &[f64],
        moments_y: &[f64],
    ) -> Vec<f64> {
        let rt_x = RTransform::from_moments(moments_x);
        let rt_y = RTransform::from_moments(moments_y);
        let rt_sum = rt_x.add(&rt_y);
        rt_sum.to_moments()
    }

    /// Free additive convolution from free cumulants.
    pub fn additive_convolution_from_cumulants(
        cumulants_x: &[f64],
        cumulants_y: &[f64],
    ) -> FreeCumulants {
        let rt_x = RTransform::from_cumulants(cumulants_x.to_vec());
        let rt_y = RTransform::from_cumulants(cumulants_y.to_vec());
        let rt_sum = rt_x.add(&rt_y);
        FreeCumulants {
            cumulants: rt_sum.cumulants,
        }
    }

    /// Free multiplicative convolution of two distributions given by moments.
    ///
    /// If X and Y are freely independent with given moments, compute
    /// the moments of XY.
    ///
    /// This uses: S_{XY}(z) = S_X(z) * S_Y(z)
    pub fn multiplicative_convolution_from_moments(
        moments_x: &[f64],
        moments_y: &[f64],
    ) -> STransform {
        let s_x = STransform::from_moments(moments_x);
        let s_y = STransform::from_moments(moments_y);
        s_x.multiply(&s_y)
    }

    /// Free additive convolution of two semicircle laws.
    ///
    /// Semicircle(σ₁²) ⊕ Semicircle(σ₂²) = Semicircle(σ₁² + σ₂²)
    pub fn additive_semicircle(sigma1_sq: f64, sigma2_sq: f64) -> FreeCumulants {
        FreeCumulants {
            cumulants: vec![0.0, sigma1_sq + sigma2_sq],
        }
    }

    /// Free additive convolution of a semicircle with an arbitrary distribution.
    pub fn additive_with_semicircle(
        cumulants: &[f64],
        sigma_sq: f64,
    ) -> FreeCumulants {
        let rt_dist = RTransform::from_cumulants(cumulants.to_vec());
        let rt_sc = RTransform::from_cumulants(vec![0.0, sigma_sq]);
        let rt_sum = rt_dist.add(&rt_sc);
        FreeCumulants {
            cumulants: rt_sum.cumulants,
        }
    }

    /// Compute the eigenvalue distribution of A + B where A and B are
    /// freely independent (asymptotically, when they are large random matrices
    /// from independent subsystems).
    ///
    /// Returns the free cumulants of A + B.
    pub fn merge_additive(cumulants_a: &[f64], cumulants_b: &[f64]) -> Vec<f64> {
        let n = cumulants_a.len().max(cumulants_b.len());
        let mut result = vec![0.0; n];
        for (i, r) in result.iter_mut().enumerate() {
            let a = if i < cumulants_a.len() { cumulants_a[i] } else { 0.0 };
            let b = if i < cumulants_b.len() { cumulants_b[i] } else { 0.0 };
            *r = a + b;
        }
        result
    }

    /// Compute the eigenvalue distribution of A * B where A and B are
    /// freely independent.
    ///
    /// Returns the S-transform of A * B.
    pub fn merge_multiplicative(s_transform_a: &STransform, s_transform_b: &STransform) -> STransform {
        s_transform_a.multiply(s_transform_b)
    }

    /// Given two fleet belief matrices with known eigenvalue distributions
    /// (as moments), predict the eigenvalue distribution of their sum
    /// without computing the actual matrix sum.
    pub fn predict_sum_spectrum(moments_a: &[f64], moments_b: &[f64]) -> Vec<f64> {
        Self::additive_convolution_from_moments(moments_a, moments_b)
    }

    /// Given two fleet belief matrices with known eigenvalue distributions,
    /// predict the eigenvalue distribution of their product.
    pub fn predict_product_spectrum(moments_a: &[f64], moments_b: &[f64]) -> STransform {
        Self::multiplicative_convolution_from_moments(moments_a, moments_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_additive_convolution_semicircles() {
        // Semicircle(1) + Semicircle(1) = Semicircle(2)
        let moments1 = vec![0.0, 1.0, 0.0, 2.0];
        let moments2 = vec![0.0, 1.0, 0.0, 2.0];
        let result = FreeConvolution::additive_convolution_from_moments(&moments1, &moments2);
        // Mean should be 0
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);
        // Second moment should be 2
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-10);
        // Fourth moment should be C_2 * (2)^2 = 2 * 4 = 8
        assert_relative_eq!(result[3], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_additive_convolution_shifted() {
        // Shifted semicircle: mean 3, var 1
        let moments = vec![3.0, 10.0];
        let result = FreeConvolution::additive_convolution_from_moments(&moments, &moments);
        assert_relative_eq!(result[0], 6.0, epsilon = 1e-10); // means add
    }

    #[test]
    fn test_additive_from_cumulants() {
        let fc = FreeConvolution::additive_convolution_from_cumulants(
            &[0.0, 1.0],  // semicircle(1)
            &[0.0, 2.0],  // semicircle(2)
        );
        assert_relative_eq!(fc.get(1), 0.0);
        assert_relative_eq!(fc.get(2), 3.0);
    }

    #[test]
    fn test_additive_semicircle() {
        let fc = FreeConvolution::additive_semicircle(1.0, 2.0);
        assert_relative_eq!(fc.get(2), 3.0);
    }

    #[test]
    fn test_additive_with_semicircle() {
        let fc = FreeConvolution::additive_with_semicircle(&[2.0, 1.0], 1.0);
        assert_relative_eq!(fc.get(1), 2.0);
        assert_relative_eq!(fc.get(2), 2.0);
    }

    #[test]
    fn test_multiplicative_convolution_mp() {
        // MP(1,1) ⊗ MP(1,1): S(z) = 1/(1+z) * 1/(1+z) = 1/(1+z)²
        let moments1 = vec![1.0, 2.0, 5.0]; // MP(1,1) moments: C_k
        let moments2 = vec![1.0, 2.0, 5.0];
        let s_result = FreeConvolution::multiplicative_convolution_from_moments(&moments1, &moments2);
        // S(0) = 1/mean = 1 (since both have mean 1)
        // Actually S(0) = 1/m_1 = 1 for each, product = 1
        assert_relative_eq!(s_result.evaluate(0.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_merge_additive() {
        let result = FreeConvolution::merge_additive(&[1.0, 2.0], &[3.0, 4.0]);
        assert_relative_eq!(result[0], 4.0);
        assert_relative_eq!(result[1], 6.0);
    }

    #[test]
    fn test_merge_additive_different_lengths() {
        let result = FreeConvolution::merge_additive(&[1.0, 2.0, 3.0], &[4.0]);
        assert_eq!(result.len(), 3);
        assert_relative_eq!(result[0], 5.0);
        assert_relative_eq!(result[1], 2.0);
        assert_relative_eq!(result[2], 3.0);
    }

    #[test]
    fn test_predict_sum_spectrum() {
        let result = FreeConvolution::predict_sum_spectrum(
            &[0.0, 1.0],
            &[0.0, 1.0],
        );
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_product_spectrum() {
        let s = FreeConvolution::predict_product_spectrum(
            &[1.0, 2.0],
            &[1.0, 2.0],
        );
        assert_relative_eq!(s.evaluate(0.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_additive_convolution_preserves_sums() {
        // κ_1 should always add (means add for free convolution)
        let m1 = vec![5.0, 30.0];
        let m2 = vec![3.0, 10.0];
        let result = FreeConvolution::additive_convolution_from_moments(&m1, &m2);
        assert_relative_eq!(result[0], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_additive_semicircle_identity() {
        // Semicircle(0) is the zero distribution
        let fc = FreeConvolution::additive_semicircle(1.0, 0.0);
        assert_relative_eq!(fc.get(2), 1.0);
    }
}
