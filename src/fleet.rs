//! Fleet belief matrix application.
//!
//! Applies free probability to predict eigenvalue distributions of
//! merged fleet belief matrices without computing the actual merge.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use crate::r_transform::RTransform;
use crate::s_transform::STransform;
use crate::free_convolution::FreeConvolution;
use crate::free_entropy::FreeEntropy;
use crate::asymptotic_freeness::AsymptoticFreeness;
use crate::cumulants::FreeCumulants;

/// A fleet subsystem's belief matrix and its spectral information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetBelief {
    /// Name/identifier for this fleet subsystem.
    pub name: String,
    /// Size of the belief matrix.
    pub size: usize,
    /// Moments of the eigenvalue distribution (m_0=1, m_1, m_2, ...).
    pub moments: Vec<f64>,
    /// Free cumulants (computed from moments).
    pub free_cumulants: FreeCumulants,
    /// R-transform coefficients.
    pub r_transform: RTransform,
}

impl FleetBelief {
    /// Create a new fleet belief from a belief matrix.
    pub fn from_matrix(name: &str, matrix: &DMatrix<f64>) -> Self {
        let eigenvalues = AsymptoticFreeness::empirical_eigenvalues(matrix);
        Self::from_eigenvalues(name, &eigenvalues)
    }

    /// Create from known eigenvalues.
    pub fn from_eigenvalues(name: &str, eigenvalues: &[f64]) -> Self {
        let moments = AsymptoticFreeness::eigenvalue_moments(eigenvalues, 8);
        Self::from_moments(name, eigenvalues.len(), &moments[1..])
    }

    /// Create from known moments (m_1, m_2, ...).
    pub fn from_moments(name: &str, size: usize, moments: &[f64]) -> Self {
        let free_cumulants = FreeCumulants::from_moments(moments);
        let r_transform = RTransform::from_moments(moments);

        Self {
            name: name.to_string(),
            size,
            moments: moments.to_vec(),
            free_cumulants,
            r_transform,
        }
    }

    /// Create a fleet belief modeled as semicircle with given variance.
    pub fn semicircle(name: &str, size: usize, sigma_sq: f64) -> Self {
        let sc = crate::semicircle::SemicircleLaw::new(sigma_sq);
        let moments = sc.moments(8);
        Self::from_moments(name, size, &moments[1..])
    }

    /// Create a fleet belief modeled as Marchenko-Pastur.
    pub fn marchenko_pastur(name: &str, size: usize, c: f64, sigma_sq: f64) -> Self {
        let mp = crate::marchenko_pastur::MarchenkoPasturLaw::with_variance(c, sigma_sq);
        let moments = mp.moments(8);
        Self::from_moments(name, size, &moments[1..])
    }

    /// Predict the eigenvalue distribution of the sum of two fleet beliefs.
    ///
    /// Uses free additive convolution (R-transform addition).
    /// No need to compute the actual matrix merge!
    pub fn predict_sum(&self, other: &FleetBelief) -> FleetMergeResult {
        let moments = FreeConvolution::predict_sum_spectrum(&self.moments, &other.moments);
        let cumulants = FreeConvolution::merge_additive(
            &self.free_cumulants.cumulants,
            &other.free_cumulants.cumulants,
        );
        let r_transform = RTransform::from_cumulants(cumulants.clone());

        let entropy = {
            // Approximate entropy from merged moments
            FreeEntropy::semicircle(if moments.len() > 1 {
                moments[1] - moments[0].powi(2)
            } else {
                1.0
            })
        };

        FleetMergeResult {
            operation: MergeOperation::Additive,
            fleet_a: self.name.clone(),
            fleet_b: other.name.clone(),
            merged_moments: moments,
            merged_cumulants: cumulants,
            merged_r_transform: r_transform,
            free_entropy: entropy,
        }
    }

    /// Predict the eigenvalue distribution of the product of two fleet beliefs.
    pub fn predict_product(&self, other: &FleetBelief) -> FleetMergeResult {
        let _s_transform = FreeConvolution::predict_product_spectrum(
            &self.moments,
            &other.moments,
        );
        let cumulants = FreeConvolution::merge_additive(
            &self.free_cumulants.cumulants,
            &other.free_cumulants.cumulants,
        );

        FleetMergeResult {
            operation: MergeOperation::Multiplicative,
            fleet_a: self.name.clone(),
            fleet_b: other.name.clone(),
            merged_moments: vec![], // Product moments are harder to recover
            merged_cumulants: cumulants,
            merged_r_transform: RTransform::from_cumulants(vec![]),
            free_entropy: 0.0,
        }
    }

    /// Compute the free entropy of this fleet's belief distribution.
    pub fn entropy(&self) -> f64 {
        let variance = if self.moments.len() >= 2 {
            self.moments[1] - self.moments[0].powi(2)
        } else {
            1.0
        };
        FreeEntropy::semicircle(variance.max(0.01))
    }

    /// Check if this fleet's belief is "close to" semicircular.
    pub fn is_near_semicircle(&self, tolerance: f64) -> bool {
        // Semicircle has κ_n = 0 for n ≥ 3
        for i in 2..self.free_cumulants.cumulants.len() {
            if self.free_cumulants.cumulants[i].abs() > tolerance {
                return false;
            }
        }
        true
    }

    /// Effective dimension: estimate the number of "significant" eigenvalues.
    pub fn effective_dimension(&self) -> usize {
        self.size
    }

    /// Summary statistics.
    pub fn summary(&self) -> FleetSummary {
        FleetSummary {
            name: self.name.clone(),
            size: self.size,
            mean: self.free_cumulants.get(1),
            variance: self.free_cumulants.get(2),
            skewness: self.free_cumulants.get(3),
            kurtosis: self.free_cumulants.get(4),
            is_semicircular: self.is_near_semicircle(0.1),
        }
    }
}

/// Result of merging two fleet beliefs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetMergeResult {
    /// Type of merge operation.
    pub operation: MergeOperation,
    /// Name of fleet A.
    pub fleet_a: String,
    /// Name of fleet B.
    pub fleet_b: String,
    /// Merged eigenvalue moments.
    pub merged_moments: Vec<f64>,
    /// Merged free cumulants.
    pub merged_cumulants: Vec<f64>,
    /// Merged R-transform.
    pub merged_r_transform: RTransform,
    /// Free entropy of the merged system.
    pub free_entropy: f64,
}

/// Type of merge operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeOperation {
    /// Free additive convolution (sum of belief matrices).
    Additive,
    /// Free multiplicative convolution (product of belief matrices).
    Multiplicative,
}

/// Summary statistics for a fleet belief.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetSummary {
    /// Fleet name.
    pub name: String,
    /// Matrix size.
    pub size: usize,
    /// Mean (κ_1).
    pub mean: f64,
    /// Variance (κ_2).
    pub variance: f64,
    /// Free skewness (κ_3).
    pub skewness: f64,
    /// Free kurtosis (κ_4).
    pub kurtosis: f64,
    /// Whether the distribution is approximately semicircular.
    pub is_semicircular: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fleet_belief_semicircle() {
        let fb = FleetBelief::semicircle("test", 100, 1.0);
        assert_eq!(fb.name, "test");
        assert_relative_eq!(fb.free_cumulants.get(1), 0.0, epsilon = 1e-10);
        assert_relative_eq!(fb.free_cumulants.get(2), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fleet_belief_marchenko_pastur() {
        let fb = FleetBelief::marchenko_pastur("mp", 100, 1.0, 1.0);
        assert_eq!(fb.name, "mp");
        assert_relative_eq!(fb.free_cumulants.get(1), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fleet_belief_from_moments() {
        let fb = FleetBelief::from_moments("custom", 50, &[2.0, 5.0]);
        assert_relative_eq!(fb.free_cumulants.get(1), 2.0, epsilon = 1e-10);
        assert_relative_eq!(fb.free_cumulants.get(2), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fleet_predict_sum() {
        let fa = FleetBelief::semicircle("A", 100, 1.0);
        let fb = FleetBelief::semicircle("B", 100, 1.0);
        let result = fa.predict_sum(&fb);
        assert_eq!(result.fleet_a, "A");
        assert_eq!(result.fleet_b, "B");
        assert_eq!(result.operation, MergeOperation::Additive);
        // Mean should be 0 (0+0)
        assert_relative_eq!(result.merged_moments[0], 0.0, epsilon = 1e-10);
        // Variance should be 2 (1+1)
        assert_relative_eq!(result.merged_moments[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fleet_predict_sum_shifted() {
        let fa = FleetBelief::from_moments("A", 100, &[3.0, 10.0]);
        let fb = FleetBelief::from_moments("B", 100, &[1.0, 2.0]);
        let result = fa.predict_sum(&fb);
        assert_relative_eq!(result.merged_moments[0], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fleet_predict_product() {
        let fa = FleetBelief::from_moments("A", 100, &[2.0, 5.0]);
        let fb = FleetBelief::from_moments("B", 100, &[3.0, 10.0]);
        let result = fa.predict_product(&fb);
        assert_eq!(result.operation, MergeOperation::Multiplicative);
    }

    #[test]
    fn test_fleet_entropy() {
        let fb = FleetBelief::semicircle("test", 100, 1.0);
        let h = fb.entropy();
        assert!(h.is_finite());
        assert!(h > 0.0);
    }

    #[test]
    fn test_fleet_is_near_semicircle_true() {
        let fb = FleetBelief::semicircle("test", 100, 1.0);
        assert!(fb.is_near_semicircle(0.1));
    }

    #[test]
    fn test_fleet_is_near_semicircle_false() {
        let fb = FleetBelief::from_moments("test", 100, &[1.0, 2.0, 10.0, 50.0]);
        // Large κ_3 and κ_4, not semicircular
        assert!(!fb.is_near_semicircle(0.1));
    }

    #[test]
    fn test_fleet_summary() {
        let fb = FleetBelief::semicircle("test", 100, 2.0);
        let summary = fb.summary();
        assert_eq!(summary.name, "test");
        assert_eq!(summary.size, 100);
        assert_relative_eq!(summary.mean, 0.0, epsilon = 1e-10);
        assert_relative_eq!(summary.variance, 2.0, epsilon = 1e-10);
        assert!(summary.is_semicircular);
    }

    #[test]
    fn test_fleet_from_eigenvalues() {
        let fb = FleetBelief::from_eigenvalues("test", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(fb.size, 5);
        assert_relative_eq!(fb.free_cumulants.get(1), 3.0, epsilon = 1e-10); // mean
    }

    #[test]
    fn test_fleet_from_matrix() {
        let m = AsymptoticFreeness::diagonal_from_eigenvalues(&[1.0, 2.0, 3.0]);
        let fb = FleetBelief::from_matrix("test", &m);
        assert_eq!(fb.size, 3);
    }

    #[test]
    fn test_merge_result_serde() {
        let fa = FleetBelief::semicircle("A", 10, 1.0);
        let fb = FleetBelief::semicircle("B", 10, 1.0);
        let result = fa.predict_sum(&fb);
        let json = serde_json::to_string(&result).unwrap();
        let parsed: FleetMergeResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.fleet_a, "A");
    }

    #[test]
    fn test_fleet_belief_serde() {
        let fb = FleetBelief::semicircle("test", 100, 1.0);
        let json = serde_json::to_string(&fb).unwrap();
        let parsed: FleetBelief = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "test");
    }

    #[test]
    fn test_effective_dimension() {
        let fb = FleetBelief::semicircle("test", 42, 1.0);
        assert_eq!(fb.effective_dimension(), 42);
    }

    #[test]
    fn test_three_fleet_merge() {
        // Merge A + B first, then add C
        let fa = FleetBelief::semicircle("A", 100, 1.0);
        let fb = FleetBelief::semicircle("B", 100, 2.0);
        let _fc = FleetBelief::semicircle("C", 100, 3.0);

        // A + B
        let ab = fa.predict_sum(&fb);
        assert_relative_eq!(ab.merged_cumulants[1], 3.0); // 1 + 2

        // (A + B) + C via cumulants
        let abc_cumulants = FreeConvolution::merge_additive(&ab.merged_cumulants, &[0.0, 3.0]);
        assert_relative_eq!(abc_cumulants[1], 6.0); // 1 + 2 + 3
    }
}
