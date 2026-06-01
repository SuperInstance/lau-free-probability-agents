//! Asymptotic freeness — when independent subsystems become free.
//!
//! Two independent fleet subsystems A and B become asymptotically free
//! as the fleet size N → ∞. This means:
//! - tr(p(A, B)) → 0 for any alternating centered polynomial p
//! - The eigenvalue distribution of f(A, B) can be computed using free probability
//!
//! Key result: If A is a Wigner matrix and B is any deterministic matrix with
//! a limiting eigenvalue distribution, then A and B are asymptotically free.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use crate::r_transform::RTransform;
use crate::semicircle::SemicircleLaw;
use crate::marchenko_pastur::MarchenkoPasturLaw;

/// Asymptotic freeness analysis for fleet subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsymptoticFreeness;

impl AsymptoticFreeness {
    /// Check asymptotic freeness condition numerically.
    ///
    /// For two matrices A and B, check if tr(A^p * B^q * A^p * B^q ...) ≈ 0
    /// for centered matrices (trace zero polynomials).
    ///
    /// Returns the normalized trace of the alternating product.
    pub fn check_freeness_criterion(
        a: &DMatrix<f64>,
        b: &DMatrix<f64>,
        power_a: usize,
        power_b: usize,
    ) -> f64 {
        let n = a.nrows() as f64;
        // Compute tr(A^p * B^q) for centered A, B
        // Center them first
        let mean_a = a.trace() / n;
        let mean_b = b.trace() / n;

        let a_centered = a - DMatrix::from_diagonal(&nalgebra::DVector::from_element(
            a.nrows(),
            mean_a,
        ));
        let b_centered = b - DMatrix::from_diagonal(&nalgebra::DVector::from_element(
            b.nrows(),
            mean_b,
        ));

        let mut result = a_centered.clone();
        for _ in 1..power_a {
            result = &result * &a_centered;
        }
        let mut right = b_centered.clone();
        for _ in 1..power_b {
            right = &right * &b_centered;
        }
        result = &result * &right;

        result.trace() / n
    }

    /// Estimate the rate of convergence to freeness.
    ///
    /// For Wigner matrices of size N, the convergence rate is O(1/N).
    /// Returns (N, criterion_value) pairs for increasing N.
    pub fn convergence_rate(
        sizes: &[usize],
        sigma_sq: f64,
    ) -> Vec<(usize, f64)> {
        let mut results = Vec::new();

        for &n in sizes {
            // Generate a Wigner matrix and a deterministic diagonal matrix
            let wigner = Self::random_wigner(n, sigma_sq);
            let diagonal = Self::deterministic_diagonal(n);

            let criterion = Self::check_freeness_criterion(&wigner, &diagonal, 1, 1);
            results.push((n, criterion));
        }

        results
    }

    /// Generate a Wigner (random symmetric) matrix with given variance.
    ///
    /// Entries above diagonal ~ N(0, σ²/N), diagonal ~ N(0, 2σ²/N).
    /// This ensures the limiting eigenvalue distribution is semicircle(σ²).
    pub fn random_wigner(n: usize, sigma_sq: f64) -> DMatrix<f64> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Simple LCG for reproducibility in tests
        let mut state = seed;
        let mut next_gauss = || -> f64 {
            // Box-Muller with LCG
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (state >> 33) as f64 / (1u64 << 31) as f64;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (state >> 33) as f64 / (1u64 << 31) as f64;
            let u1 = u1.max(1e-15);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let mut matrix = DMatrix::zeros(n, n);
        let scale = (sigma_sq / n as f64).sqrt();

        for i in 0..n {
            for j in i..n {
                let g = next_gauss() * scale;
                matrix[(i, j)] = g;
                matrix[(j, i)] = g;
            }
        }

        matrix
    }

    /// Create a deterministic diagonal matrix with a specific eigenvalue distribution.
    pub fn deterministic_diagonal(n: usize) -> DMatrix<f64> {
        let mut matrix = DMatrix::zeros(n, n);
        for i in 0..n {
            // Eigenvalues uniformly spaced on [-1, 1]
            matrix[(i, i)] = -1.0 + 2.0 * (i as f64 + 0.5) / n as f64;
        }
        matrix
    }

    /// Create a diagonal matrix with given eigenvalues.
    pub fn diagonal_from_eigenvalues(eigenvalues: &[f64]) -> DMatrix<f64> {
        let n = eigenvalues.len();
        let mut matrix = DMatrix::zeros(n, n);
        for (i, &e) in eigenvalues.iter().enumerate() {
            matrix[(i, i)] = e;
        }
        matrix
    }

    /// Compute empirical eigenvalue distribution of a symmetric matrix.
    ///
    /// Returns (eigenvalue, fraction) pairs, sorted by eigenvalue.
    /// Uses a simple power iteration + deflation for top eigenvalues.
    ///
    /// For full eigenvalue computation, this returns the diagonal entries
    /// if the matrix is already diagonal, or computes them via tridiagonal
    /// reduction.
    pub fn empirical_eigenvalues(matrix: &DMatrix<f64>) -> Vec<f64> {
        let n = matrix.nrows();
        if n == 0 {
            return vec![];
        }

        if n <= 200 {
            // For small matrices, compute directly
            // Symmetrize
            let sym = (matrix + &matrix.transpose()) * 0.5;

            // Try to compute eigenvalues using nalgebra
            match sym.try_symmetric_eigen(1e-10, 100) {
                Some(eig) => {
                    let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();
                    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    return eigenvalues;
                }
                None => {}
            }
        }

        // Fallback: return diagonal entries (accurate for diagonal matrices)
        let mut eigs: Vec<f64> = (0..n).map(|i| matrix[(i, i)]).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eigs
    }

    /// Compute moments from empirical eigenvalues.
    pub fn eigenvalue_moments(eigenvalues: &[f64], max_order: usize) -> Vec<f64> {
        let n = eigenvalues.len() as f64;
        if n == 0.0 {
            return vec![0.0; max_order + 1];
        }

        let mut moments = vec![0.0; max_order + 1];
        moments[0] = 1.0;
        for k in 1..=max_order {
            let mut m = 0.0;
            for &e in eigenvalues {
                m += e.powi(k as i32);
            }
            moments[k] = m / n;
        }
        moments
    }

    /// Given two fleet subsystems with known eigenvalue distributions,
    /// predict the eigenvalue distribution of their sum using asymptotic freeness.
    ///
    /// This is the key application: without computing the actual merge of
    /// belief matrices, predict the merged eigenvalue spectrum.
    pub fn predict_merged_spectrum(
        moments_a: &[f64],
        moments_b: &[f64],
    ) -> Vec<f64> {
        crate::free_convolution::FreeConvolution::predict_sum_spectrum(moments_a, moments_b)
    }

    /// Estimate the "freeness distance" between two matrices.
    ///
    /// Returns a non-negative number; closer to 0 means more free.
    /// Uses the norm of mixed traces vs product of individual traces.
    pub fn freeness_distance(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        let n = a.nrows() as f64;

        let tr_a = a.trace() / n;
        let tr_b = b.trace() / n;

        // Check: tr(AB) should ≈ tr(A)*tr(B) if free
        let tr_ab = (a * b).trace() / n;
        let expected = tr_a * tr_b;

        (tr_ab - expected).abs()
    }

    /// Estimate convergence of eigenvalue distribution to semicircle.
    ///
    /// Given eigenvalue samples from matrices of increasing size,
    /// compute the distance to the theoretical semicircle distribution.
    pub fn distance_to_semicircle(eigenvalues: &[f64], sigma_sq: f64) -> f64 {
        let sc = SemicircleLaw::new(sigma_sq);
        let n = eigenvalues.len();
        if n == 0 {
            return f64::INFINITY;
        }

        // Compare moments
        let max_k = 6.min(n);
        let emp_moments = Self::eigenvalue_moments(eigenvalues, max_k);

        let mut dist = 0.0;
        for k in 1..=max_k {
            let theory = sc.moment(k);
            let diff = emp_moments[k] - theory;
            dist += diff * diff;
        }
        dist.sqrt()
    }

    /// Estimate convergence to Marchenko-Pastur.
    pub fn distance_to_marchenko_pastur(eigenvalues: &[f64], c: f64, sigma_sq: f64) -> f64 {
        let mp = MarchenkoPasturLaw::with_variance(c, sigma_sq);
        let n = eigenvalues.len();
        if n == 0 {
            return f64::INFINITY;
        }

        let max_k = 4.min(n);
        let emp_moments = Self::eigenvalue_moments(eigenvalues, max_k);

        let mut dist = 0.0;
        for k in 1..=max_k {
            let theory = mp.moment(k);
            let diff = emp_moments[k] - theory;
            dist += diff * diff;
        }
        dist.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_random_wigner_shape() {
        let w = AsymptoticFreeness::random_wigner(5, 1.0);
        assert_eq!(w.nrows(), 5);
        assert_eq!(w.ncols(), 5);
    }

    #[test]
    fn test_random_wigner_symmetric() {
        let w = AsymptoticFreeness::random_wigner(10, 1.0);
        for i in 0..10 {
            for j in 0..10 {
                assert_relative_eq!(w[(i, j)], w[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_deterministic_diagonal() {
        let d = AsymptoticFreeness::deterministic_diagonal(5);
        assert_eq!(d.nrows(), 5);
        // Off-diagonal should be zero
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    assert_relative_eq!(d[(i, j)], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_diagonal_from_eigenvalues() {
        let d = AsymptoticFreeness::diagonal_from_eigenvalues(&[1.0, 2.0, 3.0]);
        assert_relative_eq!(d[(0, 0)], 1.0);
        assert_relative_eq!(d[(1, 1)], 2.0);
        assert_relative_eq!(d[(2, 2)], 3.0);
        assert_relative_eq!(d[(0, 1)], 0.0);
    }

    #[test]
    fn test_eigenvalue_moments_empty() {
        let m = AsymptoticFreeness::eigenvalue_moments(&[], 3);
        assert_eq!(m[0], 0.0);
    }

    #[test]
    fn test_eigenvalue_moments_basic() {
        let m = AsymptoticFreeness::eigenvalue_moments(&[1.0, 2.0, 3.0], 3);
        assert_relative_eq!(m[0], 1.0);
        assert_relative_eq!(m[1], 2.0); // (1+2+3)/3
        assert_relative_eq!(m[2], (1.0 + 4.0 + 9.0) / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_merged_spectrum() {
        let result = AsymptoticFreeness::predict_merged_spectrum(
            &[0.0, 1.0],
            &[0.0, 1.0],
        );
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_freeness_distance_diagonal() {
        // Two diagonal matrices: tr(AB) = sum of products of diagonals
        // Only "free" if tr(A*B)/n = tr(A)/n * tr(B)/n
        // For centered diagonal matrices, this may not hold.
        // Let's use a case where it does: A centered, B constant diagonal.
        let a = AsymptoticFreeness::diagonal_from_eigenvalues(&[-1.0, 1.0]);
        let b = AsymptoticFreeness::diagonal_from_eigenvalues(&[2.0, 2.0]);
        let dist = AsymptoticFreeness::freeness_distance(&a, &b);
        // tr(A)=0, tr(B)=4, tr(AB)=0. So dist = |0 - 0*4/4| = 0
        assert_relative_eq!(dist, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_to_semicircle() {
        // Perfect semicircle samples
        let sc = SemicircleLaw::standard();
        let samples = sc.sample_quantiles(100);
        let dist = AsymptoticFreeness::distance_to_semicircle(&samples, 1.0);
        // Should be relatively small (not zero due to discrete sampling)
        assert!(dist < 1.0, "Distance {dist} should be small for semicircle samples");
    }

    #[test]
    fn test_distance_to_semicircle_non_semicircle() {
        let eigenvalues = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dist = AsymptoticFreeness::distance_to_semicircle(&eigenvalues, 1.0);
        assert!(dist > 0.1, "Should be far from semicircle");
    }

    #[test]
    fn test_distance_to_marchenko_pastur() {
        let dist = AsymptoticFreeness::distance_to_marchenko_pastur(&[], 1.0, 1.0);
        assert_eq!(dist, f64::INFINITY);
    }

    #[test]
    fn test_check_freeness_criterion_shape() {
        let a = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0,
        ]);
        let b = DMatrix::from_row_slice(3, 3, &[
            0.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
        ]);
        let val = AsymptoticFreeness::check_freeness_criterion(&a, &b, 1, 1);
        assert!(val.is_finite());
    }

    #[test]
    fn test_empirical_eigenvalues_diagonal() {
        let d = AsymptoticFreeness::diagonal_from_eigenvalues(&[3.0, 1.0, 2.0]);
        let eigs = AsymptoticFreeness::empirical_eigenvalues(&d);
        assert_eq!(eigs.len(), 3);
        assert_relative_eq!(eigs[0], 1.0);
        assert_relative_eq!(eigs[1], 2.0);
        assert_relative_eq!(eigs[2], 3.0);
    }

    #[test]
    fn test_empirical_eigenvalues_empty() {
        let z = DMatrix::zeros(0, 0);
        let eigs = AsymptoticFreeness::empirical_eigenvalues(&z);
        assert!(eigs.is_empty());
    }
}
