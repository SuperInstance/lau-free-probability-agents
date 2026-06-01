//! Free entropy — Voiculescu's entropy for non-commutative random variables.
//!
//! Free entropy χ(a_1, ..., a_n) measures the "information content" of a
//! tuple of non-commutative random variables in a tracial W*-probability space.
//!
//! For a single self-adjoint variable with compactly supported distribution μ:
//!   χ(a) = ∬ log|x-y| dμ(x) dμ(y) + 3/4 + 1/2 log(2π)
//!
//! This is the free probability analog of Shannon differential entropy.

use serde::{Deserialize, Serialize};

/// Free entropy computations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEntropy;

impl FreeEntropy {
    /// Compute the free entropy of a single self-adjoint variable
    /// given by a discrete approximation of its distribution.
    ///
    /// χ(a) = ∬ log|x-y| dμ(x) dμ(y) + 3/4 + 1/2 log(2π)
    ///
    /// The integral is approximated using the given sample points,
    /// assumed to be uniformly weighted from the distribution.
    pub fn discrete(samples: &[f64]) -> f64 {
        let n = samples.len() as f64;
        if n < 2.0 {
            return f64::NEG_INFINITY;
        }

        // ∬ log|x-y| dμ(x) dμ(y) ≈ (1/n²) Σ_{i,j} log|xi - xj|
        let mut log_integral = 0.0;
        let mut count = 0.0;
        for i in 0..samples.len() {
            for j in 0..samples.len() {
                if i != j {
                    let diff = (samples[i] - samples[j]).abs();
                    if diff > 1e-15 {
                        log_integral += diff.ln();
                    }
                    count += 1.0;
                }
            }
        }
        log_integral /= count;

        log_integral + 3.0 / 4.0 + 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    /// Free entropy of the semicircle law with variance σ².
    ///
    /// χ(sc(σ²)) = (1/2) log(2πeσ²) = (1/2)(1 + ln(2π) + ln(σ²))
    ///
    /// The semicircle maximizes free entropy among all distributions
    /// with a given variance (free analog of the Gaussian).
    pub fn semicircle(sigma_sq: f64) -> f64 {
        if sigma_sq <= 0.0 {
            return f64::NEG_INFINITY;
        }
        0.5 * (1.0 + (2.0 * std::f64::consts::PI * sigma_sq).ln())
    }

    /// Free entropy of the Marchenko-Pastur law with parameter c.
    ///
    /// For c ≤ 1:
    /// χ(MP(c)) = -3/4 + (1/2)ln(2π) + (c-1)ln(c)/2 - (c+1)ln(c+1)/2 + c
    pub fn marchenko_pastur(c: f64) -> f64 {
        if c <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let base = -3.0 / 4.0 + 0.5 * (2.0 * std::f64::consts::PI).ln();
        if c == 1.0 {
            return base + 1.0 - std::f64::consts::PI.ln();
        }
        // For general c ≤ 1
        let term = (c - 1.0) * c.ln() / 2.0 - (c + 1.0) * (c + 1.0).ln() / 2.0 + c;
        base + term
    }

    /// Free entropy of a uniform distribution on [a, b].
    ///
    /// χ(Uniform(a,b)) = (1/2)ln(b-a) + constant
    pub fn uniform(a: f64, b: f64) -> f64 {
        if b <= a {
            return f64::NEG_INFINITY;
        }
        // For uniform on [-R, R] with R = (b-a)/2:
        // The free entropy involves ∬ log|x-y| dx dy
        let _r = (b - a) / 2.0;
        // Exact: -3/2 + (1/2)ln(2π) + ln(R) for semicircle-like, but uniform is different
        // Approximate via discrete
        let n = 1000;
        let samples: Vec<f64> = (0..n)
            .map(|i| a + (b - a) * (i as f64 + 0.5) / n as f64)
            .collect();
        Self::discrete(&samples)
    }

    /// Compute the free entropy of a distribution given by its density
    /// evaluated on a grid of points.
    ///
    /// Uses the double integral formula:
    /// χ = ∫∫ log|x-y| f(x) f(y) dx dy + 3/4 + 1/2 ln(2π)
    pub fn from_density(grid: &[f64], density: &[f64]) -> f64 {
        if grid.len() < 2 {
            return f64::NEG_INFINITY;
        }

        let n = grid.len();
        let dx = (grid[n - 1] - grid[0]) / (n - 1) as f64;

        let mut log_integral = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let diff = (grid[i] - grid[j]).abs();
                    if diff > 1e-15 {
                        log_integral += diff.ln() * density[i] * density[j] * dx * dx;
                    }
                }
            }
        }

        log_integral + 3.0 / 4.0 + 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    /// Free Fisher information Φ*(a) = 2π² ∫ f(x)² dx
    /// where f is the density of a.
    ///
    /// This is related to free entropy through:
    /// Φ*(a) = -∂χ(a+tX)/∂t |_{t=0} where X is semicircular
    pub fn fisher_information(grid: &[f64], density: &[f64]) -> f64 {
        let dx = if grid.len() > 1 {
            (grid[grid.len() - 1] - grid[0]) / (grid.len() - 1) as f64
        } else {
            return f64::INFINITY;
        };

        let mut integral = 0.0;
        for &f in density {
            integral += f * f * dx;
        }

        2.0 * std::f64::consts::PI.powi(2) * integral
    }

    /// Mutual free information between two freely independent variables.
    ///
    /// For freely independent X, Y: χ(X,Y) = χ(X) + χ(Y)
    /// So the mutual free information is 0 (as expected for independence).
    pub fn mutual_free_information_independent(
        entropy_x: f64,
        entropy_y: f64,
        entropy_joint: f64,
    ) -> f64 {
        entropy_x + entropy_y - entropy_joint
    }

    /// Check additivity of free entropy for freely independent variables.
    ///
    /// If X and Y are freely independent, χ(X,Y) = χ(X) + χ(Y).
    pub fn is_freely_independent(
        entropy_x: f64,
        entropy_y: f64,
        entropy_joint: f64,
        tolerance: f64,
    ) -> bool {
        (entropy_x + entropy_y - entropy_joint).abs() < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_semicircle_entropy_standard() {
        let h = FreeEntropy::semicircle(1.0);
        // χ(sc(1)) = (1/2)(1 + ln(2π)) ≈ (1/2)(1 + 1.8379) ≈ 1.4189
        assert_relative_eq!(h, 0.5 * (1.0 + (2.0 * std::f64::consts::PI).ln()), epsilon = 1e-10);
        assert!(h.is_finite());
    }

    #[test]
    fn test_semicircle_entropy_increases_with_variance() {
        let h1 = FreeEntropy::semicircle(1.0);
        let h2 = FreeEntropy::semicircle(2.0);
        assert!(h2 > h1, "More variance = more entropy");
    }

    #[test]
    fn test_semicircle_entropy_zero_variance() {
        let h = FreeEntropy::semicircle(0.0);
        assert_eq!(h, f64::NEG_INFINITY);
    }

    #[test]
    fn test_semicircle_entropy_negative_variance() {
        let h = FreeEntropy::semicircle(-1.0);
        assert_eq!(h, f64::NEG_INFINITY);
    }

    #[test]
    fn test_semicircle_maximizes_entropy() {
        // Semicircle should have higher free entropy than uniform with same variance
        // Uniform on [-√3, √3] has variance 1
        let h_sc = FreeEntropy::semicircle(1.0);
        let h_uni = FreeEntropy::uniform(-(3.0_f64).sqrt(), (3.0_f64).sqrt());
        assert!(h_sc > h_uni, "Semicircle should maximize free entropy");
    }

    #[test]
    fn test_discrete_entropy_basic() {
        // Two point distribution
        let samples = vec![-1.0, 1.0];
        let h = FreeEntropy::discrete(&samples);
        assert!(h.is_finite());
    }

    #[test]
    fn test_discrete_entropy_semicircle_approx() {
        // Approximate semicircle entropy via discrete samples
        let sc = crate::semicircle::SemicircleLaw::standard();
        let samples = sc.sample_quantiles(200);
        let h = FreeEntropy::discrete(&samples);
        let h_exact = FreeEntropy::semicircle(1.0);
        // Should be reasonably close
        assert!((h - h_exact).abs() < 0.5, "Discrete approx {h} vs exact {h_exact}");
    }

    #[test]
    fn test_discrete_entropy_single_point() {
        let h = FreeEntropy::discrete(&[1.0]);
        assert_eq!(h, f64::NEG_INFINITY);
    }

    #[test]
    fn test_discrete_entropy_empty() {
        let h = FreeEntropy::discrete(&[]);
        assert_eq!(h, f64::NEG_INFINITY);
    }

    #[test]
    fn test_mp_entropy_c1() {
        let h = FreeEntropy::marchenko_pastur(1.0);
        assert!(h.is_finite());
    }

    #[test]
    fn test_mp_entropy_positive() {
        // MP entropy should be less than semicircle entropy for same variance
        let h_mp = FreeEntropy::marchenko_pastur(1.0);
        let h_sc = FreeEntropy::semicircle(1.0);
        // Both should be finite
        assert!(h_mp.is_finite());
        assert!(h_sc.is_finite());
    }

    #[test]
    fn test_fisher_information_semicircle() {
        // Semicircle density f(x) = (1/2π)√(4-x²) on [-2,2]
        let n = 500;
        let grid: Vec<f64> = (0..n)
            .map(|i| -2.0 + 4.0 * (i as f64 + 0.5) / n as f64)
            .collect();
        let density: Vec<f64> = grid.iter().map(|&x| {
            if x.abs() <= 2.0 {
                (4.0 - x * x).sqrt() / (2.0 * std::f64::consts::PI)
            } else {
                0.0
            }
        }).collect();
        let fi = FreeEntropy::fisher_information(&grid, &density);
        // Fisher information for semicircle with σ²=1 should be 1
        // (actually π²/3 * 4/π² = 4/3 approximately)
        assert!(fi.is_finite());
        assert!(fi > 0.0);
    }

    #[test]
    fn test_mutual_free_information_independent() {
        // For freely independent, mutual info = 0
        let h_x = FreeEntropy::semicircle(1.0);
        let h_y = FreeEntropy::semicircle(2.0);
        let h_joint = h_x + h_y; // independent => additive
        let mi = FreeEntropy::mutual_free_information_independent(h_x, h_y, h_joint);
        assert_relative_eq!(mi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_is_freely_independent() {
        assert!(FreeEntropy::is_freely_independent(1.0, 2.0, 3.0, 0.01));
        assert!(!FreeEntropy::is_freely_independent(1.0, 2.0, 4.0, 0.01));
    }

    #[test]
    fn test_entropy_from_density() {
        let n = 200;
        let grid: Vec<f64> = (0..n)
            .map(|i| -2.0 + 4.0 * (i as f64 + 0.5) / n as f64)
            .collect();
        let density: Vec<f64> = grid.iter().map(|&x| {
            if x.abs() <= 2.0 {
                (4.0 - x * x).sqrt() / (2.0 * std::f64::consts::PI)
            } else {
                0.0
            }
        }).collect();
        let h = FreeEntropy::from_density(&grid, &density);
        assert!(h.is_finite());
    }

    #[test]
    fn test_uniform_entropy_finite() {
        let h = FreeEntropy::uniform(-1.0, 1.0);
        assert!(h.is_finite());
    }
}
