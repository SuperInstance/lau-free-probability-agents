//! Free cumulants, non-crossing partitions, and Catalan numbers.
//!
//! Free cumulants are the fundamental objects in free probability, playing the
//! role that classical cumulants play in classical probability. They are computed
//! via Möbius inversion on the lattice of non-crossing partitions.

use serde::{Deserialize, Serialize};

/// Catalan number computation and related utilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalan;

impl Catalan {
    /// Compute the n-th Catalan number C_n.
    ///
    /// C_n = (1/(n+1)) * binomial(2n, n)
    ///
    /// C_0=1, C_1=1, C_2=2, C_3=5, C_4=14, C_5=42, C_6=132, ...
    pub fn number(n: usize) -> u128 {
        if n == 0 {
            return 1;
        }
        // C_n = binom(2n, n) / (n+1)
        let binom = Self::binomial(2 * n, n);
        binom / (n as u128 + 1)
    }

    /// Generate Catalan numbers up to C_n.
    pub fn sequence(n: usize) -> Vec<u128> {
        (0..=n).map(Self::number).collect()
    }

    /// Number of non-crossing partitions of {1, ..., n} equals C_n.
    pub fn non_crossing_partition_count(n: usize) -> u128 {
        Self::number(n)
    }

    /// Number of non-crossing pair partitions of {1, ..., 2n}.
    pub fn non_crossing_pair_partition_count(n: usize) -> u128 {
        Self::number(n)
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
}

/// Non-crossing partitions of {1, 2, ..., n}.
///
/// A partition π of {1,...,n} is non-crossing if there do not exist
/// a < b < c < d with a,c in one block and b,d in another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonCrossingPartitions;

impl NonCrossingPartitions {
    /// Generate all non-crossing partitions of {1, ..., n}.
    /// Returns partitions as vectors of blocks, where each block is a sorted Vec<usize>.
    ///
    /// Uses brute-force enumeration + filtering for correctness.
    /// Only used for small n (≤ 8) in moment-cumulant formulas.
    pub fn generate(n: usize) -> Vec<Vec<Vec<usize>>> {
        if n == 0 {
            return vec![vec![]];
        }
        let all = Self::all_partitions(n);
        all.into_iter().filter(|p| Self::is_non_crossing(p)).collect()
    }

    /// Generate all set partitions of {1, ..., n} using recursion.
    fn all_partitions(n: usize) -> Vec<Vec<Vec<usize>>> {
        if n == 0 {
            return vec![vec![]];
        }
        // Restricted growth string approach
        let mut result = Vec::new();
        let mut rg = vec![0usize; n];
        Self::enumerate_rg(&mut rg, 0, n, &mut result);
        result
    }

    fn enumerate_rg(
        rg: &mut Vec<usize>,
        pos: usize,
        n: usize,
        result: &mut Vec<Vec<Vec<usize>>>,
    ) {
        if pos == n {
            // Convert restricted growth string to partition
            let max_block = *rg.iter().max().unwrap_or(&0);
            let mut partition: Vec<Vec<usize>> = (0..=max_block).map(|_| Vec::new()).collect();
            for i in 0..n {
                partition[rg[i]].push(i + 1);
            }
            result.push(partition);
            return;
        }

        let max_so_far = if pos == 0 { 0 } else { *rg[..pos].iter().max().unwrap() };
        for b in 0..=max_so_far + 1 {
            if b > pos { break; }
            rg[pos] = b;
            Self::enumerate_rg(rg, pos + 1, n, result);
        }
    }

    /// Count non-crossing partitions of {1, ..., n}.
    pub fn count(n: usize) -> u128 {
        Catalan::number(n)
    }

    /// Check if a given partition is non-crossing.
    pub fn is_non_crossing(partition: &[Vec<usize>]) -> bool {
        for i in 0..partition.len() {
            for j in (i + 1)..partition.len() {
                if Self::blocks_cross(&partition[i], &partition[j]) {
                    return false;
                }
            }
        }
        true
    }

    fn blocks_cross(a: &[usize], b: &[usize]) -> bool {
        // Two blocks cross if there exist a1, a2 in A and b1, b2 in B
        // with a1 < b1 < a2 < b2 or b1 < a1 < b2 < a2.
        for i in 0..a.len() {
            for j in (i + 1)..a.len() {
                let (lo_a, hi_a) = if a[i] < a[j] { (a[i], a[j]) } else { (a[j], a[i]) };
                for k in 0..b.len() {
                    for l in (k + 1)..b.len() {
                        let (lo_b, hi_b) = if b[k] < b[l] { (b[k], b[l]) } else { (b[l], b[k]) };
                        if (lo_a < lo_b && lo_b < hi_a && hi_a < hi_b)
                            || (lo_b < lo_a && lo_a < hi_b && hi_b < hi_a)
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Compute the Möbius function μ(0_n, 1_n) on the lattice of non-crossing partitions.
    ///
    /// For the non-crossing partition lattice, μ(0_n, 1_n) = (-1)^(n-1) * C_{n-1}.
    pub fn moebius_function(n: usize) -> i128 {
        if n == 0 {
            return 1;
        }
        let sign: i128 = if (n - 1) % 2 == 0 { 1 } else { -1 };
        sign * Catalan::number(n - 1) as i128
    }
}

/// Free cumulants κ_n of a non-commutative random variable.
///
/// Free cumulants are defined via the moment-cumulant formula:
/// m_n = Σ_{π ∈ NC(n)} Π_{B ∈ π} κ_{|B|}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeCumulants {
    /// κ_n for n = 1, 2, 3, ... (indexed starting from 1).
    pub cumulants: Vec<f64>,
}

impl FreeCumulants {
    /// Create from a vector of moments m_1, m_2, ..., m_n.
    pub fn from_moments(moments: &[f64]) -> Self {
        let n = moments.len();
        if n == 0 {
            return Self { cumulants: vec![] };
        }

        let mut cumulants = vec![0.0; n];

        for k in 1..=n {
            let nc = NonCrossingPartitions::generate(k);

            let mut correction = 0.0;
            for partition in &nc {
                // Skip the one-block partition
                if partition.len() == 1 {
                    continue;
                }

                let mut product = 1.0;
                for block in partition {
                    let block_size = block.len();
                    product *= cumulants[block_size - 1];
                }
                correction += product;
            }

            cumulants[k - 1] = moments[k - 1] - correction;
        }

        Self { cumulants }
    }

    /// Convert free cumulants back to moments.
    pub fn to_moments(&self) -> Vec<f64> {
        let n = self.cumulants.len();
        if n == 0 {
            return vec![];
        }

        let mut moments = vec![0.0; n];

        for k in 1..=n {
            let nc = NonCrossingPartitions::generate(k);
            let mut total = 0.0;

            for partition in &nc {
                let mut product = 1.0;
                for block in partition {
                    let block_size = block.len();
                    if block_size <= n {
                        product *= self.cumulants[block_size - 1];
                    }
                }
                total += product;
            }

            moments[k - 1] = total;
        }

        moments
    }

    /// Get κ_n (1-indexed).
    pub fn get(&self, n: usize) -> f64 {
        if n == 0 || n > self.cumulants.len() {
            0.0
        } else {
            self.cumulants[n - 1]
        }
    }

    /// Free cumulants of the semicircle law with variance σ².
    pub fn semicircle(sigma_sq: f64) -> Self {
        Self {
            cumulants: vec![0.0, sigma_sq],
        }
    }

    /// Free cumulants of the Marchenko-Pastur law with parameter c.
    pub fn marchenko_pastur(c: f64, sigma_sq: f64, order: usize) -> Self {
        let mut cumulants = Vec::with_capacity(order);
        for n in 1..=order {
            cumulants.push(c.powi((n - 1) as i32) * sigma_sq.powi(n as i32));
        }
        Self { cumulants }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_catalan_numbers() {
        let expected = vec![1u128, 1, 2, 5, 14, 42, 132, 429, 1430, 4862];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(Catalan::number(i), exp, "C_{i}");
        }
    }

    #[test]
    fn test_catalan_sequence() {
        let seq = Catalan::sequence(5);
        assert_eq!(seq, vec![1, 1, 2, 5, 14, 42]);
    }

    #[test]
    fn test_nc_partition_count_matches_catalan() {
        for n in 0..=6 {
            assert_eq!(
                NonCrossingPartitions::count(n),
                Catalan::number(n),
                "NC count for n={n}"
            );
        }
    }

    #[test]
    fn test_nc_partitions_n0() {
        let parts = NonCrossingPartitions::generate(0);
        assert_eq!(parts.len(), 1);
        assert!(parts[0].is_empty());
    }

    #[test]
    fn test_nc_partitions_n1() {
        let parts = NonCrossingPartitions::generate(1);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], vec![vec![1]]);
    }

    #[test]
    fn test_nc_partitions_n2() {
        let parts = NonCrossingPartitions::generate(2);
        assert_eq!(parts.len(), 2); // C_2 = 2
    }

    #[test]
    fn test_nc_partitions_n3() {
        let parts = NonCrossingPartitions::generate(3);
        assert_eq!(parts.len(), 5); // C_3 = 5
    }

    #[test]
    fn test_nc_partitions_n4() {
        let parts = NonCrossingPartitions::generate(4);
        assert_eq!(parts.len(), 14); // C_4 = 14
    }

    #[test]
    fn test_nc_partitions_n5() {
        let parts = NonCrossingPartitions::generate(5);
        assert_eq!(parts.len(), 42); // C_5 = 42
    }

    #[test]
    fn test_nc_partitions_n6() {
        let parts = NonCrossingPartitions::generate(6);
        assert_eq!(parts.len(), 132); // C_6 = 132
    }

    #[test]
    fn test_is_non_crossing_true() {
        let partition = vec![vec![1, 2], vec![3, 4]];
        assert!(NonCrossingPartitions::is_non_crossing(&partition));
    }

    #[test]
    fn test_is_non_crossing_false() {
        let partition = vec![vec![1, 3], vec![2, 4]];
        assert!(!NonCrossingPartitions::is_non_crossing(&partition));
    }

    #[test]
    fn test_is_non_crossing_nested() {
        let partition = vec![vec![1, 4], vec![2, 3]];
        assert!(NonCrossingPartitions::is_non_crossing(&partition));
    }

    #[test]
    fn test_moebius_function() {
        assert_eq!(NonCrossingPartitions::moebius_function(0), 1);
        assert_eq!(NonCrossingPartitions::moebius_function(1), 1);
        assert_eq!(NonCrossingPartitions::moebius_function(2), -1);
        assert_eq!(NonCrossingPartitions::moebius_function(3), 2);
        assert_eq!(NonCrossingPartitions::moebius_function(4), -5);
        assert_eq!(NonCrossingPartitions::moebius_function(5), 14);
    }

    #[test]
    fn test_free_cumulants_from_moments_semicircle() {
        let moments = vec![0.0, 1.0, 0.0, 2.0, 0.0, 5.0];
        let fc = FreeCumulants::from_moments(&moments);
        assert_relative_eq!(fc.get(1), 0.0, epsilon = 1e-10);
        assert_relative_eq!(fc.get(2), 1.0, epsilon = 1e-10);
        assert_relative_eq!(fc.get(3), 0.0, epsilon = 1e-10);
        assert_relative_eq!(fc.get(4), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cumulants_roundtrip() {
        let moments = vec![1.0, 3.0, 7.0, 15.0];
        let fc = FreeCumulants::from_moments(&moments);
        let recovered = fc.to_moments();
        for (m, r) in moments.iter().zip(recovered.iter()) {
            assert_relative_eq!(*m, *r, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cumulants_empty() {
        let fc = FreeCumulants::from_moments(&[]);
        assert!(fc.cumulants.is_empty());
        assert_eq!(fc.to_moments(), Vec::<f64>::new());
    }

    #[test]
    fn test_semicircle_cumulants() {
        let fc = FreeCumulants::semicircle(1.0);
        assert_relative_eq!(fc.get(1), 0.0);
        assert_relative_eq!(fc.get(2), 1.0);
        assert_relative_eq!(fc.get(3), 0.0);
    }

    #[test]
    fn test_mp_cumulants() {
        let fc = FreeCumulants::marchenko_pastur(1.0, 1.0, 4);
        for n in 1..=4 {
            assert_relative_eq!(fc.get(n), 1.0);
        }
    }

    #[test]
    fn test_cumulants_from_simple_moments() {
        let moments = vec![3.0, 10.0];
        let fc = FreeCumulants::from_moments(&moments);
        assert_relative_eq!(fc.get(1), 3.0, epsilon = 1e-10);
        assert_relative_eq!(fc.get(2), 1.0, epsilon = 1e-10);
    }
}
