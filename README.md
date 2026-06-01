# lau-free-probability-agents

**Voiculescu's free probability theory applied to agent fleet systems** — free convolution, semicircle law, Marchenko-Pastur law, R-transform, S-transform, free cumulants via non-crossing partitions, free entropy, asymptotic freeness, and fleet belief matrix prediction.

## What This Does

In classical probability, independent random variables have distributions that combine via ordinary convolution. In **free probability** (Voiculescu), "freely independent" random matrices combine via **free convolution** — a completely different operation governed by the R-transform and S-transform. This is the mathematics of large random matrices, which is exactly what fleet belief/covariance matrices become as fleet size grows.

This crate provides:

- **Semicircle law** — the "free Gaussian": eigenvalue distribution of large random symmetric matrices (Wigner matrices)
- **Marchenko-Pastur law** — eigenvalue distribution of sample covariance matrices (agent belief covariance)
- **R-transform** — free analog of log Fourier transform; R_{X+Y}(z) = R_X(z) + R_Y(z)
- **S-transform** — for free multiplicative convolution; S_{XY}(z) = S_X(z) · S_Y(z)
- **Free cumulants** — computed via Möbius inversion on non-crossing partitions (Catalan numbers)
- **Free convolution** — additive (⊕) and multiplicative (⊗) convolution of freely independent distributions
- **Free entropy** — Voiculescu's non-commutative entropy χ(a₁,...,aₙ)
- **Asymptotic freeness** — verification and convergence rate for independent fleet subsystems
- **Fleet belief matrices** — predict eigenvalue distributions of merged fleets *without computing the actual merge*

## Key Idea

When you have N agents, each with a d×d belief/covariance matrix, and you want to merge two fleets of size N→∞, the merged eigenvalue distribution is determined entirely by the **free cumulants** of each fleet. You don't need to compute the actual merged matrix — you just add the R-transforms (additive merge) or multiply the S-transforms (multiplicative merge).

This is the same mathematics that governs:
- Eigenvalues of large random matrices (random matrix theory)
- Spectral properties of wireless MIMO channels
- Risk in large financial portfolios
- Population dynamics in large ecosystems

## Install

```toml
[dependencies]
lau-free-probability-agents = "0.1.0"
```

## Quick Start

```rust
use lau_free_probability_agents::*;

// Two fleet subsystems with different belief distributions
let fleet_a = FleetBelief::semicircle("sensors", 100, 1.0);
let fleet_b = FleetBelief::marchenko_pastur("actuators", 100, 0.5, 2.0);

// Predict the eigenvalue distribution of their merge
// WITHOUT computing the actual merged matrix!
let merge_result = fleet_a.predict_sum(&fleet_b);
println!("Merged moments: {:?}", merge_result.moments);
println!("Merged free entropy: {:.4}", merge_result.entropy);

// The semicircle law (free Gaussian)
let sc = SemicircleLaw::new(1.0);
println!("2nd moment (variance): {:.4}", sc.moment(2)); // = σ² = 1.0
println!("4th moment: {:.4}", sc.moment(4));           // = 2σ⁴ = 2.0

// Marchenko-Pastur for agent covariance matrices
let mp = MarchenkoPasturLaw::new(0.5);
println!("Support: [{:.3}, {:.3}]", mp.lambda_min(), mp.lambda_max());

// Free cumulants from moments
let fc = FreeCumulants::from_moments(&[0.0, 1.0, 0.0, 2.0]);
println!("κ₁={:?}, κ₂={:?}", fc.cumulant(1), fc.cumulant(2));
```

## API Reference

### `semicircle` — Wigner Semicircle Law

| Type | Description |
|------|-------------|
| `SemicircleLaw` | The "free Gaussian" with variance σ². Radius R = 2σ. |

**Methods:**
- `standard()`, `new(sigma_sq)`
- `radius() → f64`, `sigma() → f64`, `support() → (f64, f64)`
- `pdf(x) → f64` — Density: (2/πR²)√(R²−x²)
- `cdf(x) → f64` — Exact closed form
- `moment(n) → f64` — n-th moment (0 for odd n, Catalan × σ^n for even n)
- `moments(k) → Vec<f64>` — First k moments
- `variance() → f64`, `mean() → f64`
- `sample(num_samples) → Vec<f64>`

---

### `marchenko_pastur` — Marchenko-Pastur Law

| Type | Description |
|------|-------------|
| `MarchenkoPasturLaw` | Eigenvalue distribution of sample covariance, aspect ratio c = p/n. |

**Methods:**
- `new(c)`, `with_variance(c, sigma_sq)`
- `lambda_min() → f64`, `lambda_max() → f64` — Support bounds: σ²(1±√c)²
- `support() → (f64, f64)`, `has_point_mass() → bool`, `point_mass_weight() → f64`
- `pdf(x) → f64`, `cdf(x) → f64`
- `moment(n) → f64`, `moments(k) → Vec<f64>`
- `mean() → f64`, `variance() → f64`
- `sample(num_samples) → Vec<f64>`

---

### `r_transform` — R-Transform (Free Additive Convolution)

| Type | Description |
|------|-------------|
| `RTransform` | R(z) = Σ κₙ z^{n−1}. Free analog of log characteristic function. |

**Methods:**
- `from_cumulants(cumulants)`, `from_moments(moments)`
- `evaluate(z) → f64` — Evaluate R(z)
- `cumulant(n) → f64` — Get κₙ
- `add(&other) → RTransform` — R_{X+Y} = R_X + R_Y (free additive convolution)
- `scale(c) → RTransform` — R_{cX}(z) with κₙ(cX) = c^n κₙ(X)
- `to_moments() → Vec<f64>` — Convert back to moments

---

### `s_transform` — S-Transform (Free Multiplicative Convolution)

| Type | Description |
|------|-------------|
| `STransform` | S(z) for free multiplicative convolution. S_{XY} = S_X · S_Y. |

**Methods:**
- `from_moments(moments)`, `from_free_cumulants(&cumulants)`
- `marchenko_pastur(c, sigma_sq) → STransform` — Exact MP S-transform
- `evaluate(z) → f64`
- `multiply(&other) → STransform` — S_{XY} = S_X · S_Y
- `coefficients() → &[f64]`

---

### `cumulants` — Free Cumulants and Non-Crossing Partitions

| Type | Description |
|------|-------------|
| `FreeCumulants` | Free cumulants κ₁, κ₂, ... computed from moments via non-crossing partition Möbius inversion. |
| `Catalan` | Catalan numbers Cₙ = (1/(n+1)) binom(2n,n). |
| `NonCrossingPartitions` | Generate all non-crossing partitions of {1,...,n}. |

**`FreeCumulants`:**
- `from_moments(moments)`, `from_cumulants(cumulants)`
- `cumulant(n) → f64`, `cumulants() → &[f64]`
- `to_moments() → Vec<f64>` — Inverse transform
- `n_moments(n) → Vec<f64>` — First n moments from cumulants

**`Catalan`:**
- `number(n) → u128`, `sequence(n) → Vec<u128>`
- `non_crossing_partition_count(n) → u128`

**`NonCrossingPartitions`:**
- `generate(n) → Vec<Vec<Vec<usize>>>` — All NC partitions
- `is_non_crossing(&partition) → bool`
- `all_partitions(n) → Vec<Vec<Vec<usize>>>` — All set partitions

---

### `free_convolution` — Free Convolution Operations

| Type | Description |
|------|-------------|
| `FreeConvolution` | Static methods for additive ⊕ and multiplicative ⊗ free convolution. |

**Methods:**
- `additive_convolution_from_moments(&moments_x, &moments_y) → Vec<f64>`
- `additive_convolution_from_cumulants(&kx, &ky) → FreeCumulants`
- `multiplicative_convolution_from_moments(&moments_x, &moments_y) → STransform`
- `additive_semicircle(σ₁², σ₂²) → FreeCumulants` — SC(σ₁²) ⊕ SC(σ₂²) = SC(σ₁²+σ₂²)
- `additive_with_semicircle(&cumulants, σ²) → FreeCumulants`
- `merge_additive(&kx, &ky) → Vec<f64>` — Direct cumulant addition
- `predict_sum_spectrum(&mx, &my) → Vec<f64>` — Predict merged eigenvalue moments

---

### `free_entropy` — Voiculescu's Free Entropy

| Type | Description |
|------|-------------|
| `FreeEntropy` | Static methods for computing χ(a₁,...,aₙ). |

**Methods:**
- `discrete(samples) → f64` — From empirical samples: ∬ log|x−y| dμ(x)dμ(y) + 3/4 + ½ln(2π)
- `semicircle(sigma_sq) → f64` — χ(SC) = ½(1 + ln(2πσ²))
- `marchenko_pastur(c) → f64` — Exact formula
- `uniform(a, b) → f64` — Uniform distribution on [a,b]
- `from_moments(moments) → f64` — Approximate from moment sequence
- `conditional(&joint, &marginal) → f64` — Conditional entropy

---

### `asymptotic_freeness` — Asymptotic Freeness Verification

| Type | Description |
|------|-------------|
| `AsymptoticFreeness` | Static methods for checking and analyzing asymptotic freeness. |

**Methods:**
- `check_freeness_criterion(&a, &b, power_a, power_b) → f64` — Numerical check of alternating product trace
- `convergence_rate(&sizes, sigma_sq) → Vec<(usize, f64)>` — Track convergence as N grows
- `random_wigner(n, sigma_sq) → DMatrix<f64>` — Generate Wigner random matrix
- `deterministic_diagonal(n) → DMatrix<f64>` — Deterministic diagonal test matrix
- `empirical_eigenvalues(&matrix) → Vec<f64>` — Compute eigenvalues
- `eigenvalue_moments(&eigenvalues, k) → Vec<f64>` — Moments from eigenvalues
- `predict_sum_eigenvalues(&a, &b) → Vec<f64>` — Free convolution prediction

---

### `fleet` — Fleet Belief Matrix Application

| Type | Description |
|------|-------------|
| `FleetBelief` | A fleet subsystem's belief matrix with precomputed moments, free cumulants, and R-transform. |
| `FleetMergeResult` | Result of merging two fleet beliefs: predicted moments, cumulants, R-transform, entropy. |

**`FleetBelief`:**
- `from_matrix(name, &matrix)`, `from_eigenvalues(name, &eigs)`, `from_moments(name, size, &moments)`
- `semicircle(name, size, σ²)`, `marchenko_pastur(name, size, c, σ²)` — Model-based construction
- `predict_sum(&other) → FleetMergeResult` — Free additive convolution
- `predict_product(&other) → FleetMergeResult` — Free multiplicative convolution
- `entropy() → f64` — Free entropy of this subsystem

## How It Works

1. **Moments → Free Cumulants**: Given moments m₁, m₂, ..., compute free cumulants κ₁, κ₂, ... via Möbius inversion on the lattice of non-crossing partitions. The key formula: mₙ = Σ_{π∈NC(n)} Π_{B∈π} κ_{|B|}.

2. **R-transform**: R(z) = κ₁ + κ₂z + κ₃z² + ... The free cumulants are the Taylor coefficients of R.

3. **Free Additive Convolution**: R_{X+Y}(z) = R_X(z) + R_Y(z). Just add the cumulants! This predicts the eigenvalue distribution of A + B when A and B are freely independent.

4. **S-transform**: S(z) encodes multiplicative structure. S_{XY}(z) = S_X(z) · S_Y(z). This predicts the eigenvalue distribution of AB.

5. **Asymptotic Freeness**: Wigner matrices are asymptotically free from any deterministic matrix. So random fleet belief matrices become free as fleet size → ∞.

6. **Fleet Application**: Given two fleet belief matrices, extract their eigenvalue moments, compute free cumulants, add R-transforms, and predict the merged eigenvalue distribution — all without computing the actual merged matrix.

## The Math

### Free Probability (Voiculescu)

Two non-commutative random variables X, Y in a tracial W*-probability space are **freely independent** if tr(p₁(X)q₁(Y)p₂(X)q₂(Y)...) = 0 for any centered polynomials p, q. This is the non-commutative analog of independence.

### Semicircle Law (Wigner)

The eigenvalue density of an N×N Wigner random matrix converges to the semicircle distribution f(x) = (2/πR²)√(R²−x²) as N→∞. The 2k-th moment is C_k · σ^{2k} where C_k is the k-th Catalan number. All odd moments vanish.

### Marchenko-Pastur Law

For X an n×p matrix with i.i.d. entries, the eigenvalues of (1/n)X^TX converge to the Marchenko-Pastur distribution with support [σ²(1−√c)², σ²(1+√c)²] where c = p/n.

### R-Transform

The Cauchy transform G(z) = ∫ (z−t)^{-1} dμ(t) and the R-transform are functional inverses: G(R(z) + 1/z) = z. For the semicircle: R(z) = σ²z (only κ₂ ≠ 0).

### Free Cumulants and Non-Crossing Partitions

Free cumulants κₙ are defined by the moment-cumulant formula using non-crossing partitions. The number of NC partitions of {1,...,n} is the Catalan number Cₙ. The Möbius function of the NC partition lattice gives the inversion formula.

### Free Entropy

χ(μ) = ∬ log|x−y| dμ(x) dμ(y) + 3/4 + ½log(2π). The semicircle maximizes free entropy for given variance (free analog of the Gaussian max-entropy property).

## License

MIT
