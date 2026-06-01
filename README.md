# lau-free-probability-agents

**Voiculescu's free probability theory applied to agent systems — free convolution, semicircle law, Marchenko-Pastur law, R-transform, S-transform, free cumulants, free entropy, and asymptotic freeness for fleet belief matrices.**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-134-green.svg)]()

## What This Does

This crate applies **free probability theory** — the mathematics of large random matrices — to multi-agent systems. When a fleet of agents maintains belief matrices (covariance matrices, weight matrices, policy matrices), the eigenvalue distributions of these matrices follow universal laws from free probability. The crate lets you:

- **Predict eigenvalue spectra of merged fleets** without computing the actual matrix merge (via R-transforms and free convolution)
- **Classify fleet belief structure** using the semicircle law (random beliefs) and Marchenko-Pastur law (structured beliefs)
- **Compute free entropy** — the "information content" of a fleet's belief matrix
- **Verify asymptotic freeness** — whether independent subsystems become free (commute in distribution) as fleet size grows

**134 tests** cover every module from Catalan numbers through fleet belief prediction.

## Key Idea

In classical probability, independent random variables combine via classical convolution. In **free probability**, the analog is *free convolution*. Large random matrices that are independent (from different subsystems) become "freely independent" as their size grows to infinity. This means:

```
R_{X+Y}(z) = R_X(z) + R_Y(z)    (free additive convolution via R-transform)
S_{XY}(z) = S_X(z) · S_Y(z)      (free multiplicative convolution via S-transform)
```

This is incredibly powerful: you can predict the eigenvalue distribution of a merged fleet by adding/multiplying transforms, without ever computing the actual large matrix.

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-free-probability-agents = { git = "https://github.com/SuperInstance/lau-free-probability-agents" }
```

### Dependencies

- `nalgebra` — linear algebra (eigenvalues, matrices)
- `serde` / `serde_json` — serialization
- `num-complex` — complex number arithmetic for Cauchy/Stieltjes transforms

## Quick Start

```rust
use lau_free_probability_agents::*;

// The semicircle law — "free Gaussian" — eigenvalue distribution of large random symmetric matrices
let sc = SemicircleLaw::standard();
println!("Radius: {}", sc.radius());       // 2.0
println!("PDF at 0: {}", sc.pdf(0.0));     // 2/π ≈ 0.637
println!("Variance: {}", sc.variance());   // 1.0
println!("4th moment: {}", sc.moment(4));  // C₂·σ⁴ = 2

// Marchenko-Pastur law — eigenvalue distribution of sample covariance matrices
let mp = MarchenkoPasturLaw::new(0.5);  // c = p/n = 0.5
println!("Support: [{:.3}, {:.3}]", mp.lambda_min(), mp.lambda_max());
// [0.086, 2.086]

// Free cumulants from moments
let moments = vec![1.0, 0.0, 1.0, 0.0, 2.0];  // semicircle moments
let fc = FreeCumulants::from_moments(&moments);
println!("Free cumulants: {:?}", fc.cumulants);
// [0.0, 1.0, 0.0, 0.0] — only κ₂ is nonzero for semicircle

// R-transform: R(z) = Σ κₙ zⁿ⁻¹
let rt = RTransform::from_cumulants(vec![0.0, 2.0]);
assert_eq!(rt.evaluate(0.5), 1.0);  // κ₂ · z = 2.0 × 0.5

// Free additive convolution: Semicircle(1) ⊕ Semicircle(1) = Semicircle(2)
let conv = FreeConvolution::additive_semicircle(1.0, 1.0);
assert_eq!(conv.cumulants[1], 2.0);

// Fleet belief prediction
let fleet_a = FleetBelief::semicircle("subsystem-a", 100, 1.0);
let fleet_b = FleetBelief::marchenko_pastur("subsystem-b", 100, 0.5, 1.0);
let merge = fleet_a.predict_sum(&fleet_b);
// Predict eigenvalue spectrum of the merged fleet — no matrix computation needed!
```

## API Reference

### `semicircle` — Wigner Semicircle Law

The free probability analog of the Gaussian distribution.

| Method | Description |
|--------|-------------|
| `SemicircleLaw::standard()` | σ² = 1, radius = 2 |
| `SemicircleLaw::new(σ²)` | Custom variance |
| `.radius()` | R = 2σ |
| `.pdf(x)` | Density: `f(x) = (2/πR²)√(R² - x²)` |
| `.cdf(x)` | Cumulative distribution |
| `.moment(n)` | n-th moment (odd=0, even=Catalan) |
| `.moments(k)` | First k moments |

Key property: all free cumulants vanish except κ₂ = σ².

### `marchenko_pastur` — Marchenko-Pastur Law

Eigenvalue distribution of sample covariance matrices `(1/n)X^TX` where X is n×p with i.i.d. entries.

| Method | Description |
|--------|-------------|
| `MarchenkoPasturLaw::new(c)` | Aspect ratio c = p/n |
| `.lambda_min()` / `.lambda_max()` | Support bounds: `[σ²(1-√c)², σ²(1+√c)²]` |
| `.pdf(x)` | Density on the support |
| `.cdf(x)` | Cumulative distribution |
| `.moment(n)` | n-th moment |

### `cumulants` — Free Cumulants & Non-Crossing Partitions

| Type | Description |
|------|-------------|
| `Catalan` | Catalan numbers Cₙ = (1/(n+1))binom(2n,n) |
| `NonCrossingPartitions` | Generate all NC partitions of {1,...,n} |
| `FreeCumulants` | Free cumulants computed via moment-cumulant formula using NC partitions |

The moment-cumulant formula:
```
mₙ = Σ_{π ∈ NC(n)} Π_{B ∈ π} κ_{|B|}
```
where the sum is over non-crossing partitions and `κ` are free cumulants. Möbius inversion on the NC lattice gives κ from m.

### `r_transform` — R-Transform (Free Additive)

```
R(z) = κ₁ + κ₂z + κ₃z² + ...
```

| Method | Description |
|--------|-------------|
| `RTransform::from_cumulants(κ)` | Create from free cumulants |
| `RTransform::from_moments(m)` | Create via moment-cumulant formula |
| `.evaluate(z)` | R(z) |
| `.add(other)` | Free additive convolution: R_{X+Y} = R_X + R_Y |
| `.to_moments()` | Convert back to moment sequence |

### `s_transform` — S-Transform (Free Multiplicative)

```
S_{XY}(z) = S_X(z) · S_Y(z)
```

| Method | Description |
|--------|-------------|
| `STransform::from_moments(m)` | Create from moments |
| `STransform::marchenko_pastur(c, σ²)` | Exact MP S-transform: `S(z) = 1/(σ²(1 + cz))` |
| `.multiply(other)` | Free multiplicative convolution |
| `.evaluate(z)` | S(z) |

### `free_convolution` — Free Convolution Operations

| Method | Description |
|--------|-------------|
| `additive_convolution_from_moments(mₓ, m_y)` | Free additive convolution ⊕ |
| `multiplicative_convolution_from_moments(mₓ, m_y)` | Free multiplicative convolution ⊗ |
| `additive_semicircle(σ₁², σ₂²)` | SC(σ₁²) ⊕ SC(σ₂²) = SC(σ₁² + σ₂²) |
| `merge_additive(κₓ, κ_y)` | Merge fleets using additive convolution |

### `free_entropy` — Voiculescu's Free Entropy

The free analog of Shannon differential entropy for non-commutative random variables.

| Method | Description |
|--------|-------------|
| `FreeEntropy::discrete(samples)` | χ(a) = ∬log|x-y| dμ(x)dμ(y) + 3/4 + ½log(2π) |
| `FreeEntropy::semicircle(σ²)` | χ(sc) = ½log(2πeσ²) — maximizes free entropy |
| `FreeEntropy::marchenko_pastur(c)` | MP free entropy |

The semicircle law maximizes free entropy for a given variance (free CLT).

### `asymptotic_freeness` — Asymptotic Freeness Testing

| Method | Description |
|--------|-------------|
| `check_freeness_criterion(A, B, p, q)` | Test `tr(Ã^p B̃^q) ≈ 0` for centered matrices |
| `convergence_rate(sizes, σ²)` | Estimate O(1/N) convergence to freeness |
| `empirical_eigenvalues(M)` | Compute eigenvalues of a matrix |
| `eigenvalue_moments(λ, k)` | Moments of eigenvalue distribution |

Key result: a Wigner matrix and any deterministic matrix with a limiting spectrum are asymptotically free.

### `fleet` — Fleet Belief Matrix Application

| Type | Description |
|------|-------------|
| `FleetBelief` | A fleet subsystem's spectral information: moments, free cumulants, R-transform |
| `FleetBelief::from_matrix(name, M)` | Create from a belief matrix |
| `FleetBelief::semicircle(name, n, σ²)` | Model as semicircle |
| `FleetBelief::marchenko_pastur(name, n, c, σ²)` | Model as MP |
| `.predict_sum(other)` | Predict eigenvalue spectrum of merged fleet via free convolution |
| `.free_entropy()` | Free entropy of the fleet belief |

## How It Works

### 1. Model Fleet Belief Matrices

Each subsystem's belief matrix (covariance, weights, policy) has an eigenvalue distribution. In the large-N limit, this converges to a deterministic law.

### 2. Compute Free Cumulants via Non-Crossing Partitions

Free cumulants are computed by Möbius inversion on the lattice of non-crossing partitions of {1,...,n}. The number of such partitions is the n-th Catalan number Cₙ.

### 3. Build R- and S-Transforms

The R-transform encodes free cumulants as a power series and linearizes free additive convolution. The S-transform linearizes free multiplicative convolution.

### 4. Predict Merged Fleet Spectra

To predict the eigenvalue distribution when two fleets merge:
- Compute R-transforms of each fleet's spectrum
- Add them: `R_{merged} = R_A + R_B`
- Invert back to get the merged eigenvalue distribution

No large matrix multiplication required.

### 5. Verify Asymptotic Freeness

Check that independent subsystems satisfy the freeness condition: `tr(Ã^p B̃^q) → 0` as N → ∞, where Ã, B̃ are centered. Convergence rate is O(1/N).

## The Math

### Free vs. Classical Probability

| Classical | Free |
|-----------|------|
| Independent random variables | Freely independent random matrices |
| Moments factor: E[XⁿYᵐ] = E[Xⁿ]E[Yᵐ] | Only alternating centered moments vanish |
| Cumulants via all partitions | Free cumulants via non-crossing partitions |
| Fourier transform | R-transform |
| Central limit → Gaussian | Free CLT → Semicircle |
| Shannon entropy | Free entropy (Voiculescu) |

### The R-Transform

The Cauchy transform:
```
G(z) = ∫ (z - t)⁻¹ dμ(t)
```

The R-transform is the functional inverse of G, shifted:
```
G(R(z) + 1/z) = z
```

Its series expansion: `R(z) = Σ κₙ zⁿ⁻¹`

### Free Cumulants and Non-Crossing Partitions

The moment-cumulant formula:
```
mₙ = Σ_{π ∈ NC(n)} Π_{B ∈ π} κ_{|B|}
```

Example for n=2:
```
m₂ = κ₂ + κ₁²  →  κ₂ = m₂ - m₁²
```

For the semicircle: κ₁ = 0, κ₂ = σ², κₙ = 0 for n ≥ 3.

### Marchenko-Pastur Law

For X ∈ ℝ^{n×p} with i.i.d. entries (mean 0, variance σ²), eigenvalues of `(1/n)X^TX` converge to:
```
f(x) = (1/(2πσ²c)) · √((λ₊ - x)(x - λ₋)) / x
```
where λ± = σ²(1 ± √c)² and c = p/n.

### Free Entropy

For a self-adjoint variable with distribution μ:
```
χ(a) = ∬ log|x-y| dμ(x)dμ(y) + 3/4 + ½log(2π)
```

The semicircle maximizes free entropy: `χ(sc(σ²)) = ½log(2πeσ²)`.

## Project Structure

```
src/
├── lib.rs                # Crate root, re-exports
├── semicircle.rs         # Wigner semicircle law
├── marchenko_pastur.rs   # Marchenko-Pastur law
├── cumulants.rs          # Free cumulants, Catalan numbers, non-crossing partitions
├── r_transform.rs        # R-transform (free additive convolution)
├── s_transform.rs        # S-transform (free multiplicative convolution)
├── free_convolution.rs   # Free convolution operations
├── free_entropy.rs       # Voiculescu's free entropy
├── asymptotic_freeness.rs # Asymptotic freeness testing
└── fleet.rs              # Fleet belief matrix application
```

## License

MIT
