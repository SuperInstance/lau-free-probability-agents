# lau-free-probability-agents

Voiculescu's free probability theory applied to agent systems.

In classical probability, independent random variables have product distributions. In free probability, "free" random matrices have their **own** multiplication rule — the **free convolution**. This is the mathematics of large random matrices, which is exactly what fleet belief matrices become.

## Core Concepts

### Free Convolution
- **Additive (R-transform):** `R_{X+Y}(z) = R_X(z) + R_Y(z)` — combine eigenvalue spectra of summed matrices
- **Multiplicative (S-transform):** `S_{XY}(z) = S_X(z) * S_Y(z)` — combine eigenvalue spectra of multiplied matrices

### Classical Laws
- **Semicircle law:** The "free Gaussian" — eigenvalue distribution of large random symmetric matrices (Wigner matrices)
- **Marchenko-Pastur law:** Eigenvalue distribution of sample covariance matrices (agent belief covariance)

### Transforms
- **R-transform:** Free analog of the cumulant generating function
- **S-transform:** Free analog of the moment generating function
- **Free cumulants:** Computed via non-crossing partitions and Catalan numbers

### Advanced
- **Free entropy:** Voiculescu's entropy for non-commutative random variables
- **Asymptotic freeness:** Two independent fleet subsystems become free as fleet size → ∞

## Fleet Application

Predict the eigenvalue distribution of merged fleets **without computing the merge**:

```rust
use lau_free_probability_agents::{FleetBelief, FreeConvolution};

// Create fleet beliefs from known distributions
let fleet_a = FleetBelief::semicircle("sensors", 1000, 1.0);
let fleet_b = FleetBelief::marchenko_pastur("actuators", 500, 0.8, 2.0);

// Predict merged eigenvalue spectrum using free additive convolution
let merged = fleet_a.predict_sum(&fleet_b);
println!("Merged mean: {}", merged.merged_moments[0]);
println!("Merged variance: {}", merged.merged_moments[1]);
```

## Installation

```toml
[dependencies]
lau-free-probability-agents = "0.1"
```

## License

MIT
