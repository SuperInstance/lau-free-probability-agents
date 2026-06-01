//! # lau-free-probability-agents
//!
//! Voiculescu's free probability theory applied to agent systems.
//!
//! In classical probability, independent random variables have product distributions.
//! In free probability, "free" random matrices have their own multiplication rule —
//! the free convolution. This is the mathematics of large random matrices, which is
//! exactly what fleet belief matrices become.
//!
//! # Core Concepts
//!
//! - **Free convolution**: R-transform (additive) and S-transform (multiplicative)
//! - **Semicircle law**: The "free Gaussian" — eigenvalue distribution of large random symmetric matrices
//! - **Marchenko-Pastur law**: Eigenvalue distribution of sample covariance (agent belief covariance)
//! - **Free cumulants**: Non-crossing partitions, Catalan numbers
//! - **Free entropy**: Voiculescu's entropy for non-commutative random variables
//! - **Asymptotic freeness**: Two independent fleet subsystems become free as fleet size → ∞

pub mod cumulants;
pub mod semicircle;
pub mod marchenko_pastur;
pub mod r_transform;
pub mod s_transform;
pub mod free_convolution;
pub mod free_entropy;
pub mod asymptotic_freeness;
pub mod fleet;

/// Re-export core types
pub use cumulants::{FreeCumulants, Catalan, NonCrossingPartitions};
pub use semicircle::SemicircleLaw;
pub use marchenko_pastur::MarchenkoPasturLaw;
pub use r_transform::RTransform;
pub use s_transform::STransform;
pub use free_convolution::FreeConvolution;
pub use free_entropy::FreeEntropy;
pub use asymptotic_freeness::AsymptoticFreeness;
pub use fleet::FleetBelief;
