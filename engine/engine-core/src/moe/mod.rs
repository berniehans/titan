//! MoE expert-streaming module (Phase 7).
//!
//! Sub-modules:
//! - `profile`: Hardware bandwidth profiles (`benchbw.json`), policy resolvers, and fetch fraction calculators.
//! - `benchbw`: STREAM, linear PCIe, and overlapped CPU/PCIe bandwidth profiling micro-benchmarks.

pub mod benchbw;
pub mod profile;

pub use benchbw::*;
pub use profile::*;
