//! MoE expert-streaming module (Phase 7).
//!
//! Sub-modules:
//! - `profile`: Hardware bandwidth profiles (`benchbw.json`), policy resolvers, and fetch fraction calculators.
//! - `benchbw`: STREAM, linear PCIe, and overlapped CPU/PCIe bandwidth profiling micro-benchmarks.

pub mod benchbw;
pub mod expert_bank;
pub mod profile;
pub mod slot_cache;

pub use benchbw::*;
pub use expert_bank::*;
pub use profile::*;
pub use slot_cache::*;
