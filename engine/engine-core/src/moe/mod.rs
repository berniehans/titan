//! MoE expert-streaming module (Phase 7).
//!
//! Sub-modules:
//! - `profile`: Hardware bandwidth profiles (`benchbw.json`), policy resolvers, and fetch fraction calculators.
//! - `benchbw`: STREAM, linear PCIe, and overlapped CPU/PCIe bandwidth profiling micro-benchmarks.

pub mod benchbw;
pub mod budget_planner;
pub mod cpu_executor;
pub mod expert_bank;
pub mod profile;
pub mod slot_cache;

pub use benchbw::*;
pub use budget_planner::*;
pub use cpu_executor::*;
pub use expert_bank::*;
pub use profile::*;
pub use slot_cache::*;
