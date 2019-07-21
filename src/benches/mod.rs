//! This module is used entirely for benchmarking internals
//! that should be excluded from running throught `cargo test`
//! which includes `#[bench]` marked crate-internal benches by default.
//!
//! So for benchmarks to run the user currently has to type in
//! `cargo bench --features run_internal_benches`.

pub use prelude::*;

pub use test::{black_box, Bencher};

mod neural_net;
