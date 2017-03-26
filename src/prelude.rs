//! The prophet prelude publicly imports all propet modules the user
//! needs in order to create, train and use neural networks.

#[doc(no_inline)]
pub use activation::Activation;

#[doc(no_inline)]
pub use neural_net::NeuralNet;

#[doc(no_inline)]
pub use traits::{Predict};

#[doc(no_inline)]
pub use topology::{Topology, TopologyBuilder, Layer};

#[doc(no_inline)]
pub use mentor::{Mentor, Sample, SampleView, LogConfig, Scheduling, Criterion};

#[doc(no_inline)]
pub use errors::{Result, ErrorKind};

// #[doc(no_inline)]
// pub use mentor2::configs::{LogConfig, Scheduling, Criterion};