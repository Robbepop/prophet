#![warn(missing_docs)]

//! A neural net implementation focused on performance.
//!
//! This library features a simple interface to a neural net,
//! which exists mainly of two functions that are defined as traits within the module ```traits```.
//!
//! A neural network can be trained by giving it some input data and some expected target values.
//! This is called managed learning because the user of the library has to feed the network with the
//! expected results.
//! After several iterations the neural net might improve itself and will eventually improve at predicting the expected results.
//! 
//! ```rust,no_run
//! # use prophet::error_stats::ErrorStats;
//! # trait Disciple {
//! 	fn train(&mut self, input: &[f32], expected: &[f32]) -> ErrorStats;
//! # }
//! ```
//! 
//! After a successful training session, the user might be able to use the neural net to predict expected values.
//! 
//! ```rust,no_run
//! # trait Prophet {
//! 	fn predict(&mut self, input: &[f32]) -> &[f32];
//! # }
//! ```
//!
//! # Example
//! 
//! The code below demonstrates how to train a neural net to be a logical-OR operator.
//!
//! ```rust
//! use prophet::prelude::*;
//!
//! let config  = LearnConfig::new(
//! 	0.25,                // learning_rate
//! 	0.5,                 // learning_momentum
//! 	ActivationFn::tanh() // activation function + derivate
//! );
//! let mut net = ConvNeuralNet::new(config, &[2, 3, 2, 1]);
//! // layer_sizes: - input layer which expects two values
//! //              - two hidden layers with 3 and 2 neurons
//! //              - output layer with one neuron
//! 
//! // now train the neural net how to be an OR-operator
//! let f = -1.0; // represents false
//! let t =  1.0; // represents true
//! for _ in 0..1000 { // make some iterations
//! 	net.train(&[f, f], &[f]); // ⊥ ∧ ⊥ → ⊥
//! 	net.train(&[f, t], &[t]); // ⊥ ∧ ⊤ → ⊤
//! 	net.train(&[t, f], &[t]); // ⊤ ∧ ⊥ → ⊤
//! 	net.train(&[t, t], &[t]); // ⊤ ∧ ⊤ → ⊤
//! }
//! // check if the neural net has successfully learned it by checking how close
//! // the latest ```avg_error``` is to ```0.0```:
//! assert!(net.latest_error_stats().avg_error() < 0.05);
//! ```

extern crate rand;
extern crate num;
extern crate ndarray;
extern crate itertools;

#[cfg(test)]
extern crate time;

pub mod traits;
pub mod activation_fn;
pub mod conv_neural_net;
pub mod error_stats;
pub mod learn_config;

/// The prophet prelude publicly imports all propet modules the user needs in order to
/// create, train and use neural networks.
pub mod prelude {
	pub use traits::{Prophet, Disciple};
	pub use conv_neural_net::{ConvNeuralNet};
	pub use error_stats::{ErrorStats};
	pub use learn_config::{LearnConfig};
	pub use activation_fn::{BaseFn, DerivedFn, ActivationFn};
}
