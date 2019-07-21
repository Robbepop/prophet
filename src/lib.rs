#![cfg_attr(all(feature = "benches", test), feature(test))]
#![warn(missing_docs)]

//! A neural net implementation focused on sequential performance.
//!
//! The API works as follows:  
//!
//! * The general shape of neural networks is defined with a topology.
//! * Topologies can be consumed by a mentor to train it with given training samples.
//! * After successful training the neural net's `predict` method can be used to predict data.
//!
//! Currently this library only supports supervised learning and fully connected layers.
//!
//! # Example
//!
//! The code below demonstrates how to train a neural net to be a logical-OR operator.
//!
//! ```rust
//! #[macro_use]
//! extern crate prophet;
//!
//! use prophet::prelude::*;
//! use Activation::Tanh;
//!
//! # fn main() {
//! let (t, f)  = (1.0, -1.0);
//! // static samples are easily generated with this macro!
//! let train_samples = samples_vec![
//! 	[f, f] => f, // ⊥ ∧ ⊥ → ⊥
//! 	[f, t] => t, // ⊥ ∧ ⊤ → ⊤
//! 	[t, f] => t, // ⊤ ∧ ⊥ → ⊤
//! 	[t, t] => t  // ⊤ ∧ ⊤ → ⊤
//! ];
//!
//! // create the topology for our neural network
//! let top = Topology::input(2) // has two input neurons
//! 	.layer(4, Tanh)          // with 4 neurons in the first hidden layer
//! 	.layer(3, Tanh)          // and 3 neurons in the second hidden layer
//! 	.output(1, Tanh);        // and 1 neuron in the output layer
//!
//! let mut net = top.train(train_samples)
//! 	.learn_rate(0.25)    // use the given learn rate
//! 	.learn_momentum(0.6) // use the given learn momentum
//! 	.log_config(LogConfig::Iterations(100)) // log state every 100 iterations
//! 	.scheduling(Scheduling::Random)         // use random sample scheduling
//! 	.criterion(Criterion::RecentMSE(0.0001))  // train until the recent MSE is below 0.0001
//!
//! 	.go()      // start the training session
//! 	.unwrap(); // be ashamed to unwrap a Result
//!
//! // PROFIT! now you can use the neural network to predict data!
//!
//! assert_eq!(net.predict(&[f, f])[0].round(), f);
//! assert_eq!(net.predict(&[f, t])[0].round(), t);
//! # }
//! ```
//!
//! # Example
//!
//! A more minimalistic example code for the same logical-OR operation:
//!
//! ```rust
//! # #[macro_use]
//! # extern crate prophet;
//! # use prophet::prelude::*;
//! # use Activation::Tanh;
//! # fn main() {
//! # let (t, f)  = (1.0, -1.0);
//! // create the topology for our neural network
//! let mut net = Topology::input(2) // has two input neurons
//! 	.layer(3, Tanh)              // with 3 neurons in the first hidden layer
//! 	.layer(2, Tanh)              // and 2 neurons in the second hidden layer
//! 	.output(1, Tanh)             // and 1 neuron in the output layer
//!
//! 	// train it for the given samples
//! 	.train(samples_vec![
//! 		[f, f] => f, // ⊥ ∧ ⊥ → ⊥
//! 		[f, t] => t, // ⊥ ∧ ⊤ → ⊤
//! 		[t, f] => t, // ⊤ ∧ ⊥ → ⊤
//! 		[t, t] => t  // ⊤ ∧ ⊤ → ⊤
//! 	])
//! 	.go()      // start the training session
//! 	.unwrap(); // and unwrap the Result
//!
//! assert_eq!(net.predict(&[f, f])[0].round(), f);
//! assert_eq!(net.predict(&[f, t])[0].round(), t);
//! # }
//! ```

extern crate rand;

#[cfg(test)]
extern crate num;

#[macro_use]
extern crate ndarray;
extern crate itertools;
extern crate ndarray_rand;

#[cfg(feature = "serde_support")]
extern crate serde;

#[cfg(feature = "serde_support")]
#[macro_use]
extern crate serde_derive;

#[macro_use]
extern crate log;

#[cfg(test)]
#[macro_use]
extern crate approx;

#[cfg(all(feature = "benches", test))]
extern crate test;

#[cfg(all(feature = "benches", test))]
mod benches;

mod activation;
mod errors;
mod neural_net;
mod traits;
mod utils;

pub(crate) mod layer;

mod nn;
pub mod trainer;

mod mentor;
pub mod topology;
pub mod topology_v4;

pub use crate::activation::Activation;
pub use crate::neural_net::NeuralNet;

pub use crate::mentor::configs::{Criterion, LogConfig, Scheduling};
pub use crate::mentor::samples::{Sample, SampleView};
pub use crate::mentor::training::{Mentor, MentorBuilder};

pub use crate::errors::{ErrorKind, Result};
pub use crate::traits::Predict;

/// The prophet prelude publicly imports all propet modules the user
/// needs in order to create, train and use neural networks.
pub mod prelude {
	#[doc(no_inline)]
	pub use crate::activation::Activation;

	#[doc(no_inline)]
	pub use crate::neural_net::NeuralNet;

	#[doc(no_inline)]
	pub use crate::traits::Predict;

	#[doc(no_inline)]
	pub use crate::topology::{Layer, Topology, TopologyBuilder};

	#[doc(no_inline)]
	pub use crate::errors::{ErrorKind, Result};

	#[doc(no_inline)]
	pub use crate::mentor::configs::{Criterion, LogConfig, Scheduling};

	#[doc(no_inline)]
	pub use crate::mentor::training::{Mentor, MentorBuilder};

	#[doc(no_inline)]
	pub use crate::mentor::samples::{Sample, SampleView};
}
