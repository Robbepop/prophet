//! Provides operations, data structures and error definitions for Disciple objects
//! which form the basis for topologies of neural networks.

use std::slice::Iter;
use activation::Activation;

/// Represents the topology element for a fully connected layer
/// with input neurons, output neurons and an activation function.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Layer {
	Input(usize),
	FullyConnected(usize),
	Activation(pub Activation, usize),
	Convolution((usize, usize)),
	Pooling((usize, usize))
}

impl Layer {
	/// Returns the len (in counts of neurons; without bias) of this layer.
	pub fn len(&self) -> usize {
		match self {
			&Input(size) => size,
			&FullyConnected(size) => size,
			&Activation(size, _) => size
		}
	}
}

/// Used to build topologies and do some minor compile-time and 
/// runtime checks to enforce validity of the topology as a shape for neural nets.
/// 
/// Can be used by `Mentor` types to train it and become a trained neural network
/// with which the user can predict data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyBuilder {
	layers: Vec<Layer>
}

impl TopologyBuilder {
	/// Fully connects to the previous layer with the given amount of neurons.
	fn fully_connect(mut self, size: usize) -> TopologyBuilder {
		assert!(size >= 1, "cannot define a zero-sized hidden layer");
		self.layers.push(Layer::FullyConnected(size));
		self
	}

	/// Pushes an activation layer to the topology.
	/// 
	/// Activation layers always have the size of the previous layer.
	fn activation(mut self, activation: Activation) -> TopologyBuilder {
		self.layers.push(Layer::Activation(self.last_len(), activation));
		self
	}

	/// Returns the length of the last pushed layer.
	/// 
	/// Useful for layers like activation layers which adopt their size
	/// from their previous layer.
	fn last_len(&self) -> usize {
		self.layers
			.last()
			.expect("a finished disciple must have a valid last layer!")
			.len()
	}

	/// Returns the fully constructed topology.
	/// 
	/// No further modifications to the topology are possible after this operation.
	fn done() -> Topology {
		Topology{ layers: self.layers }
	}
}

/// Represents the neural network topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topology {
	layers: Vec<Layer>
}

impl Topology {
	/// Creates a new topology.
	/// 
	/// # Panics
	/// 
	/// If size is zero.
	pub fn input(size: usize) -> TopologyBuilder {
		assert!(size >= 1, "cannot define a zero-sized input layer");
		TopologyBuilder{
			layers: vec![Layer::Input(size)]
		}
	}

	/// Returns the number of input neurons.
	///
	/// Used by mentors to validate their sample sizes.
	pub fn input_len(&self) -> usize {
		self.layers
			.first()
			.expect("a finished disciple must have a valid first layer!")
			.len()
	}

	/// Returns the number of output neurons.
	///
	/// Used by mentors to validate their sample sizes.
	pub fn output_len(&self) -> usize {
		self.layers
			.last()
			.expect("a finished disciple must have a valid last layer!")
			.len()
	}
}

/// Padding configuration for kernel operations in convolution layers.
enum Padding {
	/// Zero-padding
	/// 
	/// Fill empty areas with zeroes.
	Zero,

	/// Exact-padding
	/// 
	/// Ignore areas with one or more empty values entirely.
	Exact,

	/// Ignoring-padding
	/// 
	/// Ignores empty elements on kernel computation.
	Ignore
}

/// Pooling kind for the pooling operation in convolution layers.
enum Pooling {
	/// Compute the minimum element within a pool.
	Min,

	/// Compute the maximum element within a pool
	Max,

	/// Compute the mean amongst all elements within a pool.
	Mean
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn topology_2d() {
		use PoolingKind::Max;
		use self::Activation::{Logistic, Identity, ReLU, Tanh};
		let top = Topology::input2d((120, 80))
			.convolution((10, 10)) // PARAMS: conv-size, feature-count, padding
			.pooling((5, 5))       // PARAMS: pool-size, pooling-kind, step-size
			                       // FEATURE SIZE: 24 x 16
			.activation(ReLU)      // Activation works for 1d and 2d inputs!

			.convolution((8, 8))
			.pooling((4, 4))       // FEATURE SIZE: 6 x 4
			.activation(ReLU)

			.convolution((3, 3))
			.pooling((3, 2))       // FEATURE SIZE: 2 x 2
			.activation(ReLU)

			.pooling((2, 2))       // FEATURE SIZE: 1 x 1
			.activation(ReLU)

			.fully_connect(5) // Note: First call to `fully_connect` after
			                  // a series of convolution will convert the 1x1-sized
			                  // feature maps into a one-dimensional array that
			                  // can be fully connected against.
			                  // This operation is implicit.
			                  // Panics if the feature maps aren't of size 1x1.
			.activation(Logistic)
			.fully_connect(10).activation(Identity)
			.fully_connect(10).activation(ReLU)
			.fully_connect( 1).activation(Tanh)

			.done() // Checks validity of topology and makes it immutable.

		assert_eq!(top.input_len() , 2);
		assert_eq!(top.output_len(), 1);
	}

	#[test]
	fn topology_1d() {
		use self::Activation::{ReLU, Tanh};
		let top = Topology::input(2)
			.fully_connect( 5).activation(ReLU)
			.fully_connect(10).activation(ReLU)
			.fully_connect(10).activation(ReLU)
			.fully_connect( 5).activation(Tanh)
			.done()

		assert_eq!(top.input_len() , 2);
		assert_eq!(top.output_len(), 5);
	}
}
