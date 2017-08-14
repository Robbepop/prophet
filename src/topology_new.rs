//! Provides operations, data structures and error definitions for Disciple objects
//! which form the basis for topologies of neural networks.

use activation::Activation;
use errors::{Result, Error};

/// Represents the topology element for a fully connected layer
/// with input neurons, output neurons and an activation function.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Layer {
	/// The input layer of a neural network.
	/// 
	/// Every neural network must have exactly one input layer.
	Input(LayerSize),
	/// A layer that fully connects both adjacent layers.
	FullyConnected(LayerSize),
	/// A layer that computes a given activation function for all elements.
	Activation(Activation, LayerSize)
}

impl Layer {
	/// Returns the len (in counts of neurons; without bias) of this layer.
	pub fn len(&self) -> LayerSize {
		use self::Layer::*;
		match *self {
			Input(size)          |
			FullyConnected(size) |
			Activation(_, size)  => size
		}
	}
}

/// Represents the number of neurons within a layer of a topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerSize(usize);

impl LayerSize {
	/// Creates a new `LayerSize` with the given number of neurons.
	/// 
	/// # Errors
	/// 
	/// - If the given size is equal to 0 (zero).
	pub fn from_usize(size: usize) -> Result<LayerSize> {
		if size == 0 {
			return Err(Error::zero_layer_size())
		}
		Ok(LayerSize(size))
	}

	/// Returns the represented number of neurons as `usize`.
	pub fn to_usize(self) -> usize {
		self.0
	}
}

impl From<usize> for LayerSize {
	fn from(size: usize) -> LayerSize {
		LayerSize::from_usize(size)
			.expect("This implementation expects the user to provide valid input.")
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
	pub fn fully_connect<S>(mut self, size: S) -> TopologyBuilder
		where S: Into<LayerSize>
	{
		self.layers.push(Layer::FullyConnected(size.into()));
		self
	}

	/// Pushes an activation layer to the topology.
	/// 
	/// Activation layers always have the size of the previous layer.
	pub fn activation(mut self, activation: Activation) -> TopologyBuilder {
		let last_len = self.last_len();
		self.layers.push(Layer::Activation(activation, last_len));
		self
	}

	/// Returns the length of the last pushed layer.
	/// 
	/// Useful for layers like activation layers which adopt their size
	/// from their previous layer.
	fn last_len(&self) -> LayerSize {
		self.layers
			.last()
			.expect("a finished disciple must have a valid last layer!")
			.len()
	}

	/// Returns the fully constructed topology.
	/// 
	/// No further modifications to the topology are possible after this operation.
	fn done(self) -> Topology {
		Topology{ layers: self.layers }
	}
}

/// Represents the neural network topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topology {
	layers: Vec<Layer>
}

impl Topology {
	/// Creates a new topology with an input layer of the given size.
	/// 
	/// # Panics
	/// 
	/// If size is zero.
	pub fn input<S>(size: S) -> TopologyBuilder
		where S: Into<LayerSize>
	{
		TopologyBuilder{
			layers: vec![Layer::Input(size.into())]
		}
	}

	/// Returns the number of input neurons.
	///
	/// Used by mentors to validate their sample sizes.
	pub fn input_len(&self) -> LayerSize {
		self.layers
			.first()
			.expect("a finished disciple must have a valid first layer!")
			.len()
	}

	/// Returns the number of output neurons.
	///
	/// Used by mentors to validate their sample sizes.
	pub fn output_len(&self) -> LayerSize {
		self.layers
			.last()
			.expect("a finished disciple must have a valid last layer!")
			.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn topology_1d() {
		use self::Activation::{ReLU, Tanh};
		let top = Topology::input(2)
			.fully_connect( 5).activation(ReLU)
			.fully_connect(10).activation(ReLU)
			.fully_connect(10).activation(ReLU)
			.fully_connect( 5).activation(Tanh)
			.done();

		assert_eq!(top.input_len() , LayerSize(2));
		assert_eq!(top.output_len(), LayerSize(5));
	}
}
