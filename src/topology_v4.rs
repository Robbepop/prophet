//! Provides operations, data structures and error definitions for Disciple objects
//! which form the basis for topologies of neural networks.

use std::vec;

use activation::Activation;
use errors::{Result, Error};

/// This interface represents the bare minimum of what an abstracted layer has to offer.
pub trait Layer {
	/// Returns the length of the input signal of this `Layer`.
	fn input_len(&self) -> LayerSize;

	/// Returns the length of the output signal of this `Layer`.
	fn output_len(&self) -> LayerSize;
}

/// An abstracted `FullyConnectedLayer`.
/// 
/// This is a layer that reshapes the neighbouring neural layer sizes
/// by fully connecting their neuron signals to each other.
/// 
/// This abstract layer includes all information required in order to
/// construct a concrete `FullyConnectedLayer` with default settings.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FullyConnectedLayer{
	/// The input signal length of this `FullyConnectedLayer`
	inputs: LayerSize,
	/// The output signal length of this `FullyConnectedLayer`
	outputs: LayerSize
}

impl FullyConnectedLayer {
	/// Creates a new abstracted `FullyConnectedLayer` with the given
	/// input and output signal lengths.
	pub(crate) fn new<I, O>(inputs: I, outputs: O) -> FullyConnectedLayer
		where I: Into<LayerSize>,
		      O: Into<LayerSize>
	{
		FullyConnectedLayer{
			inputs: inputs.into(),
			outputs: outputs.into()
		}
	}
}

impl Layer for FullyConnectedLayer {
	#[inline]
	fn input_len(&self) -> LayerSize {
		self.inputs
	}

	#[inline]
	fn output_len(&self) -> LayerSize {
		self.outputs
	}
}

/// An abstracted `ActivationLayer`.
/// 
/// This layer simply forwards its input signal tranformed by its activation function.
/// 
/// This abstract layer includes all information required in order to
/// construct a concrete `ActivationLayer` with default settings.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ActivationLayer{
	size: LayerSize,
	act: Activation
}

impl ActivationLayer {
	/// Creates a new abstracted `ActivationLayer` with the given
	/// input signal length and activation function.
	pub(crate) fn new<S>(size: S, act: Activation) -> ActivationLayer
		where S: Into<LayerSize>
	{
		ActivationLayer{size: size.into(), act}
	}

	/// Returns the activation function of this `ActivationLayer`.
	pub(crate) fn activation_fn(&self) -> Activation {
		self.act
	}
}

impl Layer for ActivationLayer {
	#[inline]
	fn input_len(&self) -> LayerSize {
		self.size
	}

	#[inline]
	fn output_len(&self) -> LayerSize {
		self.input_len()
	}
}

/// Represents an abstract layer that may be any of the supported abstract layers.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AnyLayer {
	/// An abstracted `FullyConnectedLayer`.
	FullyConnected(FullyConnectedLayer),
	/// An abstracted `ActivationLayer`.
	Activation(ActivationLayer)
}

impl Layer for AnyLayer {
	#[inline]
	fn input_len(&self) -> LayerSize {
		use self::AnyLayer::*;
		match *self {
			FullyConnected(layer) => layer.input_len(),
			Activation(layer) => layer.input_len()
		}
	}

	#[inline]
	fn output_len(&self) -> LayerSize {
		use self::AnyLayer::*;
		match *self {
			FullyConnected(layer) => layer.output_len(),
			Activation(layer) => layer.output_len()
		}
	}
}

/// Represents the number of neurons within a layer of a topology.
/// 
/// Note: This does not respect bias neurons! They are implicitely
///       added in later stages of neural network construction.
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
	pub fn into_usize(self) -> usize {
		self.0
	}
}

impl From<usize> for LayerSize {
	/// Creates a new `LayerSize` with the given number of neurons.
	/// 
	/// # Panics
	/// 
	/// - If the given size is equal to 0 (zero).
	fn from(size: usize) -> LayerSize {
		LayerSize::from_usize(size)
			.expect("This implementation expects the user to provide valid input.")
	}
}

/// Builds up topologies and do some minor compile-time and 
/// runtime checks to enforce validity of the topology as a shape for neural nets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topology {
	layers: Vec<AnyLayer>
}

impl Topology {
	/// Creates a new topology with an input layer of the given size.
	/// 
	/// # Panics
	/// 
	/// If size is zero.
	pub fn input<S>(size: S) -> InitializingTopology
		where S: Into<LayerSize>
	{
		InitializingTopology{input_len: size.into()}
	}
}

/// This represents a `Topology` during its initialization.
/// 
/// Note: This structure is always temporary and just used to correctly
///       propagate the input signal length information to the real topology builder.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct InitializingTopology {
	input_len: LayerSize
}

/// Types that can build-up topologies for neural networks incrementally.
pub trait TopologyBuilder {
	/// The underlying topology builder type.
	type Builder: TopologyBuilder;

	/// Adds another abstracted fully connected layer to this topology.
	fn fully_connect<S>(self, size: S) -> Self::Builder
		where S: Into<LayerSize>;

	/// Adds another abstracted activation layer to this topology.
	fn activation(self, act: Activation) -> Self::Builder;
}

impl TopologyBuilder for InitializingTopology {
	type Builder = Topology;

	fn fully_connect<S>(self, size: S) -> Self::Builder
		where S: Into<LayerSize>
	{
		Topology{
			layers: vec![
				AnyLayer::FullyConnected(
					FullyConnectedLayer::new(self.input_len, size.into()))
			]
		}
	}

	fn activation(self, act: Activation) -> Self::Builder {
		Topology{
			layers: vec![
				AnyLayer::Activation(
					ActivationLayer::new(self.input_len, act))
			]
		}
	}
}

impl TopologyBuilder for Topology {
	type Builder = Topology;

	fn fully_connect<S>(mut self, size: S) -> Self::Builder
		where S: Into<LayerSize>
	{
		let last_len = self.last_len();
		self.layers.push(
			AnyLayer::FullyConnected(
				FullyConnectedLayer::new(last_len, size.into())
			)
		);
		self
	}

	fn activation(mut self, act: Activation) -> Self::Builder {
		let last_len = self.last_len();
		self.layers.push(
			AnyLayer::Activation(
				ActivationLayer::new(last_len, act)
			)
		);
		self
	}
}

impl Topology {
	/// Returns the length of the last pushed layer.
	/// 
	/// Useful for layers like activation layers which adopt their size
	/// from their previous layer.
	fn last_len(&self) -> LayerSize {
		self.layers
			.last()
			.expect("a finished disciple must have a valid last layer!")
			.output_len()
	}
}

impl Topology {
	/// Returns the number of input neurons.
	///
	/// Used by mentors to validate their sample sizes.
	pub fn input_len(&self) -> LayerSize {
		self.layers
			.first()
			.expect("a finished disciple must have a valid first layer!")
			.input_len()
	}

	/// Returns the number of output neurons.
	///
	/// Used by mentors to validate their sample sizes.
	pub fn output_len(&self) -> LayerSize {
		self.layers
			.last()
			.expect("a finished disciple must have a valid last layer!")
			.output_len()
	}
}

impl IntoIterator for Topology {
	type Item = AnyLayer;
	type IntoIter = vec::IntoIter<AnyLayer>;

	fn into_iter(self) -> Self::IntoIter {
		self.layers.into_iter()
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
			.fully_connect( 5).activation(Tanh);

		assert_eq!(top.input_len() , LayerSize(2));
		assert_eq!(top.output_len(), LayerSize(5));
	}
}
