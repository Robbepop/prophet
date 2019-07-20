//! Provides operations, data structures and error definitions for Disciple objects
//! which form the basis for topologies of neural networks.

use std::vec;

use crate::activation::Activation;
use crate::errors::{Result, Error};

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
	pub fn new(size: usize) -> Result<LayerSize> {
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
	/// Creates a new `LayerSize` with the given number of neurons.
	/// 
	/// # Panics
	/// 
	/// - If the given size is equal to 0 (zero).
	fn from(size: usize) -> LayerSize {
		LayerSize::new(size)
			.expect("This implementation expects the user to provide valid input.")
	}
}

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

impl From<FullyConnectedLayer> for AnyLayer {
	fn from(layer: FullyConnectedLayer) -> AnyLayer {
		AnyLayer::FullyConnected(layer)
	}
}

impl From<ActivationLayer> for AnyLayer {
	fn from(layer: ActivationLayer) -> AnyLayer {
		AnyLayer::Activation(layer)
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
				AnyLayer::from(
					FullyConnectedLayer::new(self.input_len, size.into()))
			]
		}
	}

	fn activation(self, act: Activation) -> Self::Builder {
		Topology{
			layers: vec![
				AnyLayer::from(
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
			AnyLayer::from(
				FullyConnectedLayer::new(last_len, size.into())
			)
		);
		self
	}

	fn activation(mut self, act: Activation) -> Self::Builder {
		let last_len = self.last_len();
		self.layers.push(
			AnyLayer::from(
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

	#[inline]
	fn into_iter(self) -> Self::IntoIter {
		self.layers.into_iter()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn activation_fns() -> Vec<Activation> {
		use crate::activation::Activation::*;
		vec![
			Identity,
			BinaryStep,
			Logistic,
			Tanh,
			ArcTan,
			SoftSign,
			ReLU,
			SoftPlus,
			BentIdentity,
			Sinusoid,
			Gaussian
		]
	}

	mod layer_size {
		use super::*;

		#[test]
		fn new_ok() {
			assert_eq!(LayerSize::new(1), Ok(LayerSize(1)));
			assert_eq!(LayerSize::new(42), Ok(LayerSize(42)));
			assert_eq!(LayerSize::new(1337), Ok(LayerSize(1337)));
		}

		#[test]
		fn new_fail() {
			assert_eq!(LayerSize::new(0), Err(Error::zero_layer_size()));
		}

		#[test]
		fn from_ok() {
			assert_eq!(LayerSize::from(1), LayerSize(1));
			assert_eq!(LayerSize::from(42), LayerSize(42));
			assert_eq!(LayerSize::from(1337), LayerSize(1337));
		}

		#[test]
		#[should_panic]
		fn from_fail() {
			LayerSize::from(0);
		}

		#[test]
		fn to_usize() {
			for i in 1..100 {
				assert_eq!(LayerSize(i).to_usize(), i);
			}
		}
	}

	mod fully_connected_layer {
		use super::*;

		#[test]
		fn new() {
			assert_eq!(
				FullyConnectedLayer::new(LayerSize(3), LayerSize(4)),
				FullyConnectedLayer{
					inputs: LayerSize(3),
					outputs: LayerSize(4)
				}
			);
			assert_eq!(
				FullyConnectedLayer::new(3, 4),
				FullyConnectedLayer{
					inputs: LayerSize(3),
					outputs: LayerSize(4)
				}
			);
		}

		#[test]
		fn input_len() {
			assert_eq!(FullyConnectedLayer::new(1, 1).input_len(), LayerSize(1));
			assert_eq!(FullyConnectedLayer::new(3, 4).input_len(), LayerSize(3));
			assert_eq!(FullyConnectedLayer::new(42, 1337).input_len(), LayerSize(42));
		}

		#[test]
		fn output_len() {
			assert_eq!(FullyConnectedLayer::new(1, 1).output_len(), LayerSize(1));
			assert_eq!(FullyConnectedLayer::new(3, 4).output_len(), LayerSize(4));
			assert_eq!(FullyConnectedLayer::new(42, 1337).output_len(), LayerSize(1337));
		}
	}

	mod activation_layer {
		use super::*;
		use crate::activation::Activation::*;

		#[test]
		fn new() {
			assert_eq!(
				ActivationLayer::new(LayerSize(3), Tanh),
				ActivationLayer{
					size: LayerSize(3),
					act: Tanh
				}
			);
			assert_eq!(
				ActivationLayer::new(42, ReLU),
				ActivationLayer{
					size: LayerSize(42),
					act: ReLU
				}
			);
		}

		#[test]
		fn activation_fn() {
			for act in activation_fns() {
				for n in 1..10 {
					assert_eq!(ActivationLayer::new(n, act).activation_fn(), act);
				}
			}
		}

		#[test]
		fn input_len() {
			for act in activation_fns() {
				for n in 1..10 {
					assert_eq!(ActivationLayer::new(n, act).input_len(), LayerSize(n));
				}
			}
		}

		#[test]
		fn output_len() {
			for act in activation_fns() {
				for n in 1..10 {
					assert_eq!(ActivationLayer::new(n, act).input_len(), LayerSize(n));
				}
			}
		}

		#[test]
		fn input_len_eq_output_len() {
			for act in activation_fns() {
				for n in 1..10 {
					let layer = ActivationLayer::new(n, act);
					assert_eq!(layer.input_len(), layer.output_len());
				}
			}
		}
	}

	mod any_layer {
		use super::*;
		use crate::activation::Activation::*;

		#[test]
		fn from_fully_connected_layer() {
			assert_eq!(
				AnyLayer::from(FullyConnectedLayer::new(3, 4)),
				AnyLayer::FullyConnected(FullyConnectedLayer{
					inputs: LayerSize(3),
					outputs: LayerSize(4)
				})
			);
		}

		#[test]
		fn from_activation_layer() {
			assert_eq!(
				AnyLayer::from(ActivationLayer::new(3, Tanh)),
				AnyLayer::Activation(ActivationLayer{
					size: LayerSize(3),
					act: Tanh
				})
			);
		}

		#[test]
		fn input_len() {
			for i in 1..10 {
				for o in 1..10 {
					assert_eq!(AnyLayer::from(FullyConnectedLayer::new(i, o)).input_len(), LayerSize(i));
				}
				for act in activation_fns() {
					assert_eq!(AnyLayer::from(ActivationLayer::new(i, act)).input_len(), LayerSize(i));
				}
			}
		}

		#[test]
		fn output_len() {
			for i in 1..10 {
				for o in 1..10 {
					assert_eq!(AnyLayer::from(FullyConnectedLayer::new(i, o)).output_len(), LayerSize(o));
				}
				for act in activation_fns() {
					assert_eq!(AnyLayer::from(ActivationLayer::new(i, act)).output_len(), LayerSize(i));
				}
			}
		}
	}

	mod topology {
		use super::*;
		use crate::activation::Activation::*;

		fn simple_dummy_topology() -> Topology {
			Topology::input(1).fully_connect( 1)
		}

		fn medium_dummy_topology() -> Topology {
			Topology::input(2)
				.fully_connect(3).activation(Tanh)
				.fully_connect(1).activation(Tanh)
		}

		fn complex_dummy_topology() -> Topology {
			Topology::input(10)
				.fully_connect(42)
				.fully_connect(1337)
				.activation(Tanh)
				.activation(ReLU)
				.fully_connect(11).activation(Logistic)
				.fully_connect(7).activation(Gaussian)
		}

		#[test]
		fn input() {
			for n in 1..10 {
				assert_eq!(
					Topology::input(LayerSize(n)),
					InitializingTopology{input_len: LayerSize(n)}
				);
				assert_eq!(
					Topology::input(LayerSize(n)),
					Topology::input(n)
				);
			}
		}

		#[test]
		fn fully_connect() {
			fn check_top_with_cfg(top: Topology, size: usize) {
				let initial_output_len = top.output_len();
				assert_eq!(
					top.fully_connect(size).into_iter().last().unwrap(),
					AnyLayer::FullyConnected(FullyConnectedLayer{
						inputs: initial_output_len,
						outputs: LayerSize(size)
					})
				);
			}
			let test_sizes = &[1, 3, 7, 13, 42, 1337];
			for &size in test_sizes {
				check_top_with_cfg(simple_dummy_topology(), size);
				check_top_with_cfg(medium_dummy_topology(), size);
				check_top_with_cfg(complex_dummy_topology(), size);
			}
		}

		#[test]
		fn activation() {
			fn check_top_with_cfg(top: Topology, act: Activation) {
				let initial_output_len = top.output_len();
				assert_eq!(
					top.activation(act).into_iter().last().unwrap(),
					AnyLayer::Activation(ActivationLayer{
						size: initial_output_len,
						act
					})
				);
			}
			for act in activation_fns() {
				check_top_with_cfg(simple_dummy_topology(), act);
				check_top_with_cfg(medium_dummy_topology(), act);
				check_top_with_cfg(complex_dummy_topology(), act);
			}
		}

		#[test]
		fn check_simple() {
			assert_eq!(
				simple_dummy_topology(),
				Topology{
					layers: vec![
						AnyLayer::from(FullyConnectedLayer::new(1, 1))
					]
				}
			);
		}

		#[test]
		fn check_medium() {
			assert_eq!(
				medium_dummy_topology(),
				Topology{
					layers: vec![
						AnyLayer::from(FullyConnectedLayer::new(2, 3)),
						AnyLayer::from(ActivationLayer::new(3, Tanh)),
						AnyLayer::from(FullyConnectedLayer::new(3, 1)),
						AnyLayer::from(ActivationLayer::new(1, Tanh))
					]
				}
			);
		}

		#[test]
		fn check_complex() {
			assert_eq!(
				complex_dummy_topology(),
				Topology{
					layers: vec![
						AnyLayer::from(FullyConnectedLayer::new(10, 42)),
						AnyLayer::from(FullyConnectedLayer::new(42, 1337)),
						AnyLayer::from(ActivationLayer::new(1337, Tanh)),
						AnyLayer::from(ActivationLayer::new(1337, ReLU)),
						AnyLayer::from(FullyConnectedLayer::new(1337, 11)),
						AnyLayer::from(ActivationLayer::new(11, Logistic)),
						AnyLayer::from(FullyConnectedLayer::new(11, 7)),
						AnyLayer::from(ActivationLayer::new(7, Gaussian))
					]
				}
			);
		}

		#[test]
		fn input_len() {
			assert_eq!(simple_dummy_topology().input_len() , LayerSize( 1));
			assert_eq!(medium_dummy_topology().input_len() , LayerSize( 2));
			assert_eq!(complex_dummy_topology().input_len(), LayerSize(10));
		}

		#[test]
		fn output_len() {
			assert_eq!(simple_dummy_topology().output_len() , LayerSize(1));
			assert_eq!(medium_dummy_topology().output_len() , LayerSize(1));
			assert_eq!(complex_dummy_topology().output_len(), LayerSize(7));
		}

		#[test]
		fn into_iter_simple() {
			let mut iter = simple_dummy_topology().into_iter();
			assert_eq!(iter.next(), Some(AnyLayer::from(FullyConnectedLayer::new(1, 1))));
			assert_eq!(iter.next(), None);
		}

		#[test]
		fn into_iter_medium() {
			let mut iter = medium_dummy_topology().into_iter();
			assert_eq!(iter.next(), Some(AnyLayer::from(FullyConnectedLayer::new(2, 3))));
			assert_eq!(iter.next(), Some(AnyLayer::from(ActivationLayer::new(3, Tanh))));
			assert_eq!(iter.next(), Some(AnyLayer::from(FullyConnectedLayer::new(3, 1))));
			assert_eq!(iter.next(), Some(AnyLayer::from(ActivationLayer::new(1, Tanh))));
			assert_eq!(iter.next(), None);
		}

		#[test]
		fn into_iter_complex() {
			let mut iter = complex_dummy_topology().into_iter();

			assert_eq!(iter.next(), Some(AnyLayer::from(FullyConnectedLayer::new(10, 42))));
			assert_eq!(iter.next(), Some(AnyLayer::from(FullyConnectedLayer::new(42, 1337))));
			assert_eq!(iter.next(), Some(AnyLayer::from(ActivationLayer::new(1337, Tanh))));
			assert_eq!(iter.next(), Some(AnyLayer::from(ActivationLayer::new(1337, ReLU))));
			assert_eq!(iter.next(), Some(AnyLayer::from(FullyConnectedLayer::new(1337, 11))));
			assert_eq!(iter.next(), Some(AnyLayer::from(ActivationLayer::new(11, Logistic))));
			assert_eq!(iter.next(), Some(AnyLayer::from(FullyConnectedLayer::new(11, 7))));
			assert_eq!(iter.next(), Some(AnyLayer::from(ActivationLayer::new(7, Gaussian))));

			assert_eq!(iter.next(), None);
		}
	}
}
