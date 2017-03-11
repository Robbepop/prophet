//! Provides operations, data structures and error definitions for Disciple objects
//! which form the basis for topologies of neural networks.

use std::slice::Iter;
use std::marker::PhantomData;

/// This is the current compile-time state of the `Disciple`.
/// Used for compile-time checking of construction constrains.
/// E.g. to make it impossible to define multiple output layers,
/// no input-layers etc..
pub trait InitializationState {}

mod state {
	/// States that the `Disciple` is under construction and may mutate.
	pub struct UnderConstruction();

	/// States that the construction of the `Disciple` is finished.
	/// No further modifications are possible!
	pub struct Finished();

	impl super::InitializationState for UnderConstruction {}
	impl super::InitializationState for Finished {}
}
use self::state::{UnderConstruction, Finished};

/// Represents the neural network topology.
/// 
/// This is in fact a compile-time-checked builder for neural network topologies.
/// Can be used by `Mentor` types to train it and become a trained neural network
/// with which the user can predict data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disciple<S: InitializationState> {
	layer_sizes: Vec<usize>,
	phantom: PhantomData<S>
}

impl Disciple<UnderConstruction> {
	/// Creates a new `Disciple` with the given amount of input neurons.
	/// 
	/// Bias-Neurons are not included in the given number!
	pub fn with_input(size: usize) -> Disciple<UnderConstruction> {
		Disciple{
			layer_sizes: vec![size],
			phantom    : PhantomData::default()
		}
	}

	/// Adds a hidden layer to this `Disciple` with the given amount of neurons.
	/// 
	/// Bias-Neurons are not included in the given number!
	pub fn add_layer(mut self, size: usize) -> Disciple<UnderConstruction> {
		self.layer_sizes.push(size);
		self
	}

	/// Adds some hidden layers to this `Disciple` with the given amount of neurons.
	/// 
	/// Bias-Neurons are not included in the given number!
	pub fn add_layers(mut self, sizes: &[usize]) -> Disciple<UnderConstruction> {
		for &size in sizes {
			self.layer_sizes.push(size);
		}
		self
	}

	/// Finishes constructing a `Disciple` by defining its output layer neurons.
	/// 
	/// Bias-Neurons are not included in the given number!
	pub fn with_output(mut self, size: usize) -> Disciple<Finished> {
		self.layer_sizes.push(size);
		Disciple{
			layer_sizes: self.layer_sizes,
			phantom    : PhantomData::default()
		}
	}
}

impl Disciple<Finished> {
	/// Iterates over the layer sizes of this Disciple's topology definition.
	pub fn iter_layer_sizes<'a>(&'a self) -> Iter<'a, usize> {
		self.layer_sizes.iter()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn construction() {
		let dis = Disciple::with_input(2)
			.add_layer(5)
			.add_layers(&[10, 10])
			.with_output(5);
		let mut it  = dis
			.iter_layer_sizes()
			.map(|&size| size);
		assert_eq!(it.next(), Some(2));
		assert_eq!(it.next(), Some(5));
		assert_eq!(it.next(), Some(10));
		assert_eq!(it.next(), Some(10));
		assert_eq!(it.next(), Some(5));
	}
}
