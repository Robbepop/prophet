//! Provides traits to serve as common interface for neural network implementations.

use num::Float;
use error_stats::ErrorStats;

/// Representative for neural network implementations that are only able to predict data,
/// but have no ability to further improve by training themselves.
/// 
/// This might come in handy for future versions of this library to provide a way to 
/// convert a trainable neural network into a static one to optimize for space since
/// there are many constructs in a neural network that are only needed for training purpose.
pub trait Prophet {
	/// The type used internally to store weights, intermediate results etc.
	/// 
	/// In future versions of this library it might be possible to create neural networks
	/// that can be parameterize over a given set of floating point types. For example
	/// to create the possibility to use ```f32``` or ```f64``` internally.
	/// 
	/// Eventual language updates like adding ```f16``` or ```f128``` float primitives
	/// would enhance this functionality furthermore.
	type Elem: Float;

	/// Predicts resulting data based on the given input data and on the data that was
	/// used to train this neural network, eventually.
	/// 
	/// The length of the resulting slice of predicted values is equal to the count 
	/// of neurons in the output layer of the implementing neural network.
	/// 
	/// # Panics
	/// 
	/// If the element count of the given input slice is not equal to the neuron count 
	/// of the input layer for the implementing neural network.
	fn predict<'b, 'a: 'b>(&'a mut self, input: &'b [Self::Elem]) -> &'b [Self::Elem];
}

/// Representative for neural network implementations that have the capability to 
/// learn from training data and predict data based on the learned examples.
pub trait Disciple {
	/// The type used internally to store weights, intermediate results etc.
	/// 
	/// In future versions of this library it might be possible to create neural networks
	/// that can be parameterize over a given set of floating point types. For example
	/// to create the possibility to use ```f32``` or ```f64``` internally.
	/// 
	/// Eventual language updates like adding ```f16``` or ```f128``` float primitives
	/// would enhance this functionality furthermore.
	type Elem: Float;

	/// Trains the neural network with the given input data based on the target expected values.
	/// 
	/// Returns an ErrorStats object that stores useful information about the learning process.
	/// 
	/// # Panics
	/// 
	/// If the element count of the input slice is not equal to the neuron count of the 
	/// input layer for the implementing neural network.
	/// 
	/// If the element count of the expected slice is not equal to the neuron count of the
	/// output layer for the implementing neural network.
	fn train(&mut self, input: &[Self::Elem], expected: &[Self::Elem]) -> ErrorStats;
}
