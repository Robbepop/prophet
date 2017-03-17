//! Provides traits to serve as common interface for neural network implementations.

use ndarray::*;
use error_stats::ErrorStats;

/// Types that can predict data based on a one-dimensional input data range.
pub trait Predict<I> {
	/// Predicts data based on given input data.
	fn predict(&mut self, input: I) -> ArrayView1<f32>;
}

/// Types that can propagate through gradient descent.
/// Used by learning procedures.
///
/// This trait should only be used internally!
pub trait UpdateGradients<T> {
	/// Performs gradient descent within the neural network.
	fn update_gradients(&mut self, target: T);
}

/// Types that can adjust their internal weights.
/// Used by learning procedures.
///
/// This trait should only be used internally!
pub trait UpdateWeights<I> {
	/// Updates weights based on the given input data and the current gradients.
	fn update_weights(&mut self, input: I, rate: f32, momentum: f32);
}

/// Temporarly convenience trait for training a neural network.
/// 
/// Will be replaced by a combination of `Predict`, `UpdateGradients`
/// and `UpdateWeights` once the new `Mentor` system has landed.
pub trait Train<I, T> {
	/// Trains the implementor given the input and its expected values.
	fn train(&mut self, input: I, target_values: T) -> ErrorStats;
}
