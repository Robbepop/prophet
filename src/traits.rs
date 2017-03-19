//! Provides traits to serve as common interface for neural network implementations.

use ndarray::*;
use error_stats::ErrorStats;
use errors::Result;
use errors::ErrorKind::{InvalidLearnRate, InvalidLearnMomentum};

/// Learn rate.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct LearnRate(pub f32);

impl LearnRate {
	/// Returns learn rate from the given `f64` if valid.
	/// 
	/// `rate` has to be in `(0,1)` to form a valid `LearnRate`
	pub fn from_f64(rate: f64) -> Result<LearnRate> {
		if rate > 0.0 && rate < 1.0 {
			Ok(LearnRate(rate as f32))
		}
		else {
			Err(InvalidLearnRate)
		}
	}
}

impl Default for LearnRate {
	fn default() -> Self {
		LearnRate(0.3)
	}
}

/// Learn momentum.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct LearnMomentum(pub f32);

impl LearnMomentum {
	/// Returns learn momentum from the given `f64` if valid.
	/// 
	/// `momentum` has to be in `(0,1)` to form a valid `LearnMomentum`
	pub fn from_f64(momentum: f64) -> Result<LearnMomentum> {
		if momentum > 0.0 && momentum < 1.0 {
			Ok(LearnMomentum(momentum as f32))
		}
		else {
			Err(InvalidLearnMomentum)
		}
	}
}

impl Default for LearnMomentum {
	fn default() -> Self {
		LearnMomentum(0.5)
	}
}


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
	fn update_weights(&mut self, input: I, rate: LearnRate, momentum: LearnMomentum);
}

/// Temporarly convenience trait for training a neural network.
/// 
/// Will be replaced by a combination of `Predict`, `UpdateGradients`
/// and `UpdateWeights` once the new `Mentor` system has landed.
pub trait Train<I, T> {
	/// Trains the implementor given the input and its expected values.
	fn train(&mut self, input: I, target_values: T) -> ErrorStats;
}
