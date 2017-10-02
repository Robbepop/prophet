//! Provides traits to serve as common interface for neural network implementations.

use ndarray::*;

use sample::SupervisedSample;
use utils::{LearnRate, LearnMomentum};

/// Types that can predict data based on a one-dimensional input data range.
pub trait Predict<I> {
	/// Predicts data based on given input data.
	fn predict(&mut self, input: I) -> ArrayView1<f32>;
}

/// Types that can propagate through gradient descent.
/// Used by learning procedures.
///
/// This trait should only be used internally!
pub(crate) trait UpdateGradients<T> {
	/// Performs gradient descent within the neural network.
	fn update_gradients(&mut self, expected: T);
}

/// Types that can adjust their internal weights.
/// Used by learning procedures.
///
/// This trait should only be used internally!
pub(crate) trait UpdateWeights {
	/// Updates weights based on the given input data and the current gradients.
	fn update_weights(&mut self, rate: LearnRate, momentum: LearnMomentum);
}

pub(crate) trait PredictSupervised<S>
	where S: SupervisedSample
{
	type Finalizer: OptimizeSupervised;

	fn predict_supervised(&mut self, sample: &S) -> Self::Finalizer;
}

pub(crate) trait OptimizeSupervised {
	fn optimize_supervised(&mut self, lr: LearnRate, lm: LearnMomentum);
}

pub mod prelude {
	#[doc(no_inline)]
	pub use super::Predict;

	#[doc(no_inline)]
	pub(crate) use super::{
		UpdateGradients,
		UpdateWeights
	};
}
