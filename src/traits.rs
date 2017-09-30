//! Provides traits to serve as common interface for neural network implementations.

use ndarray::*;

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

use layer::utils::{UnbiasedSignalView};

pub(crate) trait PredictWithTarget<'i, 'e, I, E>
	where I: Into<UnbiasedSignalView<'i>>,
	      E: Into<UnbiasedSignalView<'e>>,
	      Self: Sized
{
	fn predict_with_target<'nn>(&'nn mut self, _input: I, _expected: E) -> ReadyToOptimizePredict<'nn, 'i, 'e, I, E, Self> {
		unimplemented!()
	}
}

use std::marker::PhantomData;

pub(crate) struct ReadyToOptimizePredict<'nn, 'i, 'e, I: 'i, E: 'e, NN: 'nn>
	where NN: PredictWithTarget<'i, 'e, I, E>,
	      I : Into<UnbiasedSignalView<'i>>,
	      E : Into<UnbiasedSignalView<'e>>
{
	nn: &'nn mut NN,
	marker: PhantomData<(&'i I, &'e E)>
}

pub(crate) trait OptimizePredict {
	fn optimize_predict(&mut self, _lr: LearnRate, _lm: LearnMomentum) {
		unimplemented!()
	}
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
