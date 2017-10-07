use trainer::SupervisedSample;
use utils::{
	LearnRate,
	LearnMomentum
};

/// This trait can be implemented by neural networks to make them capable to
/// train them via supervised learning strategies implemented in this library.
/// 
/// Neural networks may be trained by some other strategy, however.
/// 
/// Note that currently there is no other method available besides supervised learning.
pub trait PredictSupervised<S>
	where S: SupervisedSample
{
	/// The type that implements the final optimization of the neural network.
	/// 
	/// The main purpose of this associated type is to enpasulate optimization
	/// to enforce at least one supervised predicts before every optimization sweep.
	/// 
	/// This also supports batched learning where neural nets predict data supervised 
	/// multiple times before optimizing once. This is a performance related learning
	/// strategy and exchanges better performance for slightly worse results in some cases.
	type Finalizer: OptimizeSupervised;

	/// Predicts the sample input data and adjust itself via the sample's expected values.
	/// 
	/// Returns an optimizer structure that's only purpose is to optimize the neural
	/// network after it has predicted via the given sample.
	/// 
	/// Note that this procedure by itself does not improve the neural network by
	/// for example adjusting its weights. For that, use the downstreamed optimization routine.
	fn predict_supervised(self, sample: &S) -> Self::Finalizer;
}

/// Trait trait has to be implemented for the `Finalizer` associated type of the
/// `PredictSupervised` trait. Its mere purpose is to decouple training via supervised learning
/// given sample data and learning from the train-procedure.
/// 
/// Types implementing this trait normally are reference types that directly influence 
/// for example the weights of their referenced neural network to optimize it towards the
/// expected sample values of the foregun supervised prediction passes.
pub trait OptimizeSupervised {
	type Evaluator: EvaluateSupervised;

	/// Optimizes the neural network respective to the foregun supervised predicition passes
	/// using the sample input signal and expected signal.
	/// With this design it is possible to accumulate multiple prediction passes before carrying
	/// out a single optimization pass: This is also called batch learning.
	fn optimize_supervised(self, lr: LearnRate, lm: LearnMomentum) -> Self::Evaluator;
}

pub trait EvaluateSupervised {
	type Stats;

	fn stats(self) -> Self::Stats;
}
