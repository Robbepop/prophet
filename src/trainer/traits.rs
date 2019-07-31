use crate::{
    trainer::SupervisedSample,
    utils::{
        LearnMomentum,
        LearnRate,
    },
};

/// This trait can be implemented by neural networks to make them capable to
/// train them via supervised learning strategies implemented in this library.
///
/// Neural networks may be trained by some other strategy, however.
///
/// Note that currently there is no other method available besides supervised learning.
pub trait PredictSupervised<S>
where
    S: SupervisedSample,
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

/// Trait that has to be implemented for the `Finalizer` associated type of the
/// `PredictSupervised` trait. Its mere purpose is to decouple training via supervised learning
/// given sample data and learning from the train-procedure.
///
/// Types implementing this trait normally are reference types that directly influence
/// for example the weights of their referenced neural network to optimize it towards the
/// expected sample values of the foregun supervised prediction passes.
pub trait OptimizeSupervised {
    /// A type that can be used to query statistics about the learning process.
    type Evaluator: EvaluateSupervised;

    /// Optimizes the neural network respective to the foregun supervised predicition passes
    /// using the sample input signal and expected signal.
    /// With this design it is possible to accumulate multiple prediction passes before carrying
    /// out a single optimization pass: This is also called batch learning.
    fn optimize_supervised(self, lr: LearnRate, lm: LearnMomentum) -> Self::Evaluator;
}

/// Types that act as query and statistics interface to supervise the learning process.
///
/// In particular this is currently used in supervised learning to make it possible to query the latest
/// mean-squared-error after every iteration of the learning process.
pub trait EvaluateSupervised {
    /// The statistics object encapsulating the stats to be queried.
    ///
    /// # Example
    ///
    /// For supervised learning this type is just a thin wrapper around a value representing the mean-squared-error.
    ///
    /// Note that this can be the void-type to hide statistics from users.
    type Stats;

    /// Fetches the latest statistics of the learning procedure.
    fn stats(self) -> Self::Stats;
}
