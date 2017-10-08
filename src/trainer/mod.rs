//! The trainer module helps users of this library to setup a working training process
//! and makes it much easier to train neural networks with given constraints and goals.
//! 
//! Currently, only the supervised learning strategy is supported as well as the
//! mean-squared-error LOSS approach.
//! 
//! To invoke a training session a user first has to create a neural network or a topology
//! for a neural network. The preferred way is to use the topology module to setup a 
//! topology which describes a neural network and use the `train()` method on it.
//! This returns an initial `InitializingMentor` object that acts as a setup-helper
//! to construct a `Mentor` and thus a training session.
//! 
//! Note that it is possible to provide the trainer with a halting condition that tells the
//! training process when to stop training as well as a logging condition that tells the 
//! training process when to log the entire training state and some statistics.
//! 
//! It is recommended to look into the detailed documentation to find out about all
//! the possible settings and configurations.

pub mod condition;
#[macro_use]
mod sample;
mod mentor;
mod traits;
mod utils;

pub use self::sample::{
	SupervisedSample,
	Sample,
	SampleCollection,
	SampleGen,
	SequentialSampleScheduler,
	RandomSampleScheduler
};

pub use self::condition::{
	TrainingState,
	TrainCondition
};

pub use self::mentor::{
	Mentor,
	InitializingMentor,
	MentorBuilder
};

pub use self::traits::{
	PredictSupervised,
	OptimizeSupervised,
	EvaluateSupervised
};

pub use self::utils::{
	MeanSquaredError
};
