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
	MentorBuilder,
	Context
};

pub use self::traits::{
	PredictSupervised,
	OptimizeSupervised,
	EvaluateSupervised
};

pub use self::utils::{
	MeanSquaredError
};
