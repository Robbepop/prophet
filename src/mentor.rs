use ndarray::prelude::*;
use rand::*;

use ::topology::*;
use ::error_stats::*;
use ::neural_net::*;

/// Possible errors during mentoring.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ErrorKind {
	/// Occures when invalid sample input sizes are recognized.
	InvalidSampleInputSize,

	/// Occures when invalid sample target sizes are recognized.
	InvalidSampleTargetSize,

	/// Occures when the learning rate is not within the valid
	/// range of `(0,1)`.
	InvalidLearnRate,

	/// Occures when the learning momentum is not within the
	/// valid range of `(0,1)`.
	InvalidLearnMomentum,

	/// Occures when the specified average net error
	/// criterion is invalid.
	InvalidAvgNetError,

	/// Occures when the specified mean squared error
	/// criterion is invalid.
	InvalidMeanSquaredError
}
use self::ErrorKind::*;

/// Cirterias after which the learning process holds.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Criterion {
	/// Stop after the given amount of milliseconds.
	TimeOut(u64),

	/// Stop after the given amount of learning iterations.
	Iterations(u64),

	/// Stop when the mean square error drops below the given value.
	MeanSquaredError(f64),

	/// Stop as soon as the average net error (on MSE basis)
	/// drops below the given value.
	AvgNetError(f64),
}

impl Criterion {
	/// Checks if this criterion is valid.
	fn check_validity(&self) -> Result<()> {
		use self::Criterion::*;
		match *self {
			TimeOut(_)    => Ok(()),
			Iterations(_) => Ok(()),
			MeanSquaredError(mse) => {
				if mse > 0.0 && mse < 1.0 {
					Ok(())
				}
				else {
					Err(InvalidMeanSquaredError)
				}
			},
			AvgNetError(avg) => {
				if avg > 0.0 && avg < 1.0 {
					Ok(())
				}
				else {
					Err(InvalidAvgNetError)
				}
			}
		}
	}
}

/// Learning rate configuration.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LearnRate {
	/// Automatically adapt learn rate during learning.
	Adapt,

	/// Use the given fixed learn rate.
	Fixed(f64)
}

impl LearnRate {
	/// Checks if this learn rate is valid.
	fn check_validity(&self) -> Result<()> {
		use self::LearnRate::*;
		match *self {
			Adapt => Ok(()),
			Fixed(rate) => {
				if rate > 0.0 && rate < 1.0 {
					Ok(())
				}
				else {
					Err(InvalidLearnRate)
				}
			}
		}
	}
}

/// Learning momentum configuration.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LearnMomentum {
	/// Automatically adapt learn momentum during learning.
	Adapt,

	/// Use the given fixed learn momentum.
	Fixed(f64)
}

impl LearnMomentum {
	/// Checks if this learn momentum is valid.
	fn check_validity(&self) -> Result<()> {
		use self::LearnMomentum::*;
		match *self {
			Adapt => Ok(()),
			Fixed(momentum) => {
				if momentum > 0.0 && momentum < 1.0 {
					Ok(())
				}
				else {
					Err(InvalidLearnMomentum)
				}
			}
		}
	}
}

/// Sample scheduling strategy while learning.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SchedulingKind {
	/// Pick samples randomly.
	/// 
	/// This usually is a good approach to defeat sample-pattern learning.
	Random,

	/// Pick samples in order.
	/// 
	/// This maybe useful for testing purposes.
	Iterative
}

/// A scheduler for indices with a scheduling strategy.
/// 
/// Used by `SampleScheduler` to pick samples with different scheduling strategies.
#[derive(Clone)]
pub enum Scheduler {
	/// Samples randomly.
	Random(ThreadRng),

	/// Samples iteratively.
	Iterative(u64)
}

use ::std::fmt::{Debug, Formatter};
impl Debug for Scheduler {
	fn fmt(&self, f: &mut Formatter) -> ::std::fmt::Result {
		use self::Scheduler::*;
		match self {
			&Random(_) => write!(f, "Scheduler::Random(_)"),
			&Iterative(x) => write!(f, "Scheduler::Iterative({})", x)
		}
	}
}

impl Scheduler {
	/// Creates a new `Scheduler` from a given scheduling strategy.
	fn from_kind(kind: SchedulingKind) -> Self {
		use self::SchedulingKind::*;
		match kind {
			Random    => Scheduler::Random(thread_rng()),
			Iterative => Scheduler::Iterative(0)
		}
	}

	/// Returns the next scheduled index.
	/// 
	/// The returned index is then used by the `SampleScheduler`
	/// to pick the associated sample.
	fn next(&mut self, num_samples: usize) -> usize {
		use self::Scheduler::*;
		match self {
			&mut Random(ref mut rng) => rng.gen_range(0, num_samples),
			&mut Iterative(ref mut cur)  => {
				let next = *cur as usize % num_samples;
				*cur += 1;
				next
			}
		}
	}
}

/// Organizes the scheduling of samples with different strategies.
#[derive(Debug, Clone)]
pub struct SampleScheduler {
	samples  : Vec<Sample>,
	scheduler: Scheduler
}

impl SampleScheduler {
	/// Creates a new `SampleScheduler` from given samples and a scheduling strategy.
	fn from_samples(kind: SchedulingKind, samples: Vec<Sample>) -> Self {
		SampleScheduler{
			samples  : samples,
			scheduler: Scheduler::from_kind(kind)
		}
	}

	/// Returns the next sample.
	fn next(&mut self) -> &Sample {
		let len_samples = self.samples.len();
		let id = self.scheduler.next(len_samples);
		&self.samples[id]
	}
}

/// Result type that are returned by some `Mentor` functionalities.
pub type Result<T> = ::std::result::Result<T, ErrorKind>;

// /// Mentors are objects that train a given disciple structure
// /// resulting in a prophet structure that can be used to predict
// /// data.
// /// The static type of the trainable `Disciple` and the resuting `Prophet`
// /// has to be known.
// /// 
// /// Mentors define different criteria under which a disciple is
// /// meant to be fully (or well-enough) trained to become a prophet.
// /// 
// /// A naive implementation is the `AvgNetErrorMentor` that simply
// /// trains its disciple until the average net error decreases below 
// /// a given value. For this the mentor requires some sample training pieces.
// trait Mentor {
// 	type D: Disciple;
// 	type P: Prophet;

/// A sample used to train a disciple during supervised learinng.
#[derive(Debug, Clone)]
pub struct Sample {
	/// The input parameter of this `Sample`.
	pub input : Array1<f32>,

	/// The expected target values of this `Sample`.
	pub target: Array1<f32>
}

impl<Arr> From<(Arr, Arr)> for Sample
	where Arr: Into<Array1<f32>>
{
	fn from(from: (Arr, Arr)) -> Sample {
		Sample{
			input : from.0.into(),
			target: from.1.into()
		}
	}
}

/// Builder follows the builder pattern to incrementally
/// build properties for the real Mentor and delay computations
/// until the `go` routine is called.
#[derive(Debug, Clone)]
pub struct Builder {
	err_stats : ErrorStats,
	learn_rate: LearnRate,
	learn_mom : LearnMomentum,
	criterion : Criterion,
	scheduling: SchedulingKind,
	disciple  : Topology<Finished>,
	samples   : Vec<Sample>
}

impl Builder {
	/// Creates a new mentor for the given disciple and
	/// with the given sample collection (training data).
	pub fn new(disciple: Topology<Finished>, samples: Vec<Sample>) -> Builder {
		Builder{
			err_stats : ErrorStats::default(),
			learn_rate: LearnRate::Adapt,
			learn_mom : LearnMomentum::Adapt,
			criterion : Criterion::AvgNetError(0.05),
			scheduling: SchedulingKind::Random,
			disciple  : disciple,
			samples   : samples
		}
	}

	/// Use the given criterion.
	/// 
	/// Default criterion is `AvgNetError(0.05)`.
	pub fn criterion(mut self, criterion: Criterion) -> Builder {
		self.criterion = criterion;
		self
	}

	/// Use the given learn rate.
	/// 
	/// Default learn rate is adapting behaviour.
	pub fn learn_rate(mut self, learn_rate: LearnRate) -> Builder {
		self.learn_rate = learn_rate;
		self
	}

	/// Use the given learn momentum.
	/// 
	/// Default learn momentum is `0.5`.
	pub fn learn_momentum(mut self, learn_mom: LearnMomentum) -> Builder {
		self.learn_mom = learn_mom;
		self
	}

	/// Use the given scheduling routine.
	/// 
	/// Default scheduling routine is to pick random samples.
	pub fn scheduling(mut self, kind: SchedulingKind) -> Builder {
		self.scheduling = kind;
		self
	}

	/// Validate all sample input and target sizes.
	fn validate_samples(&self) -> Result<()> {
		let req_inputs  = self.disciple.len_input();
		let req_outputs = self.disciple.len_output();
		for sample in self.samples.iter() {
			if sample.input.len() != req_inputs {
				return Err(InvalidSampleInputSize)
			}
			if sample.target.len() != req_outputs {
				return Err(InvalidSampleTargetSize)
			}
		}
		Ok(())
	}

	/// Checks invariants about the given settings for the learning procedure
	/// such as checking if learn rate is within bounds or the samples are
	/// of correct sizes for the underlying neural network etc.
	/// 
	/// Then starts the learning procedure and returns the fully trained
	/// neural network (Prophet) that is capable to predict data if no 
	/// errors occured while training it.
	pub fn go(self) -> Result<NeuralNet> {
		self.criterion.check_validity()?;
		self.learn_rate.check_validity()?;
		self.learn_mom.check_validity()?;
		self.validate_samples()?;
		Mentor::from(self).train()
	}
}

impl Topology<Finished> {
    /// Iterates over the layer sizes of this Disciple's topology definition.
    pub fn train(self, samples: Vec<Sample>) -> Builder {
        Builder::new(self, samples)
    }
}

/// A Mentor is an object type that is able to train a Disciple
/// to become a fully qualified and useable Prophet.
#[derive(Debug, Clone)]
struct Mentor {
	err_stats : ErrorStats,
	learn_rate: LearnRate,
	learn_mom : LearnMomentum,
	criterion : Criterion,
	disciple  : NeuralNet,
	scheduler : SampleScheduler
}

impl Mentor {
	fn train(self) -> Result<NeuralNet> {
		Ok(self.disciple)
	}
}

impl From<Builder> for Mentor {
	fn from(builder: Builder) -> Mentor {
		Mentor{
			err_stats : builder.err_stats,
			learn_rate: builder.learn_rate,
			learn_mom : builder.learn_mom,
			criterion : builder.criterion,
			disciple  : builder.disciple.into(),
			scheduler : SampleScheduler::from_samples(builder.scheduling, builder.samples)
		}
	}
}
