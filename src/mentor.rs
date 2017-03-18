//! The Mentor is used to create and train neural networks in order to
//! prevent a situation where a neural network is defined and used to predict
//! data without any training beforehand to verify a certain metric of quality
//! for the predicted data.
//!
//! In future versions of this crate it shall be impossible to create new
//! neural network instances without using a Mentor to train it beforehand.

use ndarray::prelude::*;
use rand::*;
use chrono::prelude::*;
// use chrono::Duration;

use std::time::{SystemTime, Duration};

use topology::*;
use error_stats::*;
use neural_net::*;

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
	InvalidRecentMSE,

	/// Occures when the specified mean squared error
	/// criterion is invalid.
	InvalidLatestMSE,
}
use self::ErrorKind::*;

/// Cirterias after which the learning process holds.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Criterion {
	/// Stop after the given duration of time.
	TimeOut(Duration),

	/// Stop after the given amount of learning iterations.
	Iterations(u64),

	/// Stop when the latest mean square error drops below the given value.
	LatestMSE(f64),

	/// Stop as soon as the recent mean squared error
	/// drops below the given value.
	RecentMSE(f64),
}

impl Criterion {
	/// Checks if this criterion is valid.
	fn check_validity(&self) -> Result<()> {
		use self::Criterion::*;
		match *self {
			TimeOut(_) => Ok(()),
			Iterations(_) => Ok(()),
			LatestMSE(mse) => {
				if mse > 0.0 && mse < 1.0 {
					Ok(())
				} else {
					Err(InvalidLatestMSE)
				}
			}
			RecentMSE(recent) => {
				if recent > 0.0 && recent < 1.0 {
					Ok(())
				} else {
					Err(InvalidRecentMSE)
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
	Fixed(f64),
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
				} else {
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
	Fixed(f64),
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
				} else {
					Err(InvalidLearnMomentum)
				}
			}
		}
	}
}

/// Sample scheduling strategy while learning.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Scheduling {
	/// Pick samples randomly.
	///
	/// This usually is a good approach to defeat sample-pattern learning.
	Random,

	/// Pick samples in order.
	///
	/// This maybe useful for testing purposes.
	Iterative,
}

/// A scheduler for indices with a scheduling strategy.
///
/// Used by `SampleScheduler` to pick samples with different scheduling strategies.
#[derive(Clone)]
pub enum Scheduler {
	/// Samples randomly.
	Random(ThreadRng),

	/// Samples iteratively.
	Iterative(u64),
}

use std::fmt::{Debug, Formatter};
impl Debug for Scheduler {
	fn fmt(&self, f: &mut Formatter) -> ::std::fmt::Result {
		use self::Scheduler::*;
		match self {
			&Random(_) => write!(f, "Scheduler::Random(_)"),
			&Iterative(x) => write!(f, "Scheduler::Iterative({})", x),
		}
	}
}

impl Scheduler {
	/// Creates a new `Scheduler` from a given scheduling strategy.
	fn from_kind(kind: Scheduling) -> Self {
		use self::Scheduling::*;
		match kind {
			Random => Scheduler::Random(thread_rng()),
			Iterative => Scheduler::Iterative(0),
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
			&mut Iterative(ref mut cur) => {
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
	scheduler: Scheduler,
}

impl SampleScheduler {
	/// Creates a new `SampleScheduler` from given samples and a scheduling strategy.
	fn from_samples(kind: Scheduling, samples: Vec<Sample>) -> Self {
		SampleScheduler {
			samples: samples,
			scheduler: Scheduler::from_kind(kind),
		}
	}

	/// Returns the next sample.
	fn next(&mut self) -> SampleView {
		let len_samples = self.samples.len();
		let id = self.scheduler.next(len_samples);
		(&self.samples[id]).into()
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
	pub input: Array1<f32>,

	/// The expected target values of this `Sample`.
	pub target: Array1<f32>,
}

impl<Arr> From<(Arr, Arr)> for Sample
    where Arr: Into<Array1<f32>>
{
	fn from(from: (Arr, Arr)) -> Sample {
		Sample {
			input: from.0.into(),
			target: from.1.into(),
		}
	}
}

/// A sample used to train a disciple during supervised learinng.
#[derive(Debug, Clone)]
pub struct SampleView<'a> {
	/// The input parameter of this `SampleView`.
	pub input: ArrayView1<'a, f32>,

	/// The expected target values of this `SampleView`.
	pub target: ArrayView1<'a, f32>,
}

impl<'a> From<&'a Sample> for SampleView<'a> {
	fn from(from: &'a Sample) -> SampleView<'a> {
		SampleView {
			input: from.input.view(),
			target: from.target.view(),
		}
	}
}

/// Builder follows the builder pattern to incrementally
/// build properties for the real Mentor and delay computations
/// until the `go` routine is called.
#[derive(Debug, Clone)]
pub struct Builder {
	deviation : Deviation,
	learn_rate: LearnRate,
	learn_mom : LearnMomentum,
	criterion : Criterion,
	scheduling: Scheduling,
	disciple  : Topology,
	samples   : Vec<Sample>,
}

impl Builder {
	/// Creates a new mentor for the given disciple and
	/// with the given sample collection (training data).
	pub fn new(disciple: Topology, samples: Vec<Sample>) -> Builder {
		Builder {
			deviation : Deviation::default(),
			learn_rate: LearnRate::Adapt,
			learn_mom : LearnMomentum::Adapt,
			criterion : Criterion::RecentMSE(0.05),
			scheduling: Scheduling::Random,
			disciple  : disciple,
			samples   : samples,
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
	pub fn scheduling(mut self, kind: Scheduling) -> Builder {
		self.scheduling = kind;
		self
	}

	/// Validate all sample input and target sizes.
	fn validate_samples(&self) -> Result<()> {
		let req_inputs = self.disciple.len_input();
		let req_outputs = self.disciple.len_output();
		for sample in self.samples.iter() {
			if sample.input.len() != req_inputs {
				return Err(InvalidSampleInputSize);
			}
			if sample.target.len() != req_outputs {
				return Err(InvalidSampleTargetSize);
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

impl Topology {
	/// Iterates over the layer sizes of this Disciple's topology definition.
	pub fn train(self, samples: Vec<Sample>) -> Builder {
		Builder::new(self, samples)
	}
}

/// Handles deviations of predicted and target values of
/// the neural network under training.
/// 
/// This is especially useful when using `MeanSquaredError`
/// or `AvgNetError` criterions.
#[derive(Debug, Copy, Clone)]
struct Deviation {
	latest_mse   : f64,
	recent_mse   : f64,
	recent_factor: f64,
}

impl Deviation {
	/// Creates a new deviation instance.
	///
	/// ***Panics*** If the given smoothing factor is ∉ *(0, 1)*.
	pub fn new(recent_factor: f64) -> Self {
		assert!(0.0 < recent_factor && recent_factor < 1.0);
		Deviation{
			latest_mse   : 0.0,
			recent_mse   : 0.0,
			recent_factor: recent_factor,
		}
	}

	/// Calculates mean squared error based on the given actual and expected data.
	fn update_mse<F>(&mut self, actual: ArrayView1<F>, expected: ArrayView1<F>)
		where F: NdFloat
	{
		use std::ops::Div;
		use itertools::multizip;
		self.latest_mse = multizip((actual.iter(), expected.iter()))
			.map(|(&actual, &expected)| {
				let dx = expected - actual;
				(dx * dx).to_f64().unwrap()
			})
			.sum::<f64>()
			.div(actual.len() as f64)
			.sqrt();
	}

	/// Calculates recent mean squared error based on the recent factor smoothing.
	fn update_recent_mse(&mut self) {
		self.recent_mse = self.recent_factor * self.recent_mse
			+ (1.0 - self.recent_factor) * self.latest_mse;
	}

	/// Updates the current mean squared error and associated data.
	pub fn update<F>(&mut self, actual: ArrayView1<F>, expected: ArrayView1<F>)
		where F: NdFloat
	{
		self.update_mse(actual, expected);
		self.update_recent_mse();
	}

	/// Gets the latest mean squared error.
	pub fn latest_mse(&self) -> f64 {
		self.latest_mse
	}

	/// Gets the recent mean squared error.
	pub fn recent_mse(&self) -> f64 {
		self.recent_mse
	}
}

impl Default for Deviation {
	fn default() -> Self {
		Deviation::new(0.95)
	}
}

/// A very simple type that can count upwards and
/// is comparable to other instances of itself.
///
/// Used by `Mentor` to manage iteration number.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
struct Iteration(u64);

impl Iteration {
	/// Bumps the iteration count by 1.
	fn bump(&mut self) {
		self.0 += 1
	}
}

/// A Mentor is an object type that is able to train a Disciple
/// to become a fully qualified and useable Prophet.
#[derive(Debug, Clone)]
struct Mentor {
	cfg       : Config,
	disciple  : NeuralNet,
	scheduler : SampleScheduler,
	deviation : Deviation,
	iterations: Iteration,
	starttime : SystemTime,
	learn_rate: f32,
	learn_mom : f32
}

/// Config parameters for mentor objects used throughtout a training session.
#[derive(Debug, Copy, Clone)]
struct Config {
	pub learn_rate: LearnRate,
	pub learn_mom : LearnMomentum,
	pub criterion : Criterion
}

use traits::{Predict, UpdateGradients, UpdateWeights};

impl Mentor {
	fn is_done(&self) -> bool {
		use mentor::Criterion::*;
		match self.cfg.criterion {
			TimeOut(duration) => {
				return self.starttime.elapsed().unwrap() >= duration
			},
			Iterations(limit) => {
				return self.iterations.0 == limit
			},
			LatestMSE(target) => {
				return self.deviation.latest_mse() <= target
			}
			RecentMSE(target) => {
				return self.deviation.recent_mse() <= target
			}
		}
		false
	}

	fn session(&mut self) {
		let sample = self.scheduler.next();
		{
			let output = self.disciple.predict(sample.input);
			self.deviation.update(output, sample.target);
		}
		self.disciple.update_gradients(sample.target);
		self.disciple.update_weights(sample.input, self.learn_rate, self.learn_mom);
		self.iterations.bump();
	}

	fn update_learn_rate(&mut self) {
		use self::LearnRate::*;
		match self.cfg.learn_rate {
			Adapt => {
				// not yet implemented
			}
			Fixed(_) => {
				// nothing to do here!
			}
		}
	}

	fn update_learn_momentum(&mut self) {
		use self::LearnMomentum::*;
		match self.cfg.learn_mom {
			Adapt => {
				// not yet implemented
			}
			Fixed(_) => {
				// nothing to do here!
			}
		}
	}

	fn train(mut self) -> Result<NeuralNet> {
		while !self.is_done() {
			self.update_learn_rate();
			self.update_learn_momentum();
			self.session()
		}
		Ok(self.disciple)
	}
}

impl From<Builder> for Mentor {
	fn from(builder: Builder) -> Mentor {
		Mentor {
			disciple : NeuralNet::from(builder.disciple),
			scheduler: SampleScheduler::from_samples(builder.scheduling, builder.samples),

			cfg: Config{
				learn_rate: builder.learn_rate,
				learn_mom : builder.learn_mom,
				criterion : builder.criterion
			},

			learn_rate: match builder.learn_rate {
				LearnRate::Adapt    => 0.3,
				LearnRate::Fixed(r) => r as f32
			},

			learn_mom: match builder.learn_mom {
				LearnMomentum::Adapt    => 0.5,
				LearnMomentum::Fixed(m) => m as f32
			},

			iterations: Iteration::default(),
			starttime : SystemTime::now(),
			deviation : builder.deviation,
		}
	}
}
