//! The Mentor is used to create and train neural networks in order to
//! prevent a situation where a neural network is defined and used to predict
//! data without any training beforehand to verify a certain metric of quality
//! for the predicted data.
//!
//! In future versions of this crate it shall be impossible to create new
//! neural network instances without using a Mentor to train it beforehand.

use ndarray::prelude::*;
use rand::*;

use std::time::{SystemTime, Duration};

use errors::Result;
use errors::ErrorKind::*;
use topology::*;
use neural_net::*;
use traits::{
	LearnRate,
	LearnMomentum,
	Predict,
	UpdateGradients,
	UpdateWeights
};

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
pub enum LearnRateConfig {
	/// Automatically adapt learn rate during learning.
	Adapt,

	/// Use the given fixed learn rate.
	Fixed(LearnRate),
}

impl LearnRateConfig {
	/// Checks if this learn rate is valid.
	fn check_validity(&self) -> Result<()> {
		use self::LearnRateConfig::*;
		match *self {
			Adapt => Ok(()),
			Fixed(rate) => {
				if rate.0 > 0.0 && rate.0 < 1.0 {
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
pub enum LearnMomentumConfig {
	/// Automatically adapt learn momentum during learning.
	Adapt,

	/// Use the given fixed learn momentum.
	Fixed(LearnMomentum),
}

impl LearnMomentumConfig {
	/// Checks if this learn momentum is valid.
	fn check_validity(&self) -> Result<()> {
		use self::LearnMomentumConfig::*;
		match *self {
			Adapt => Ok(()),
			Fixed(momentum) => {
				if momentum.0 > 0.0 && momentum.0 < 1.0 {
					Ok(())
				} else {
					Err(InvalidLearnMomentum)
				}
			}
		}
	}
}

/// Logging interval for logging stats during the learning process.
/// 
/// Default logging configuration is to never log anything.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LogConfig {
	/// Never log anything.
	Never,

	/// Log in intervals based on the given duration.
	TimeSteps(Duration),

	/// Log every given number of training iterations.
	Iterations(u64)
}

impl Default for LogConfig {
	fn default() -> Self {
		LogConfig::Never
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
enum Scheduler {
	/// Samples randomly.
	Random(ThreadRng),

	/// Samples iteratively.
	Iterative(u64),
}

impl ::std::fmt::Debug for Scheduler {
	fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
		use self::Scheduler::*;
		match self {
			&Random(_)    => write!(f, "Scheduler::Random(_)"),
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
			&mut Random(ref mut rng) => {
				rng.gen_range(0, num_samples)
			},
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
struct SampleScheduler {
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

/// A sample used to train a disciple during supervised learning.
#[derive(Debug, Clone)]
pub struct Sample {
	/// The input parameter of this `Sample`.
	pub input: Array1<f32>,

	/// The expected target values of this `Sample`.
	pub target: Array1<f32>,
}

impl<A1, A2> From<(A1, A2)> for Sample
    where A1: Into<Vec<f32>>,
          A2: Into<Vec<f32>>
{
	fn from(from: (A1, A2)) -> Sample {
		Sample {
			input : Array1::from_vec(from.0.into()),
			target: Array1::from_vec(from.1.into()),
		}
	}
}

/// A sample view used to train a disciple during supervised learning.
/// 
/// Views are non-owning.
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

/// Mentor follows the builder pattern to incrementally
/// build properties for the training session and delay any
/// expensive computations until the go routine is called.
#[derive(Debug, Clone)]
pub struct Mentor {
	deviation : Deviation,
	learn_rate: LearnRateConfig,
	learn_mom : LearnMomentumConfig,
	criterion : Criterion,
	scheduling: Scheduling,
	disciple  : Topology,
	samples   : Vec<Sample>,
	log_config: LogConfig
}

impl Mentor {
	/// Creates a new mentor for the given disciple and
	/// with the given sample collection (training data).
	pub fn new(disciple: Topology, samples: Vec<Sample>) -> Self {
		Mentor {
			deviation : Deviation::default(),
			learn_rate: LearnRateConfig::Adapt,
			learn_mom : LearnMomentumConfig::Adapt,
			criterion : Criterion::RecentMSE(0.05),
			scheduling: Scheduling::Random,
			disciple  : disciple,
			samples   : samples,
			log_config: LogConfig::Never
		}
	}

	/// Use the given criterion.
	///
	/// Default criterion is `AvgNetError(0.05)`.
	pub fn criterion(mut self, criterion: Criterion) -> Self {
		self.criterion = criterion;
		self
	}

	/// Use the given fixed learn rate.
	///
	/// Default learn rate is adapting behaviour.
	/// 
	/// ***Panics*** if given learn rate is invalid!
	pub fn learn_rate(mut self, learn_rate: f64) -> Self {
		self.learn_rate = LearnRateConfig::Fixed(
			LearnRate::from_f64(learn_rate)
				.expect("expected valid learn rate"));
		self
	}

	/// Use the given fixed learn momentum.
	///
	/// Default learn momentum is fixed at `0.5`.
	/// 
	/// ***Panics*** if given learn momentum is invalid
	pub fn learn_momentum(mut self, learn_momentum: f64) -> Self {
		self.learn_mom = LearnMomentumConfig::Fixed(
			LearnMomentum::from_f64(learn_momentum)
				.expect("expected valid learn momentum"));
		self
	}

	/// Use the given scheduling routine.
	///
	/// Default scheduling routine is to pick random samples.
	pub fn scheduling(mut self, kind: Scheduling) -> Self {
		self.scheduling = kind;
		self
	}

	/// Use the given logging configuration.
	/// 
	/// Default logging configuration is to never log anything.
	pub fn log_config(mut self, config: LogConfig) -> Self {
		self.log_config = config;
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
		Training::from(self).train()
	}
}

impl Topology {
	/// Iterates over the layer sizes of this Disciple's topology definition.
	pub fn train(self, samples: Vec<Sample>) -> Mentor {
		Mentor::new(self, samples)
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
			recent_mse   : 1.0,
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

/// Config parameters for mentor objects used throughtout a training session.
#[derive(Debug, Copy, Clone)]
struct Config {
	pub learn_rate: LearnRateConfig,
	pub learn_mom : LearnMomentumConfig,
	pub criterion : Criterion
}

/// Status during the learning process.
#[derive(Debug, Copy, Clone)]
struct Stats {
	/// Number of samples learned so far.
	pub iterations  : u64,

	/// Time passed since beginning of the training.
	pub elapsed_time: Duration,

	/// The latest mean squared error.
	pub latest_mse  : f64,

	/// The recent mean squared error.
	pub recent_mse  : f64
}

/// Logger facility for stats logging during the learning process.
#[derive(Debug, Clone)]
enum Logger {
	Never,
	TimeSteps{
		last_log: SystemTime,
		interval: Duration
	},
	Iterations(u64)
}

impl From<LogConfig> for Logger {
	fn from(config: LogConfig) -> Self {
		use self::LogConfig::*;
		match config {
			Never => Logger::Never,
			TimeSteps(duration) => Logger::TimeSteps{
				last_log: SystemTime::now(),
				interval: duration
			},
			Iterations(interval) => Logger::Iterations(interval)
		}
	}
}

impl Logger {
	fn log(stats: Stats) {
		info!("{:?}\n", stats);
	}

	fn try_log(&mut self, stats: Stats) {
		use self::Logger::*;
		match self {
			&mut TimeSteps{ref mut last_log, interval} => {
				if last_log.elapsed().expect("expected valid duration") >= interval {
					Self::log(stats);
					*last_log = SystemTime::now();
				}
			},
			&mut Iterations(interval) => {
				if stats.iterations % interval == 0 {
					Self::log(stats)
				}
			},
			_ => {
				// nothing to do here!
			}
		}
	}
}

/// A training session trains a neural network and stops only
/// after the neural networks training stats meet certain 
/// predefined criteria.
#[derive(Debug, Clone)]
struct Training {
	cfg       : Config,
	disciple  : NeuralNet,
	scheduler : SampleScheduler,
	deviation : Deviation,
	iterations: Iteration,
	starttime : SystemTime,
	learn_rate: LearnRate,
	learn_mom : LearnMomentum,
	logger    : Logger
}

impl Training {
	fn is_done(&self) -> bool {
		use mentor::Criterion::*;
		match self.cfg.criterion {
			TimeOut(duration) => {
				self.starttime.elapsed().unwrap() >= duration
			},
			Iterations(limit) => {
				self.iterations.0 == limit
			},
			LatestMSE(target) => {
				self.deviation.latest_mse() <= target
			}
			RecentMSE(target) => {
				self.deviation.recent_mse() <= target
			}
		}
	}

	fn session(&mut self) {
		{
			let sample = self.scheduler.next();
			{
				let output = self.disciple.predict(sample.input);
				self.deviation.update(output, sample.target);
			}
			self.disciple.update_gradients(sample.target);
			self.disciple.update_weights(sample.input, self.learn_rate, self.learn_mom);
			self.iterations.bump();
		}
		self.try_log();
	}

	fn update_learn_rate(&mut self) {
		use self::LearnRateConfig::*;
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
		use self::LearnMomentumConfig::*;
		match self.cfg.learn_mom {
			Adapt => {
				// not yet implemented
			}
			Fixed(_) => {
				// nothing to do here!
			}
		}
	}

	fn stats(&self) -> Stats {
		Stats{
			iterations  : self.iterations.0,
			elapsed_time: self.starttime.elapsed().expect("time must be valid!"),
			latest_mse  : self.deviation.latest_mse(),
			recent_mse  : self.deviation.recent_mse()
		}
	}

	fn try_log(&mut self) {
		let stats = self.stats();
		self.logger.try_log(stats)
	}

	fn train(mut self) -> Result<NeuralNet> {
		loop {
			self.update_learn_rate();
			self.update_learn_momentum();
			self.session();
			if self.is_done() { break }
		}
		Ok(self.disciple)
	}
}

impl From<Mentor> for Training {
	fn from(builder: Mentor) -> Training {
		Training {
			disciple : NeuralNet::from(builder.disciple),
			scheduler: SampleScheduler::from_samples(builder.scheduling, builder.samples),

			cfg: Config{
				learn_rate: builder.learn_rate,
				learn_mom : builder.learn_mom,
				criterion : builder.criterion
			},

			learn_rate: match builder.learn_rate {
				LearnRateConfig::Adapt    => LearnRate::default(),
				LearnRateConfig::Fixed(r) => r
			},

			learn_mom: match builder.learn_mom {
				LearnMomentumConfig::Adapt    => LearnMomentum::default(),
				LearnMomentumConfig::Fixed(m) => m
			},

			iterations: Iteration::default(),
			starttime : SystemTime::now(),
			deviation : builder.deviation,

			logger: Logger::from(builder.log_config)
		}
	}
}

/// Creates a vector of samples.
/// 
/// Given the following definitions
/// 
/// ```rust,no_run
/// let t =  1.0;
/// let f = -1.0;
/// ```
/// ... this macro invokation ...
/// 
/// ```rust
/// # #[macro_use]
/// # extern crate prophet;
/// # use prophet::prelude::*;
/// # fn main() {
/// # let t =  1.0;
/// # let f = -1.0;
/// let samples = samples![
/// 	[f, f] => [f],
/// 	[t, f] => [t],
/// 	[f, t] => [t],
/// 	[t, t] => [f]
/// ];
/// # }
/// ```
/// 
/// ... will expand to this
/// 
/// ```rust,no_run
/// # #[macro_use]
/// # extern crate prophet;
/// # use prophet::prelude::*;
/// # fn main() {
/// # let t =  1.0;
/// # let f = -1.0;
/// let samples = vec![
/// 	Sample::from((vec![f, f], vec![f])),
/// 	Sample::from((vec![t, f], vec![t])),
/// 	Sample::from((vec![f, t], vec![t])),
/// 	Sample::from((vec![t, t], vec![f])),
/// ];
/// # }
/// ```
#[macro_export]
macro_rules! samples {
	[
		$(
			[ $($i:expr),+ ] => [ $($e:expr),+ ]
		),+
	] => {
		vec![$(
			Sample::from(
				(
					vec![$($i),+],
					vec![$($e),+]
				)
			)
		),+]
	};
}

#[cfg(test)]
mod tests {
	use super::*;

	fn validate_impl(mut net: NeuralNet, samples: Vec<Sample>, rounded: bool) {
		use itertools::{Itertools, multizip};
		for sample in samples.into_iter() {
			let predicted = net.predict(sample.input.view());
			multizip((predicted.iter(), sample.target.iter()))
				.foreach(|(&predicted, &expected)| {
					if rounded {
						assert_eq!(predicted.round(), expected);
					}
					else {
						relative_eq!(predicted, expected);
					}
				});
		}
	}

	fn validate_rounded(net: NeuralNet, samples: Vec<Sample>) {
		validate_impl(net, samples, true)
	}

	fn validate_exact(net: NeuralNet, samples: Vec<Sample>) {
		validate_impl(net, samples, false)
	}

	#[test]
	fn xor() {
		use activation::Activation::Tanh;

		let (t, f) = (1.0, -1.0);
		let samples = samples![
			[f, f] => [f],
			[t, f] => [t],
			[f, t] => [t],
			[t, t] => [f]
		];

		let net = Topology::input(2)
			.layer(4, Tanh)
			.layer(3, Tanh)
			.output(1, Tanh)

			.train(samples.clone())
			.go()
			.unwrap();

		validate_rounded(net, samples);
	}

	#[test]
	fn train_constant() {
		use activation::Activation::Identity;

		// samples to train the net with
		let learn_samples = samples![
			[0.0] => [1.0],
			[0.2] => [1.0],
			[0.4] => [1.0],
			[0.6] => [1.0],
			[0.8] => [1.0],
			[1.0] => [1.0]
		];

		// samples to test the trained net with
		let test_samples = samples![
			[0.1] => [1.0],
			[0.3] => [1.0],
			[0.5] => [1.0],
			[0.7] => [1.0],
			[0.9] => [1.0]
		];

		let net = Topology::input(1)
			.output(1, Identity)

			.train(learn_samples)
			.go()
			.unwrap();

		validate_rounded(net, test_samples)
	}

	#[test]
	fn train_and() {
		use activation::Activation::Tanh;

		let (t, f) = (1.0, -1.0);
		let samples = samples![
			[f, f] => [f],
			[f, t] => [f],
			[t, f] => [f],
			[t, t] => [t]
		];

		let net = Topology::input(2)
			.output(1, Tanh)

			.train(samples.clone())
			.go()
			.unwrap();

		validate_rounded(net, samples)
	}

	#[test]
	fn train_triple_add() {
		use activation::Activation::Identity;
		use rand::*;

		let count_learn_samples = 10_000;
		let count_test_samples  = 10;

		let mut rng = thread_rng();

		// generate learn samples
		let mut learn_samples = Vec::with_capacity(count_learn_samples);
		for _ in 0..count_learn_samples {
			let a = rng.next_f32();
			let b = rng.next_f32();
			let c = rng.next_f32();
			learn_samples.push(Sample::from((vec![a, b, c], vec![a + b + c])))
		}

		// generate test samples
		let mut test_samples = Vec::with_capacity(count_test_samples);
		for _ in 0..count_test_samples {
			let a = rng.next_f32();
			let b = rng.next_f32();
			let c = rng.next_f32();
			test_samples.push(Sample::from((vec![a, b, c], vec![a + b + c])))
		}

		let net = Topology::input(3)
			.output(1, Identity)

			.train(learn_samples)
			.log_config(LogConfig::Iterations(100))
			.go()
			.unwrap();

		validate_exact(net, test_samples)
	}
}
