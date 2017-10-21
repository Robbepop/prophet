use trainer::TrainingState;
use nn::NeuralNet;
use utils::{
	LearnRate,
	LearnMomentum
};
use trainer::sample::SampleGen;
use trainer::{
	TrainCondition
};
use errors::{
	Error,
	Result,
	MentorBuilderDoubledField,
	MentorBuilderInvalidArgument
};
use trainer::condition;
use trainer::MeanSquaredError;

use std::time;

/// The actual state describing the current context during a training process.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Context {
	time_started: time::Instant,
	iteration: usize,
	epochs_passed: usize,
	epoch_len: usize,
	latest_mse: MeanSquaredError,
	batch_len: usize,
	batch_bit: usize,
	lr: LearnRate,
	lm: LearnMomentum
}

impl Context {
	/// Creates a new `Context` from the given `LearnRate` and `LearnMomentum`
	/// and initializes the other values with their respective defaults.
	pub fn new(
		epoch_len: usize,
		batch_len: usize,
		lr: LearnRate,
		lm: LearnMomentum
	)
		-> Context
	{
		Context{
			time_started: time::Instant::now(),
			iteration: 0,
			epochs_passed: 0,
			epoch_len,
			latest_mse: MeanSquaredError::new(1.0).unwrap(),
			batch_len,
			batch_bit: 0,
			lr, lm
		}
	}

	/// Adjusts stats to the next step.
	/// 
	/// This adjusts iteration count, current epoch, as well as the current batch.
	#[inline]
	pub fn next(&mut self) {
		self.next_iteration();
		self.next_batch();
	}

	/// Increases the iteration counter by `1` (one).
	#[inline]
	fn next_iteration(&mut self) {
		self.iteration += 1;
		if self.iteration % self.epoch_len == 0 {
			self.next_epoch()
		}
	}

	/// Increases the counter representing the epochs passed by `1` (one).
	#[inline]
	fn next_epoch(&mut self) {
		self.epochs_passed += 1
	}

	/// Increases the batch bit by `1` (one) also resets to zero `0` whenever the batch len is reached..
	#[inline]
	fn next_batch(&mut self) {
		self.batch_bit += 1;
		self.batch_bit %= self.batch_len;
	}

	/// Upates the latest mean squared error (MSE).
	#[inline]
	pub fn update_mse(&mut self, latest_mse: MeanSquaredError) {
		self.latest_mse = latest_mse
	}

	/// Returns `true` if the latest batch was finished and the neural net is ready to 
	/// optimize itself according to the latest accumulated supervised predicitions.
	/// 
	/// Note that this always returns `true` for non-batched learning.
	#[inline]
	pub fn is_batch_finished(&self) -> bool {
		self.batch_bit == 0
	}
}

impl TrainingState for Context {
	#[inline]
	fn time_started(&self) -> time::Instant {
		self.time_started
	}

	#[inline]
	fn iterations(&self) -> usize {
		self.iteration
	}

	#[inline]
	fn epochs_passed(&self) -> usize {
		self.epochs_passed
	}

	#[inline]
	fn latest_mse(&self) -> MeanSquaredError {
		self.latest_mse
	}
}

/// A `Mentor` can be used to train a given neural network.
/// 
/// It manages the training process, guarantees certain result qualities
/// and offers great flexibility for customizing the training process.
/// 
/// Note that users of this library do not depend on this structure
/// in order to train neural networks. The `Mentor` and its API simply
/// provides an easy-to-use interface for a training procedure based on
/// the API of this library.
#[derive(Debug, Clone, PartialEq)]
pub struct Mentor<Sampler, StopWhen, LogWhen>
	where StopWhen: TrainCondition,
	      LogWhen : TrainCondition,
	      Sampler : SampleGen
{
	nn: NeuralNet,
	sample_gen: Sampler,
	stop_when: StopWhen,
	log_when: LogWhen,
	ctx: Context
}

mod marker {
	pub trait UnsetTrainCondition {}
	pub trait UnsetSampleGen {}

	#[derive(Debug, Copy, Clone, PartialEq)]
	pub struct Unset;

	impl UnsetTrainCondition for Unset {}
	impl UnsetSampleGen for Unset {}
}
use self::marker::Unset;
pub type UninitializedMentor = InitializingMentor<Unset, Unset, Unset>;

/// Used to construct fully-fledged `Mentor`s using the well-known builder pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct InitializingMentor<Sampler, StopWhen, LogWhen> {
	nn: NeuralNet,
	sample_gen: Sampler,
	stop_when: StopWhen,
	log_when: LogWhen,
	epoch_len: Option<usize>,
	batch_len: Option<usize>,
	lr: Option<LearnRate>,
	lm: Option<LearnMomentum>
}

pub trait Train {
	fn train(self) -> UninitializedMentor;
}

impl<T> Train for T where T: Into<NeuralNet> {
	fn train(self) -> UninitializedMentor {
		UninitializedMentor::new(self)
	}
}

impl<Sampler, StopWhen, LogWhen> Mentor<Sampler, StopWhen, LogWhen>
	where StopWhen: TrainCondition,
	      LogWhen : TrainCondition,
	      Sampler : SampleGen
{
	// /// Initiates a training session for the given neural network or topology.
	// pub fn train<NN>(nn: NN) -> UninitializedMentor
	// 	where NN: Into<NeuralNet>
	// {
	// 	UninitializedMentor::new(nn)
	// }

	/// Creates a new `Mentor` from the given `InitializingMentor`.
	/// 
	/// This also initializes values that were not provided by the user
	/// to their defaults.
	/// 
	/// Note that this does not start the training process by itself.
	/// 
	/// # Errors
	/// 
	/// - If some required property was not set in the builer.
	fn from_builder(builder: InitializingMentor<Sampler, StopWhen, LogWhen>) -> Result<Self> {
		let ctx = Context::new(
			builder.epoch_len.unwrap_or_else(|| builder.sample_gen.finite_len().unwrap_or(1)),
			builder.batch_len.unwrap_or(1),
			builder.lr.unwrap_or_else(|| LearnRate::from(0.15)),
			builder.lm.unwrap_or_else(|| LearnMomentum::from(0.0))
		);
		Ok(Mentor{
			nn: builder.nn,
			sample_gen: builder.sample_gen,
			stop_when: builder.stop_when,
			log_when: builder.log_when,
			ctx
		})
	}

	fn log_statistics(&mut self) {
		use log::LogLevel;
		if log_enabled!(LogLevel::Info) && self.log_when.evaluate(&self.ctx) {
			info!(
				"\n\
				 =======================================================\n\
				 Prophet Library :: Log of `prophet::trainer::Mentor`:  \n\
				  * Context                                             \n\
				 {:?}\n\
				 -------------------------------------------------------\n\
				  * Neural Net                                          \n\
				 {:?}\n\
				 =======================================================\n\
				",
				self.ctx, self.nn
			)
		}
	}

	/// Starts the training procedure for the given `Mentor` settings.
	/// 
	/// Note that this call may take a while depending on the requested quality of training result.
	/// 
	/// Returns the fully trained neural network when the training is finished.
	pub fn finish(mut self) -> Result<NeuralNet> {
		use trainer::{
			PredictSupervised,
			OptimizeSupervised,
			EvaluateSupervised
		};
		while !self.stop_when.evaluate(&self.ctx) {
			self.ctx.next();
			{
				let sample = self.sample_gen.next_sample();
				let latest_mse = if self.ctx.is_batch_finished() {
					self.nn.predict_supervised(sample)
					       .optimize_supervised(self.ctx.lr, self.ctx.lm)
					       .stats()
				}
				else {
					self.nn.predict_supervised(sample)
					       .stats()
				};
				self.ctx.update_mse(latest_mse);
			}
			self.log_statistics();
		}
		Ok(self.nn)
	}
}

/// Types that implement this trait can build up `Mentor`s.
/// 
/// This trait's main purpose is to allow for `MentorBuilder` and
/// `Result<MentorBuilder>` to be unified in the streamlined process of building
/// with method chaining to no longer require using `.unwrap()` or similar
/// unwrapping functionality after each configuring step during the build
/// process.
pub trait MentorBuilderOrError {
	/// The mentor builder type.
	type Builder;

	/// Returns a `MentorBuilder` wrapped as a `Result<MentorBuilder>`.
	fn builder_or_error(self) -> Result<Self::Builder>;
}

impl<Sampler, StopWhen, LogWhen> MentorBuilderOrError for InitializingMentor<Sampler, StopWhen, LogWhen> {
	type Builder = Self;

	#[inline]
	fn builder_or_error(self) -> Result<Self::Builder> {
		Ok(self)
	}
}

impl<Sampler, StopWhen, LogWhen> MentorBuilderOrError for Result<InitializingMentor<Sampler, StopWhen, LogWhen>> {
	type Builder = InitializingMentor<Sampler, StopWhen, LogWhen>;

	#[inline]
	fn builder_or_error(self) -> Result<Self::Builder> {
		self
	}
}

/// Types that build fully-fledged `Mentor`s via the well-known builder-pattern.
/// 
/// The main purpose of this trait is to allow for greater flexibility in the building
/// process so that users may leave away some manual error handling to improve usability
/// of the mentor building process.
pub trait MentorBuilder<StopWhen, LogWhen, Sampler> {
	/// The current state of the builder type.
	type Builder: MentorBuilder<StopWhen, LogWhen, Sampler>;

	/// Sets the learning rate that is used throughout the training session.
	/// 
	/// Note: It isn't yet possible to adjust the learning rate after this phase.
	/// 
	/// # Errors
	/// 
	/// - If the learning rate was already set for this builder.
	fn learn_rate<LR>(self, lr: LR) -> Result<Self::Builder>
		where LR: Into<LearnRate>;

	/// Sets the learn momentum that is used throughout the training session.
	/// 
	/// Note: It isn't yet possible to adjust the learn momentum after this phase.
	/// 
	/// # Errors
	/// 
	/// - If the learn momentum was already set for this builder.
	fn learn_momentum<LM>(self, lm: LM) -> Result<Self::Builder>
		where LM: Into<LearnMomentum>;

	/// Sets the epoch length that is used statistics purposes throughout the training session.
	/// 
	/// Defaults to the length of the sample set if it is finite; else defaults to one (`1`).
	/// 
	/// # Errors
	/// 
	/// - If the given epoch length is zero (`0`). Epoch length must be a positive number.
	fn epoch_len(self, epoch_len: usize) -> Result<Self::Builder>;

	/// Sets the batch length that is used for batched learning purposes throughout the training session.
	/// 
	/// Defaults to one (`1`) if no user-provided batch size is provided.
	/// 
	/// # Errors
	/// 
	/// - If the given epoch length is zero (`0`). Epoch length must be a positive number.
	fn batch_len(self, batch_len: usize) -> Result<Self::Builder>;
}

pub trait MentorBuilderSampleGen<StopWhen, LogWhen> {
	/// Sets the sample generator that is used throughout the training session.
	fn sample_gen<Sampler>(self, sample_gen: Sampler) -> Result<InitializingMentor<Sampler, StopWhen, LogWhen>>
		where Sampler: SampleGen;
}

pub trait MentorBuilderStopWhen<Sampler, LogWhen> {
	/// Sets the halting condition that is used to query when the training shall end.
	fn stop_when<StopWhen>(self, stop_when: StopWhen) -> Result<InitializingMentor<Sampler, StopWhen, LogWhen>>
		where StopWhen: TrainCondition;
}

pub trait MentorBuilderLogWhen<Sampler, StopWhen> {
	/// Sets the logging condition that is used to query when the training state shall be logged.
	fn log_when<LogWhen>(self, log_when: LogWhen) -> Result<InitializingMentor<Sampler, StopWhen, LogWhen>>
		where LogWhen: TrainCondition;
}

pub trait MentorBuilderFinish<Sampler, StopWhen, LogWhen> {
	/// Finishes the building process and tries to create a full-fledged mentor
	/// that is capable of training the given neural network using the settings
	/// specified during the building phase.
	fn start(self) -> Result<NeuralNet>;
}

impl UninitializedMentor {
	/// Create a new `InitializingMentor` for the given `NeuralNet`.
	/// 
	/// This initiates the configuration and settings phase of a training session
	/// via automated mentoring. The entire process is formed according to the
	/// well-known builder pattern that guides you through the creation process.
	pub fn new<NN>(nn: NN) -> UninitializedMentor
		where NN: Into<NeuralNet>
	{
		InitializingMentor{
			nn: nn.into(),
			sample_gen: Unset,
			stop_when: Unset,
			log_when: Unset,
			epoch_len: None,
			batch_len: None,
			lr: None,
			lm: None
		}
	}
}

impl<T, StopWhen, LogWhen> MentorBuilderSampleGen<StopWhen, LogWhen> for T
	where T: MentorBuilderOrError<Builder = InitializingMentor<Unset, StopWhen, LogWhen>>
{
	fn sample_gen<Sampler>(self, sample_gen: Sampler) -> Result<InitializingMentor<Sampler, StopWhen, LogWhen>>
		where Sampler: SampleGen
	{
		let this = self.builder_or_error()?;
		Ok(InitializingMentor{
			nn: this.nn,
			sample_gen: sample_gen,
			stop_when: this.stop_when,
			log_when: this.log_when,
			epoch_len: this.epoch_len,
			batch_len: this.batch_len,
			lr: this.lr,
			lm: this.lm
		})
	}
}

impl<T, Sampler, LogWhen> MentorBuilderStopWhen<Sampler, LogWhen> for T
	where T: MentorBuilderOrError<Builder = InitializingMentor<Sampler, Unset, LogWhen>>
{
	fn stop_when<StopWhen>(self, stop_when: StopWhen) -> Result<InitializingMentor<Sampler, StopWhen, LogWhen>>
		where StopWhen: TrainCondition
	{
		let this = self.builder_or_error()?;
		Ok(InitializingMentor{
			nn: this.nn,
			sample_gen: this.sample_gen,
			stop_when: stop_when,
			log_when: this.log_when,
			epoch_len: this.epoch_len,
			batch_len: this.batch_len,
			lr: this.lr,
			lm: this.lm
		})
	}
}

impl<T, Sampler, StopWhen> MentorBuilderLogWhen<Sampler, StopWhen> for T
	where T: MentorBuilderOrError<Builder = InitializingMentor<Sampler, StopWhen, Unset>>
{
	fn log_when<LogWhen>(self, log_when: LogWhen) -> Result<InitializingMentor<Sampler, StopWhen, LogWhen>>
		where LogWhen: TrainCondition
	{
		let this = self.builder_or_error()?;
		Ok(InitializingMentor{
			nn: this.nn,
			sample_gen: this.sample_gen,
			stop_when: this.stop_when,
			log_when: log_when,
			epoch_len: this.epoch_len,
			batch_len: this.batch_len,
			lr: this.lr,
			lm: this.lm
		})
	}
}

impl<T, Sampler, StopWhen> MentorBuilderFinish<Sampler, StopWhen, Unset> for T
	where T: MentorBuilderOrError<Builder = InitializingMentor<Sampler, StopWhen, Unset>>,
	      Sampler: SampleGen,
	      StopWhen: TrainCondition
{
	fn start(self) -> Result<NeuralNet> {
		let this = self.builder_or_error()?;
		let this = this.log_when(condition::Never)?;
		Mentor::from_builder(this)?.finish()
	}
}

impl<T, Sampler, StopWhen, LogWhen> MentorBuilderFinish<Sampler, StopWhen, LogWhen> for T
	where T: MentorBuilderOrError<Builder = InitializingMentor<Sampler, StopWhen, LogWhen>>,
	      Sampler: SampleGen,
	      StopWhen: TrainCondition,
	      LogWhen: TrainCondition
{
	fn start(self) -> Result<NeuralNet> {
		let this = self.builder_or_error()?;
		Mentor::from_builder(this)?.finish()
	}
}

impl<T, Sampler, StopWhen, LogWhen> MentorBuilder<Sampler, StopWhen, LogWhen> for T
	where T: MentorBuilderOrError<Builder = InitializingMentor<Sampler, StopWhen, LogWhen>>
{
	type Builder = InitializingMentor<Sampler, StopWhen, LogWhen>;

	fn learn_rate<LR>(self, lr: LR) -> Result<Self::Builder>
		where LR: Into<LearnRate>
	{
		let mut this = self.builder_or_error()?;
		match this.lr {
			None => {
				this.lr = Some(lr.into());
				Ok(this)
			}
			Some(old_lr) => {
				use self::MentorBuilderDoubledField::LearnRate;
				Err(Error::mentor_builder_initialized_field_twice(LearnRate)
					.with_annotation(format!("Already set learn rate to {:?}.", old_lr.to_f32())))
			}
		}
	}

	fn learn_momentum<LM>(self, lm: LM) -> Result<Self::Builder>
		where LM: Into<LearnMomentum>
	{
		let mut this = self.builder_or_error()?;
		match this.lm {
			None => {
				this.lm = Some(lm.into());
				Ok(this)
			}
			Some(old_lm) => {
				use self::MentorBuilderDoubledField::LearnMomentum;
				Err(Error::mentor_builder_initialized_field_twice(LearnMomentum)
					.with_annotation(format!("Already set learn momentum to {:?}.", old_lm.to_f32())))
			}
		}
	}

	fn epoch_len(self, epoch_len: usize) -> Result<Self::Builder> {
		if epoch_len == 0 {
			use self::MentorBuilderInvalidArgument::EpochLen;
			return Err(Error::mentor_builder_invalid_argument(EpochLen))
		}
		let mut this = self.builder_or_error()?;
		match this.epoch_len {
			None => {
				this.epoch_len = Some(epoch_len);
				Ok(this)
			}
			Some(old_epoch_len) => {
				use self::MentorBuilderDoubledField::EpochLen;
				Err(Error::mentor_builder_initialized_field_twice(EpochLen)
					.with_annotation(format!("Already set epoch length to {:?}.", old_epoch_len)))
			}
		}
	}

	fn batch_len(self, batch_len: usize) -> Result<Self::Builder> {
		if batch_len == 0 {
			use self::MentorBuilderInvalidArgument::BatchLen;
			return Err(Error::mentor_builder_invalid_argument(BatchLen))
		}
		let mut this = self.builder_or_error()?;
		match this.batch_len {
			None => {
				this.batch_len = Some(batch_len);
				Ok(this)
			}
			Some(old_batch_len) => {
				use self::MentorBuilderDoubledField::BatchLen;
				Err(Error::mentor_builder_initialized_field_twice(BatchLen)
					.with_annotation(format!("Already set batch length to {:?}.", old_batch_len)))
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	#[ignore]
	fn xor() {
		use Activation::Tanh;
		use trainer::RandomSampleScheduler;
		use trainer::condition;
		use topology_v4::{
			Topology,
			TopologyBuilder
		};
		use std::time;

		use trainer::sample::{
			Sample,
			SampleCollection
		};

		println!("Starting unit-test for XOR-training.");
		println!(" - Creating samples ...");

		let (t, f) = (1.0, -1.0);
		let samples = samples![
			[f, f] => f,
			[t, f] => t,
			[f, t] => t,
			[t, t] => f
		];

		println!(" - Sample creation finished.");
		println!(" - Creating topology ...");

		let top = Topology::input(2)
			.fully_connect(2).activation(Tanh)
			.fully_connect(1).activation(Tanh);

		println!(" - Creating neural net from topology ...");

		let net = NeuralNet::from_topology(top).unwrap();

		println!(" - Starting setup of learning process ...");

		let training = net
			.train()
			.sample_gen(RandomSampleScheduler::new(samples))
			.learn_rate(0.3)
			.learn_momentum(0.5)
			.log_when(condition::TimeInterval::new(time::Duration::from_secs(1)))
			.stop_when(condition::BelowRecentMSE::new(0.9, 0.05).unwrap()).unwrap();

		println!(" - Start learning ...");
	
		training.start().unwrap();

		println!(" - Finished learning.");

		// validate_rounded(net, samples);
	}

	mod context {
		use super::*;

		#[test]
		fn new() {
			fn assert_for(epoch_len: usize, batch_len: usize, lr: LearnRate, lm: LearnMomentum) {
				let mut actual = Context::new(epoch_len, batch_len, lr, lm);
				// Hack to counter uncontrollable time shifts:
				use std::time;
				let test_time_started = time::Instant::now();
				actual.time_started = test_time_started;
				// Define what is expected, use the hack for time started.
				let expected = Context{
					time_started: test_time_started,
					iteration: 0,
					epochs_passed: 0,
					epoch_len,
					latest_mse: MeanSquaredError::new(1.0).unwrap(),
					batch_len,
					batch_bit: 0,
					lr,
					lm
				};
				assert_eq!(expected, actual);
			}
			for &epoch_len in &[1, 2, 5, 10, 25] {
				for &batch_len in &[1, 2, 5, 10, 25] {
					for lr in [0.01, 0.1, 0.3, 0.5, 1.0].iter().map(|lr| *lr).map(LearnRate::from) {
						for lm in [0.0, 0.25, 0.33, 0.5].iter().map(|lm| *lm).map(LearnMomentum::from) {
							assert_for(epoch_len, batch_len, lr, lm);
						}
					}
				}
			}
		}

		#[test]
		fn next() {
			fn assert_for(epoch_len: usize, batch_len: usize, lr: LearnRate, lm: LearnMomentum) {
				fn assert_next_step(ctx: &mut Context, expected: &mut Context) {
					let epoch_len = ctx.epoch_len;
					let batch_len = ctx.batch_len;
					ctx.next();
					expected.iteration += 1;
					if (expected.iteration % epoch_len) == 0 {
						expected.epochs_passed += 1;
					}
					expected.batch_bit += 1;
					expected.batch_bit %= batch_len;
					assert_eq!(ctx, expected);
				}
				let mut ctx = Context::new(epoch_len, batch_len, lr, lm);
				let mut expected = ctx.clone();
				for _ in 0..10 {
					assert_next_step(&mut ctx, &mut expected);
				}
			}
			for &epoch_len in &[1, 2, 5, 10, 25] {
				for &batch_len in &[1, 2, 5, 10, 25] {
					for lr in [0.01, 0.1, 0.3, 0.5, 1.0].iter().map(|lr| *lr).map(LearnRate::from) {
						for lm in [0.0, 0.25, 0.33, 0.5].iter().map(|lm| *lm).map(LearnMomentum::from) {
							assert_for(epoch_len, batch_len, lr, lm);
						}
					}
				}
			}
		}

		#[test]
		fn update_mse() {
			fn assert_for(mse: MeanSquaredError) {
				let mut ctx = Context::new(1, 1, LearnRate::from(0.1), LearnMomentum::from(0.5));
				ctx.update_mse(mse);
				assert_eq!(ctx.latest_mse, mse);
			}
			for mse in [0.0, 0.5, 1.0, 42.0, 77.7, 13.37].iter().map(|mse| MeanSquaredError::from(*mse)) {
				assert_for(mse)
			}
		}

		#[test]
		fn is_batch_finished() {
			{
				let mut ctx = Context::new(1, 1, LearnRate::from(0.5), LearnMomentum::from(0.5));
				assert_eq!(ctx.is_batch_finished(), true);
				ctx.next();
				assert_eq!(ctx.is_batch_finished(), true);
			}
			{
				let mut ctx = Context::new(1, 2, LearnRate::from(0.5), LearnMomentum::from(0.5));
				assert_eq!(ctx.is_batch_finished(), true);
				ctx.next();
				assert_eq!(ctx.is_batch_finished(), false);
				ctx.next();
				assert_eq!(ctx.is_batch_finished(), true);
			}
		}

		#[test]
		fn time_started() {
			let ctx = Context::new(1, 1, LearnRate::from(0.5), LearnMomentum::from(0.5));
			let test_time_started = ctx.time_started;
			assert_eq!(ctx.time_started(), test_time_started);
		}

		#[test]
		fn iterations() {
			let n = 10;
			let mut ctx = Context::new(1, 1, LearnRate::from(0.5), LearnMomentum::from(0.5));
			assert_eq!(ctx.iterations(), 0);
			for _ in 0..n {
				ctx.next();
			}
			assert_eq!(ctx.iterations(), n);
		}

		#[test]
		fn epochs_passed() {
			fn assert_for(epoch_len: usize) {
				let n = 10;
				let mut ctx = Context::new(epoch_len, 1, LearnRate::from(0.5), LearnMomentum::from(0.5));
				assert_eq!(ctx.epochs_passed(), 0);
				for _ in 0..n {
					ctx.next();
				}
				assert_eq!(ctx.epochs_passed(), n / epoch_len);
			}
			for &epoch_len in &[1, 2, 3, 5, 10, 42] {
				assert_for(epoch_len)
			}
		}

		#[test]
		fn latest_mse() {
			let mut ctx = Context::new(1, 1, LearnRate::from(0.5), LearnMomentum::from(0.5));
			assert_eq!(ctx.latest_mse(), MeanSquaredError::from(1.0));
			ctx.update_mse(MeanSquaredError::from(0.5));
			assert_eq!(ctx.latest_mse(), MeanSquaredError::from(0.5));
		}
	}

	mod mentor_builder {
		use super::*;

		use topology_v4::{
			Topology,
			TopologyBuilder
		};

		fn dummy_topology() -> Topology {
			Topology::input(1).fully_connect( 1)
		}

		fn dummy_builder() -> UninitializedMentor {
			InitializingMentor::new(dummy_topology())
		}

		#[test]
		fn new() {
			let builder = dummy_builder();
			assert_eq!(builder.sample_gen, Unset);
			assert_eq!(builder.stop_when, Unset);
			assert_eq!(builder.log_when, Unset);
			assert!(builder.epoch_len.is_none());
			assert!(builder.batch_len.is_none());
			assert!(builder.lr.is_none());
			assert!(builder.lm.is_none());
		}

		#[test]
		fn learn_rate() {
			use utils::LearnRate;
			let fst_lr = LearnRate::from(0.5);
			let snd_lr = LearnRate::from(1.0);
			let b = dummy_builder();
			assert!(b.lr.is_none());
			let b = b.learn_rate(fst_lr);
			assert!(b.is_ok());
			let b = b.unwrap();
			assert_eq!(b.lr, Some(fst_lr));
			assert_eq!(
				b.learn_rate(snd_lr),
				Err(Error::mentor_builder_initialized_field_twice(MentorBuilderDoubledField::LearnRate)
					.with_annotation(format!("Already set learn rate to {:?}.", fst_lr.to_f32())))
			);
		}

		#[test]
		fn learn_rate_init() {
			assert_eq!(dummy_builder().lr, None)
		}

		#[test]
		fn learn_rate_ok() {
			let new_lr = LearnRate::from(0.5);
			assert_eq!(dummy_builder().learn_rate(new_lr).unwrap().lr, Some(new_lr))
		}

		#[test]
		fn learn_rate_fail() {
			let old_lr = LearnRate::from(0.5);
			let new_lr = LearnRate::from(1.0);
			let b = dummy_builder().learn_rate(old_lr).unwrap();
			assert_eq!(
				b.learn_rate(new_lr),
				Err(Error::mentor_builder_initialized_field_twice(MentorBuilderDoubledField::LearnRate)
					.with_annotation(format!("Already set learn rate to {:?}.", old_lr.to_f32())))
			)
		}

		#[test]
		fn learn_momentum_init() {
			assert_eq!(dummy_builder().lm, None)
		}

		#[test]
		fn learn_momentum_ok() {
			let new_lm = LearnMomentum::from(0.5);
			assert_eq!(dummy_builder().learn_momentum(new_lm).unwrap().lm, Some(new_lm))
		}

		#[test]
		fn learn_momentum_fail() {
			let old_lm = LearnMomentum::from(0.5);
			let new_lm = LearnMomentum::from(1.0);
			let b = dummy_builder().learn_momentum(old_lm).unwrap();
			assert_eq!(
				b.learn_momentum(new_lm),
				Err(Error::mentor_builder_initialized_field_twice(MentorBuilderDoubledField::LearnMomentum)
					.with_annotation(format!("Already set learn momentum to {:?}.", old_lm.to_f32())))
			)
		}

		#[test]
		fn epoch_len_init() {
			assert_eq!(dummy_builder().epoch_len, None)
		}

		#[test]
		fn epoch_len_ok() {
			let new_epoch_len = 10;
			assert_eq!(dummy_builder().epoch_len(new_epoch_len).unwrap().epoch_len, Some(new_epoch_len))
		}

		#[test]
		fn epoch_len_fail_zero() {
			assert_eq!(
				dummy_builder().epoch_len(0),
				Err(Error::mentor_builder_invalid_argument(MentorBuilderInvalidArgument::EpochLen))
			)
		}

		#[test]
		fn epoch_len_fail_double() {
			let old_epoch_len = 42;
			let new_epoch_len = 1337;
			let b = dummy_builder().epoch_len(old_epoch_len).unwrap();
			assert_eq!(
				b.epoch_len(new_epoch_len),
				Err(Error::mentor_builder_initialized_field_twice(MentorBuilderDoubledField::EpochLen)
					.with_annotation(format!("Already set epoch length to {:?}.", old_epoch_len)))
			)
		}

		#[test]
		fn batch_len_init() {
			assert_eq!(dummy_builder().batch_len, None)
		}

		#[test]
		fn batch_len_ok() {
			let new_batch_len = 10;
			assert_eq!(dummy_builder().batch_len(new_batch_len).unwrap().batch_len, Some(new_batch_len))
		}

		#[test]
		fn batch_len_fail_zero() {
			assert_eq!(
				dummy_builder().batch_len(0),
				Err(Error::mentor_builder_invalid_argument(MentorBuilderInvalidArgument::BatchLen))
			)
		}

		#[test]
		fn batch_len_fail_double() {
			let old_batch_len = 42;
			let new_batch_len = 1337;
			let b = dummy_builder().batch_len(old_batch_len).unwrap();
			assert_eq!(
				b.batch_len(new_batch_len),
				Err(Error::mentor_builder_initialized_field_twice(MentorBuilderDoubledField::BatchLen)
					.with_annotation(format!("Already set batch length to {:?}.", old_batch_len)))
			)
		}

		#[test]
		fn sample_gen_init() {
			assert_eq!(dummy_builder().sample_gen, Unset)
		}

		#[test]
		fn sample_gen_ok() {
			use trainer::sample::{Sample, SampleCollection, SequentialSampleScheduler};
			let dummy_samples = samples![ [0.0] => 1.0 ];
			let new_sample_gen = SequentialSampleScheduler::new(dummy_samples);
			assert_eq!(
				dummy_builder().sample_gen(new_sample_gen.clone()).unwrap().sample_gen,
				new_sample_gen
			)
		}

		#[test]
		fn stop_when_init() {
			assert_eq!(dummy_builder().stop_when, Unset)
		}

		#[test]
		fn stop_when_ok() {
			let new_stop_when = condition::Always;
			assert_eq!(
				dummy_builder().stop_when(new_stop_when).unwrap().stop_when,
				new_stop_when
			)
		}

		#[test]
		fn log_when_init() {
			assert_eq!(dummy_builder().log_when, Unset)
		}

		#[test]
		fn log_when_ok() {
			let new_log_when = condition::Always;
			assert_eq!(
				dummy_builder().log_when(new_log_when).unwrap().log_when,
				new_log_when
			)
		}
	}

	mod mentor {
		use super::*;

		use topology_v4::{
			Topology,
			TopologyBuilder
		};
		use trainer::sample::SequentialSampleScheduler;

		fn dummy_topology() -> Topology {
			Topology::input(1).fully_connect( 1)
		}

		fn dummy_sample_gen() -> SequentialSampleScheduler {
			use trainer::sample::{Sample, SampleCollection};
			let dummy_samples = samples![ [0.0] => 1.0 ];
			SequentialSampleScheduler::new(dummy_samples)
		}

		fn dummy_builder() -> InitializingMentor<SequentialSampleScheduler, condition::Always, Unset> {
			InitializingMentor::new(dummy_topology())
				.sample_gen(dummy_sample_gen())
				.stop_when(condition::Always)
				.unwrap()
		}

		#[test]
		#[ignore]
		fn from_builder_ok() {
			// assert_eq!(
			// 	Mentor::from_builder(dummy_builder()),
			// 	Mentor{
			// 		nn: NeuralNet::from_topology(dummy_topology()).unwrap(),
			// 		sample_gen: dummy_sample_gen(),
			// 		stop_when: condition::Always,
			// 		log_when: condition::Never,
			// 		ctx: Context::new(1, 1, LearnRate::from(0.15), LearnMomentum::from(0.0))
			// 	}
			// )

			// fn from_builder(builder: InitializingMentor<Sampler, StopWhen, LogWhen>) -> Result<Self> {
			// 	let ctx = Context::new(
			// 		builder.epoch_len.unwrap_or_else(|| builder.sample_gen.finite_len().unwrap_or(1)),
			// 		builder.batch_len.unwrap_or(1),
			// 		builder.lr.unwrap_or_else(|| LearnRate::from(0.15)),
			// 		builder.lm.unwrap_or_else(|| LearnMomentum::from(0.0))
			// 	);
			// 	Ok(Mentor{
			// 		nn: builder.nn,
			// 		sample_gen: builder.sample_gen,
			// 		stop_when: builder.stop_when,
			// 		log_when: builder.log_when,
			// 		ctx
			// 	})
			// }
		}
	}
}
