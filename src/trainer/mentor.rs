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
use errors::{Result};
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
		lm: LearnMomentum) -> Context {
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
	pub fn batch_finished(&self) -> bool {
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
#[derive(Debug)]
pub struct Mentor {
	nn: NeuralNet,
	sample_gen: Box<SampleGen>,
	log_when: Box<TrainCondition>,
	stop_when: Box<TrainCondition>,
	ctx: Context
}

/// Used to construct fully-fledged `Mentor`s using the well-known builder pattern.
#[derive(Debug)]
pub struct InitializingMentor {
	nn: NeuralNet,
	sample_gen: Option<Box<SampleGen>>,
	stop_when: Option<Box<TrainCondition>>,
	log_when: Option<Box<TrainCondition>>,
	epoch_len: Option<usize>,
	batch_len: Option<usize>,
	lr: Option<LearnRate>,
	lm: Option<LearnMomentum>
}

impl Mentor {
	/// Initiates a training session for the given neural network or topology.
	pub fn train<NN>(nn: NN) -> InitializingMentor
		where NN: Into<NeuralNet>
	{
		InitializingMentor::new(nn.into())
	}

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
	fn from_builder(builder: InitializingMentor) -> Result<Mentor> {
		let sample_gen =
			match builder.sample_gen {
				Some(sample_gen) => sample_gen,
				None => {
					panic!("No sample gen specified during building process!")
				}
			};
		let stop_when = 
			match builder.stop_when {
				Some(stop_when) => stop_when,
				None => {
					panic!("No halting condition specified during building process!")
				}
			};
		let log_when =
			match builder.log_when {
				Some(log_when) => log_when,
				None => {
					Box::new(condition::Always)
				}
			};
		let ctx = Context::new(
			builder.epoch_len.unwrap_or_else(|| sample_gen.finite_len().unwrap_or(1)),
			builder.batch_len.unwrap_or(1),
			builder.lr.unwrap_or_else(|| LearnRate::from(0.15)),
			builder.lm.unwrap_or_else(|| LearnMomentum::from(0.0))
		);
		Ok(Mentor{
			nn: builder.nn,
			sample_gen,
			stop_when,
			log_when,
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
				let latest_mse = if self.ctx.batch_finished() {
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
	/// Returns a `MentorBuilder` wrapped as a `Result<MentorBuilder>`.
	fn builder_or_error(self) -> Result<InitializingMentor>;
}

impl MentorBuilderOrError for InitializingMentor {
	#[inline]
	fn builder_or_error(self) -> Result<InitializingMentor> {
		Ok(self)
	}
}

impl MentorBuilderOrError for Result<InitializingMentor> {
	#[inline]
	fn builder_or_error(self) -> Result<InitializingMentor> {
		self
	}
}

/// Types that build fully-fledged `Mentor`s via the well-known builder-pattern.
/// 
/// The main purpose of this trait is to allow for greater flexibility in the building
/// process so that users may leave away some manual error handling to improve usability
/// of the mentor building process.
pub trait MentorBuilder: Sized {
	/// Finishes the building process and tries to create a full-fledged mentor
	/// that is capable of training the given neural network using the settings
	/// specified during the building phase.
	fn start(self) -> Result<NeuralNet>;

	/// Sets the learning rate that is used throughout the training session.
	/// 
	/// Note: It isn't yet possible to adjust the learning rate after this phase.
	/// 
	/// # Errors
	/// 
	/// - If the learning rate was already set for this builder.
	fn learn_rate<LR>(self, lr: LR) -> Result<InitializingMentor>
		where LR: Into<LearnRate>;

	/// Sets the learn momentum that is used throughout the training session.
	/// 
	/// Note: It isn't yet possible to adjust the learn momentum after this phase.
	/// 
	/// # Errors
	/// 
	/// - If the learn momentum was already set for this builder.
	fn learn_momentum<LM>(self, lm: LM) -> Result<InitializingMentor>
		where LM: Into<LearnMomentum>;

	/// Sets the epoch length that is used statistics purposes throughout the training session.
	/// 
	/// Defaults to the length of the sample set if it is finite; else defaults to one (`1`).
	/// 
	/// # Errors
	/// 
	/// - If the given epoch length is zero (`0`). Epoch length must be a positive number.
	fn epoch_len(self, epoch_len: usize) -> Result<InitializingMentor>;

	/// Sets the batch length that is used for batched learning purposes throughout the training session.
	/// 
	/// Defaults to one (`1`) if no user-provided batch size is provided.
	/// 
	/// # Errors
	/// 
	/// - If the given epoch length is zero (`0`). Epoch length must be a positive number.
	fn batch_len(self, batch_len: usize) -> Result<InitializingMentor>;

	/// Sets the sample generator that is used throughout the training session.
	/// 
	/// # Errors
	/// 
	/// - If a sample generator was already set for this builder.
	fn sample_gen<G>(self, sample_gen: G) -> Result<InitializingMentor>
		where G: SampleGen + 'static;

	/// Sets the halting condition that is used to query when the training shall end.
	/// 
	/// # Errors
	/// 
	/// - If a halting condition was already set for this builder.
	fn stop_when<C>(self, stop_when: C) -> Result<InitializingMentor>
		where C: TrainCondition + 'static;

	/// Sets the logging condition that is used to query when the training state shall be logged.
	/// 
	/// # Errors
	/// 
	/// - If a logging condition was already set for this builder.
	fn log_when<C>(self, log_when: C) -> Result<InitializingMentor>
		where C: TrainCondition + 'static;
}

impl InitializingMentor {
	/// Create a new `InitializingMentor` for the given `NeuralNet`.
	/// 
	/// This initiates the configuration and settings phase of a training session
	/// via automated mentoring. The entire process is formed according to the
	/// well-known builder pattern that guides you through the creation process.
	pub fn new(nn: NeuralNet) -> Self {
		InitializingMentor{nn,
			sample_gen: None,
			stop_when: None,
			log_when: None,
			epoch_len: None,
			batch_len: None,
			lr: None,
			lm: None
		}
	}
}

impl<MB> MentorBuilder for MB where MB: MentorBuilderOrError {
	fn start(self) -> Result<NeuralNet> {
		Mentor::from_builder(self.builder_or_error()?)?.finish()
	}

	fn learn_rate<LR>(self, lr: LR) -> Result<InitializingMentor>
		where LR: Into<LearnRate>
	{
		let mut this = self.builder_or_error()?;
		match this.lr {
			None => {
				this.lr = Some(lr.into());
				Ok(this)
			}
			Some(old_lr) => {
				// TODO: Do proper error handling here:
				panic!("Already set learn rate to {:?}. Cannot set twice!", old_lr.to_f32());
			}
		}
	}

	fn learn_momentum<LM>(self, lm: LM) -> Result<InitializingMentor>
		where LM: Into<LearnMomentum>
	{
		let mut this = self.builder_or_error()?;
		match this.lm {
			None => {
				this.lm = Some(lm.into());
				Ok(this)
			}
			Some(old_lm) => {
				// TODO: Do proper error handling here:
				panic!("Already set learn momentum to {:?}. Cannot set twice!", old_lm.to_f32());
			}
		}
	}

	fn epoch_len(self, epoch_len: usize) -> Result<InitializingMentor> {
		if epoch_len == 0 {
			// TODO: Do proper error handling here:
			panic!("Cannot set epoch length to zero (`0`). Epoch length must be a positive number.")
		}
		let mut this = self.builder_or_error()?;
		match this.epoch_len {
			None => {
				this.epoch_len = Some(epoch_len);
				Ok(this)
			}
			Some(old_epoch_len) => {
				// TODO: Do proper error handling here:
				panic!("Already set epoch length to {:?}. Cannot set it twice!", old_epoch_len);
			}
		}
	}

	fn batch_len(self, batch_len: usize) -> Result<InitializingMentor> {
		if batch_len == 0 {
			// TODO: Do proper error handling here:
			panic!("Cannot set batch length to zero (`0`). Batch length must be a positive number.")
		}
		let mut this = self.builder_or_error()?;
		match this.batch_len {
			None => {
				this.batch_len = Some(batch_len);
				Ok(this)
			}
			Some(old_batch_len) => {
				// TODO: Do proper error handling here:
				panic!("Already set batch length to {:?}. Cannot set it twice!", old_batch_len);
			}
		}
	}

	fn sample_gen<G>(self, sample_gen: G) -> Result<InitializingMentor>
		where G: SampleGen + 'static
	{
		let mut this = self.builder_or_error()?;
		match this.sample_gen {
			None => {
				this.sample_gen = Some(Box::new(sample_gen));
				Ok(this)
			}
			Some(_) => {
				// TODO: Do proper error handling here:
				panic!("Already set a sample generator. Confused which one to use. Cannot set twice.");
			}
		}
	}

	fn stop_when<C>(self, stop_when: C) -> Result<InitializingMentor>
		where C: TrainCondition + 'static
	{
		let mut this = self.builder_or_error()?;
		match this.stop_when {
			None => {
				this.stop_when = Some(Box::new(stop_when));
				Ok(this)
			}
			Some(_) => {
				// TODO: Do proper error handling here:
				panic!("Already set a halting condition. Confused which one to use. Cannot set twice.");
			}
		}
	}

	fn log_when<C>(self, log_when: C) -> Result<InitializingMentor>
		where C: TrainCondition + 'static
	{
		let mut this = self.builder_or_error()?;
		match this.log_when {
			None => {
				this.log_when = Some(Box::new(log_when));
				Ok(this)
			}
			Some(_) => {
				// TODO: Do proper error handling here:
				panic!("Already set a logging condition. Confused which one to use. Cannot set twice.");
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

		let training = Mentor::train(net)
			.sample_gen(RandomSampleScheduler::new(samples))
			.learn_rate(0.3)
			.learn_momentum(0.5)
			.log_when(condition::TimeInterval::once_in(time::Duration::from_secs(1)))
			.stop_when(condition::BelowRecentMSE::new(0.9, 0.05).unwrap());

		println!(" - Start learning ...");
	
		training.start().unwrap();

		println!(" - Finished learning.");

		// validate_rounded(net, samples);
	}
}
