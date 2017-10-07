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

use std::time;

/// The actual state describing the current context during a training process.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Context {
	time_started: time::Instant,
	iteration: usize,
	epochs_passed: usize,
	epoch_len: usize,
	latest_mse: f64,
	lr: LearnRate,
	lm: LearnMomentum
}

impl Context {
	/// Creates a new `Context` from the given `LearnRate` and `LearnMomentum`
	/// and initializes the other values with their respective defaults.
	pub fn new(
		epoch_len: usize,
		lr: LearnRate,
		lm: LearnMomentum) -> Context {
		Context{
			time_started: time::Instant::now(),
			iteration: 0,
			epochs_passed: 0,
			epoch_len, // TODO: Better way to handle a sane default value.
			latest_mse: 1.0,
			lr, lm
		}
	}

	/// Increases the iteration counter by `1` (one).
	#[inline]
	pub fn next_iteration(&mut self) {
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

	/// Upates the latest mean squared error (MSE).
	#[inline]
	pub fn update_mse(&mut self, latest_mse: f64) {
		self.latest_mse = latest_mse
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
	fn latest_mse(&self) -> f64 {
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
pub struct MentorBuilder {
	nn: NeuralNet,
	sample_gen: Option<Box<SampleGen>>,
	stop_when: Option<Box<TrainCondition>>,
	log_when: Option<Box<TrainCondition>>,
	epoch_len: Option<usize>,
	lr: Option<LearnRate>,
	lm: Option<LearnMomentum>
}

impl Mentor {
	/// Initiates a training session for the given neural network or topology.
	pub fn train<NN>(nn: NN) -> MentorBuilder
		where NN: Into<NeuralNet>
	{
		MentorBuilder::new(nn.into())
	}

	/// Creates a new `Mentor` from the given `MentorBuilder`.
	/// 
	/// This also initializes values that were not provided by the user
	/// to their defaults.
	/// 
	/// Note that this does not start the training process by itself.
	/// 
	/// # Errors
	/// 
	/// - If some required property was not set in the builer.
	fn from_builder(builder: MentorBuilder) -> Result<Mentor> {
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
			builder.epoch_len.unwrap_or_else(|| sample_gen.len().unwrap_or(1)),
			builder.lr.unwrap_or(LearnRate::from(0.15)),
			builder.lm.unwrap_or(LearnMomentum::from(0.0))
		);
		Ok(Mentor{
			nn: builder.nn,
			sample_gen,
			stop_when,
			log_when,
			ctx
		})
	}

	/// Starts the training procedure for the given `Mentor` settings.
	/// 
	/// Note that this call may take a while depending on the requested quality of training result.
	/// 
	/// Returns the fully trained neural network when the training is finished.
	pub fn finish(mut self) -> Result<NeuralNet> {
		use trainer::{
			PredictSupervised,
			OptimizeSupervised
		};
		while !self.stop_when.evaluate(&self.ctx) {
			let sample = self.sample_gen.next_sample();
			self.nn.predict_supervised(sample)
				.optimize_supervised(self.ctx.lr, self.ctx.lm);
			if self.log_when.evaluate(&self.ctx) {
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
			// TODO: Fix bug/misdesign that the latest MSE (or general LOSS deviate)
			//       is not communicated back to the trainer since it is not returned anywhere.
			//       This requires a slight redesign of the trait API.
			self.ctx.next_iteration();
		}
		Ok(self.nn)
	}
}

impl MentorBuilder {
	/// Create a new `MentorBuilder` for the given `NeuralNet`.
	/// 
	/// This initiates the configuration and settings phase of a training session
	/// via automated mentoring. The entire process is formed according to the
	/// well-known builder pattern that guides you through the creation process.
	pub fn new(nn: NeuralNet) -> Self {
		MentorBuilder{nn,
			sample_gen: None,
			stop_when: None,
			log_when: None,
			epoch_len: None,
			lr: None,
			lm: None
		}
	}

	/// Finishes the building process and tries to create a full-fledged mentor
	/// that is capable of training the given neural network using the settings
	/// specified during the building phase.
	pub fn start(self) -> Result<NeuralNet> {
		Mentor::from_builder(self)?.finish()
	}

	/// Sets the learning rate that is used throughout the training session.
	/// 
	/// Note: It isn't yet possible to adjust the learning rate after this phase.
	/// 
	/// # Errors
	/// 
	/// - If the learning rate was already set for this builder.
	pub fn learn_rate<LR>(mut self, lr: LR) -> Result<Self>
		where LR: Into<LearnRate>
	{
		match self.lr {
			None => {
				self.lr = Some(lr.into());
				Ok(self)
			}
			Some(old_lr) => {
				// TODO: Do proper error handling here:
				panic!("Already set learn rate to {:?}. Cannot set twice!", old_lr.to_f32());
			}
		}
	}

	/// Sets the learn momentum that is used throughout the training session.
	/// 
	/// Note: It isn't yet possible to adjust the learn momentum after this phase.
	/// 
	/// # Errors
	/// 
	/// - If the learn momentum was already set for this builder.
	pub fn learn_momentum<LM>(mut self, lm: LM) -> Result<Self>
		where LM: Into<LearnMomentum>
	{
		match self.lm {
			None => {
				self.lm = Some(lm.into());
				Ok(self)
			}
			Some(old_lm) => {
				// TODO: Do proper error handling here:
				panic!("Already set learn momentum to {:?}. Cannot set twice!", old_lm.to_f32());
			}
		}
	}

	/// Sets the epoch length that is used for batched learning purposes throughout the training session.
	/// 
	/// Note: Batch learning isn't yet supported by this library. So this value is useless right now.
	///       However, keep in mind to set the epoch length when this trainer cannot infer a default value.
	///       This is the case when using a `SampleGen` that is not limited to a finite set of samples.
	/// 
	/// # Errors
	/// 
	/// - If the given epoch length is zero (`0`). Epoch length must be a positive number.
	pub fn epoch_len(mut self, epoch_len: usize) -> Result<Self> {
		if epoch_len == 0 {
			// TODO: Do proper error handling here:
			panic!("Cannot set epoch length to zero (`0`). Epoch length must be a positive number.")
		}
		match self.epoch_len {
			None => {
				self.epoch_len = Some(epoch_len);
				Ok(self)
			}
			Some(old_epoch_len) => {
				// TODO: Do proper error handling here:
				panic!("Already set epoch length to {:?}. Cannot set it twice!", old_epoch_len);
			}
		}
	}

	/// Sets the sample generator that is used throughout the training session.
	/// 
	/// # Errors
	/// 
	/// - If a sample generator was already set for this builder.
	pub fn sample_gen<G>(mut self, sample_gen: G) -> Result<Self>
		where G: SampleGen + 'static
	{
		match self.sample_gen {
			None => {
				self.sample_gen = Some(Box::new(sample_gen));
				Ok(self)
			}
			Some(_) => {
				// TODO: Do proper error handling here:
				panic!("Already set a sample generator. Confused which one to use. Cannot set twice.");
			}
		}
	}

	/// Sets the halting condition that is used to query when the training shall end.
	/// 
	/// # Errors
	/// 
	/// - If a halting condition was already set for this builder.
	pub fn stop_when<C>(mut self, stop_when: C) -> Result<Self>
		where C: TrainCondition + 'static
	{
		match self.stop_when {
			None => {
				self.stop_when = Some(Box::new(stop_when));
				Ok(self)
			}
			Some(_) => {
				// TODO: Do proper error handling here:
				panic!("Already set a halting condition. Confused which one to use. Cannot set twice.");
			}
		}
	}

	/// Sets the logging condition that is used to query when the training state shall be logged.
	/// 
	/// # Errors
	/// 
	/// - If a logging condition was already set for this builder.
	pub fn log_when<C>(mut self, log_when: C) -> Result<Self>
		where C: TrainCondition + 'static
	{
		match self.log_when {
			None => {
				self.log_when = Some(Box::new(log_when));
				Ok(self)
			}
			Some(_) => {
				// TODO: Do proper error handling here:
				panic!("Already set a logging condition. Confused which one to use. Cannot set twice.");
			}
		}
	}
}
