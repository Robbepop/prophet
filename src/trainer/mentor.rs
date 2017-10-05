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

use log::Log;

use std::time;
use std::fmt::Debug;

/// The actual state describing the current context during a training process.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Context {
	time_started: time::Instant,
	iteration: usize,
	epochs_passed: usize,
	latest_mse: f64,
	lr: LearnRate,
	lm: LearnMomentum
}

impl Context {
	/// Creates a new `Context` from the given `LearnRate` and `LearnMomentum`
	/// and initializes the other values with their respective defaults.
	pub fn new(lr: LearnRate, lm: LearnMomentum) -> Context {
		Context{
			time_started: time::Instant::now(),
			iteration: 0,
			epochs_passed: 0,
			latest_mse: 1.0,
			lr, lm
		}
	}

	/// Increases the iteration counter by `1` (one).
	#[inline]
	pub fn next_iteration(&mut self) {
		self.iteration += 1
	}

	/// Increases the counter representing the epochs passed by `1` (one).
	#[inline]
	pub fn next_epoch(&mut self) {
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
	fn epochs(&self) -> usize {
		self.epochs_passed
	}

	#[inline]
	fn latest_mse(&self) -> f64 {
		self.latest_mse
	}
}

trait DebuggableLog: Log + Debug {}

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
	logger: Option<Box<DebuggableLog>>,
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
	logger: Option<Box<DebuggableLog>>,
	log_when: Option<Box<TrainCondition>>,
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
		Ok(Mentor{
			nn: builder.nn,
			sample_gen: match builder.sample_gen {
				Some(sample_gen) => sample_gen,
				None => {
					panic!("No sample gen specified during building process!")
				}
			},
			stop_when: match builder.stop_when {
				Some(stop_when) => stop_when,
				None => {
					panic!("No halting condition specified during building process!")
				}
			},
			logger: match builder.logger {
				Some(logger) => Some(logger),
				None => None
			},
			log_when: match builder.log_when {
				Some(log_when) => log_when,
				None => {
					Box::new(condition::Always)
				}
			},
			ctx: Context::new(
				builder.lr.unwrap_or(LearnRate::from(0.15)),
				builder.lm.unwrap_or(LearnMomentum::from(0.0))
			)
		})
	}

	pub fn finish(self) -> Result<NeuralNet> {
		unimplemented!() // TODO: Implement the training procedure.
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
			logger: None,
			log_when: None,
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
}
