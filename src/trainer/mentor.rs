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
use errors::{Error, Result};

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

#[derive(Debug)]
pub struct Mentor {
	nn: NeuralNet,
	sample_gen: Box<SampleGen>,
	logger: Option<Box<DebuggableLog>>,
	log_when: Box<TrainCondition>,
	stop_when: Box<TrainCondition>,
	ctx: Context
}

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
