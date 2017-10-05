use std::time::{SystemTime};

use neural_net::NeuralNet;
use utils::{
	LearnRate,
	LearnMomentum
};
use traits::{
	Predict,
	UpdateGradients,
	UpdateWeights
};
use errors::{Result, Error};
use topology::Topology;
use mentor::configs::{
	LearnRateConfig,
	LearnMomentumConfig,
	Criterion,
	LogConfig,
	Scheduling
};
use mentor::samples::{SampleScheduler};
use mentor::deviation::Deviation;
use mentor::logger::{Stats, Logger};
use mentor::samples::Sample;


impl Topology {
	/// Iterates over the layer sizes of this Disciple's topology definition.
	pub fn train(self, samples: Vec<Sample>) -> MentorBuilder {
		Mentor::new(self, samples)
	}
}



mod state {
	pub trait LearnRateConfigState {}
	pub trait LearnMomentumConfigState {}
	pub trait CriterionConfigState {}
	pub trait SchedulingConfigState {}
	pub trait LogConfigState {}

	#[derive(Debug, Copy, Clone)]
	pub struct Unset;
	#[derive(Debug, Copy, Clone)]
	pub struct Set;

	impl LearnRateConfigState for Unset {}
	impl LearnMomentumConfigState for Unset {}
	impl CriterionConfigState for Unset {}
	impl SchedulingConfigState for Unset {}
	impl LogConfigState for Unset {}

	impl LearnRateConfigState for Set {}
	impl LearnMomentumConfigState for Set {}
	impl CriterionConfigState for Set {}
	impl SchedulingConfigState for Set {}
	impl LogConfigState for Set {}
}
use self::state::{
	LearnRateConfigState,
	LearnMomentumConfigState,
	CriterionConfigState,
	SchedulingConfigState,
	LogConfigState,

	Unset,
	Set
};
use std::marker::PhantomData;

/// A fresh mentor which is completely uninitialized, yet.
pub type MentorBuilder = Mentor<Unset, Unset, Unset, Unset, Unset>;

/// Mentor follows the builder pattern to incrementally
/// build properties for the training session and delay any
/// expensive computations until the go routine is called.
#[derive(Debug, Clone)]
pub struct Mentor<
	LR: LearnRateConfigState,
	LM: LearnMomentumConfigState,
	CR: CriterionConfigState,
	SC: SchedulingConfigState,
	LG: LogConfigState >
{
	learn_rate: LearnRateConfig,
	learn_mom : LearnMomentumConfig,
	criterion : Criterion,
	scheduling: Scheduling,
	disciple  : Topology,
	samples   : Vec<Sample>,
	log_config: LogConfig,

	phantom   : PhantomData<(LR, LM, CR, SC, LG)>
}

impl MentorBuilder {
	/// Creates a new mentor for the given disciple and
	/// with the given sample collection (training data).
	fn new(disciple: Topology, samples: Vec<Sample>) -> MentorBuilder {
		Mentor {
			learn_rate: LearnRateConfig::Adapt,
			learn_mom : LearnMomentumConfig::Adapt,
			criterion : Criterion::RecentMSE(0.05),
			scheduling: Scheduling::Random,
			disciple  : disciple,
			samples   : samples,
			log_config: LogConfig::Never,
			phantom   : PhantomData
		}
	}
}

impl<LR1, LM1, CR1, SC1, LG1> Mentor<LR1, LM1, CR1, SC1, LG1>
	where
		LR1: LearnRateConfigState,
		LM1: LearnMomentumConfigState,
		CR1: CriterionConfigState,
		SC1: SchedulingConfigState,
		LG1: LogConfigState
{
	/// Switches the compile-time type-based state of this mentor.
	/// 
	/// This is a no-op at runtime!
	fn switch_state<
		LR2: LearnRateConfigState,
		LM2: LearnMomentumConfigState,
		CR2: CriterionConfigState,
		SC2: SchedulingConfigState,
		LG2: LogConfigState>
	(self) -> Mentor<LR2, LM2, CR2, SC2, LG2> {
		Mentor{
			learn_rate: self.learn_rate,
			learn_mom : self.learn_mom,
			criterion : self.criterion,
			scheduling: self.scheduling,
			disciple  : self.disciple,
			samples   : self.samples,
			log_config: self.log_config,
			phantom   : PhantomData
		}
	}
}

impl<LM, CR, SC, LG> Mentor<Unset, LM, CR, SC, LG>
	where
		LM: LearnMomentumConfigState,
		CR: CriterionConfigState,
		SC: SchedulingConfigState,
		LG: LogConfigState
{
	/// Use the given fixed learn rate.
	///
	/// Default learn rate is adapting behaviour.
	/// 
	/// ***Panics*** if given learn rate is invalid!
	pub fn learn_rate<LR>(mut self, learn_rate: LR) -> Mentor<Set, LM, CR, SC, LG>
		where LR: Into<LearnRate>
	{
		self.learn_rate = LearnRateConfig::Fixed(learn_rate.into());
		self.switch_state()
	}
}

impl<LR, CR, SC, LG> Mentor<LR, Unset, CR, SC, LG>
	where
		LR: LearnRateConfigState,
		CR: CriterionConfigState,
		SC: SchedulingConfigState,
		LG: LogConfigState
{
	/// Use the given fixed learn momentum.
	///
	/// Default learn momentum is fixed at `0.5`.
	/// 
	/// ***Panics*** if given learn momentum is invalid
	pub fn learn_momentum<LM>(mut self, learn_momentum: LM) -> Mentor<LR, Set, CR, SC, LG>
		where LM: Into<LearnMomentum>
	{
		self.learn_mom = LearnMomentumConfig::Fixed(learn_momentum.into());
		self.switch_state()
	}
}

impl<LR, LM, SC, LG> Mentor<LR, LM, Unset, SC, LG>
	where
		LR: LearnRateConfigState,
		LM: LearnMomentumConfigState,
		SC: SchedulingConfigState,
		LG: LogConfigState
{
	/// Use the given criterion.
	///
	/// Default criterion is `AvgNetError(0.05)`.
	pub fn criterion(mut self, criterion: Criterion) -> Mentor<LR, LM, Set, SC, LG> {
		self.criterion = criterion;
		self.switch_state()
	}
}

impl<LR, LM, CR, LG> Mentor<LR, LM, CR, Unset, LG>
	where
		LR: LearnRateConfigState,
		LM: LearnMomentumConfigState,
		CR: CriterionConfigState,
		LG: LogConfigState
{
	/// Use the given scheduling routine.
	///
	/// Default scheduling routine is to pick random samples.
	pub fn scheduling(mut self, kind: Scheduling) -> Mentor<LR, LM, CR, Set, LG> {
		self.scheduling = kind;
		self.switch_state()
	}
}

impl<LR, LM, CR, SC> Mentor<LR, LM, CR, SC, Unset>
	where
		LR: LearnRateConfigState,
		LM: LearnMomentumConfigState,
		CR: CriterionConfigState,
		SC: SchedulingConfigState,
{
	/// Use the given logging configuration.
	/// 
	/// Default logging configuration is to never log anything.
	pub fn log_config(mut self, config: LogConfig) -> Mentor<LR, LM, CR, SC, Set> {
		self.log_config = config;
		self.switch_state()
	}
}

impl<LR, LM, CR, SC, LG> Mentor<LR, LM, CR, SC, LG>
	where
		LR: LearnRateConfigState,
		LM: LearnMomentumConfigState,
		CR: CriterionConfigState,
		SC: SchedulingConfigState,
		LG: LogConfigState
{
	/// Validate all sample input and target sizes.
	fn validate_samples(&self) -> Result<()> {
		let req_inputs = self.disciple.len_input();
		let req_outputs = self.disciple.len_output();
		for sample in &self.samples {
			if sample.input.len() != req_inputs {
				return Err(Error::unmatching_input_sample_size(sample.input.len(), req_inputs));
			}
			if sample.target.len() != req_outputs {
				return Err(Error::unmatching_target_sample_size(sample.target.len(), req_outputs));
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
		self.validate_samples()?;
		self.start_training().start()
	}

	/// Consumes this mentor and starts a training session.
	/// 
	/// This process computes all required structures for the training session.
	fn start_training(self) -> Training {
		Training {
			disciple : NeuralNet::from_topology(self.disciple),
			scheduler: SampleScheduler::from_samples(self.scheduling, self.samples),

			cfg: Config{
				learn_rate: self.learn_rate,
				learn_mom : self.learn_mom,
				criterion : self.criterion
			},

			learn_rate: match self.learn_rate {
				LearnRateConfig::Adapt    => LearnRate::from(0.3),
				LearnRateConfig::Fixed(r) => r
			},

			learn_mom: match self.learn_mom {
				LearnMomentumConfig::Adapt    => LearnMomentum::from(0.5),
				LearnMomentumConfig::Fixed(m) => m
			},

			iterations: Iteration::default(),
			starttime : SystemTime::now(),
			deviation : Deviation::default(),

			logger: Logger::from(self.log_config)
		}
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

/// A training session trains a neural network and stops only
/// after the neural networks training stats meet certain 
/// predefined criteria.
#[derive(Debug, Clone)]
pub struct Training {
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
		use mentor::configs::Criterion::*;
		match self.cfg.criterion {
			TimeOut(duration) => {
				self.starttime.elapsed().unwrap() >= duration
			},
			Iterations(limit) => {
				self.iterations.0 == limit
			},
			RecentMSE(target) => {
				self.deviation.recent_mse() <= target
			}
		}
	}

	fn session(&mut self) {
		{
			let sample = self.scheduler.next_sample();
			{
				let output = self.disciple.predict(sample.input);
				self.deviation.update(output, sample.target);
			}
			self.disciple.update_gradients(sample.target);
			self.disciple.update_weights(self.learn_rate, self.learn_mom);
			self.iterations.bump();
		}
		self.try_log();
	}

	fn update_learn_rate(&mut self) {
		use self::LearnRateConfig::*;
		match self.cfg.learn_rate {
			Adapt => {
				// TODO: not yet implemented
			}
			Fixed(_) => return // nothing to do here!
		}
	}

	fn update_learn_momentum(&mut self) {
		use self::LearnMomentumConfig::*;
		match self.cfg.learn_mom {
			Adapt => {
				// TODO: not yet implemented
			}
			Fixed(_) => return // nothing to do here!
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

	fn start(mut self) -> Result<NeuralNet> {
		loop {
			self.update_learn_rate();
			self.update_learn_momentum();
			self.session();
			if self.is_done() { break }
		}
		Ok(self.disciple)
	}
}
