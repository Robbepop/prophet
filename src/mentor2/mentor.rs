use neural_net::NeuralNet;
use mentor2::configs::{
	Criterion,
	LearnRateConfig,
	LearnMomentumConfig,
	Scheduling,
	LogConfig
};
use traits::{LearnRate, LearnMomentum};
use errors::ErrorKind::{InvalidSampleInputSize, InvalidSampleTargetSize};
use errors::Result;
use topology::Topology;
use mentor2::samples::Sample;

// impl Topology {
// 	/// Iterates over the layer sizes of this Disciple's topology definition.
// 	pub fn train(self, samples: Vec<Sample>) -> Mentor {
// 		Mentor::new(self, samples)
// 	}
// }

/// Mentor follows the builder pattern to incrementally
/// build properties for the training session and delay any
/// expensive computations until the go routine is called.
#[derive(Debug, Clone)]
pub struct Mentor {
	// deviation : Deviation,
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
			// deviation : Deviation::default(),
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
		self.validate_samples()?;
		self.start_training().start()
	}
}
