use ndarray::ArrayView1;

use traits::prelude::*;
use layer::utils::prelude::*;
use layer::{
	HasOutputSignal,
	ProcessInputSignal,
	CalculateOutputErrorSignal,
	ApplyErrorSignalCorrection
};
use layer::{ContainerLayer};
use layer;
use utils::{LearnRate, LearnMomentum};
use topology_v4::{
	Topology
};
use errors::{Result};
use sample::SupervisedSample;

#[derive(Debug, Clone, PartialEq)]
pub struct NeuralNet {
	input: BiasedSignalBuffer,
	layers: ContainerLayer
}

#[derive(Debug)]
pub(crate) struct ReadyToOptimizeSupervised<'nn> {
	nn: &'nn mut NeuralNet
}

impl NeuralNet {
	/// Creates a new neural network from the given topology.
	pub fn from_topology(top: Topology) -> Result<Self> {
		Ok(NeuralNet{
			input: BiasedSignalBuffer::zeros_with_bias(
				top.input_len().into_usize())?,
			layers: ContainerLayer::from_vec(top
				.into_iter()
				.map(|layer| layer::Layer::from(layer))
				.collect()
			)?
		})
	}
}

impl From<Topology> for NeuralNet {
	fn from(top: Topology) -> NeuralNet {
		NeuralNet::from_topology(top)
			.expect("Expected a valid topology as input to construct a new NeuralNet.")
	}
}

impl<'a, I> Predict<I> for NeuralNet
	where I: Into<UnbiasedSignalView<'a>>
{
	/// Implementation for inputs that do not respect a bias value.
	fn predict(&mut self, input: I) -> ArrayView1<f32> {
		let input = input.into();
		self.input.unbias_mut().assign(&input).unwrap(); // TODO: do proper error handling
		self.layers.process_input_signal(self.input.view());
		self.layers.output_signal().into_unbiased().into_data()
	}
}

impl<'nn, S> PredictSupervised<S> for &'nn mut NeuralNet
	where S: SupervisedSample
{
	type Finalizer = ReadyToOptimizeSupervised<'nn>;

	fn predict_supervised(self, sample: &S) -> Self::Finalizer {
		self.input.unbias_mut().assign(&sample.input()).unwrap();
		self.layers.process_input_signal(self.input.view());
		self.layers.calculate_output_error_signal(sample.expected());
		self.layers.propagate_error_signal_internally();
		ReadyToOptimizeSupervised{nn: self}
	}
}

impl<'nn> OptimizeSupervised for ReadyToOptimizeSupervised<'nn> {
	fn optimize_supervised(self, lr: LearnRate, lm: LearnMomentum) {
		self.nn.layers.apply_error_signal_correction(self.nn.input.view(), lr, lm)
	}
}
