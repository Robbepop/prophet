use ndarray::ArrayView1;

use traits::prelude::*;
use layer::utils::prelude::*;
use layer::{
	HasOutputSignal,
	ProcessInputSignal
};
use layer::{ContainerLayer};
use layer;
use utils::{LearnRate, LearnMomentum};
use topology_v4;
use topology_v4::{
	Topology
};
use errors::{Result};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NeuralNet {
	input: BiasedSignalBuffer,
	layers: ContainerLayer
}

#[derive(Debug)]
pub(crate) struct ReadyToOptimizePredict<'nn> {
	nn: &'nn mut NeuralNet
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
