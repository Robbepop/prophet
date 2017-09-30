use ndarray::ArrayView1;

use traits::prelude::*;
use layer::utils::prelude::*;
use layer::{
	HasOutputSignal,
	ProcessInputSignal
};
use layer::{ContainerLayer};
use utils::{LearnRate, LearnMomentum};

#[derive(Debug, Clone, PartialEq)]
struct NeuralNet {
	input: BiasedSignalBuffer,
	layers: ContainerLayer
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

impl<'a, T> UpdateGradients<T> for NeuralNet
	where T: Into<UnbiasedSignalView<'a>>
{
	/// Implementation for inputs that do not respect a bias value.
	fn update_gradients(&mut self, _target_values: T) {
		unimplemented!()
	}
}

impl UpdateWeights for NeuralNet
{
	fn update_weights(
		&mut self, 
		_rate: LearnRate,
		_momentum: LearnMomentum
	) {
		unimplemented!()
	}
}
