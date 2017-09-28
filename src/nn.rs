use ndarray::ArrayView1;

use layer::utils::prelude::*;
use layer::{ContainerLayer};
use traits::prelude::*;
use utils::{LearnRate, LearnMomentum};

#[derive(Debug, Clone, PartialEq)]
struct NeuralNet {
	input: BiasedSignalBuffer,
	layers: ContainerLayer
}

impl<'a, I> Predict<I> for NeuralNet
	where I: Into<ArrayView1<'a, f32>>
{
	/// Implementation for inputs that do not respect a bias value.
	fn predict(&mut self, _input: I) -> ArrayView1<f32> {
		unimplemented!()
	}
}

impl<'a> Predict<BiasedSignalView<'a>> for NeuralNet {
	/// Implementation for inputs that do respect a bias value.
	fn predict(&mut self, _input: BiasedSignalView) -> ArrayView1<f32> {
		unimplemented!()
	}
}

impl<'a, A> UpdateGradients<A> for NeuralNet
	where A: Into<ArrayView1<'a, f32>>
{
	/// Implementation for inputs that do not respect a bias value.
	fn update_gradients(&mut self, _target_values: A) {
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
