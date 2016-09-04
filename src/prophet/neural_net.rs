use num::Float;

use prophet::error_stats::ErrorStats;

pub trait NeuralNet {
	type Elem: Float;

	fn predict<'b, 'a: 'b>(&'a mut self, input: &'b [Self::Elem]) -> &'b [Self::Elem];
}

pub trait TrainableNeuralNet: NeuralNet {
	fn train(&mut self, input: &[Self::Elem], expected: &[Self::Elem]) -> ErrorStats;
}
