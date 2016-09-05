use std::vec::Vec;

use rand::*;
use ndarray::prelude::*;
use ndarray::{Shape};
use itertools::{Zip};

use prophet::error_stats::{ErrorStats};
use prophet::activation_fn::{ActivationFn, BaseDerivedActivationFn};
use prophet::neural_net::{NeuralNet, TrainableNeuralNet};

type Array1D<F> = Array<F, Ix>;
type Array2D<F> = Array<F, (Ix, Ix)>;

pub struct ConvNeuralLayer {
	weights:       Array2D<f32>,
	delta_weights: Array2D<f32>,
	outputs:       Array1D<f32>,
	gradients:     Array1D<f32>
}

pub struct ConvNeuralNet {
	layers: Vec<ConvNeuralLayer>,
	act_fns: BaseDerivedActivationFn<f32>,
	learning_rate: f32,
	learning_momentum: f32,
	error_stats: ErrorStats
}

impl ConvNeuralLayer {
	/// Creates a ConvNeuralLayer by consuming the given vector vec.
	/// vec is required to also include the weights for the bias neuron!
	/// 
	/// This constructor should only be used internally for testing purpose
	/// and is not meant to be used outside of this crate.
	fn from_vec(inputs: Ix, outputs: Ix, vec: Vec<f32>) -> Self {
		assert!(inputs >= 2 && outputs >= 1);
		let shape = Shape::from((outputs, inputs));
		// Need one more gradient for the bias neuron.
		let count_gradients = outputs + 1;
		ConvNeuralLayer{
			weights: Array2D::from_shape_vec(shape, vec).unwrap(),
			delta_weights: Array2D::default(shape),
			outputs: Array::default(outputs),
			gradients: Array::zeros(count_gradients)
		}
	}

	/// Creates a ConvNeuralLayer with randomized weights.
	/// Implicitely creates weights for the bias neuron,
	/// so the dimensions of the weights matrix is equal to
	/// (output)x(input+1).
	/// The weights are randomized within the open interval (0,1).
	/// This excludes 0.0 and 1.0 as weights.
	/// Other optional intervals may come with a future update!
	pub fn random(inputs: Ix, outputs: Ix) -> Self {
		assert!(inputs >= 1 && outputs >= 1);
		let inputs = inputs + 1; // implicitely add bias!
		let elems = inputs * outputs;
		let buffer = thread_rng().gen_iter::<Open01<f32>>()
			.take(elems)
			.map(|Open01(val)| val)
			.collect::<Vec<f32>>();
		ConvNeuralLayer::from_vec(inputs, outputs, buffer)
	}

	fn count_rows(&self) -> Ix { let (rows, _) = self.weights.dim(); rows }
	fn count_columns(&self) -> Ix { let (_, cols) = self.weights.dim(); cols }
	fn count_outputs(&self) -> Ix { self.outputs.dim() }
	fn count_gradients(&self) -> Ix { self.gradients.dim() }

	/// Returns this layer's output as readable slice.
	fn output_as_slice(&self) -> &[f32] {
		self.outputs.as_slice().unwrap()
	}

	/// Takes input slice and performs a feed forward procedure
	/// using the given activation function.
	/// Output of this operation will be stored within this layer
	/// and be returned as readable slice.
	/// 
	/// Expects:
	///  - input with n elements
	/// Requires:
	///  - weight matrix with m rows and (n+1) columns
	/// Asserts:
	///  - output with m elements
	pub fn feed_forward<'a>(
		&'a mut self,
		input: &[f32],
		activation_fn: ActivationFn<f32>
	)
		-> &'a [f32]
	{
		debug_assert_eq!(self.count_rows(), self.count_outputs());
		debug_assert_eq!(self.count_columns(), input.len() + 1);

		for (weights_row, output) in Zip::new((self.weights.outer_iter(), self.outputs.iter_mut())) {
			*output = activation_fn(Zip::new((weights_row.iter(),
			                                  input.iter().chain(&[1.0])))
				.fold(0.0, |sum, (w, i)| sum + w*i));
		};
		self.output_as_slice()
	}

	/// Used internally in the output layer to initialize gradients for the back propagation phase.
	/// Sets the gradient for the bias neuron to zero - hopefully this is the correct behaviour.
	fn calculate_output_gradients(&mut self, target_values: &[f32], act_fn_dx: ActivationFn<f32>) -> &Self {
		debug_assert_eq!(self.count_outputs(), target_values.len());
		debug_assert_eq!(self.count_gradients(), target_values.len() + 1); // no calculation for bias!

		for (gradient, target, &output) in Zip::new(
			(self.gradients.iter_mut(), target_values.iter(), self.outputs.iter()))
		{
			*gradient = (target - output) * act_fn_dx(output);
		}
		// gradient of bias should be set equal to zero during object initialization already.
		self
	}

	/// Sets all gradient values in this layer to zero.
	/// This is required as initialization step before propagating gradients
	/// for the efficient implementation of this library.
	fn reset_gradients(&mut self) {
		for gradient in self.gradients.iter_mut() {
			*gradient = 0.0;
		}

		debug_assert_eq!(self.gradients.iter().sum::<f32>(), 0.0);
	}

	/// Applies the given activation function on all gradients of this layer.
	fn apply_activation(&mut self, act_fn_dx: ActivationFn<f32>) {
		debug_assert_eq!(self.count_gradients(), self.count_outputs() + 1);

		for (mut gradient, output) in Zip::new((self.gradients.iter_mut(),
		                                        self.outputs.iter().chain(&[1.0]))) {
			*gradient *= act_fn_dx(*output);
		}
	}

	/// Back propagate gradients from the previous layer (in reversed order) to this layer
	/// using the given activation function.
	/// This also computes the gradient for the bias neuron.
	/// Returns readable reference to self to allow chaining.
	fn propagate_gradients(&mut self, prev: &ConvNeuralLayer, act_fn_dx: ActivationFn<f32>) -> &Self {
		debug_assert_eq!(prev.count_rows(), prev.count_gradients() - 1);
		debug_assert_eq!(prev.count_columns(), self.count_gradients());

		self.reset_gradients();

		for (prev_weights_row, prev_gradient) in Zip::new((prev.weights.outer_iter(),
		                                                   prev.gradients.iter()))
		{
			for (mut gradient, weight) in Zip::new((self.gradients.iter_mut(),
			                                        prev_weights_row.iter()))
			{
				*gradient += weight * prev_gradient;
			}
		}

		self.apply_activation(act_fn_dx);
		self // for chaining in a fold expression
	}

	/// Updates the connection weights of this layer.
	/// This operation is usually used after successful computation of gradients.
	fn update_weights(&mut self,
	                  prev_outputs: &[f32],
	                  train_rate: f32,
	                  learning_momentum: f32) -> &[f32]
	{
		debug_assert_eq!(prev_outputs.len() + 1, self.count_columns());
		debug_assert_eq!(self.count_gradients(), self.count_rows() + 1);

		for (mut weights_row, mut delta_weights_row, gradient) in Zip::new((self.weights.outer_iter_mut(),
		                                                                    self.delta_weights.outer_iter_mut(),
		                                                                    self.gradients.iter())) {
			for (prev_output, weight, delta_weight) in Zip::new((prev_outputs.iter().chain(&[1.0]),
			                                                     weights_row.iter_mut(),
			                                                     delta_weights_row.iter_mut())) {
				*delta_weight =
					// Individual input, magnified by the gradient and train rate
					train_rate * prev_output * gradient
					// Also add momentum which is a fraction of the previous delta weight
					+ learning_momentum * *delta_weight;
				*weight += *delta_weight;
			}
		}

		self.output_as_slice()
	}
}



impl ConvNeuralNet {
	fn from_vec(
		learning_rate: f32,
		learning_momentum: f32,
		act_fn: BaseDerivedActivationFn<f32>,
		layers: Vec<ConvNeuralLayer>
	)
		-> Self
	{
		ConvNeuralNet{
			layers: layers,
			act_fns: act_fn,
			learning_rate: learning_rate,
			learning_momentum: learning_momentum,
			error_stats: ErrorStats::default()
		}
	}

	pub fn new(
		learning_rate: f32,
		learning_momentum: f32,
		act_fn: BaseDerivedActivationFn<f32>,
		layer_sizes: &[Ix]
	)
		-> Self
	{
		let buffer = layer_sizes.windows(2)
			.map(|inout| (inout[0], inout[1]))
			.map(|(inputs, outputs)| ConvNeuralLayer::random(inputs, outputs))
			.collect::<Vec<ConvNeuralLayer>>();
		ConvNeuralNet::from_vec(learning_rate, learning_momentum, act_fn, buffer)
	}

	fn output_layer(&self) -> &ConvNeuralLayer {
		self.layers.last().unwrap()
	}

	fn output_layer_mut(&mut self) -> &mut ConvNeuralLayer {
		self.layers.last_mut().unwrap()
	}

	fn overall_net_error(&self, target_values: &[f32]) -> f32 {
		let outputs = self.output_layer().output_as_slice();
		let sum = Zip::new((outputs.iter(), target_values))
			.map(|(output, target)| { let dx = target - output; dx*dx })
			.sum::<f32>();
		(sum / outputs.len() as f32).sqrt()
	}

	fn latest_error_stats(&self) -> ErrorStats {
		self.error_stats
	}

	fn propagate_gradients(&mut self, target_values: &[f32]) {
		let act_fn_dx = self.act_fns.derived; // because of borrow checker bugs

		if let Some((&mut ref mut last, ref mut tail)) = self.layers.split_last_mut() {
			tail.iter_mut()
				.rev()
				.fold(last.calculate_output_gradients(target_values, act_fn_dx),
				      |prev, layer| layer.propagate_gradients(prev, act_fn_dx));
		}
	}

	fn update_weights(&mut self, input: &[f32]) {
		let learn_rate     = self.learning_rate;
		let learn_momentum = self.learning_momentum;

		self.layers.iter_mut()
			.fold(input, |prev_output, layer| layer.update_weights(prev_output, learn_rate, learn_momentum));
	}

	fn update_error_stats(&mut self, target_values: &[f32]) -> ErrorStats {
		let latest_error = self.overall_net_error(target_values);
		self.error_stats.update(latest_error);
		self.error_stats
	}
}

impl NeuralNet for ConvNeuralNet {
	type Elem = f32;

	fn predict<'b, 'a: 'b>(&'a mut self, input: &'b [Self::Elem]) -> &'b [Self::Elem] {
		let act_fn = self.act_fns.base; // cannot be used in the fold as self.activation_fn
		self.layers.iter_mut()
			.fold(input, |out, layer| layer.feed_forward(out, act_fn))
	}
}

impl TrainableNeuralNet for ConvNeuralNet {
	fn train(&mut self, input: &[Self::Elem], target_values: &[Self::Elem]) -> ErrorStats {
		self.predict(input);
		self.propagate_gradients(target_values);
		self.update_weights(input);
		self.update_error_stats(target_values)
	}
}

#[cfg(test)]
mod tests {
	use prophet::neural_net::{
		NeuralNet,
		TrainableNeuralNet
	};
	use prophet::activation_fn::{
		BaseDerivedActivationFn
	};
	use super::{
		ConvNeuralNet,
		ConvNeuralLayer
	};

	#[test]
	fn train_xor() {
		let mut net = ConvNeuralNet::new(0.15, 0.4, BaseDerivedActivationFn::<f32>::tanh(), &[2, 4, 3, 1]);
		let t =  1.0;
		let f = -1.0;
		let print = false;
		for _ in 0..200 {
			if print {
				println!("(f,f) => {}"  , net.train(&[f, f], &[f]));
				println!("(f,t) => {}"  , net.train(&[f, t], &[t]));
				println!("(t,f) => {}"  , net.train(&[t, f], &[t]));
				println!("(t,t) => {}\n", net.train(&[t, t], &[f]));
			}
			else {
				net.train(&[f, f], &[f]);
				net.train(&[f, t], &[t]);
				net.train(&[t, f], &[t]);
				net.train(&[t, t], &[f]);
			}
		}
		assert!(net.latest_error_stats().avg_error() < 0.05);
	}

	#[test]
	fn train_constant() {
		let mut net = ConvNeuralNet::new(0.25, 0.5, BaseDerivedActivationFn::<f32>::identity(), &[1, 1]);
		let mut vx = vec![0.0; 1];
		let print = false;
		for _ in 0..100 {
			for &x in &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
				vx[0] = x;
				if print {
					println!("{} => {}", x, net.train(vx.as_slice(), &[1.0]));
				}
				else {
					net.train(vx.as_slice(), &[1.0]);
				}
			}
		}
		assert!(net.latest_error_stats().avg_error() < 0.05);
	}

	#[test]
	fn train_and() {
		let mut net = ConvNeuralNet::new(0.15, 0.5, BaseDerivedActivationFn::<f32>::tanh(), &[2, 3, 3, 1]);
		let f = -1.0;
		let t =  1.0;
		let print = false;
		for _ in 0..200 {
			if print {
				println!("(f, f) => {}"  , net.train(&[f, f], &[f]));
				println!("(f, t) => {}"  , net.train(&[f, t], &[f]));
				println!("(t, f) => {}"  , net.train(&[t, f], &[f]));
				println!("(t, t) => {}\n", net.train(&[t, t], &[t]));
			}
			else {
				net.train(&[f, f], &[f]);
				net.train(&[f, t], &[f]);
				net.train(&[t, f], &[f]);
				net.train(&[t, t], &[t]);
			}
		}
		assert!(net.latest_error_stats().avg_error() < 0.05);
	}
}