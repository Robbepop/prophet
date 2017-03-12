//! An implementation of a neural network that can be used to learn from target data
//! and to predict results after feeding it some training data.

use std::vec::Vec;

use rand::distributions::Range;
use ndarray_rand::RandomExt;
use ndarray::prelude::*;
use ndarray::{Shape};
use itertools::{multizip, Itertools};

use learn_config::{LearnConfig};
use error_stats::{ErrorStats};
use traits::{
	Prophet,
	Disciple
};
use activation_fn::{BaseFn, DerivedFn};

/// A fully connected layer within a neural net.
/// 
/// The layer constists of a weights and a delta-weights matrix with equal dimensions
/// and an output and gradients vector of equal size.
/// 
/// The values stored within the n-th column of the two weights matrices are respective to 
/// the weights of the n-th input neuron to all other neurons of the next neuron layer.
/// 
/// And respective are all values stored within the n-th row or the two weights matrices
/// interpreted as the weights of all incoming connections to the n-th neuron in the next 
/// neuron layer.
/// 
/// For predicting only the weights matrix and the outputs vector is required.
/// 
/// The outputs and gradients vectors are just used as frequently used intermediate buffers
/// which should speed up computation.
/// 
/// The data structure is organized in a way that it mainly represents the connections
/// between two adjacent neuron layers.
/// It owns its output but needs a reference to its input.
/// This design was chosen to be the most modular one and enables to avoid copy-overhead in 
/// all currently developed situations.
/// 
/// Besides that this design allows to completely avoid heap memory allocations after 
/// setting up the objects initially.
#[derive(Debug, Clone, PartialEq)]
struct FullyConnectedLayer {
	weights:       Array2<f32>,
	delta_weights: Array2<f32>,
	outputs:       Array1<f32>,
	gradients:     Array1<f32>
}

/// A neural net.
/// 
/// Can be trained with testing data and afterwards be used to predict results.
/// 
/// Neural nets in this implementation constists of several stacked neural layers
/// and organized the data flow between them.
/// 
/// For example when the user uses ```predict``` from ```NeuralNet``` this
/// object organizes the input data throughout all of its owned layers and pipes
/// the result in the last layer back to the user.
pub struct NeuralNet {
	/// the layers within this ```NeuralNet```
	layers: Vec<FullyConnectedLayer>,

	/// the config that handles all the parameters to tune the learning process
	pub config: LearnConfig,

	/// holds error stats for the user to query the current learning state of the network
	error_stats: ErrorStats
}

impl FullyConnectedLayer {
	/// Creates a FullyConnectedLayer with randomized weights.
	/// 
	/// Implicitely creates weights for the bias neuron,
	/// so the dimensions of the weights matrix is equal to
	/// (output)x(input+1).
	/// 
	/// The weights are randomized within the open interval (0,1).
	/// This excludes 0.0 and 1.0 as weights.
	/// Other optional intervals may come with a future update!
	fn random(inputs: Ix, outputs: Ix) -> Self {
		assert!(inputs >= 1 && outputs >= 1);

		let inputs          = inputs  + 1; // implicitely add bias!
		let count_gradients = outputs + 1;
		let shape           = Shape::from(Dim([outputs, inputs]));

		FullyConnectedLayer{
			weights:       Array2::random(shape, Range::new(0.0, 1.0)),
			delta_weights: Array2::default(shape),
			outputs:       Array1::default(outputs),
			gradients:     Array1::zeros(count_gradients)
		}
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
	fn feed_forward<'a>(
		&'a mut self,
		input: &[f32],
		activation_fn: BaseFn<f32>
	)
		-> &'a [f32]
	{
		debug_assert_eq!(self.count_rows(), self.count_outputs());
		debug_assert_eq!(self.count_columns(), input.len() + 1);

		multizip((self.outputs.iter_mut(), self.weights.outer_iter()))
			.foreach(|(output, weights_row)| {
				*output = activation_fn(
					multizip((weights_row.iter(), input.iter().chain(&[1.0])))
						.map(|(w, i)| w*i)
						.sum()
				) 
			});

		self.output_as_slice()
	}

	/// Used internally in the output layer to initialize gradients for the back propagation phase.
	/// Sets the gradient for the bias neuron to zero - hopefully this is the correct behaviour.
	fn calculate_output_gradients(&mut self, target_values: &[f32], act_fn_dx: DerivedFn<f32>) -> &Self {
		debug_assert_eq!(self.count_outputs(), target_values.len());
		debug_assert_eq!(self.count_gradients(), target_values.len() + 1); // no calculation for bias!

		multizip((self.gradients.iter_mut(), target_values.iter(), self.outputs.iter()))
			.foreach(|(gradient, target, &output)| {
				*gradient = (target - output) * act_fn_dx(output)
			});

		// gradient of bias should be set equal to zero during object initialization already.
		self
	}

	/// Sets all gradient values in this layer to zero.
	/// This is required as initialization step before propagating gradients
	/// for the efficient implementation of this library.
	fn reset_gradients(&mut self) {
		self.gradients.fill(0.0);
		debug_assert!(self.gradients.iter().all(|&g| g == 0.0));
	}

	/// Applies the given activation function on all gradients of this layer.
	fn apply_activation(&mut self, act_fn_dx: DerivedFn<f32>) {
		debug_assert_eq!(self.count_gradients(), self.count_outputs() + 1);

		multizip((self.gradients.iter_mut(), self.outputs.iter().chain(&[1.0])))
			.foreach(|(gradient, &output)| *gradient *= act_fn_dx(output));
	}

	/// Back propagate gradients from the previous layer (in reversed order) to this layer
	/// using the given activation function.
	/// This also computes the gradient for the bias neuron.
	/// Returns readable reference to self to allow chaining.
	fn propagate_gradients(&mut self, prev: &FullyConnectedLayer, act_fn_dx: DerivedFn<f32>) -> &Self {
		debug_assert_eq!(prev.count_rows(), prev.count_gradients() - 1);
		debug_assert_eq!(prev.count_columns(), self.count_gradients());

		self.reset_gradients();

		multizip((prev.weights.outer_iter(), prev.gradients.iter()))
			.foreach(|(prev_weights_row, prev_gradient)| {
				multizip((self.gradients.iter_mut(), prev_weights_row.iter()))
					.foreach(|(gradient, weight)| {
						*gradient += weight * prev_gradient
					})
			});

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

		for (mut weights_row, mut delta_weights_row, gradient) in multizip((self.weights.outer_iter_mut(),
		                                                                    self.delta_weights.outer_iter_mut(),
		                                                                    self.gradients.iter())) {
			for (prev_output, weight, delta_weight) in multizip((prev_outputs.iter().chain(&[1.0]),
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



impl NeuralNet {
	fn from_vec(
		learn_config: LearnConfig,
		layers: Vec<FullyConnectedLayer>
	)
		-> Self
	{
		NeuralNet{
			layers: layers,
			config: learn_config,
			error_stats: ErrorStats::default()
		}
	}

	/// Creates a new instance of a ```NeuralNet```.
	/// 
	///  - ```layer_sizes``` define the count of neurons (without bias) per neural layer.
	///  - ```learning_rate``` and ```learning_momentum``` describe the acceleration and momentum
	/// with which the created neural net will be learning. These values can be changed later during
	/// the lifetime of the object if needed.
	///  - ```act_fn``` represents the pair of activation function and derivate used throughout the
	/// neural net layers.
	/// 
	/// Weights between the neural layers are initialized to ```(0,1)```.
	/// 
	/// # Examples
	///
	/// ```
	/// use prophet::prelude::*;
	///
	/// let config  = LearnConfig::new(
	/// 	0.15,                // learning_rate
	/// 	0.4,                 // learning_momentum
	/// 	ActivationFn::tanh() // activation function + derivate
	/// );
	/// let mut net = NeuralNet::new(config, &[2, 4, 3, 1]);
	/// // layer_sizes: - input layer which expects two values
	/// //              - two hidden layers with 4 and 3 neurons
	/// //              - output layer with one neuron
	/// 
	/// // now train the neural net how to be an XOR-operator
	/// let f = -1.0; // represents false
	/// let t =  1.0; // represents true
	/// for _ in 0..1000 {
	/// 	net.train(&[f, f], &[f]);
	/// 	net.train(&[f, t], &[t]);
	/// 	net.train(&[t, f], &[t]);
	/// 	net.train(&[t, t], &[f]);
	/// }
	/// // now check if the neural net has successfully learned it by checking how close
	/// // the latest ```avg_error``` is to ```0.0```:
	/// assert!(net.latest_error_stats().avg_error() < 0.05);
	/// ```
	pub fn new(
		config: LearnConfig,
		layer_sizes: &[Ix]
	)
		-> Self
	{
		let buffer = layer_sizes
			.iter()
			.tuple_windows::<(_, _)>()
			.map(|(&inputs, &outputs)| FullyConnectedLayer::random(inputs, outputs))
			.collect::<Vec<FullyConnectedLayer>>();
		NeuralNet::from_vec(config, buffer)
	}

	fn output_layer(&self) -> &FullyConnectedLayer {
		self.layers.last().unwrap()
	}

	fn overall_net_error(&self, target_values: &[f32]) -> f32 {
		let outputs = self.output_layer().output_as_slice();
		let sum = multizip((outputs.iter(), target_values))
			.map(|(output, target)| { let dx = target - output; dx*dx })
			.sum::<f32>();
		(sum / outputs.len() as f32).sqrt()
	}

	/// Returns the ```ErrorStats``` that were generated by the latest call to ```train```.
	/// 
	/// Returns a default constructed ```ErrorStats``` object when this neural net
	/// was never trained before.
	pub fn latest_error_stats(&self) -> ErrorStats {
		self.error_stats
	}

	fn propagate_gradients(&mut self, target_values: &[f32]) {
		let act_fn_dx = self.config.act_fn.derived_fn(); // because of borrow checker bugs

		if let Some((&mut ref mut last, ref mut tail)) = self.layers.split_last_mut() {
			tail.iter_mut()
				.rev()
				.fold(last.calculate_output_gradients(target_values, act_fn_dx),
				      |prev, layer| layer.propagate_gradients(prev, act_fn_dx));
		}
	}

	fn update_weights(&mut self, input: &[f32]) {
		let learn_rate     = self.config.learn_rate();
		let learn_momentum = self.config.learn_momentum();

		self.layers
			.iter_mut()
			.fold(input, |prev_output, layer| layer.update_weights(prev_output, learn_rate, learn_momentum));
	}

	fn update_error_stats(&mut self, target_values: &[f32]) -> ErrorStats {
		let latest_error = self.overall_net_error(target_values);
		self.error_stats.update(latest_error);
		self.error_stats
	}
}

impl Prophet for NeuralNet {
	type Elem = f32;

	fn predict<'b, 'a: 'b>(&'a mut self, input: &'b [Self::Elem]) -> &'b [Self::Elem] {
		let act_fn = self.config.act_fn.base_fn(); // cannot be used in the fold as self.activation_fn
		self.layers
			.iter_mut()
			.fold(input, |out, layer| layer.feed_forward(out, act_fn))
	}
}

impl Disciple for NeuralNet {
	type Elem = f32;

	fn train(&mut self, input: &[Self::Elem], target_values: &[Self::Elem]) -> ErrorStats {
		self.predict(input);
		self.propagate_gradients(target_values);
		self.update_weights(input);
		self.update_error_stats(target_values)
	}
}

#[cfg(test)]
mod tests {
	use learn_config::{
		LearnConfig
	};
	use traits::*;
	use activation_fn::{
		ActivationFn
	};
	use super::{
		NeuralNet
	};

	#[test]
	fn train_xor() {
		let config  = LearnConfig::new(0.15, 0.4, ActivationFn::<f32>::tanh());
		let mut net = NeuralNet::new(config, &[2, 4, 3, 1]);
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
		let config  = LearnConfig::new(0.25, 0.5, ActivationFn::<f32>::identity());
		let mut net = NeuralNet::new(config, &[1, 1]);
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
		let config  = LearnConfig::new(0.15, 0.5, ActivationFn::<f32>::tanh());
		let mut net = NeuralNet::new(config, &[2, 1]);
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

	#[test]
	fn train_triple_add() {
		use rand::*;
		let config  = LearnConfig::new(0.25, 0.5, ActivationFn::<f32>::identity());
		let mut net = NeuralNet::new(config, &[3, 1]);
		let mut gen = thread_rng();
		let print   = false;
		for _ in 0..200 {
			let a = gen.next_f32();
			let b = gen.next_f32();
			let c = gen.next_f32();
			if print {
				println!("{} + {} + {} (= {}) => {}", a, b, c, a+b+c, net.train(&[a, b, c], &[a+b+c]));
			}
			else {
				net.train(&[a, b, c], &[a+b+c]);
			}
		}
		assert!(net.latest_error_stats().avg_error() < 0.05);
	}
}


#[cfg(all(feature = "bench", test))]
mod bench {
	use super::*;
	use test::{Bencher, black_box};

	use traits::*;
	use learn_config::{LearnConfig};
	use activation_fn::{ActivationFn};
	use super::{NeuralNet};

	#[bench]
	fn bench_giant(bencher: &mut Bencher) {
		let config  = LearnConfig::new(0.25, 0.5, ActivationFn::<f32>::tanh());
		let mut net = NeuralNet::new(config, &[2, 1000, 1000, 1]);
		let f = -1.0;
		let t =  1.0;

		bencher.iter(|| {
			net.train(&[f, f], &[f]);
			net.train(&[f, t], &[f]);
			net.train(&[t, f], &[f]);
			net.train(&[t, t], &[t]);
		});
	}
}
