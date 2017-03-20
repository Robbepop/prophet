//! An implementation of a neural network that can be used to learn from target data
//! and to predict results after feeding it some training data.

use std::vec::Vec;

use rand::distributions::Range;
use ndarray_rand::RandomExt;
use ndarray::prelude::*;
use ndarray::Shape;
use itertools::{multizip, Itertools};

use traits::{LearnRate, LearnMomentum, Predict, UpdateGradients, UpdateWeights};
use activation::Activation;
use topology::*;

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
	weights      : Array2<f32>,
	delta_weights: Array2<f32>,
	outputs      : Array1<f32>,
	gradients    : Array1<f32>,
	activation   : Activation,
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
#[derive(Debug, Clone)]
pub struct NeuralNet {
	/// the layers within this ```NeuralNet```
	layers: Vec<FullyConnectedLayer>,
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
	fn random(inputs: Ix, outputs: Ix, activation: Activation) -> Self {
		assert!(inputs >= 1 && outputs >= 1);

		let inputs = inputs + 1; // implicitely add bias!
		let count_gradients = outputs + 1;
		let shape = Shape::from(Dim([outputs, inputs]));

		FullyConnectedLayer {
			weights:       Array2::random(shape, Range::new(0.0, 1.0)),
			delta_weights: Array2::default(shape),
			outputs:       Array1::default(outputs),
			gradients:     Array1::zeros(count_gradients),
			activation:    activation,
		}
	}

	fn count_rows(&self) -> Ix {
		let (rows, _) = self.weights.dim();
		rows
	}

	fn count_columns(&self) -> Ix {
		let (_, cols) = self.weights.dim();
		cols
	}

	fn count_outputs(&self) -> Ix {
		self.outputs.dim()
	}

	fn count_gradients(&self) -> Ix {
		self.gradients.dim()
	}

	/// Returns this layer's output as read-only view.
	fn output_view(&self) -> ArrayView1<f32> {
		self.outputs.view()
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
	fn feed_forward(&mut self,
	                input: ArrayView1<f32>)
	                -> ArrayView1<f32> {
		debug_assert_eq!(self.count_rows(), self.count_outputs());
		debug_assert_eq!(self.count_columns(), input.len() + 1);

		let act = self.activation;
		multizip((self.outputs.iter_mut(), self.weights.outer_iter()))
			.foreach(|(output, weights_row)| {
				*output = act.base(
					multizip((weights_row.iter(), input.iter().chain(&[1.0])))
						.map(|(w, i)| w * i)
						.sum())
			});

		self.output_view()
	}

	/// Used internally in the output layer to initialize gradients for the back propagation phase.
	/// Sets the gradient for the bias neuron to zero - hopefully this is the correct behaviour.
	fn calculate_output_gradients(&mut self,
	                              target_values: ArrayView1<f32>)
	                              -> &Self {
		debug_assert_eq!(self.count_outputs()  , target_values.len());
		debug_assert_eq!(self.count_gradients(), target_values.len() + 1); // no calculation for bias!

		let act = self.activation;
		multizip((self.gradients.iter_mut(), target_values.iter(), self.outputs.iter()))
			.foreach(|(gradient, target, &output)| { *gradient = (target - output) * act.derived(output) });

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
	fn apply_activation(&mut self) {
		debug_assert_eq!(self.count_gradients(), self.count_outputs() + 1);

		let act = self.activation;
		multizip((self.gradients.iter_mut(), self.outputs.iter().chain(&[1.0])))
			.foreach(|(gradient, &output)| *gradient *= act.derived(output));
	}

	/// Back propagate gradients from the previous layer (in reversed order) to this layer
	/// using the given activation function.
	/// This also computes the gradient for the bias neuron.
	/// Returns readable reference to self to allow chaining.
	fn propagate_gradients(&mut self,
	                       prev: &FullyConnectedLayer)
	                       -> &Self {
		debug_assert_eq!(prev.count_rows(), prev.count_gradients() - 1);
		debug_assert_eq!(prev.count_columns(), self.count_gradients());

		self.reset_gradients();

		multizip((prev.weights.outer_iter(), prev.gradients.iter()))
			.foreach(|(prev_weights_row, prev_gradient)| {
				multizip((self.gradients.iter_mut(), prev_weights_row.iter()))
					.foreach(|(gradient, weight)| *gradient += weight * prev_gradient)
			});

		self.apply_activation();
		self // for chaining in a fold expression
	}

	/// Updates the connection weights of this layer.
	/// This operation is usually used after successful computation of gradients.
	fn update_weights(&mut self,
	                  prev_outputs: ArrayView1<f32>,
	                  learn_rate  : LearnRate,
	                  learn_mom   : LearnMomentum)
	                  -> ArrayView1<f32> {
		debug_assert_eq!(prev_outputs.len() + 1, self.count_columns());
		debug_assert_eq!(self.count_gradients(), self.count_rows() + 1);

		multizip((self.weights.outer_iter_mut(),
		          self.delta_weights.outer_iter_mut(),
		          self.gradients.iter()))
			.foreach(|(mut weights_row, mut delta_weights_row, gradient)| {
				multizip((prev_outputs.iter().chain(&[1.0]),
				          weights_row.iter_mut(),
				          delta_weights_row.iter_mut()))
					.foreach(|(prev_output, weight, delta_weight)| {
						*delta_weight =
							// Individual input, magnified by the gradient and train rate
							learn_rate.0 * prev_output * gradient
							// Also add momentum which is a fraction of the previous delta weight
							+ learn_mom.0 * *delta_weight;
						*weight += *delta_weight;
					})
			});

		self.output_view()
	}
}

impl NeuralNet {
	/// Creates a new neural network from a given vector of fully connected layers.
	///
	/// This constructor should only be used internally!
	fn from_vec(layers: Vec<FullyConnectedLayer>) -> Self {
		NeuralNet {
			layers: layers
		}
	}

	/// Creates a new neural network of fully connected layers from a given topology.
	pub fn from_topology(topology: Topology) -> Self {
		NeuralNet::from_vec(topology
			.iter_layers()
			.map(|&layer| {
				FullyConnectedLayer::random(
					layer.inputs, layer.outputs, layer.activation)
			})
			.collect()
		)
	}
}

impl From<Topology> for NeuralNet {
	fn from(topology: Topology) -> Self {
		NeuralNet::from_topology(topology)
	}
}

impl<'b, A> Predict<A> for NeuralNet
	where A: Into<ArrayView1<'b, f32>>
{
	fn predict(&mut self, input: A) -> ArrayView1<f32> {
		let input  = input.into();
		if let Some((first, tail)) = self.layers.split_first_mut() {
			tail.iter_mut()
				.fold(first.feed_forward(input),
				      |prev, layer| layer.feed_forward(prev))
		} else {
			panic!("A Neural Net is guaranteed to have at least one layer so this situation \
			        should never happen!");
		}
	}
}

impl<'a, A> UpdateGradients<A> for NeuralNet
	where A: Into<ArrayView1<'a, f32>>
{
	fn update_gradients(&mut self, target_values: A) {
		if let Some((&mut ref mut last, ref mut tail)) = self.layers.split_last_mut() {
			tail.iter_mut()
				.rev()
				.fold(last.calculate_output_gradients(target_values.into()),
				      |prev, layer| layer.propagate_gradients(prev));
		}
	}
}

impl<'b, A> UpdateWeights<A> for NeuralNet
	where A: Into<ArrayView1<'b, f32>>
{
	fn update_weights(&mut self, input: A, rate: LearnRate, momentum: LearnMomentum) {
		let input = input.into();
		if let Some((first, tail)) = self.layers.split_first_mut() {
			tail.iter_mut()
				.fold(first.update_weights(input, rate, momentum),
				      |prev, layer| layer.update_weights(prev, rate, momentum));
		}
	}
}

#[cfg(all(feature = "bench", test))]
mod bench {
	use super::*;
	use test::{
		Bencher,
		// black_box
	};

	use learn_config::LearnConfig;
	use activation_fn::ActivationFn;
	use super::NeuralNet;

	#[bench]
	fn bench_giant(bencher: &mut Bencher) {

		// TODO

		// let config = LearnConfig::new(0.25, 0.5, ActivationFn::<f32>::tanh());
		// let mut net = NeuralNet::new(config, &[2, 1000, 1000, 1]);
		// let f = -1.0;
		// let t =  1.0;

		// bencher.iter(|| {
		// 	net.train(&[f, f], &[f]);
		// 	net.train(&[f, t], &[f]);
		// 	net.train(&[t, f], &[f]);
		// 	net.train(&[t, t], &[t]);
		// });
	}
}
