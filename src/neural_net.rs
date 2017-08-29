//! An implementation of a neural network that can be used to learn from target data
//! and to predict results after feeding it some training data.

use std::vec::Vec;

use rand::distributions::Range;
use ndarray_rand::RandomExt;
use ndarray::prelude::*;
use ndarray::{Zip, Ix};
use itertools::{multizip, Itertools};

use traits::{Predict, UpdateGradients, UpdateWeights};
use utils::{LearnRate, LearnMomentum};
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
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct NeuralNet {
	/// A bias-extended input buffer for user input
	/// that is not required to respect bias values.
	/// 
	/// This is needed so that the internal implementation
	/// can use the invariant of symmetrically sized buffers
	/// and weight-matrices for optimization purposes.
	/// 
	/// Later this operation could be done with a unique layer type,
	///    e.g. `InputLayer`: adds bias values to user-provided input.
	input: Array1<f32>,
	/// The actual layers of this `NeuralNet`.
	layers: Vec<FullyConnectedLayer>,
}

impl FullyConnectedLayer {
	fn with_weights(weights: Array2<f32>, activation: Activation) -> Self {

		// Implicitely add a bias neuron to all arrays and matrices.
		// 
		// In theory this is only required for gradients and both
		// weight matrices, not for outputs. However, it is done for outputs, too,
		// to create size-symmetry which simplifies implementation of
		// optimized algorithms.
		let (n_outputs, _biased_inputs) = weights.dim();
		let biased_outputs   = n_outputs + 1;
		let biased_gradients = biased_outputs;
		let biased_shape     = weights.dim();

		// Construct outputs with a `1.0` constant bias value as last element.
		let mut outputs = Array1::zeros(biased_outputs);
		outputs[n_outputs] = 1.0;

		FullyConnectedLayer{
			weights,
			delta_weights: Array2::zeros(biased_shape), // Must be initialized with zeros or else computation
			                                            // in the first iteration will be screwed!
			outputs,
			gradients: Array1::zeros(biased_gradients),
			activation: activation,
		}
	}

	/// Creates a FullyConnectedLayer with randomized weights.
	///
	/// Implicitely creates weights for the bias neuron,
	/// so the dimensions of the weights matrix is equal to
	/// (output)x(input+1).
	///
	/// The weights are randomized within the open interval (0,1).
	/// This excludes 0.0 and 1.0 as weights.
	/// Other optional intervals may come with a future update!
	fn random(n_inputs: Ix, n_outputs: Ix, activation: Activation) -> Self {
		assert!(n_inputs >= 1 && n_outputs >= 1);

		// Implicitely add a bias neuron to all arrays and matrices.
		// 
		// In theory this is only required for gradients and both
		// weight matrices, not for outputs. However, it is done for outputs, too,
		// to create size-symmetry which simplifies implementation of
		// optimized algorithms.

		let biased_inputs    = n_inputs  + 1;
		let biased_outputs   = n_outputs + 1;
		let biased_gradients = biased_outputs;
		let biased_shape     = (n_outputs, biased_inputs);

		let mut outputs = Array1::zeros(biased_outputs);
		outputs[n_outputs] = 1.0; // The last value of any outputs array represents
		                          // the value of the bias neuron which is always `1.0`
		                          // and should never change during computation.

		// println!("FullyConnectedLayer::random() :: outputs = {}", outputs);

		FullyConnectedLayer {
			weights:       Array2::random(biased_shape, Range::new(-1.0, 1.0)), // Maybe `(-1.0, 1.0)` is a sub-optimal range?
			delta_weights: Array2::zeros(biased_shape), // Must be initialized with zeros or else computation
			                                            // in the first iteration will be screwed!
			outputs,
			gradients:     Array1::zeros(biased_gradients),
			activation:    activation,
		}
	}

	/// Count output neurons of this layer.
	/// 
	/// This method should be only used for debugging purposes!
	#[inline]
	fn count_outputs(&self) -> Ix {
		self.outputs.dim()
	}

	/// Count gradients of this layer.
	/// 
	/// This method should be only used for debugging purposes!
	#[inline]
	fn count_gradients(&self) -> Ix {
		self.gradients.dim()
	}

	/// Returns this layer's output as read-only view.
	#[inline]
	fn output_view(&self) -> ArrayView1<f32> {
		self.outputs.view()
	}

	/// Takes input slice and performs a feed forward procedure
	/// using the given activation function.
	/// Output of this operation will be stored within this layer
	/// and be returned as readable slice.
	///
	/// ### Expects
	/// 
	///  - input with `n` elements
	/// 
	/// ### Requires
	/// 
	///  - weight matrix with `m` rows and `n` columns
	/// 
	/// ### Asserts
	/// 
	///  - output with `m` elements
	/// 
	/// ### Returns
	/// 
	///  - A view to the resulting output. Useful for method chaining and reductions.
	/// 
	fn feed_forward(
		&mut self,
		input: ArrayView1<f32>
	)
		-> ArrayView1<f32>
	{
		debug_assert_eq!(self.weights.rows() + 1, self.count_outputs());
		debug_assert_eq!(self.weights.cols()    , input.len());

		use ndarray::linalg::general_mat_vec_mul;

		general_mat_vec_mul(1.0, &self.weights, &input, 1.0, &mut (self.outputs.slice_mut(s![..-1])));

		self.apply_activation();

		self.output_view() // required for folding the general operation
	}

	/// Used internally in the output layer to initialize gradients for the back propagation phase.
	/// Sets the gradient for the bias neuron to zero - hopefully this is the correct behaviour.
	/// 
	/// ### Returns
	/// 
	/// `&Self` to allow for method chaining, especially reductions.
	/// 
	fn calculate_output_gradients(
		&mut self,
	    target_values: ArrayView1<f32>
	)
		-> &Self
	{
		debug_assert_eq!(self.count_outputs()  , target_values.len() + 1); // No calculation for bias neurons.
		debug_assert_eq!(self.count_gradients(), target_values.len() + 1); // see above ...

		let act = self.activation; // Required because of non-lexical borrows.

		// Just as slow as the old version below ...
		Zip::from(&mut self.gradients.slice_mut(s![..-1]))
				.and(&target_values)
				.and(&self.outputs.slice(s![..-1]))
				.apply(|gradient, &target, &output| {
			*gradient = (target - output) * act.derived(output)
		});

		// gradient of bias should be set equal to zero during object initialization already.
		self
	}

	/// Sets all gradient values in this layer to zero.
	/// This is required as initialization step before propagating gradients
	/// for the efficient implementation of this library.
	#[inline]
	fn reset_gradients(&mut self) {
		self.gradients.fill(0.0);
		debug_assert!(self.gradients.iter().all(|&g| g == 0.0));
	}

	/// Applies the given activation function on all gradients of this layer.
	/// 
	/// ### Expects
	/// 
	/// - number of gradients equals number of outputs
	/// 
	fn apply_activation(&mut self) {
		debug_assert_eq!(self.count_gradients(), self.count_outputs());

		let act = self.activation; // Required because of non-lexical borrows.

		multizip((&mut self.gradients, &self.outputs))
			.foreach(|(gradient, &output)| *gradient *= act.derived(output));
	}

	/// Back propagate gradients from the previous layer (in reversed order) to this layer
	/// using the given activation function.
	/// This also computes the gradient for the bias neuron.
	/// Returns readable reference to self to allow for method chaining.
	/// 
	/// ### Returns
	/// 
	/// `&Self` to allow for method chaining, especially reductions.
	fn propagate_gradients(
		&mut self,
	    prev: &FullyConnectedLayer
	)
	    -> &Self
	{
		debug_assert_eq!(prev.weights.rows() + 1, prev.count_gradients());
		debug_assert_eq!(prev.weights.cols()    , self.count_gradients());

		multizip((prev.weights.outer_iter(), prev.gradients.iter()))
			.foreach(|(prev_weights_row, prev_gradient)| {
				multizip((self.gradients.iter_mut(), prev_weights_row.iter()))
					.foreach(|(gradient, weight)| *gradient += weight * prev_gradient)
			});

		self.apply_activation();
		self
	}

	/// Updates the connection weights of this layer.
	/// This operation is usually used after successful computation of gradients.
	fn update_weights(
		&mut self,
	    prev_outputs: ArrayView1<f32>,
	    learn_rate  : LearnRate,
	    learn_mom   : LearnMomentum
	)
	    -> ArrayView1<f32>
	{
		debug_assert_eq!(prev_outputs.len()    , self.weights.cols()    );
		debug_assert_eq!(self.count_gradients(), self.weights.rows() + 1);

		// Compute new delta weights.
		multizip((self.delta_weights.outer_iter_mut(),
		          self.gradients.iter()))
			.foreach(|(mut delta_weights_row, gradient)| {
				multizip((prev_outputs.iter(),
				          delta_weights_row.iter_mut()))
					.foreach(|(prev_output, delta_weight)| {
						*delta_weight =
							// Individual input, magnified by the gradient and train rate
							learn_rate.0 * prev_output * gradient
							// Also add momentum which is a fraction of the previous delta weight
							+ learn_mom.0 * *delta_weight;
					});
			});

		// Add the delta weights computed above to the current real weights.
		self.weights += &self.delta_weights;

		self.reset_gradients();
		self.output_view()
	}
}

impl NeuralNet {
	/// Creates a new neural network from a given vector of fully connected layers.
	///
	/// This constructor should only be used internally!
	fn from_vec(inputs: usize, layers: Vec<FullyConnectedLayer>) -> Self {
		assert!(!layers.is_empty());
		let mut biased_input = Array1::zeros(inputs + 1);
		biased_input[inputs] = 1.0; // Set bias value which is always `1.0`.
		                                // This initial value should never be overwritten.
		debug_assert_eq!(biased_input[inputs], 1.0);
		NeuralNet {
			input : biased_input,
			layers: layers
		}
	}

	/// Creates a new neural network of fully connected layers from a given topology.
	pub fn from_topology(topology: Topology) -> Self {
		NeuralNet::from_vec(
			topology.len_input(),
			topology
				.iter_layers()
				.map(|&layer| {
					FullyConnectedLayer::random(
						layer.inputs, layer.outputs, layer.activation)
				})
				.collect()
		)
	}
}

impl<'b, A> Predict<A> for NeuralNet
	where A: Into<ArrayView1<'b, f32>>
{
	fn predict(&mut self, input: A) -> ArrayView1<f32> {
		let input = input.into();

		debug_assert_eq!(input.len() + 1, self.input.len());
		debug_assert_eq!(self.input[self.input.len() - 1], 1.0);

		// Copy the user provided inputs into a buffer that is
		// extended to additionally store the bias values (which is always `1.0`).
		// This is used by the implementation internals for optimizations.
		self.input.slice_mut(s![..-1]).assign(&input);

		debug_assert_eq!(input.len() + 1, self.input.len());
		debug_assert_eq!(self.input[self.input.len() - 1], 1.0);

		// Compute the feed forward from first to last layer.
		if let Some((first, tail)) = self.layers.split_first_mut() {
			tail.iter_mut()
				.fold(first.feed_forward(self.input.view()), |prev, layer| {
					layer.feed_forward(prev)
				})
		} else {
			unreachable!("Since neural networks can never have an empty stack of layers 
				          this code should never be reachable.");
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

		debug_assert_eq!(input.len() + 1, self.input.len());

		// Copy the user provided inputs into a buffer that is
		// extended to additionally store the bias values (which is always `1.0`).
		// This is used by the implementation internals for optimizations.
		self.input.slice_mut(s![..-1]).assign(&input);
		// Update the weights of this neural network from first to last layer.
		if let Some((first, tail)) = self.layers.split_first_mut() {
			tail.iter_mut()
				.fold(first.update_weights(self.input.view(), rate, momentum),
				      |prev, layer| layer.update_weights(prev, rate, momentum));
		}
	}
}

#[cfg(test)]
mod tests {
	pub use super::*;

	mod fully_connected_layer {
		use super::*;

		#[test]
		fn feed_forward() {
			use self::Activation::{Identity};
			let mut layer = FullyConnectedLayer::with_weights(
				Array2::from_shape_fn((3, 4), |(r, c)| ((4*r+c) + 1) as f32), Identity);
			let applier = Array1::linspace(1.0, 4.0, 4);
			let outputs = layer.feed_forward(applier.view()).to_owned();
			let targets = Array1::from_vec(vec![30.0, 70.0, 110.0, 1.0]);
			// println!("layer =\n{:?}", layer.weights);
			// println!("applier =\n{:?}", applier);
			// println!("outputs =\n{:?}", outputs);
			// println!("targets =\n{:?}", targets);
			assert_eq!(outputs, targets);
		}

		#[test]
		fn update_output_gradients() {
			// TODO
		}

		#[test]
		fn update_gradients() {
			// TODO
		}

		#[test]
		fn update_weights() {
			// TODO
		}
	}
}
