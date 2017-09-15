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
		use std::iter;

		// Implicitely add a bias neuron to all arrays and matrices.
		// 
		// In theory this is only required for gradients and both
		// weight matrices, not for outputs. However, it is done for outputs, too,
		// to create size-symmetry which simplifies implementation of
		// optimized algorithms.
		let (n_outputs, _)   = weights.dim();
		let biased_outputs   = n_outputs + 1;
		let biased_gradients = biased_outputs;
		let biased_shape     = weights.dim();

		FullyConnectedLayer{
			weights,

			// Must be initialized with zeros or else computation
			// in the first iteration will be screwed!
			delta_weights: Array2::zeros(biased_shape),

			// Construct outputs with a `1.0` constant bias value as last element.
			outputs: Array1::from_iter(iter::repeat(0.0).take(n_outputs).chain(iter::once(1.0))),

			// Gradients must be initialized with zeros to prevent accidentally
			// compute invalid gradients on the first iteration.
			gradients: Array1::zeros(biased_gradients),

			// Initialize the activation function. TODO: Should be moved into its own layer.
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

		let biased_inputs = n_inputs  + 1;
		let biased_shape  = (n_outputs, biased_inputs);

		FullyConnectedLayer::with_weights(
			Array2::random(biased_shape, Range::new(-1.0, 1.0)), activation)
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

	/// Returns this layer's output as read-only view.
	#[inline]
	#[cfg(test)]
	fn gradients_view(&self) -> ArrayView1<f32> {
		self.gradients.view()
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

		self.apply_activation_to_outputs();

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

	/// Applies this layer's activation function on all gradients of this layer.
	/// 
	/// ### Expects
	/// 
	/// - number of gradients equals number of outputs
	/// 
	fn apply_activation_to_gradients(&mut self) {
		debug_assert_eq!(self.count_gradients(), self.count_outputs());

		let act = self.activation; // Required because of non-lexical borrows.

		// NEW CODE
		Zip::from(&mut self.gradients).and(&self.outputs).apply(|g, o| *g *= act.derived(*o));

		// OLD CODE: slightly faster sequentially, but also not as easy parallelizable as new code above ...
		// multizip((&mut self.gradients, &self.outputs))
		// 	.foreach(|(gradient, &output)| *gradient *= act.derived(output));
	}

	/// Applies this layer's activation function on all outputs of this layer.
	fn apply_activation_to_outputs(&mut self) {
		let act = self.activation;
		self.outputs.slice_mut(s![..-1]).mapv_inplace(|o| act.base(o));
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

		multizip((prev.weights.genrows(), prev.gradients.iter()))
			.foreach(|(prev_weights_row, prev_gradient)| {
				multizip((self.gradients.iter_mut(), prev_weights_row.iter()))
					.foreach(|(gradient, weight)| *gradient += weight * prev_gradient)
			});

		self.apply_activation_to_gradients();
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
		debug_assert_eq!(prev_outputs.len()    , self.weights.cols());
		debug_assert_eq!(self.count_gradients(), self.delta_weights.rows() + 1);

		// Compute new delta weights.
		multizip((self.delta_weights.genrows_mut(),
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
		use std::iter;

		assert!(!layers.is_empty());

		NeuralNet {
			input : Array::from_iter(iter::repeat(0.0).take(inputs).chain(iter::once(1.0))),
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

impl UpdateWeights for NeuralNet
{
	fn update_weights(
		&mut self, 
		rate: LearnRate,
		momentum: LearnMomentum
	) {
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

		use std::iter;

		#[test]
		fn construction_invariants() {
			use self::Activation::{Identity};
			let weights = Array1::linspace(1.0, 12.0, 12).into_shape((3, 4)).unwrap();
			let layer = FullyConnectedLayer::with_weights(weights.clone(), Identity);
			assert_eq!(layer.weights, weights);
			assert_eq!(layer.delta_weights, Array1::zeros(12).into_shape((3, 4)).unwrap());
			assert_eq!(layer.gradients, Array1::zeros(4));
			let expected_outputs = Array1::from_iter(iter::repeat(0.0).take(3).chain(iter::once(1.0)));
			assert_eq!(layer.outputs, expected_outputs);
		}

		#[test]
		fn feed_forward() {
			fn assert_raw_config_with_expected(
				weights: Array2<f32>,
				activation: Activation,
				applier: Array1<f32>,
				expected: Array1<f32>
			) {
				let mut layer = FullyConnectedLayer::with_weights(weights, activation);
				let result = layer.feed_forward(applier.view()).to_owned();
				assert!(result.all_close(&expected, 1e-6));
			}

			fn assert_config_with_expected(
				weights: Array2<f32>,
				applier: Array1<f32>,
				expected: Array1<f32>
			) {
				use self::Activation::*;
				let activations = [Identity, Tanh, Logistic, SoftPlus, ReLU, Gaussian];
				for act in &activations {
					let mut expected_activated = expected.clone();
					expected_activated.slice_mut(s![..-1]).mapv_inplace(|e| act.base(e));
					assert_raw_config_with_expected(
						weights.clone(), *act,
						applier.clone(),
						expected_activated
					);
				}
			}

			assert_config_with_expected(
				Array::linspace(1.0, 12.0, 12).into_shape((3, 4)).unwrap(),
				Array::from_vec(vec![1.0, 2.0, 3.0, 1.0]),
				Array::from_vec(vec![
					1.0*1.0 +  2.0*2.0 +  3.0*3.0 +  4.0*1.0, // = 18.0,
					5.0*1.0 +  6.0*2.0 +  7.0*3.0 +  8.0*1.0, // = 46.0,
					9.0*1.0 + 10.0*2.0 + 11.0*3.0 + 12.0*1.0, // = 74.0,
					1.0 // just the bias!
				])
			);
		}

		#[test]
		fn calculate_output_gradients() {
			fn assert_raw_config_with_expected(
				outputs   : Array1<f32>,
				activation: Activation,
				targets   : Array1<f32>,
				expected_gradients: Array1<f32>
			) {
				let unbiased_dim = (outputs.dim() - 1, 1);
				let zero_weights = Array2::zeros(unbiased_dim);
				let mut layer = FullyConnectedLayer{
					weights      : zero_weights.clone(),
					delta_weights: zero_weights.clone(),
					gradients    : Array::zeros(outputs.dim()),
					outputs      : outputs.clone(),
					activation
				};
				layer.calculate_output_gradients(targets.view());
				let result_gradients = layer.gradients.clone();
				assert!(result_gradients.all_close(&expected_gradients, 1e-6));
			}

			fn assert_config_with_expected(
				outputs: Array1<f32>,
				targets: Array1<f32>,
				expected_gradients: Array1<f32>
			) {
				use self::Activation::*;
				let activations = [Identity, Tanh, Logistic, SoftPlus, ReLU, Gaussian];
				for act in &activations {
					let mut expected_gradients_activated = expected_gradients.clone();
					multizip((expected_gradients_activated.slice_mut(s![..-1]), outputs.slice(s![..-1])))
						.foreach(|(g, o)| *g *= act.derived(*o));
					assert_raw_config_with_expected(
						outputs.clone(), *act,
						targets.clone(),
						expected_gradients_activated
					);
				}
			}

			assert_config_with_expected(
				Array1::from_vec(vec![ 1.0,  2.0,  3.0,  4.0,  5.0, 1.0]),
				Array1::from_vec(vec![10.0, 10.0, 10.0, 10.0, 10.0]),
				Array1::from_vec(vec![
					10.0 - 1.0,
					10.0 - 2.0,
					10.0 - 3.0,
					10.0 - 4.0,
					10.0 - 5.0,
					0.0
				])
			);
		}

		#[test]
		fn propagate_gradients() {
			fn assert_raw_config_with_expected(
				self_outputs: Array1<f32>,
				activation: Activation,
				next_weights: Array2<f32>,
				next_gradients: Array1<f32>,
				expected_gradients: Array1<f32>
			) {
				use self::Activation::{Identity};

				let self_n_outputs = self_outputs.dim();
				let self_n_inputs  = 1;
				let self_shape     = (self_n_outputs, self_n_inputs);

				let (next_n_outputs, next_n_inputs) = next_weights.dim();
				let next_shape = next_weights.dim();

				let next_outputs = Array1::from_iter(iter::repeat(0.0)
					.take(next_n_outputs).chain(iter::once(1.0)));

				let self_zero_weights = Array1::zeros(self_n_inputs * self_n_outputs).into_shape(self_shape).unwrap();
				let next_zero_weights = Array1::zeros(next_n_inputs * next_n_outputs).into_shape(next_shape).unwrap();

				let mut self_layer = FullyConnectedLayer{
					weights      : self_zero_weights.clone(),
					delta_weights: self_zero_weights.clone(),
					outputs      : self_outputs,
					gradients    : Array1::zeros(self_n_outputs),
					activation
				};

				let next_layer = FullyConnectedLayer{
					weights      : next_weights,
					delta_weights: next_zero_weights.clone(),
					outputs      : next_outputs,
					gradients    : next_gradients,
					activation
				};

				self_layer.propagate_gradients(&next_layer);
				let result_gradients = self_layer.gradients.clone();

				assert!(result_gradients.all_close(&expected_gradients, 1e-6));
			}

			fn assert_config_with_expected(
				self_outputs  : Array1<f32>,
				next_gradients: Array1<f32>,
				next_weights  : Array2<f32>,
				expected_gradients_without_act: Array1<f32>
			) {
				use self::Activation::*;
				let activations = [Identity, Tanh, Logistic, SoftPlus, ReLU, Gaussian];
				for act in &activations {
					let mut expected_gradients = expected_gradients_without_act.clone();
					multizip((&mut expected_gradients, &self_outputs))
						.foreach(|(g, o)| *g *= act.derived(*o));
					assert_raw_config_with_expected(
						self_outputs.clone(), *act,
						next_weights.clone(),
						next_gradients.clone(),
						expected_gradients
					);
				}
			}

			assert_config_with_expected(
				Array1::from_vec(vec![ 0.25,  0.5,  0.75, 1.0]),
				Array1::from_vec(vec![10.00, 20.0, 30.00, 0.0]),
				Array1::linspace(1.0, 12.0, 12).into_shape((3, 4)).unwrap(),
				Array1::from_vec(vec![
					1.0 * 10.0 + 5.0 * 20.0 +  9.0 * 30.0, // self.gradient_1
					2.0 * 10.0 + 6.0 * 20.0 + 10.0 * 30.0, // self.gradient_2
					3.0 * 10.0 + 7.0 * 20.0 + 11.0 * 30.0, // self.gradient_3
					4.0 * 10.0 + 8.0 * 20.0 + 12.0 * 30.0  // self.gradient_4
				])
			);
		}

		#[test]
		fn update_weights() {
			fn assert_raw_config_with_expected(
				self_gradients    : Array1<f32>,
				self_weights      : Array2<f32>,
				self_delta_weights: Array2<f32>,
				prev_outputs      : Array1<f32>,
				learn_rate        : LearnRate,
				learn_momentum    : LearnMomentum,
				expected_weights  : Array2<f32>,
				expected_deltas   : Array2<f32>
			) {
				let (self_n_outputs, self_n_inputs) = self_weights.dim();
				let self_outputs = Array1::from_iter(iter::repeat(0.0)
					.take(self_n_outputs).chain(iter::once(1.0)));
				let mut self_layer = FullyConnectedLayer{
					weights      : self_weights,
					delta_weights: self_delta_weights,
					gradients    : self_gradients,
					outputs      : self_outputs,
					activation   : Activation::Identity
				};
				self_layer.update_weights(prev_outputs.view(), learn_rate, learn_momentum);

				let result_weights = self_layer.weights.clone();
				let result_deltas  = self_layer.delta_weights.clone();

				// println!("=========================");
				// println!("expected_weights = \n{:?}\n", expected_weights);
				// println!("result_weights   = \n{:?}\n\n", result_weights);

				// println!("expected_deltas = \n{:?}\n", expected_deltas);
				// println!("result_deltas   = \n{:?}\n\n", result_deltas);
				// println!("=========================");

				assert!(result_weights.all_close(&expected_weights, 1e-4));
				assert!(result_deltas.all_close(&expected_deltas, 1e-4));
			}

			fn assert_raw_config(
				self_gradients    : Array1<f32>,
				self_weights      : Array2<f32>,
				self_delta_weights: Array2<f32>,
				prev_outputs      : Array1<f32>,
				learn_rate        : LearnRate,
				learn_momentum    : LearnMomentum
			) {
				let LearnRate(lr)     = learn_rate;
				let LearnMomentum(lm) = learn_momentum;
				let mut expected_deltas = self_delta_weights.clone();
				multizip((expected_deltas.genrows_mut(), &self_gradients)).foreach(|(mut s_dw_rows, s_g)| {
					multizip((&mut s_dw_rows, &prev_outputs)).foreach(|(dw, p_o)| {
						*dw = lr * s_g * (*p_o) + lm * (*dw);
					})
				});
				let mut expected_weights = self_weights.clone();
				expected_weights += &expected_deltas;
				assert_raw_config_with_expected(
					self_gradients.clone(),
					self_weights.clone(),
					self_delta_weights.clone(),
					prev_outputs.clone(),
					learn_rate,
					learn_momentum,
					expected_weights,
					expected_deltas
				)
			}

			fn assert_config(
				self_gradients    : Array1<f32>,
				self_weights      : Array2<f32>,
				self_delta_weights: Array2<f32>,
				prev_outputs      : Array1<f32>,
			) {
				let learn_rates = [
					LearnRate(0.0),
					LearnRate(0.1),
					LearnRate(0.3),
					LearnRate(0.5),
					LearnRate(0.75),
					LearnRate(1.0)
				];
				let learn_momentums = [
					LearnMomentum(0.0),
					LearnMomentum(0.1),
					LearnMomentum(0.25),
					LearnMomentum(0.5),
					LearnMomentum(0.75),
					LearnMomentum(1.0)
				];
				use itertools::Itertools;
				for (lr, lm) in learn_rates.iter().cartesian_product(&learn_momentums) {
					assert_raw_config(
						self_gradients.clone(),
						self_weights.clone(),
						self_delta_weights.clone(),
						prev_outputs.clone(),
						*lr, *lm
					);
				}
			}

			assert_config(
				Array1::from_vec(vec![7.0, 11.0, 13.0, 17.0]),
				Array1::linspace( 1.0,  12.0, 12).into_shape((3, 4)).unwrap(),
				Array1::linspace(10.0, 120.0, 12).into_shape((3, 4)).unwrap(),
				Array1::from_vec(vec![11.0, 22.0, 33.0, 1.0])
			);
		}

		#[test]
		fn update_weights_old() {
			use self::Activation::{Identity};
			let lr = LearnRate(0.5);
			let lm = LearnMomentum(1.0);
			let outputs = Array1::from_iter(iter::repeat(0.0).take(3).chain(iter::once(1.0)));
			let mut layer = FullyConnectedLayer{
				weights      : Array1::linspace(1.0, 12.0, 12).into_shape((3, 4)).unwrap(),
				delta_weights: Array::zeros((3, 4)),
				outputs      : Array1::from_iter(iter::repeat(0.0).take(3).chain(iter::once(1.0))),
				gradients    : Array1::linspace(10.0, 40.0, 4),
				activation   : Identity
			};
			let result_outputs = layer.update_weights(outputs.view(), lr, lm).to_owned();
			let target_outputs = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
			let result_weights = layer.weights.clone();
			let target_weights = Array::from_vec(vec![
				1.0,  2.0,  3.0,  9.0,
				5.0,  6.0,  7.0, 18.0,
				9.0, 10.0, 11.0, 27.0]).into_shape((3, 4)).unwrap();
			assert_eq!(result_outputs, target_outputs);
			assert_eq!(result_weights, target_weights);
		}
	}
}
