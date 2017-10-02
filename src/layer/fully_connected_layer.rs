use layer::utils::prelude::*;
use layer::traits::prelude::*;
use errors::{Result};
use utils::{LearnRate, LearnMomentum};
use topology_v4;

#[derive(Debug, Clone, PartialEq)]
pub struct FullyConnectedLayer {
	weights     : WeightsMatrix,
	deltas      : DeltaWeightsMatrix,
	outputs     : BiasedSignalBuffer,
	error_signal: BiasedErrorSignalBuffer
}

impl FullyConnectedLayer {
	/// Creates a new `FullyConnectedLayer` with the given weights.
	/// 
	/// Note: The given weights imply the resulting layer's input and output signal lengths.
	/// 
	/// # Errors
	/// 
	/// If the implied input and output signals have a length of zero.
	pub(crate) fn with_weights(weights: WeightsMatrix) -> Result<FullyConnectedLayer> {
		let (inputs, outputs) = (weights.inputs(), weights.outputs());
		Ok(FullyConnectedLayer{
			weights,
			deltas      : DeltaWeightsMatrix::zeros(inputs, outputs)?,
			outputs     : BiasedSignalBuffer::zeros_with_bias(outputs)?,
			error_signal: BiasedErrorSignalBuffer::zeros_with_bias(outputs)?,
		})
	}

	/// Creates a new `FullyConnectedLayer` with default settings with the given
	/// lengths for the input and output signals.
	/// 
	/// # Errors
	/// 
	/// If input or output lengths are zero.
	pub fn random(inputs: usize, outputs: usize) -> Result<FullyConnectedLayer> {
		Ok(FullyConnectedLayer::with_weights(
			WeightsMatrix::random(inputs, outputs)?)?)
	}

	/// Creates a new `FullyConnectedLayer` from the given topology based abstract fully connected layer.
	/// 
	/// # Errors
	/// 
	/// If the given topology based abstract fully connected layer is invalid.
	pub fn from_top_layer(top_layer: topology_v4::FullyConnectedLayer) -> Result<FullyConnectedLayer> {
		use topology_v4::Layer;
		FullyConnectedLayer::random(
			top_layer.input_len().into_usize(), top_layer.input_len().into_usize()
		)
	}
}

impl From<topology_v4::FullyConnectedLayer> for FullyConnectedLayer {
	fn from(top_layer: topology_v4::FullyConnectedLayer) -> FullyConnectedLayer {
		FullyConnectedLayer::from_top_layer(top_layer)
			.expect("Expected a well-formed abstracted topology FullyConnectedLayer.")
	}
}

impl ProcessInputSignal for FullyConnectedLayer {
	fn process_input_signal(&mut self, input_signal: BiasedSignalView) {
		if self.output_signal().len() != input_signal.len() {
			panic!("Error: unmatching signals to layer size") // TODO: Replace this with error.
		}
		use ndarray::linalg::general_mat_vec_mul;
		general_mat_vec_mul(1.0, &self.weights.view(), &input_signal.data(), 1.0, &mut self.outputs.unbias_mut().data_mut())
	}
}

impl CalculateOutputErrorSignal for FullyConnectedLayer {
	fn calculate_output_error_signal(&mut self, target_signal: UnbiasedSignalView) {
		if self.output_signal().len() != target_signal.len() {
			panic!("Error: unmatching signals to layer size") // TODO: Replace this with error.
		}
		use ndarray::Zip;
		Zip::from(&mut self.error_signal.unbias_mut().data_mut())
			.and(&self.outputs.unbias().data())
			.and(&target_signal.data())
			.apply(|e, &o, &t| {
				*e = t - o
			}
		)
	}
}

impl PropagateErrorSignal for FullyConnectedLayer {
	fn propagate_error_signal<P>(&mut self, propagated: &mut P)
		where P: HasErrorSignal
	{
		use ndarray::Zip;
		use itertools::*;
		multizip((self.weights.genrows(), &self.error_signal.unbias())).foreach(|(s_wrow, &s_e)| {
			Zip::from(&mut propagated.error_signal_mut().unbias_mut().data_mut())
				.and(&s_wrow.view())
				.apply(|p_e, &s_w| {
					*p_e += s_w * s_e;
				})
		})
	}
}

impl ApplyErrorSignalCorrection for FullyConnectedLayer {
	fn apply_error_signal_correction(&mut self, input_signal: BiasedSignalView, lr: LearnRate, lm: LearnMomentum) {
		// use std::ops::AddAssign;
		use ndarray::Zip;
		use itertools::*;
		multizip((self.deltas.genrows_mut(), &self.error_signal.unbias())).foreach(|(mut s_drow, &s_e)| {
			Zip::from(&mut s_drow.view_mut()).and(input_signal.unbias().data()).apply(|s_dw, &p_i| {
				*s_dw = (1.0 - lm.0) * lr.0 * p_i * s_e + lm.0 * *s_dw;
			});
		});
		self.weights.apply_delta_weights(&self.deltas);
		self.error_signal.reset_to_zeros();
	}
}

impl HasOutputSignal for FullyConnectedLayer {
	fn output_signal(&self) -> BiasedSignalView {
		self.outputs.view()
	}

	fn output_signal_mut(&mut self) -> BiasedSignalViewMut {
		self.outputs.view_mut()
	}
}

impl HasErrorSignal for FullyConnectedLayer {
	fn error_signal(&self) -> BiasedErrorSignalView {
		self.error_signal.view()
	}

	fn error_signal_mut(&mut self) -> BiasedErrorSignalViewMut {
		self.error_signal.view_mut()
	}
}

impl SizedLayer for FullyConnectedLayer {
	fn inputs(&self) -> usize {
		self.weights.inputs()
	}

	fn outputs(&self) -> usize {
		self.weights.outputs()
	}
}
