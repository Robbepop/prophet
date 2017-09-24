use layer::utils::{
	SignalBuffer,
	ErrorSignalBuffer,
	WeightsMatrix,
	DeltaWeightsMatrix
};
use layer::traits::{
	SizedLayer,
	ProcessInputSignal,
	CalculateOutputErrorSignal,
	PropagateErrorSignal,
	ApplyErrorSignalCorrection,
	HasOutputSignal,
	HasErrorSignal,
};
use errors::{Result};
use utils::{LearnRate, LearnMomentum};

#[derive(Debug, Clone, PartialEq)]
pub struct FullyConnectedLayer {
	weights     : WeightsMatrix,
	deltas      : DeltaWeightsMatrix,
	outputs     : SignalBuffer,
	error_signal: ErrorSignalBuffer
}

impl FullyConnectedLayer {
	pub(crate) fn with_weights(weights: WeightsMatrix) -> Result<Self> {
		let (inputs, outputs) = (weights.inputs(), weights.outputs());
		Ok(FullyConnectedLayer{
			weights,
			deltas      : DeltaWeightsMatrix::zeros(inputs, outputs)?,
			outputs     : SignalBuffer::zeros(outputs)?,
			error_signal: ErrorSignalBuffer::zeros(outputs)?,
		})
	}

	pub fn random(inputs: usize, outputs: usize) -> Result<Self> {
		Ok(FullyConnectedLayer::with_weights(
			WeightsMatrix::random(inputs, outputs)?)?)
	}
}

impl ProcessInputSignal for FullyConnectedLayer {
	fn process_input_signal(&mut self, input_signal: &SignalBuffer) {
		if self.output_signal().len() != input_signal.len() {
			panic!("Error: unmatching signals to layer size") // TODO: Replace this with error.
		}
		use ndarray::linalg::general_mat_vec_mul;
		general_mat_vec_mul(1.0, &self.weights.view(), &input_signal.biased_view(), 1.0, &mut self.outputs.view_mut())
	}
}

impl CalculateOutputErrorSignal for FullyConnectedLayer {
	fn calculate_output_error_signal(&mut self, target_signal: &SignalBuffer) {
		if self.output_signal().len() != target_signal.len() {
			panic!("Error: unmatching signals to layer size") // TODO: Replace this with error.
		}
		use ndarray::Zip;
		Zip::from(&mut self.error_signal.view_mut())
			.and(&self.outputs.view())
			.and(&target_signal.view())
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
		multizip((self.weights.genrows(), self.error_signal.view())).foreach(|(s_wrow, &s_e)| {
			Zip::from(&mut propagated.error_signal_mut().view_mut()).and(&s_wrow.view()).apply(|p_e, &s_w| {
				*p_e += s_w * s_e;
			})
		})
	}
}

impl ApplyErrorSignalCorrection for FullyConnectedLayer {
	fn apply_error_signal_correction(&mut self, input_signal: &SignalBuffer, lr: LearnRate, lm: LearnMomentum) {
		// use std::ops::AddAssign;
		use ndarray::Zip;
		use itertools::*;
		multizip((self.deltas.genrows_mut(), self.error_signal.view())).foreach(|(mut s_drow, &s_e)| {
			Zip::from(&mut s_drow.view_mut()).and(&input_signal.view()).apply(|s_dw, &p_i| {
				*s_dw = (1.0 - lm.0) * lr.0 * p_i * s_e + lm.0 * *s_dw;
			});
		});
		self.weights.apply_delta_weights(&self.deltas);
		self.error_signal.reset_to_zeros();
	}
}

impl HasOutputSignal for FullyConnectedLayer {
	fn output_signal(&self) -> &SignalBuffer {
		&self.outputs
	}

	fn output_signal_mut(&mut self) -> &mut SignalBuffer {
		&mut self.outputs
	}
}

impl HasErrorSignal for FullyConnectedLayer {
	fn error_signal(&self) -> &ErrorSignalBuffer {
		&self.error_signal
	}

	fn error_signal_mut(&mut self) -> &mut ErrorSignalBuffer {
		&mut self.error_signal
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
