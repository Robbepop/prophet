use layer::signal_buffer::SignalBuffer;
use layer::error_signal_buffer::ErrorSignalBuffer;
use layer::weights_matrix::WeightsMatrix;
use layer::traits::{
	ProcessInputSignal,
	CalculateOutputErrorSignal,
	HasOutputSignal,
	HasErrorSignal,
};
use errors::{Result};

#[derive(Debug, Clone, PartialEq)]
pub struct FullyConnectedLayer {
	weights  : WeightsMatrix,
	deltas   : WeightsMatrix,
	outputs  : SignalBuffer,
	gradients: ErrorSignalBuffer
}

impl FullyConnectedLayer {
	pub(crate) fn with_weights(weights: WeightsMatrix) -> Result<Self> {
		let (inputs, outputs) = (weights.inputs(), weights.outputs());
		Ok(FullyConnectedLayer{
			weights,
			deltas   : WeightsMatrix::zeros(inputs, outputs)?,
			outputs  : SignalBuffer::zeros(outputs)?,
			gradients: ErrorSignalBuffer::zeros(outputs)?,
		})
	}

	pub fn random(inputs: usize, outputs: usize) -> Result<Self> {
		Ok(FullyConnectedLayer::with_weights(
			WeightsMatrix::random(inputs, outputs)?)?)
	}
}

impl ProcessInputSignal for FullyConnectedLayer {
	fn process_input_signal(&mut self, signal: &SignalBuffer) {
		if self.output_signal().len() != signal.len() {
			panic!("Error: unmatching signals to layer size") // TODO: Replace this with error. (Needs to change trait.) 
		}
		use ndarray::linalg::general_mat_vec_mul;
		general_mat_vec_mul(1.0, &self.weights.view(), &signal.biased_view(), 1.0, &mut self.outputs.view_mut())
	}
}

impl CalculateOutputErrorSignal for FullyConnectedLayer {
	fn calculate_output_error_signal(&mut self, target_signal: &SignalBuffer) {
		if self.output_signal().len() != target_signal.len() {
			panic!("Error: unmatching signals to layer size") // TODO: Replace this with error. (Needs to change trait.) 
		}
		use ndarray::Zip;
		Zip::from(&mut self.gradients.view_mut())
			.and(&self.outputs.view())
			.and(&target_signal.view())
			.apply(|g, &t, &o| {
				*g = t - o
			}
		);
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
		&self.gradients
	}

	fn error_signal_mut(&mut self) -> &mut ErrorSignalBuffer {
		&mut self.gradients
	}
}
