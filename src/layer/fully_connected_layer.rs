use layer::signal_buffer::SignalBuffer;
use layer::gradient_buffer::GradientBuffer;
use layer::weights_matrix::WeightsMatrix;
use layer::traits::{
	ProcessSignal,
	CalculateErrorGradients,
	HasOutputSignal,
	HasGradientBuffer
};
use errors::{Result};

#[derive(Debug, Clone, PartialEq)]
pub struct FullyConnectedLayer {
	weights  : WeightsMatrix,
	deltas   : WeightsMatrix,
	outputs  : SignalBuffer,
	gradients: GradientBuffer
}

impl FullyConnectedLayer {
	pub(crate) fn with_weights(weights: WeightsMatrix) -> Result<Self> {
		let (inputs, outputs) = (weights.inputs(), weights.outputs());
		Ok(FullyConnectedLayer{
			weights,
			deltas   : WeightsMatrix::zeros(inputs, outputs)?,
			outputs  : SignalBuffer::zeros(outputs)?,
			gradients: GradientBuffer::zeros(outputs)?,
		})
	}

	pub fn random(inputs: usize, outputs: usize) -> Result<Self> {
		Ok(FullyConnectedLayer::with_weights(
			WeightsMatrix::random(inputs, outputs)?)?)
	}
}

impl ProcessSignal for FullyConnectedLayer {
	fn process_signal(&mut self, signal: &SignalBuffer) {
		if self.output_signal().len() != signal.len() {
			panic!("Error: unmatching signals to layer size") // TODO: Replace this with error. (Needs to change trait.) 
		}
		use ndarray::linalg::general_mat_vec_mul;
		general_mat_vec_mul(1.0, &self.weights.view(), &signal.biased_view(), 1.0, &mut self.outputs.view_mut())
	}
}

impl CalculateErrorGradients for FullyConnectedLayer {
	fn calculate_gradient_descent(&mut self, target_signal: &SignalBuffer) {
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

impl HasGradientBuffer for FullyConnectedLayer {
	fn gradients(&self) -> &GradientBuffer {
		&self.gradients
	}

	fn gradients_mut(&mut self) -> &mut GradientBuffer {
		&mut self.gradients
	}
}
