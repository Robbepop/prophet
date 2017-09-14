use layer::signal_buffer::SignalBuffer;
use layer::gradient_buffer::GradientBuffer;
use layer::weights_matrix::WeightsMatrix;
use errors::{Result};

#[derive(Debug, Clone, PartialEq)]
struct FullyConnectedLayer {
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
			deltas: WeightsMatrix::zeros(inputs, outputs)?,
			outputs: SignalBuffer::zeros(outputs)?,
			gradients: GradientBuffer::zeros(outputs)?,
		})
	}

	pub fn random(inputs: usize, outputs: usize) -> Result<Self> {
		Ok(FullyConnectedLayer{
			weights: WeightsMatrix::random(inputs, outputs)?,
			deltas: WeightsMatrix::zeros(inputs, outputs)?,
			outputs: SignalBuffer::zeros(outputs)?,
			gradients: GradientBuffer::zeros(outputs)?,
		})
	}
}
