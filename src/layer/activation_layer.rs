use layer::signal_buffer::SignalBuffer;
use layer::gradient_buffer::GradientBuffer;
use errors::{Result};
use activation::Activation;

#[derive(Debug, Clone, PartialEq)]
struct ActivationLayer {
	outputs  : SignalBuffer,
	gradients: GradientBuffer,
	act      : Activation
}

impl ActivationLayer {
	pub fn with_activation(len: usize, act: Activation) -> Result<Self> {
		Ok(ActivationLayer{
			outputs  : SignalBuffer::zeros(len)?,
			gradients: GradientBuffer::zeros(len)?,
			act
		})
	}
}
