use layer::signal_buffer::SignalBuffer;
use layer::gradient_buffer::GradientBuffer;
use layer::traits::{
	ProcessSignal,
	CalculateErrorGradients,
	HasOutputSignal,
	HasGradientBuffer,
	PropagateGradients
};
use errors::{Result};
use activation::Activation;

/// Activation layers simply apply their activation function onto incoming signals.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationLayer {
	outputs  : SignalBuffer,
	gradients: GradientBuffer,
	act      : Activation
}

impl ActivationLayer {
	/// Creates a new `ActivationLayer` with the given number of inputs
	/// and outputs (not respecting the bias input and output) and given
	/// an activation function.
	pub fn with_activation(len: usize, act: Activation) -> Result<Self> {
		Ok(ActivationLayer{
			outputs  : SignalBuffer::zeros(len)?,
			gradients: GradientBuffer::zeros(len)?,
			act
		})
	}

	pub fn len(&self) -> usize {
		self.outputs.view().dim()
	}
}

impl ProcessSignal for ActivationLayer {
	fn process_signal(&mut self, signal: &SignalBuffer) {
		if self.len() != signal.len() {
			panic!("Error: unmatching signals to layer size") // TODO: Replace this with error. (Needs to change trait.) 
		}
		let act = self.act; // Required since borrow-checker doesn't allow
		                    // using `self.act` within method-call context.
		self.outputs.view_mut().mapv_inplace(|o| act.base(o))
	}
}

impl CalculateErrorGradients for ActivationLayer {
	fn calculate_gradient_descent(&mut self, target_signals: &SignalBuffer) {
		use ndarray::Zip;

		debug_assert_eq!(self.output_signal().biased_len(), target_signals.len()); // No calculation for bias neurons.

		let act = self.act; // Required because of non-lexical borrows.
		Zip::from(&mut self.gradients.view_mut())
			.and(&self.outputs.view())
			.and(&target_signals.view())
			.apply(|g, &t, &o| {
				*g = (t - o) * act.derived(o)
			}
		);
	}
}

impl PropagateGradients for ActivationLayer {
	fn propagate_gradients<P>(&self, propagated: &mut P)
		where P: HasGradientBuffer
	{
		unimplemented!()
	}
}

// fn propagate_gradients(
// 	&mut self,
//     prev: &FullyConnectedLayer
// )
//     -> &Self
// {
// 	debug_assert_eq!(prev.weights.rows() + 1, prev.count_gradients());
// 	debug_assert_eq!(prev.weights.cols()    , self.count_gradients());

// 	multizip((prev.weights.genrows(), prev.gradients.iter()))
// 		.foreach(|(prev_weights_row, prev_gradient)| {
// 			multizip((self.gradients.iter_mut(), prev_weights_row.iter()))
// 				.foreach(|(gradient, weight)| *gradient += weight * prev_gradient)
// 		});

// 	self.apply_activation_to_gradients();
// 	self
// }


impl HasOutputSignal for ActivationLayer {
	fn output_signal(&self) -> &SignalBuffer {
		&self.outputs
	}

	fn output_signal_mut(&mut self) -> &mut SignalBuffer {
		&mut self.outputs
	}
}

impl HasGradientBuffer for ActivationLayer {
	fn gradients(&self) -> &GradientBuffer {
		&self.gradients
	}

	fn gradients_mut(&mut self) -> &mut GradientBuffer {
		&mut self.gradients
	}
}
