use layer::signal_buffer::SignalBuffer;
use layer::error_signal_buffer::ErrorSignalBuffer;
use layer::traits::{
	HasOutputSignal,
	HasErrorSignal,
	ProcessInputSignal,
	CalculateOutputErrorSignal,
	PropagateErrorSignal,
	ApplyErrorSignalCorrection,
};
use utils::{LearnRate, LearnMomentum};
use errors::{Result};
use activation::Activation;

/// Activation layers simply apply their activation function onto incoming signals.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationLayer {
	/// These are only required for a correct implementation of the back propagation
	/// algorithm where the derived activation function is applied to the net value
	/// instead of the output signal, so with the current design we have to store both.
	/// 
	/// Maybe this situation could be improved in the future by using references
	/// or shared ownership of the input with the previous layer ... but then again
	/// we had to know our previous layer.
	inputs   : SignalBuffer,
	/// The outputs of this activation layer.
	/// 
	/// This is basically equivalent to the input transformed with the activation
	/// of this layer.
	outputs  : SignalBuffer,
	/// The buffer for the back propagated error signal.
	error_signal: ErrorSignalBuffer,
	/// The activation function of this `ActivationLayer`.
	act      : Activation
}

impl ActivationLayer {
	/// Creates a new `ActivationLayer` with the given number of inputs
	/// and outputs (not respecting the bias input and output) and given
	/// an activation function.
	pub fn with_activation(len: usize, act: Activation) -> Result<Self> {
		Ok(ActivationLayer{
			inputs      : SignalBuffer::zeros(len)?,
			outputs     : SignalBuffer::zeros(len)?,
			error_signal: ErrorSignalBuffer::zeros(len)?,
			act
		})
	}

	/// Returns the length of this `ActivationLayer`.
	#[inline]
	pub fn len(&self) -> usize {
		self.outputs.view().dim()
	}
}

impl ProcessInputSignal for ActivationLayer {
	fn process_input_signal(&mut self, input_signal: &SignalBuffer) {
		if self.len() != input_signal.len() {
			panic!("Error: unmatching signals to layer size") // TODO: Replace this with error.
		}
		let act = self.act; // Required since borrow-checker doesn't allow
		                    // using `self.act` within method-call context.
		use ndarray::Zip;
		Zip::from(&mut self.inputs.view_mut())
			.and(&mut self.outputs.view_mut())
			.and(&input_signal.view())
			.apply(|s_i, s_o, &p_i| {
				*s_i = p_i; // Note: Required for correct back propagation!
				*s_o = act.base(p_i);
			});
	}
}

impl CalculateOutputErrorSignal for ActivationLayer {
	fn calculate_output_error_signal(&mut self, target_signals: &SignalBuffer) {
		if self.len() != target_signals.len() {
			// Note: Target signals do not respect bias values.
			//       We could model this in a way that `target_signals` are simply one element shorter.
			//       Or they have also `1.0` as their last element which eliminates 
			//       the resulting error signal's last element to `0.0`.
			panic!("Error: unmatching length of output signal and target signal") // TODO: Replace this with error.
		}
		use ndarray::Zip;
		Zip::from(&mut self.error_signal.view_mut())
			.and(&self.outputs.view())
			.and(&target_signals.view())
			.apply(|e, &o, &t| {
				*e = t - o
			}
		);
	}
}

impl PropagateErrorSignal for ActivationLayer {
	fn propagate_error_signal<P>(&mut self, propagated: &mut P)
		where P: HasErrorSignal
	{
		if self.len() != propagated.error_signal().len() {
			panic!("Error: unmatching signals to layer size") // TODO: Replace this with error.
		}
		use ndarray::Zip;
		let act = self.act;
		// Calculate the gradients and multiply them to the current
		// error signal and propagate the result to the next layer.
		Zip::from(&mut propagated.error_signal_mut().biased_view_mut())
			.and(&self.inputs.biased_view())
			.and(&self.error_signal.biased_view())
			.apply(|o_e, &s_n, &s_e| {
				*o_e += s_e * act.derived(s_n)
			});
		// We need to set the error signals for `ActivationLayer`s to zero
		// since it would corrupt error signal propagation without applying
		// error signal correction afterwards which is a known optimization
		// and thus used often.
		// Note: This applies only to `ActivationLayer`s!
		self.error_signal_mut().reset_to_zeros()
	}
}

impl ApplyErrorSignalCorrection for ActivationLayer {
	fn apply_error_signal_correction(&mut self, _signal: &SignalBuffer, _lr: LearnRate, _lm: LearnMomentum) {
		// Nothing to do here since there are no weights that could be updated!
	}
}

impl HasOutputSignal for ActivationLayer {
	fn output_signal(&self) -> &SignalBuffer {
		&self.outputs
	}

	fn output_signal_mut(&mut self) -> &mut SignalBuffer {
		&mut self.outputs
	}
}

impl HasErrorSignal for ActivationLayer {
	fn error_signal(&self) -> &ErrorSignalBuffer {
		&self.error_signal
	}

	fn error_signal_mut(&mut self) -> &mut ErrorSignalBuffer {
		&mut self.error_signal
	}
}
