use layer::{ActivationLayer, FullyConnectedLayer};
use layer::error_signal_buffer::ErrorSignalBuffer;
use layer::signal_buffer::SignalBuffer;
use layer::traits::{
	ProcessInputSignal,
	CalculateOutputErrorSignal,
	PropagateErrorSignal,
	ApplyErrorSignalCorrection,
	HasOutputSignal,
	HasErrorSignal,
};
use utils::{LearnRate, LearnMomentum};

#[derive(Debug, Clone, PartialEq)]
pub enum Layer {
	Activation(ActivationLayer),
	FullyConnected(FullyConnectedLayer)
}

impl From<ActivationLayer> for Layer {
	fn from(act_layer: ActivationLayer) -> Self {
		Layer::Activation(act_layer)
	}
}

impl From<FullyConnectedLayer> for Layer {
	fn from(fc_layer: FullyConnectedLayer) -> Self {
		Layer::FullyConnected(fc_layer)
	}
}

impl ProcessInputSignal for Layer {
	fn process_input_signal(&mut self, input_signal: &SignalBuffer) {
		use self::Layer::*;
		match *self {
			Activation(ref mut layer) => layer.process_input_signal(input_signal),
			FullyConnected(ref mut layer) => layer.process_input_signal(input_signal)
		}
	}
}

impl CalculateOutputErrorSignal for Layer {
	fn calculate_output_error_signal(&mut self, target_signals: &SignalBuffer) {
		use self::Layer::*;
		match *self {
			Activation(ref mut layer) => layer.calculate_output_error_signal(target_signals),
			FullyConnected(ref mut layer) => layer.calculate_output_error_signal(target_signals)
		}
	}
}

impl PropagateErrorSignal for Layer {
	fn propagate_error_signal<P>(&mut self, propagated: &mut P)
		where P: HasErrorSignal
	{
		use self::Layer::*;
		match *self {
			Activation(ref mut layer) => layer.propagate_error_signal(propagated),
			FullyConnected(ref mut layer) => layer.propagate_error_signal(propagated)
		}
	}
}

impl ApplyErrorSignalCorrection for Layer {
	fn apply_error_signal_correction(&mut self, signal: &SignalBuffer, lr: LearnRate, lm: LearnMomentum) {
		use self::Layer::*;
		match *self {
			Activation(ref mut layer) => layer.apply_error_signal_correction(signal, lr, lm),
			FullyConnected(ref mut layer) => layer.apply_error_signal_correction(signal, lr, lm)
		}
	}
}

impl HasOutputSignal for Layer {
	fn output_signal(&self) -> &SignalBuffer {
		use self::Layer::*;
		match *self {
			Activation(ref layer) => layer.output_signal(),
			FullyConnected(ref layer) => layer.output_signal()
		}
	}

	fn output_signal_mut(&mut self) -> &mut SignalBuffer {
		use self::Layer::*;
		match *self {
			Activation(ref mut layer) => layer.output_signal_mut(),
			FullyConnected(ref mut layer) => layer.output_signal_mut()
		}
	}
}

impl HasErrorSignal for Layer {
	fn error_signal(&self) -> &ErrorSignalBuffer {
		use self::Layer::*;
		match *self {
			Activation(ref layer) => layer.error_signal(),
			FullyConnected(ref layer) => layer.error_signal()
		}
	}

	fn error_signal_mut(&mut self) -> &mut ErrorSignalBuffer {
		use self::Layer::*;
		match *self {
			Activation(ref mut layer) => layer.error_signal_mut(),
			FullyConnected(ref mut layer) => layer.error_signal_mut()
		}
	}
}
