use layer::{ActivationLayer, FullyConnectedLayer, ContainerLayer};
use layer::error_signal_buffer::ErrorSignalBuffer;
use layer::signal_buffer::SignalBuffer;
use layer::traits::{
	SizedLayer,
	ProcessInputSignal,
	CalculateOutputErrorSignal,
	PropagateErrorSignal,
	ApplyErrorSignalCorrection,
	HasOutputSignal,
	HasErrorSignal,
};
use utils::{LearnRate, LearnMomentum};

use self::Layer::{Activation, FullyConnected, Container};

#[derive(Debug, Clone, PartialEq)]
pub enum Layer {
	Activation(ActivationLayer),
	FullyConnected(FullyConnectedLayer),
	Container(ContainerLayer)
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

impl From<ContainerLayer> for Layer {
	fn from(c_layer: ContainerLayer) -> Self {
		Layer::Container(c_layer)
	}
}

impl ProcessInputSignal for Layer {
	fn process_input_signal(&mut self, input_signal: &SignalBuffer) {
		match *self {
			Activation(ref mut layer) => layer.process_input_signal(input_signal),
			FullyConnected(ref mut layer) => layer.process_input_signal(input_signal),
			Container(ref mut layer) => layer.process_input_signal(input_signal)
		}
	}
}

impl CalculateOutputErrorSignal for Layer {
	fn calculate_output_error_signal(&mut self, target_signals: &SignalBuffer) {
		match *self {
			Activation(ref mut layer) => layer.calculate_output_error_signal(target_signals),
			FullyConnected(ref mut layer) => layer.calculate_output_error_signal(target_signals),
			Container(ref mut layer) => layer.calculate_output_error_signal(target_signals)
		}
	}
}

impl PropagateErrorSignal for Layer {
	fn propagate_error_signal<P>(&mut self, propagated: &mut P)
		where P: HasErrorSignal
	{
		match *self {
			Activation(ref mut layer) => layer.propagate_error_signal(propagated),
			FullyConnected(ref mut layer) => layer.propagate_error_signal(propagated),
			Container(ref mut layer) => layer.propagate_error_signal(propagated)
		}
	}
}

impl ApplyErrorSignalCorrection for Layer {
	fn apply_error_signal_correction(&mut self, signal: &SignalBuffer, lr: LearnRate, lm: LearnMomentum) {
		match *self {
			Activation(ref mut layer) => layer.apply_error_signal_correction(signal, lr, lm),
			FullyConnected(ref mut layer) => layer.apply_error_signal_correction(signal, lr, lm),
			Container(ref mut layer) => layer.apply_error_signal_correction(signal, lr, lm)
		}
	}
}

impl HasOutputSignal for Layer {
	fn output_signal(&self) -> &SignalBuffer {
		match *self {
			Activation(ref layer) => layer.output_signal(),
			FullyConnected(ref layer) => layer.output_signal(),
			Container(ref layer) => layer.output_signal()
		}
	}

	fn output_signal_mut(&mut self) -> &mut SignalBuffer {
		match *self {
			Activation(ref mut layer) => layer.output_signal_mut(),
			FullyConnected(ref mut layer) => layer.output_signal_mut(),
			Container(ref mut layer) => layer.output_signal_mut()
		}
	}
}

impl HasErrorSignal for Layer {
	fn error_signal(&self) -> &ErrorSignalBuffer {
		match *self {
			Activation(ref layer) => layer.error_signal(),
			FullyConnected(ref layer) => layer.error_signal(),
			Container(ref layer) => layer.error_signal()
		}
	}

	fn error_signal_mut(&mut self) -> &mut ErrorSignalBuffer {
		match *self {
			Activation(ref mut layer) => layer.error_signal_mut(),
			FullyConnected(ref mut layer) => layer.error_signal_mut(),
			Container(ref mut layer) => layer.error_signal_mut()
		}
	}
}

impl SizedLayer for Layer {
	fn inputs(&self) -> usize {
		match *self {
			Activation(ref layer) => layer.inputs(),
			FullyConnected(ref layer) => layer.inputs(),
			Container(ref layer) => layer.inputs()
		}
	}

	fn outputs(&self) -> usize {
		match *self {
			Activation(ref layer) => layer.outputs(),
			FullyConnected(ref layer) => layer.outputs(),
			Container(ref layer) => layer.outputs()
		}
	}
}
