use layer::{ActivationLayer, FullyConnectedLayer, ContainerLayer};
use layer::utils::prelude::*;
use layer::traits::prelude::*;
use utils::{LearnRate, LearnMomentum};
use topology_v4;

#[derive(Debug, Clone, PartialEq)]
pub enum AnyLayer {
	Activation(ActivationLayer),
	FullyConnected(FullyConnectedLayer),
	Container(ContainerLayer)
}
use self::AnyLayer::{Activation, FullyConnected, Container};

impl From<ActivationLayer> for AnyLayer {
	fn from(act_layer: ActivationLayer) -> Self {
		AnyLayer::Activation(act_layer)
	}
}

impl From<FullyConnectedLayer> for AnyLayer {
	fn from(fc_layer: FullyConnectedLayer) -> Self {
		AnyLayer::FullyConnected(fc_layer)
	}
}

impl From<ContainerLayer> for AnyLayer {
	fn from(c_layer: ContainerLayer) -> Self {
		AnyLayer::Container(c_layer)
	}
}

impl From<topology_v4::AnyLayer> for AnyLayer {
	fn from(any_top_layer: topology_v4::AnyLayer) -> AnyLayer {
		use topology_v4::AnyLayer::*;
		match any_top_layer {
			Activation(layer) => ActivationLayer::from(layer).into(),
			FullyConnected(layer) => FullyConnectedLayer::from(layer).into()
		}
	}
}

impl ProcessInputSignal for AnyLayer {
	fn process_input_signal(&mut self, input_signal: BiasedSignalView) {
		match *self {
			Activation(ref mut layer) => layer.process_input_signal(input_signal),
			FullyConnected(ref mut layer) => layer.process_input_signal(input_signal),
			Container(ref mut layer) => layer.process_input_signal(input_signal)
		}
	}
}

impl CalculateOutputErrorSignal for AnyLayer {
	fn calculate_output_error_signal(&mut self, target_signals: UnbiasedSignalView) {
		match *self {
			Activation(ref mut layer) => layer.calculate_output_error_signal(target_signals),
			FullyConnected(ref mut layer) => layer.calculate_output_error_signal(target_signals),
			Container(ref mut layer) => layer.calculate_output_error_signal(target_signals)
		}
	}
}

impl PropagateErrorSignal for AnyLayer {
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

impl ApplyErrorSignalCorrection for AnyLayer {
	fn apply_error_signal_correction(&mut self, signal: BiasedSignalView, lr: LearnRate, lm: LearnMomentum) {
		match *self {
			Activation(ref mut layer) => layer.apply_error_signal_correction(signal, lr, lm),
			FullyConnected(ref mut layer) => layer.apply_error_signal_correction(signal, lr, lm),
			Container(ref mut layer) => layer.apply_error_signal_correction(signal, lr, lm)
		}
	}
}

impl HasOutputSignal for AnyLayer {
	fn output_signal(&self) -> BiasedSignalView {
		match *self {
			Activation(ref layer) => layer.output_signal(),
			FullyConnected(ref layer) => layer.output_signal(),
			Container(ref layer) => layer.output_signal()
		}
	}

	fn output_signal_mut(&mut self) -> BiasedSignalViewMut {
		match *self {
			Activation(ref mut layer) => layer.output_signal_mut(),
			FullyConnected(ref mut layer) => layer.output_signal_mut(),
			Container(ref mut layer) => layer.output_signal_mut()
		}
	}
}

impl HasErrorSignal for AnyLayer {
	fn error_signal(&self) -> BiasedErrorSignalView {
		match *self {
			Activation(ref layer) => layer.error_signal(),
			FullyConnected(ref layer) => layer.error_signal(),
			Container(ref layer) => layer.error_signal()
		}
	}

	fn error_signal_mut(&mut self) -> BiasedErrorSignalViewMut {
		match *self {
			Activation(ref mut layer) => layer.error_signal_mut(),
			FullyConnected(ref mut layer) => layer.error_signal_mut(),
			Container(ref mut layer) => layer.error_signal_mut()
		}
	}
}

impl SizedLayer for AnyLayer {
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