use layer::layer::Layer;
use layer::signal_buffer::SignalBuffer;
use layer::error_signal_buffer::ErrorSignalBuffer;
use layer::traits::{
	SizedLayer,
	HasOutputSignal,
	HasErrorSignal,
	ProcessInputSignal,
	CalculateOutputErrorSignal,
	PropagateErrorSignal,
	ApplyErrorSignalCorrection,
};
use errors::{Result};
use utils::{LearnRate, LearnMomentum};

#[derive(Debug, Clone, PartialEq)]
pub struct ContainerLayer {
	childs: Vec<Layer>
}

impl ContainerLayer {
	pub fn from_vec(layers: Vec<Layer>) -> Result<ContainerLayer> {
		if layers.len() == 0 {
			panic!("ContainerLayer requires to contain at least one child layer."); // TODO: Rewrite as error.
		}
		Ok(ContainerLayer{
			childs: layers
		})
	}

	#[inline]
	fn input_layer(&self) -> &Layer {
		self.childs.first().unwrap()
	}

	#[inline]
	fn input_layer_mut(&mut self) -> &mut Layer {
		self.childs.first_mut().unwrap()
	}

	#[inline]
	fn output_layer(&self) -> &Layer {
		self.childs.last().unwrap()
	}

	#[inline]
	fn output_layer_mut(&mut self) -> &mut Layer {
		self.childs.last_mut().unwrap()
	}
}

impl ProcessInputSignal for ContainerLayer {
	fn process_input_signal(&mut self, input_signal: &SignalBuffer) {
		unimplemented!()
	}
}

impl CalculateOutputErrorSignal for ContainerLayer {
	fn calculate_output_error_signal(&mut self, target_signals: &SignalBuffer) {
		self.output_layer_mut().calculate_output_error_signal(target_signals)
	}
}

impl PropagateErrorSignal for ContainerLayer {
	fn propagate_error_signal<P>(&mut self, propagated: &mut P)
		where P: HasErrorSignal
	{
		unimplemented!()
	}
}

impl ApplyErrorSignalCorrection for ContainerLayer {
	fn apply_error_signal_correction(&mut self, _signal: &SignalBuffer, _lr: LearnRate, _lm: LearnMomentum) {
		unimplemented!()
	}
}

impl HasOutputSignal for ContainerLayer {
	fn output_signal(&self) -> &SignalBuffer {
		self.output_layer().output_signal()
	}

	fn output_signal_mut(&mut self) -> &mut SignalBuffer {
		self.output_layer_mut().output_signal_mut()
	}
}

impl HasErrorSignal for ContainerLayer {
	fn error_signal(&self) -> &ErrorSignalBuffer {
		self.output_layer().error_signal()
	}

	fn error_signal_mut(&mut self) -> &mut ErrorSignalBuffer {
		self.output_layer_mut().error_signal_mut()
	}
}

impl SizedLayer for ContainerLayer {
	fn inputs(&self) -> usize {
		self.input_layer().inputs()
	}

	fn outputs(&self) -> usize {
		self.output_layer().outputs()
	}
}
