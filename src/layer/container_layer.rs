use layer::layer::Layer;
use layer::utils::{
	SignalBuffer,
	ErrorSignalBuffer
};
use layer::traits::prelude::*;
use errors::{Result};
use utils::{LearnRate, LearnMomentum};

/// `ContainerLayer` is itself a neuronal layer that contains other layers in sequential order.
/// 
/// It forwards signals and information flow to its child layers in the correct order.
/// With this layer kind it is possible to stack layer hierachies and modularize layer topologies.
#[derive(Debug, Clone, PartialEq)]
pub struct ContainerLayer {
	childs: Vec<Layer>
}

impl ContainerLayer {
	/// Creates a new `ContainerLayer` from the given vector of layers.
	/// 
	/// # Errors
	/// 
	/// This fails if the given vector is empty or when the input and output sizes of the
	/// given layers within the vector do not match.
	pub fn from_vec(layers: Vec<Layer>) -> Result<ContainerLayer> {
		if layers.len() == 0 {
			panic!("ContainerLayer requires to contain at least one child layer."); // TODO: Rewrite as error.
		}
		use itertools::Itertools;
		if layers.iter().tuple_windows().any(|(l, r)| l.outputs() != r.inputs()) {
			panic!("ContainerLayer requires all given layers to match their neighbours inputs and outputs.") // TODO: Rewrite as error.
		}
		Ok(ContainerLayer{
			childs: layers
		})
	}

	/// Returns a reference to the input child layer.
	#[inline]
	fn input_layer(&self) -> &Layer {
		self.childs.first().unwrap()
	}

	/// Returns a mutable reference to the input child layer.
	#[inline]
	fn input_layer_mut(&mut self) -> &mut Layer {
		self.childs.first_mut().unwrap()
	}

	/// Returns a reference to the output child layer.
	#[inline]
	fn output_layer(&self) -> &Layer {
		self.childs.last().unwrap()
	}

	/// Returns a mutable reference to the output child layer.
	#[inline]
	fn output_layer_mut(&mut self) -> &mut Layer {
		self.childs.last_mut().unwrap()
	}

	/// Propagates the error signal from the last internal child layer to the first.
	fn propagate_error_signal_internally(&mut self) {
		if let Some((last, predecessors)) = self.childs.split_last_mut() {
			predecessors.iter_mut().rev().fold(last, |layer, prev_layer| {
				layer.propagate_error_signal(prev_layer);
				prev_layer
			});
		}
		else {
			unreachable!(
				"Reached code marked as unreachable in `ContainerLayer::propagate_error_signal: \
				 This code is unreachable since ContainerLayers cannot have an empty set of child layers");
		};
	}
}

impl ProcessInputSignal for ContainerLayer {
	fn process_input_signal(&mut self, prev_output_signal: &SignalBuffer) {
		if let Some((first, tail)) = self.childs.split_first_mut() {
			tail.iter_mut().fold({
				first.process_input_signal(prev_output_signal);
				first.output_signal()
			}, |prev_output_signal, layer| {
				layer.process_input_signal(prev_output_signal);
				layer.output_signal()
			});
		}
		else {
			unreachable!(
				"Reached code marked as unreachable in `ContainerLayer::process_input_signal: \
				 This code is unreachable since ContainerLayers cannot have an empty set of child layers");
		}
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
		self.propagate_error_signal_internally();
		self.input_layer_mut().propagate_error_signal(propagated)
	}
}

impl ApplyErrorSignalCorrection for ContainerLayer {
	fn apply_error_signal_correction(&mut self, prev_output_signal: &SignalBuffer, rate: LearnRate, momentum: LearnMomentum) {
		if let Some((first, tail)) = self.childs.split_first_mut() {
			tail.iter_mut().fold({
				first.apply_error_signal_correction(prev_output_signal, rate, momentum);
				first.output_signal()
			}, |prev_output_signal, layer| {
				layer.apply_error_signal_correction(prev_output_signal, rate, momentum);
				layer.output_signal()
			});
		}
		else {
			unreachable!(
				"Reached code marked as unreachable in `ContainerLayer::apply_error_signal_correction: \
				 This code is unreachable since ContainerLayers cannot have an empty set of child layers");
		}
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
