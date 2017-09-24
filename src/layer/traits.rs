use layer::utils::{SignalBuffer, ErrorSignalBuffer};
use utils::{LearnRate, LearnMomentum};

pub(crate) trait ProcessInputSignal {
	fn process_input_signal(&mut self, signal: &SignalBuffer);
}

pub(crate) trait HasOutputSignal {
	fn output_signal(&self) -> &SignalBuffer;
	fn output_signal_mut(&mut self) -> &mut SignalBuffer;
}

pub(crate) trait CalculateOutputErrorSignal {
	fn calculate_output_error_signal(&mut self, target_signals: &SignalBuffer);
}

pub(crate) trait HasErrorSignal {
	fn error_signal(&self) -> &ErrorSignalBuffer;
	fn error_signal_mut(&mut self) -> &mut ErrorSignalBuffer;
}

pub(crate) trait PropagateErrorSignal {
	fn propagate_error_signal<P>(&mut self, propagated: &mut P)
		where P: HasErrorSignal;
}

pub(crate) trait ApplyErrorSignalCorrection {
	fn apply_error_signal_correction(&mut self, signal: &SignalBuffer, lr: LearnRate, lm: LearnMomentum);
}

pub(crate) trait SizedLayer {
	fn inputs(&self) -> usize;
	fn outputs(&self) -> usize;
}
