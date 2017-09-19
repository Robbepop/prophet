use layer::{SignalBuffer, GradientBuffer};
use utils::{LearnRate, LearnMomentum};

pub(crate) trait ProcessSignal {
	fn process_signal(&mut self, signal: &SignalBuffer);
}

pub(crate) trait HasOutputSignal {
	fn output_signal(&self) -> &SignalBuffer;
	fn output_signal_mut(&mut self) -> &mut SignalBuffer;
}

pub(crate) trait CalculateErrorGradients {
	fn calculate_gradient_descent(&mut self, target_signals: &SignalBuffer);
}

pub(crate) trait HasGradientBuffer {
	fn gradients(&self) -> &GradientBuffer;
	fn gradients_mut(&mut self) -> &mut GradientBuffer;
}

pub(crate) trait PropagateGradients {
	fn propagate_gradients<P>(&self, propagated: &mut P)
		where P: HasGradientBuffer;
}

pub(crate) trait ApplyGradients {
	fn apply_gradients(&mut self, signal: &SignalBuffer, lr: LearnRate, lm: LearnMomentum);
}
