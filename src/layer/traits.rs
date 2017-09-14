use layer::SignalBuffer;
use utils::{LearnRate, LearnMomentum};

pub(crate) trait FeedForward {
	fn feed_forward(&mut self, signal: &SignalBuffer);
}

pub(crate) trait OutputSignal {
	fn output_signal(&self) -> &SignalBuffer;
	fn output_signal_mut(&mut self) -> &mut SignalBuffer;
}

pub(crate) trait CalculateErrorGradients {
	fn calculate_gradient_descent(&mut self, target_signals: &SignalBuffer);
}

pub(crate) trait GradientBuffer {
	fn gradients(&self) -> &GradientBuffer;
	fn gradients_mut(&mut self) -> &mut GradientBuffer;
}

pub(crate) trait PropagateGradients {
	fn propagate_gradients<P>(&mut self, propagator: P)
		where P: GradientBuffer;
}

pub(crate) trait ApplyGradients {
	fn apply_gradients(&mut self, signal: &SignalBuffer, lr: LearnRate, lm: LearnMomentum);
}
