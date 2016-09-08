//! Provides an implementation and interface to work with learning configurations
//! for neural net and layer implemenations.

use std::fmt;

use activation_fn::ActivationFn;

/// Represents a configuration that directly controls the parameters of learning for a neural net.
/// 
/// A system that tries to optimize the way a neural net learns over the duration of a training 
/// session could make a good use of this structure as an interface to the underlying parameters.
#[derive(Copy, Clone, Debug)]
pub struct LearnConfig {
	/// the rate at which the neural net is adapting to the expectations
	learn_rate: f32,
	/// the rate at which the updating of the weights is affected by the previous weights
	learn_momentum: f32,
	/// the activation function used during training and predicting
	pub act_fn: ActivationFn<f32>
}

impl LearnConfig {
	/// Creates a new instance of ```LearnConfig```.
	/// 
	/// # Panics
	/// If ```learn_rate``` and ```learn_momentum``` ∉ *(0, 1)*.
	pub fn new(learn_rate: f32, learn_momentum: f32, act_fn: ActivationFn<f32>) -> Self {
		assert!(learn_rate > 0.0 && learn_rate < 1.0);
		assert!(learn_momentum > 0.0 && learn_momentum < 1.0);
		LearnConfig{
			learn_rate: learn_rate,
			learn_momentum: learn_momentum,
			act_fn: act_fn
		}
	}

	/// Returns the current learn rate.
	pub fn learn_rate(&self) -> f32 { self.learn_rate }

	/// Returns the current learn rate.
	pub fn learn_momentum(&self) -> f32 { self.learn_momentum }

	/// Sets the learn rate of this configuration to the given value.
	/// 
	/// # Panics
	/// If the given learn rate is ∉ *(0, 1]*.
	pub fn update_learn_rate(&mut self, new_learn_rate: f32) {
		assert!(new_learn_rate > 0.0 && new_learn_rate <= 1.0);
		self.learn_rate = new_learn_rate;
	}

	/// Sets the learn momentum of this configuration to the given value.
	/// 
	/// # Panics
	/// If the given learn momentum ∉ *[0, 1]*.
	pub fn update_learn_momentum(&mut self, new_learn_momentum: f32) {
		assert!(new_learn_momentum >= 0.0 && new_learn_momentum <= 1.0);
		self.learn_momentum = new_learn_momentum;
	}
}

impl fmt::Display for LearnConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "( learn_rate = {}, learn_momentum = {}, ƒ = {} )", self.learn_rate, self.learn_momentum, self.act_fn)
    }
}

#[cfg(test)]
mod tests {
	use super::LearnConfig;
	use activation_fn::ActivationFn;

	#[test]
	fn getter() {
		let config = LearnConfig::new(0.25, 0.5, ActivationFn::tanh());
		assert_eq!(config.learn_rate, 0.25);
		assert_eq!(config.learn_momentum, 0.5);
		assert_eq!(config.act_fn, ActivationFn::<f32>::tanh());
	}
}
