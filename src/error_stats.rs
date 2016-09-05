//! Provides an implementation and interface for users of neural networks to
//! learn about their learning state and process during training sessions.

use std::fmt;

/// Stores useful information about a training run for a neural net.
#[derive(Debug, Copy, Clone)]
pub struct ErrorStats {
	net_error: f32,
	avg_error: f32,
	avg_smoothing_factor: f32
}

impl ErrorStats {
	/// Creates a new instance of ```ErrorStats```.
	pub fn new(net_error: f32, avg_error: f32, avg_smoothing_factor: f32) -> Self {
		ErrorStats{
			net_error: net_error,
			avg_error: avg_error,
			avg_smoothing_factor: avg_smoothing_factor
		}
	}

	/// Updates this ```ErrorStats``` instance with a newer lastest-net-error
	/// and keeps the internal average error calculation up-to-date using the given
	/// smoothing factor.
	pub fn update(&mut self, latest_net_error: f32) {
		self.net_error = latest_net_error;
		self.avg_error =
			(self.avg_error * self.avg_smoothing_factor + self.net_error) /
			(self.avg_smoothing_factor + 1.0);
	}

	/// Returns the latest net error that was updated via the ```update``` method.
	pub fn net_error(&self) -> f32 {
		self.net_error
	}

	/// Returns the average error that is a calculation concerning the latest
	/// net errors with which this ```ErrorStats``` was updated.
	/// 
	/// This provides a good indication of how well the neural network has learned
	/// over the last accumulation of iterations.
	pub fn avg_error(&self) -> f32 {
		self.avg_error
	}
}

impl Default for ErrorStats {
	fn default() -> Self {
		ErrorStats{
			net_error: 0.0,
			avg_error: 0.0,
			avg_smoothing_factor: 0.95
		}
	}
}

impl fmt::Display for ErrorStats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "( net_error = {}, avg_error = {} )", self.net_error, self.avg_error)
    }
}