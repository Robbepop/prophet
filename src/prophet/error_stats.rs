use std::fmt;

#[derive(Debug, Copy, Clone)]
pub struct ErrorStats {
	net_error: f32,
	avg_error: f32,
	avg_smoothing_factor: f32
}

impl ErrorStats {
	pub fn new(net_error: f32, avg_error: f32, avg_smoothing_factor: f32) -> Self {
		ErrorStats{
			net_error: net_error,
			avg_error: avg_error,
			avg_smoothing_factor: avg_smoothing_factor
		}
	}

	pub fn update(&mut self, latest_net_error: f32) {
		self.net_error = latest_net_error;
		self.avg_error =
			(self.avg_error * self.avg_smoothing_factor + self.net_error) /
			(self.avg_smoothing_factor + 1.0);
	}

	pub fn net_error(&self) -> f32 {
		self.net_error
	}

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