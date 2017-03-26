use ndarray::prelude::*;

/// Handles deviations of predicted and target values of
/// the neural network under training.
/// 
/// This is especially useful when using `MeanSquaredError`
/// or `AvgNetError` criterions.
#[derive(Debug, Copy, Clone)]
pub struct Deviation {
	latest_mse   : f64,
	recent_mse   : f64,
	recent_factor: f64,
}

impl Deviation {
	/// Creates a new deviation instance.
	///
	/// ***Panics*** If the given smoothing factor is ∉ *(0, 1)*.
	pub fn new(recent_factor: f64) -> Self {
		assert!(0.0 < recent_factor && recent_factor < 1.0);
		Deviation{
			latest_mse   : 0.0,
			recent_mse   : 1.0,
			recent_factor: recent_factor,
		}
	}

	/// Calculates mean squared error based on the given actual and expected data.
	fn update_mse<F>(&mut self, actual: ArrayView1<F>, expected: ArrayView1<F>)
		where F: NdFloat
	{
		use std::ops::Div;
		use itertools::multizip;
		self.latest_mse = multizip((actual.iter(), expected.iter()))
			.map(|(&actual, &expected)| {
				let dx = expected - actual;
				(dx * dx).to_f64().unwrap()
			})
			.sum::<f64>()
			.div(actual.len() as f64)
			.sqrt();
	}

	/// Calculates recent mean squared error based on the recent factor smoothing.
	fn update_recent_mse(&mut self) {
		self.recent_mse = self.recent_factor * self.recent_mse
			+ (1.0 - self.recent_factor) * self.latest_mse;
	}

	/// Updates the current mean squared error and associated data.
	pub fn update<F>(&mut self, actual: ArrayView1<F>, expected: ArrayView1<F>)
		where F: NdFloat
	{
		self.update_mse(actual, expected);
		self.update_recent_mse();
	}

	/// Gets the latest mean squared error.
	pub fn latest_mse(&self) -> f64 {
		self.latest_mse
	}

	/// Gets the recent mean squared error.
	pub fn recent_mse(&self) -> f64 {
		self.recent_mse
	}
}

impl Default for Deviation {
	fn default() -> Self {
		Deviation::new(0.95)
	}
}
