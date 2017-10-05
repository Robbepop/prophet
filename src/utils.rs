use errors::{Result, Error};

/// Learn rate.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct LearnRate(pub f32);

impl LearnRate {
	/// Returns learn rate from the given `f64` if valid.
	/// 
	/// `rate` has to be in `(0,1)` to form a valid `LearnRate`
	pub fn from_f64(rate: f64) -> Result<LearnRate> {
		if rate < 0.0 || 1.0 < rate {
			return Err(Error::invalid_learn_rate(rate))
		}
		Ok(LearnRate(rate as f32))
	}

	/// Returns the `f32` representation of this `LearnRate`.
	#[inline]
	pub fn to_f32(self) -> f32 { self.0 }
}

impl From<f64> for LearnRate {
	fn from(rate: f64) -> LearnRate {
		LearnRate::from_f64(rate)
			.expect("Expected valid user input (`f64`) for creating a new LearnRate.")
	}
}

impl Default for LearnRate {
	fn default() -> Self {
		LearnRate(0.3)
	}
}

/// Learn momentum.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct LearnMomentum(pub f32);

impl LearnMomentum {
	/// Returns learn momentum from the given `f64` if valid.
	/// 
	/// `momentum` has to be in `(0,1)` to form a valid `LearnMomentum`
	pub fn from_f64(momentum: f64) -> Result<LearnMomentum> {
		if momentum < 0.0 || 1.0 < momentum {
			return Err(Error::invalid_learn_momentum(momentum))
		}
		Ok(LearnMomentum(momentum as f32))
	}

	/// Returns the `f32` representation of this `LearnMomentum`.
	#[inline]
	pub fn to_f32(self) -> f32 { self.0 }
}

impl From<f64> for LearnMomentum {
	fn from(rate: f64) -> LearnMomentum {
		LearnMomentum::from_f64(rate)
			.expect("Expected valid user input (`f64`) for creating a new LearnMomentum.")
	}
}

impl Default for LearnMomentum {
	fn default() -> Self {
		LearnMomentum(0.5)
	}
}
