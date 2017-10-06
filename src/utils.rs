use errors::{Result, Error};

/// Learn rate used during supervised learning.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct LearnRate(f32);

impl LearnRate {
	/// Creates a new `LearnRate` from the given `f32`.
	/// 
	/// # Errors
	/// 
	/// - If the given `f32` is not within the valid open interval of `(0,1]`.
	pub fn new(rate: f32) -> Result<LearnRate> {
		if !(0.0 <= rate && rate <= 1.0) {
			return Err(Error::invalid_learn_rate(rate))
		}
		Ok(LearnRate(rate))
	}

	/// Returns the `f32` representation of this `LearnRate`.
	#[inline]
	pub fn to_f32(self) -> f32 { self.0 }
}

impl From<f32> for LearnRate {
	fn from(rate: f32) -> LearnRate {
		LearnRate::new(rate)
			.expect("Expected valid user input (`f32`) for creating a new LearnRate.")
	}
}

/// Learn momentum.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct LearnMomentum(f32);

impl LearnMomentum {
	/// Creates a new `LearnMomentum` from the given `f32`.
	/// 
	/// # Errors
	/// 
	/// - If the given `f32` is not within the valid open interval of `[0,1]`.
	pub fn new(momentum: f32) -> Result<LearnMomentum> {
		if !(0.0 <= momentum && momentum <= 1.0) {
			return Err(Error::invalid_learn_momentum(momentum))
		}
		Ok(LearnMomentum(momentum))
	}

	/// Returns the `f32` representation of this `LearnMomentum`.
	#[inline]
	pub fn to_f32(self) -> f32 { self.0 }
}

impl From<f32> for LearnMomentum {
	fn from(rate: f32) -> LearnMomentum {
		LearnMomentum::new(rate)
			.expect("Expected valid user input (`f32`) for creating a new LearnMomentum.")
	}
}
