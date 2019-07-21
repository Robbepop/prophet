use crate::errors::{Error, Result};

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
			return Err(Error::invalid_learn_rate(rate));
		}
		Ok(LearnRate(rate))
	}

	/// Returns the `f32` representation of this `LearnRate`.
	#[inline]
	pub fn to_f32(self) -> f32 {
		self.0
	}
}

impl From<f32> for LearnRate {
	fn from(rate: f32) -> LearnRate {
		LearnRate::new(rate)
			.expect("Expected valid user input (`f32`) for creating a new LearnRate.")
	}
}

/// Learn momentum used during supervised learning.
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
			return Err(Error::invalid_learn_momentum(momentum));
		}
		Ok(LearnMomentum(momentum))
	}

	/// Returns the `f32` representation of this `LearnMomentum`.
	#[inline]
	pub fn to_f32(self) -> f32 {
		self.0
	}
}

impl From<f32> for LearnMomentum {
	fn from(rate: f32) -> LearnMomentum {
		LearnMomentum::new(rate)
			.expect("Expected valid user input (`f32`) for creating a new LearnMomentum.")
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	const EPSILON: f32 = 1e-6;

	mod learn_rate {
		use super::*;

		#[test]
		fn new_ok() {
			assert_eq!(LearnRate::new(0.0), Ok(LearnRate(0.0)));
			assert_eq!(LearnRate::new(EPSILON), Ok(LearnRate(EPSILON)));
			assert_eq!(LearnRate::new(0.5), Ok(LearnRate(0.5)));
			assert_eq!(LearnRate::new(1.0 - EPSILON), Ok(LearnRate(1.0 - EPSILON)));
			assert_eq!(LearnRate::new(1.0), Ok(LearnRate(1.0)));
		}

		#[test]
		fn new_fail() {
			assert_eq!(LearnRate::new(-42.0), Err(Error::invalid_learn_rate(-42.0)));
			assert_eq!(
				LearnRate::new(-EPSILON),
				Err(Error::invalid_learn_rate(-EPSILON))
			);
			assert_eq!(
				LearnRate::new(1.0 + EPSILON),
				Err(Error::invalid_learn_rate(1.0 + EPSILON))
			);
			assert_eq!(
				LearnRate::new(1337.0),
				Err(Error::invalid_learn_rate(1337.0))
			);
		}

		#[test]
		fn from_ok() {
			assert_eq!(LearnRate::from(0.0), LearnRate(0.0));
			assert_eq!(LearnRate::from(EPSILON), LearnRate(EPSILON));
			assert_eq!(LearnRate::from(0.5), LearnRate(0.5));
			assert_eq!(LearnRate::from(1.0 - EPSILON), LearnRate(1.0 - EPSILON));
			assert_eq!(LearnRate::from(1.0), LearnRate(1.0));
		}

		#[test]
		#[should_panic]
		fn from_fail_01() {
			LearnRate::from(-42.0);
		}

		#[test]
		#[should_panic]
		fn from_fail_02() {
			LearnRate::from(-EPSILON);
		}

		#[test]
		#[should_panic]
		fn from_fail_03() {
			LearnRate::from(1.0 + EPSILON);
		}

		#[test]
		#[should_panic]
		fn from_fail_04() {
			LearnRate::from(1337.0);
		}

		#[test]
		fn to_f32() {
			assert_eq!(LearnRate(0.0).to_f32(), 0.0);
			assert_eq!(LearnRate(0.5).to_f32(), 0.5);
			assert_eq!(LearnRate(1.0).to_f32(), 1.0);
		}
	}

	mod learn_momentum {
		use super::*;

		#[test]
		fn new_ok() {
			assert_eq!(LearnMomentum::new(0.0), Ok(LearnMomentum(0.0)));
			assert_eq!(LearnMomentum::new(EPSILON), Ok(LearnMomentum(EPSILON)));
			assert_eq!(LearnMomentum::new(0.5), Ok(LearnMomentum(0.5)));
			assert_eq!(
				LearnMomentum::new(1.0 - EPSILON),
				Ok(LearnMomentum(1.0 - EPSILON))
			);
			assert_eq!(LearnMomentum::new(1.0), Ok(LearnMomentum(1.0)));
		}

		#[test]
		fn new_fail() {
			assert_eq!(
				LearnMomentum::new(-42.0),
				Err(Error::invalid_learn_momentum(-42.0))
			);
			assert_eq!(
				LearnMomentum::new(-EPSILON),
				Err(Error::invalid_learn_momentum(-EPSILON))
			);
			assert_eq!(
				LearnMomentum::new(1.0 + EPSILON),
				Err(Error::invalid_learn_momentum(1.0 + EPSILON))
			);
			assert_eq!(
				LearnMomentum::new(1337.0),
				Err(Error::invalid_learn_momentum(1337.0))
			);
		}

		#[test]
		fn from_ok() {
			assert_eq!(LearnMomentum::from(0.0), LearnMomentum(0.0));
			assert_eq!(LearnMomentum::from(EPSILON), LearnMomentum(EPSILON));
			assert_eq!(LearnMomentum::from(0.5), LearnMomentum(0.5));
			assert_eq!(
				LearnMomentum::from(1.0 - EPSILON),
				LearnMomentum(1.0 - EPSILON)
			);
			assert_eq!(LearnMomentum::from(1.0), LearnMomentum(1.0));
		}

		#[test]
		#[should_panic]
		fn from_fail_01() {
			LearnMomentum::from(-42.0);
		}

		#[test]
		#[should_panic]
		fn from_fail_02() {
			LearnMomentum::from(-EPSILON);
		}

		#[test]
		#[should_panic]
		fn from_fail_03() {
			LearnMomentum::from(1.0 + EPSILON);
		}

		#[test]
		#[should_panic]
		fn from_fail_04() {
			LearnMomentum::from(1337.0);
		}

		#[test]
		fn to_f32() {
			assert_eq!(LearnMomentum(0.0).to_f32(), 0.0);
			assert_eq!(LearnMomentum(0.5).to_f32(), 0.5);
			assert_eq!(LearnMomentum(1.0).to_f32(), 1.0);
		}
	}
}
