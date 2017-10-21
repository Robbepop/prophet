use ndarray::AsArray;

use errors::{Error, Result};

/// Represents a mean-squared-error value.
/// 
/// Note that a mean-squared-error can never be a negative value.
/// This is used mainly during the training process as a statistics
/// for supervised learning that may be used to query a state when the
/// training process can be halted.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MeanSquaredError(f32);

impl From<f32> for MeanSquaredError {
	fn from(mse: f32) -> MeanSquaredError {
		MeanSquaredError::new(mse).unwrap()
	}
}

impl MeanSquaredError {
	/// Creates a new `MeanSquaredError` from the given `f32` value representing it.
	/// 
	/// This is used mainly to initialize `MeanSquaredError` types.
	/// 
	/// # Errors
	/// 
	/// - If the given mse is negative.
	pub fn new(mse: f32) -> Result<MeanSquaredError> {
		if mse < 0.0 {
			return Err(Error::mse_invalid_negative_value(mse))
		}
		Ok(MeanSquaredError(mse))
	}

	/// Calculates a new `MeanSquaredError` from the two given array-like components.
	/// 
	/// Note that for performance reasons this does not calculate the real mean-squared-error
	/// as a sum of the squared differences divided by the number of elements but instead only
	/// divides by two `2` to make its deviation simpler and thus improve the overall performance.
	/// Read more here: [Link](https://de.wikipedia.org/wiki/Backpropagation#Fehlerminimierung)
	/// 
	/// # Errors
	/// 
	/// - If the array representing the actual values has a length of zero (`0`).
	/// - If the array representing the expected values has a length of zero (`0`).
	/// - If the given arrays have unmatching lengths.
	pub fn from_arrays<'a, 'e, A, E>(actual: A, expected: E) -> Result<MeanSquaredError>
		where A: AsArray<'a, f32>,
		      E: AsArray<'e, f32>
	{
		let actual = actual.into();
		let expected = expected.into();
		if actual.dim() == 0 {
			return Err(Error::mse_invalid_empty_actual_buffer())
		}
		if expected.dim() == 0 {
			return Err(Error::mse_invalid_empty_expected_buffer())
		}
		if actual.dim() != expected.dim() {
			return Err(Error::mse_unmatching_actual_and_empty_buffers(actual.dim(), expected.dim()))
		}
		use itertools;
		Ok(
			MeanSquaredError(
				itertools::multizip((actual.iter(), expected.iter()))
					.map(|(a, e)| { let diff = a - e; diff*diff })
					.sum::<f32>() / 2.0
			)
		)
	}

	/// Returns the `f32` representation of this `MeanSquaredError`.
	#[inline]
	pub fn to_f32(self) -> f32 {
		self.0
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	mod mean_squared_error {
		use super::*;

		#[test]
		fn from_ok() {
			assert_eq!(MeanSquaredError::from(0.0), MeanSquaredError(0.0));
			assert_eq!(MeanSquaredError::from(-0.0), MeanSquaredError(-0.0));
			assert_eq!(MeanSquaredError::from(1.0), MeanSquaredError(1.0));
			assert_eq!(MeanSquaredError::from(1337.0), MeanSquaredError(1337.0));
		}

		#[test]
		#[should_panic]
		fn from_failure() {
			MeanSquaredError::from(-1.0);
		}

		#[test]
		fn new_ok() {
			assert_eq!(MeanSquaredError::new(0.0), Ok(MeanSquaredError(0.0)));
			assert_eq!(MeanSquaredError::new(-0.0), Ok(MeanSquaredError(-0.0)));
			assert_eq!(MeanSquaredError::new(1.0), Ok(MeanSquaredError(1.0)));
			assert_eq!(MeanSquaredError::new(1337.0), Ok(MeanSquaredError(1337.0)));
		}

		#[test]
		fn new_failure() {
			assert_eq!(MeanSquaredError::new(-1e-8), Err(Error::mse_invalid_negative_value(-1e-8)));
			assert_eq!(MeanSquaredError::new(-1.0), Err(Error::mse_invalid_negative_value(-1.0)));
			assert_eq!(MeanSquaredError::new(-42.0), Err(Error::mse_invalid_negative_value(-42.0)));
		}

		#[test]
		fn from_arrays_empty_actual() {
			let actual = vec![];
			let expected = vec![1.0];
			assert_eq!(
				MeanSquaredError::from_arrays(&actual, &expected),
				Err(Error::mse_invalid_empty_actual_buffer())
			);
		}

		#[test]
		fn from_arrays_empty_expected() {
			let actual = vec![1.0];
			let expected = vec![];
			assert_eq!(
				MeanSquaredError::from_arrays(&actual, &expected),
				Err(Error::mse_invalid_empty_expected_buffer())
			);
		}

		#[test]
		fn from_arrays_unmatching_actual_expected() {
			{
				let actual = vec![1.0];
				let expected = vec![2.0, 3.0];
				assert_eq!(
					MeanSquaredError::from_arrays(&actual, &expected),
					Err(Error::mse_unmatching_actual_and_empty_buffers(1, 2))
				);
			}
			{
				let actual = vec![1.0, 2.0];
				let expected = vec![3.0];
				assert_eq!(
					MeanSquaredError::from_arrays(&actual, &expected),
					Err(Error::mse_unmatching_actual_and_empty_buffers(2, 1))
				);
			}
		}

		#[test]
		fn from_arrays_ok() {
			let a = (1.0, 2.0);
			let e = (3.0, 5.0);
			let actual = vec![a.0, a.1];
			let expected = vec![e.0, e.1];
			assert_eq!(
				MeanSquaredError::from_arrays(&actual, &expected),
				Ok(MeanSquaredError(0.5 * ((a.0 - e.0).powi(2) + (a.1 - e.1).powi(2))))
			);
		}

		#[test]
		fn to_f32() {
			assert_eq!(MeanSquaredError(0.0).to_f32(), 0.0);
			assert_eq!(MeanSquaredError(1e-8).to_f32(), 1e-8);
			assert_eq!(MeanSquaredError(1.0).to_f32(), 1.0);
			assert_eq!(MeanSquaredError(42.0).to_f32(), 42.0);
		}

	}
}
