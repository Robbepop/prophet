use ndarray::AsArray;

use errors::{Result};

/// Represents a mean-squared-error value.
/// 
/// Note that a mean-squared-error can never be a negative value.
/// This is used mainly during the training process as a statistics
/// for supervised learning that may be used to query a state when the
/// training process can be halted.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MeanSquaredError(f32);

impl MeanSquaredError {
	/// Creates a new `MeanSquaredError` from the given `f32` value representing it.
	/// 
	/// This is used mainly to initialize `MeanSquaredError` types.
	/// 
	/// # Errors
	/// 
	/// - If the given mse is negative.
	pub fn new(mse: f32) -> Result<MeanSquaredError> {
		if mse.is_sign_negative() {
			// TODO: Handle errors properly.
			panic!("Error: Cannot create a negative MeanSquaredError.")
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
		if actual.len() == 0 {
			/// TODO: Handle errors properly.
			panic!("Error: Array for actual data has zero (`0`) length. Required to be at least one.")
		}
		if expected.len() == 0 {
			/// TODO: Handle errors properly.
			panic!("Error: Array for expected data has zero (`0`) length. Required to be at least one.")
		}
		if actual.len() != expected.len() {
			/// TODO: Handle errors properly.
			panic!("Error: Arrays for actual data and expected data are of different sizes.")
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
