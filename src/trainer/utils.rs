use ndarray::AsArray;

use errors::{Result};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MeanSquaredError(f32);

impl MeanSquaredError {
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

	#[inline]
	pub fn to_f32(self) -> f32 {
		self.0
	}
}
