use ndarray::prelude::*;
use rand::XorShiftRng;
use rand;

use layer::utils::{
	UnbiasedSignalView,
	UnbiasedSignalBuffer
};
use errors::{Error, Result};

/// Represents a supervised sample for supervised learning purposes.
/// 
/// This requires an `expected` signal associated to ever `input` signal.
pub(crate) trait SupervisedSample {
	/// Returns the input signal of this sample.
	fn input(&self) -> UnbiasedSignalView;

	/// Returns the expected signal of this sample.
	fn expected(&self) -> UnbiasedSignalView;
}

/// A `Sample` suitable for supervised learning.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
	input: UnbiasedSignalBuffer,
	expected: UnbiasedSignalBuffer
}

impl Sample {
	pub fn new<I, E>(input: I, expected: E) -> Result<Sample>
		where I: Into<Array1<f32>>,
		      E: Into<Array1<f32>>
	{
		let input = UnbiasedSignalBuffer::from_raw(input.into())?;
		let expected = UnbiasedSignalBuffer::from_raw(expected.into())?;
		Ok(Sample{input, expected})
	}
}

impl SupervisedSample for Sample {
	#[inline]
	fn input(&self) -> UnbiasedSignalView {
		self.input.view()
	}

	#[inline]
	fn expected(&self) -> UnbiasedSignalView {
		self.expected.view()
	}
}

/// A `SampleCollection` is, as the name reads, a non-empty collection of samples.
/// 
/// Samples within a collection are uniform which means that their input length
/// and expected length are all equal.
/// 
/// `SampleCollection`s are used by `SampleScheduler`s.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleCollection{
	samples: Vec<Sample>
}

impl SampleCollection {
	/// Creates a new `SampleCollection` from the given iterator of `Sample`s.
	/// 
	/// # Errors
	/// 
	/// This fails if ...
	/// 
	/// - ... the given iterator is empty
	/// - ... any of the samples have unmatching input or expected lengths
	/// 
	/// # Note
	/// 
	/// This won't fail for duplicate samples within the iterator, however,
	/// a check for this scenario might be added in the future.
	/// 
	pub fn from_iter<I>(samples: I) -> Result<SampleCollection>
		where I: Iterator<Item = Sample>
	{
		let samples: Vec<Sample> = samples.collect();
		if let Some((first, rest)) = samples.split_first() {
			let input_len = first.input().len();
			let expected_len = first.expected().len();
			for sample in rest {
				if sample.input().len() != input_len {
					return Err(Error::unmatching_sample_input_len(input_len, sample.input().len()))
				}
				if sample.expected().len() == expected_len {
					return Err(Error::unmatching_sample_expected_len(input_len, sample.input().len()))
				}
			}
			
		}
		else {
			return Err(Error::empty_sample_collection())
		}
		Ok(SampleCollection{samples})
	}

	/// Returns the sample length of the input signal.
	/// 
	/// This must be equal to the input signal length of the neural network.
	#[inline]
	pub fn input_len(&self) -> usize {
		self.samples.first().unwrap().input().len()
	}

	/// Returns the sample length of the expected signal.
	/// 
	/// This must be equal to the output signal length of the neural network. 
	#[inline]
	pub fn expected_len(&self) -> usize {
		self.samples.first().unwrap().expected().len()
	}

	/// Returns the number of `Sample`s stored in this `SampleCollection`.
	#[inline]
	fn len(&self) -> usize {
		self.samples.len()
	}

	/// Returns `true` if this `SampleCollection` is empty, `false` otherwise.
	/// 
	/// Note: A `SampleCollection` is never empty enforced by its invariants.
	#[inline]
	pub fn is_empty(&self) -> bool {
		assert!(!self.samples.is_empty());
		self.samples.is_empty()
	}

	/// Inserts the given sample into this `SampleCollection`.
	#[inline]
	pub fn insert(&mut self, sample: Sample) {
		self.samples.push(sample)
	}

	/// Returns a slice over the `Sample`s of this `SampleCollection`.
	#[inline]
	pub fn as_slice(&self) -> &[Sample] {
		&self.samples
	}
}

/// `SampleScheduler`s simply are non-exhaustive iterators over a set of `Sample`s.
pub trait SampleScheduler {
	/// Returns the next sample for scheduling.
	fn next_sample(&mut self) -> &Sample;
}

/// A `SampleScheduler` that iterates over its finite set of samples in a sequential fashion.
#[derive(Debug, Clone)]
struct SequentialSampleScheduler{
	samples: SampleCollection,
	current: usize
}

impl SequentialSampleScheduler {
	/// Creates a new `SequentialSampleScheduler` from the given `SampleCollection`.
	pub fn new(samples: SampleCollection) -> SequentialSampleScheduler {
		SequentialSampleScheduler{samples, current: 0}
	}
}

impl SampleScheduler for SequentialSampleScheduler {
	/// Returns a reference to the next sequentially scheduled `Sample`.
	fn next_sample(&mut self) -> &Sample {
		let next = &self.samples.as_slice()[self.current];
		self.current += 1;
		self.current %= self.samples.len();
		next
	}
}

/// A `SampleScheduler` that iterates over its finite set of samples in a random fashion.
#[derive(Debug, Clone)]
struct RandomSampleScheduler{
	samples: SampleCollection,
	rng: XorShiftRng
}

impl RandomSampleScheduler {
	/// Creates a new `RandomSampleScheduler` from the given `SampleCollection`.
	pub fn new(samples: SampleCollection) -> RandomSampleScheduler {
		RandomSampleScheduler{samples, rng: rand::weak_rng()}
	}
}

impl SampleScheduler for RandomSampleScheduler {
	/// Returns a reference to the next randomly scheduled `Sample`.
	fn next_sample(&mut self) -> &Sample {
		use rand::Rng;
		self.rng.choose(self.samples.as_slice()).unwrap()
	}
}

/// Creates a new `SampleCollection`.
/// 
/// Given the following definitions
/// 
/// ```rust,no_run
/// # #[allow(unused_variables)]
/// let t =  1.0;
/// # #[allow(unused_variables)]
/// let f = -1.0;
/// ```
/// ... this macro invokation ...
/// 
/// ```rust
/// # #[macro_use]
/// # extern crate prophet;
/// # use prophet::prelude::*;
/// # fn main() {
/// # let t =  1.0;
/// # let f = -1.0;
/// # #[allow(unused_variables)]
/// let samples = samples![
/// 	[f, f] => f,
/// 	[t, f] => t,
/// 	[f, t] => t,
/// 	[t, t] => f
/// ];
/// # }
/// ```
/// 
/// ... will expand to this
/// 
/// ```rust,no_run
/// # extern crate prophet;
/// # use prophet::prelude::*;
/// # fn main() {
/// # let t =  1.0;
/// # let f = -1.0;
/// # #[allow(unused_variables)]
/// let samples = vec![
/// 	Sample::new(vec![f, f], vec![f]),
/// 	Sample::new(vec![t, f], vec![t]),
/// 	Sample::new(vec![f, t], vec![t]),
/// 	Sample::new(vec![t, t], vec![f]),
/// ];
/// # }
/// ```
#[macro_export]
macro_rules! samples {
	[ $( [ $($i:expr),+ ] => [ $($e:expr),+ ] ),+ ] => {
		SampleCollection::from_iter(vec![$(
			Sample::new(
				vec![$($i),+],
				vec![$($e),+]
			).unwrap()
		),+].into_iter())
	};

	[ $( [ $($i:expr),+ ] => $e:expr ),+ ] => {
		SampleCollection::from_iter(vec![$(
			Sample::new(
				vec![$($i),+],
				vec![$e]
			).unwrap()
		),+].into_iter())
	};

	[ $( $i:expr => [ $($e:expr),+ ] ),+ ] => {
		SampleCollection::from_iter(vec![$(
			Sample::new(
				vec![$i],
				vec![$($e),+]
			).unwrap()
		),+].into_iter())
	};

	[ $( $i:expr => $e:expr ),+ ] => {
		SampleCollection::from_iter(vec![$(
			Sample::new(
				vec![$i],
				vec![$e]
			).unwrap()
		),+].into_iter())
	};
}

#[cfg(test)]
mod tests {
	use super::*;

	mod samples_macro {
		use super::*;

		#[test]
		fn sample_and_vec_equal() {
			let expansion = samples![
				[1.0, 2.0] => [3.0],
				[4.0, 5.0] => [5.0]
			];
			let expected = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap(),
				Sample::new(vec![4.0, 5.0], vec![5.0]).unwrap(),
			].into_iter());
			assert_eq!(expansion, expected);
		}

		#[test]
		fn missing_right_brackets() {
			let with_rbrackets = samples![
				[1.0, 2.0] => [3.0],
				[4.0, 5.0] => [5.0],
				[6.0, 7.0] => [8.0]
			];
			let without_rbrackets = samples![
				[1.0, 2.0] => 3.0,
				[4.0, 5.0] => 5.0,
				[6.0, 7.0] => 8.0
			];
			assert_eq!(with_rbrackets, without_rbrackets);
		}

		#[test]
		fn missing_left_brackets() {
			let with_lbrackets = samples![
				[1.0] => [2.0, 3.0],
				[4.0] => [5.0, 6.0],
				[7.0] => [8.0, 9.0]
			];
			let without_lbrackets = samples![
				1.0 => [2.0, 3.0],
				4.0 => [5.0, 6.0],
				7.0 => [8.0, 9.0]
			];
			assert_eq!(with_lbrackets, without_lbrackets);
		}

		#[test]
		fn missing_both_brackets() {
			let with_brackets = samples![
				[1.0] => [2.0],
				[3.0] => [4.0],
				[5.0] => [6.0]
			];
			let without_brackets = samples![
				1.0 => 2.0,
				3.0 => 4.0,
				5.0 => 6.0
			];
			assert_eq!(with_brackets, without_brackets);
		}
	}
}
