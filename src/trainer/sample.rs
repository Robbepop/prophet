//! Defines samples, sample containers, sample schedulers and macros for supervised learning purposes.

use ndarray::prelude::*;
use rand::XorShiftRng;
use rand;

use layer::utils::{
	UnbiasedSignalView,
	UnbiasedSignalBuffer
};
use errors::{Error, Result};

use std::fmt::Debug;

/// Represents a supervised sample for supervised learning purposes.
/// 
/// This requires an `expected` signal associated to ever `input` signal.
pub trait SupervisedSample {
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
	/// Creates a new `Sample` suitable for supervised learning purposes
	/// from the given input singal and expected result signal.
	/// 
	/// # Errors
	/// 
	/// When the given signals are invalid, e.g. empty.
	pub fn new<I, E>(input: I, expected: E) -> Result<Sample>
		where I: Into<Array1<f32>>,
		      E: Into<Array1<f32>>
	{
		let input = UnbiasedSignalBuffer::from_raw(input.into())?; // TODO: add annotation to error
		let expected = UnbiasedSignalBuffer::from_raw(expected.into())?; // TODO: add annotation to error
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
/// `SampleCollection`s are used by `SampleGen`s.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleCollection {
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
				if sample.expected().len() != expected_len {
					return Err(Error::unmatching_sample_expected_len(expected_len, sample.expected().len()))
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
		assert!(!self.samples.is_empty()); // Enforced by its invariants to never be empty.
		self.samples.is_empty()
	}

	/// Inserts the given sample into this `SampleCollection`.
	#[inline]
	pub fn insert(&mut self, sample: Sample) -> Result<()> {
		if sample.input().len() != self.input_len() {
			return Err(Error::unmatching_sample_input_len(self.input_len(), sample.input().len()))
		}
		if sample.expected().len() != self.expected_len() {
			return Err(Error::unmatching_sample_expected_len(self.expected_len(), sample.expected().len()))
		}
		self.samples.push(sample);
		Ok(())
	}

	/// Returns a slice over the `Sample`s of this `SampleCollection`.
	#[inline]
	pub fn as_slice(&self) -> &[Sample] {
		&self.samples
	}
}

/// `SampleGen`s simply are non-exhaustive iterators over a set of `Sample`s.
pub trait SampleGen: Debug {
	/// Returns the next sample for scheduling.
	fn next_sample(&mut self) -> &Sample;

	/// Returns the length of the set of samples if it is finite; else returns `None`.
	/// 
	/// A returned `batch_len` of `None` represents a theoretically infinite set of samples
	/// that are possible to be generated by this `SampleGen`.
	fn finite_len(&self) -> Option<usize>;
}

/// A `SampleGen` that iterates over its finite set of samples in a sequential fashion.
#[derive(Debug, Clone)]
pub struct SequentialSampleScheduler{
	samples: SampleCollection,
	current: usize
}

impl SequentialSampleScheduler {
	/// Creates a new `SequentialSampleScheduler` from the given `SampleCollection`.
	pub fn new(samples: SampleCollection) -> SequentialSampleScheduler {
		SequentialSampleScheduler{samples, current: 0}
	}
}

impl SampleGen for SequentialSampleScheduler {
	/// Returns a reference to the next sequentially scheduled `Sample`.
	fn next_sample(&mut self) -> &Sample {
		let next = &self.samples.as_slice()[self.current];
		self.current += 1;
		self.current %= self.samples.len();
		next
	}

	#[inline]
	fn finite_len(&self) -> Option<usize> { Some(self.samples.len()) }
}

/// A `SampleGen` that iterates over its finite set of samples in a random fashion.
#[derive(Debug, Clone)]
pub struct RandomSampleScheduler{
	samples: SampleCollection,
	rng: XorShiftRng
}

impl RandomSampleScheduler {
	/// Creates a new `RandomSampleScheduler` from the given `SampleCollection`.
	pub fn new(samples: SampleCollection) -> RandomSampleScheduler {
		RandomSampleScheduler{samples, rng: rand::weak_rng()}
	}
}

impl SampleGen for RandomSampleScheduler {
	/// Returns a reference to the next randomly scheduled `Sample`.
	fn next_sample(&mut self) -> &Sample {
		use rand::Rng;
		self.rng.choose(self.samples.as_slice()).unwrap()
	}

	#[inline]
	fn finite_len(&self) -> Option<usize> { Some(self.samples.len()) }
}

/// Creates a new `SampleCollection`.
/// 
/// Given the following definitions ...
/// 
/// ```rust,no_run
/// # #[allow(unused_variables)]
/// let t =  1.0;
/// # #[allow(unused_variables)]
/// let f = -1.0;
/// ```
/// ... this macro invocation ...
/// 
/// ```rust
/// # #[macro_use]
/// # extern crate prophet;
/// # use prophet::prelude::*;
/// # use prophet::trainer::{Sample, SampleCollection};
/// # fn main() {
/// # let t =  1.0;
/// # let f = -1.0;
/// # #[allow(unused_variables)]
/// let samples = samples![
/// 	[f, f] => [f],
/// 	[t, f] => [t],
/// 	[f, t] => [t],
/// 	[t, t] => [f]
/// ];
/// # }
/// ```
/// 
/// ... will expand to this
/// 
/// ```rust,no_run
/// # extern crate prophet;
/// # use prophet::prelude::*;
/// # use prophet::trainer::{Sample, SampleCollection};
/// # fn main() {
/// # let t =  1.0;
/// # let f = -1.0;
/// # #[allow(unused_variables)]
/// let samples = SampleCollection::from_iter(vec![
/// 	Sample::new(vec![f, f], vec![f]).unwrap(),
/// 	Sample::new(vec![t, f], vec![t]).unwrap(),
/// 	Sample::new(vec![f, t], vec![t]).unwrap(),
/// 	Sample::new(vec![t, t], vec![f]).unwrap(),
/// ].into_iter()).unwrap();
/// # }
/// ```
/// 
/// Note that for single elements all of the
/// below macro invocations are equal:
/// 
/// ```rust
/// # #[macro_use]
/// # extern crate prophet;
/// # use prophet::prelude::*;
/// # use prophet::trainer::{Sample, SampleCollection};
/// # fn main() {
/// # let t =  1.0;
/// # let f = -1.0;
/// # #[allow(unused_variables)]
/// let samples_a = samples![
/// 	[f] => [t],
/// 	[t] => [f]
/// ];
/// # #[allow(unused_variables)]
/// let samples_b = samples![
/// 	f => [t],
/// 	t => [f]
/// ];
/// # #[allow(unused_variables)]
/// let samples_c = samples![
/// 	[f] => t,
/// 	[t] => f
/// ];
/// # #[allow(unused_variables)]
/// let samples_d = samples![
/// 	f => t,
/// 	t => f
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
		),+].into_iter()).unwrap()
	};

	[ $( [ $($i:expr),+ ] => $e:expr ),+ ] => {
		SampleCollection::from_iter(vec![$(
			Sample::new(
				vec![$($i),+],
				vec![$e]
			).unwrap()
		),+].into_iter()).unwrap()
	};

	[ $( $i:expr => [ $($e:expr),+ ] ),+ ] => {
		SampleCollection::from_iter(vec![$(
			Sample::new(
				vec![$i],
				vec![$($e),+]
			).unwrap()
		),+].into_iter()).unwrap()
	};

	[ $( $i:expr => $e:expr ),+ ] => {
		SampleCollection::from_iter(vec![$(
			Sample::new(
				vec![$i],
				vec![$e]
			).unwrap()
		),+].into_iter()).unwrap()
	};
}

/// Creates a new `Sample` for supervised learning.
/// 
/// Given the following definitions ...
/// 
/// ```rust,no_run
/// # #[allow(unused_variables)]
/// let t =  1.0;
/// # #[allow(unused_variables)]
/// let f = -1.0;
/// ```
/// ... this macro invocation ...
/// 
/// ```rust
/// # #[macro_use]
/// # extern crate prophet;
/// # use prophet::prelude::*;
/// # use prophet::trainer::{Sample};
/// # fn main() {
/// # let t =  1.0;
/// # let f = -1.0;
/// # #[allow(unused_variables)]
/// let sample = sample!([f, f] => [t]);
/// # }
/// ```
/// 
/// ... will expand to ...
/// 
/// ```rust,no_run
/// # extern crate prophet;
/// # use prophet::prelude::*;
/// # use prophet::trainer::{Sample};
/// # fn main() {
/// # let t =  1.0;
/// # let f = -1.0;
/// # #[allow(unused_variables)]
/// let sample = Sample::new(vec![f, f], vec![f]).unwrap();
/// # }
/// ```
/// 
/// For single elements these macro invocations are all equal:
/// 
/// ```rust
/// # #[macro_use]
/// # extern crate prophet;
/// # use prophet::prelude::*;
/// # use prophet::trainer::{Sample};
/// # fn main() {
/// # let t =  1.0;
/// # let f = -1.0;
/// # #[allow(unused_variables)]
/// let sample = sample!([f] => [t]);
/// let sample = sample!([f] =>  t );
/// let sample = sample!( f  => [t]);
/// let sample = sample!( f  =>  t );
/// # }
/// ```
///
#[macro_export]
macro_rules! sample {
	( [ $($i:expr),+ ] => [ $($e:expr),+ ] ) => {
		Sample::new(
			vec![$($i),+],
			vec![$($e),+]
		).unwrap()
	};

	( $i:expr => [ $($e:expr),+ ] ) => {
		Sample::new(
			vec![$i],
			vec![$($e),+]
		).unwrap()
	};

	( [ $($i:expr),+ ] => $e:expr ) => {
		Sample::new(
			vec![$($i),+],
			vec![$e]
		).unwrap()
	};

	( $i:expr => $e:expr ) => {
		Sample::new(
			vec![$i],
			vec![$e]
		).unwrap()
	};
}

#[cfg(test)]
mod tests {
	use super::*;

	mod sample {
		use super::*;

		#[test]
		fn create_ok() {
			let sample = Sample::new(
				vec![1.0, 2.0, 3.0],
				vec![10.0, 11.0]
			);
			let expected = Ok(Sample{
				input: UnbiasedSignalBuffer::from_raw(
					Array::from_vec(vec![1.0, 2.0, 3.0])).unwrap(),
				expected: UnbiasedSignalBuffer::from_raw(
					Array::from_vec(vec![10.0, 11.0])).unwrap()
			});
			assert_eq!(sample, expected);
		}

		#[test]
		fn create_empty_input() {
			let sample = Sample::new(vec![], vec![42.0]);
			let expected = Err(Error::zero_sized_signal_buffer());
			assert_eq!(sample, expected);
		}

		#[test]
		fn create_empty_expected() {
			let sample = Sample::new(vec![1337.0], vec![]);
			let expected = Err(Error::zero_sized_signal_buffer());
			assert_eq!(sample, expected);
		}

		#[test]
		fn input() {
			let sample = Sample::new(
				vec![1.0, 2.0, 3.0],
				vec![10.0, 11.0]
			).unwrap();
			let expected_input = Array::from_vec(vec![1.0, 2.0, 3.0]);
			assert!(sample.input().data().all_close(&expected_input, 0.0));
		}

		#[test]
		fn expected() {
			let sample = Sample::new(
				vec![1.0, 2.0, 3.0],
				vec![10.0, 11.0]
			).unwrap();
			let expected_expected = Array::from_vec(vec![10.0, 11.0]);
			assert!(sample.expected().data().all_close(&expected_expected, 0.0));
		}
	}

	mod sample_collection {
		use super::*;

		#[test]
		fn create_ok() {
			let samples = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap(),
				Sample::new(vec![4.0, 5.0], vec![5.0]).unwrap(),
			].into_iter());

			assert!(samples.is_ok());
			let samples = samples.unwrap();

			assert_eq!(samples.as_slice()[0].input().len(), 2);
			assert_eq!(samples.as_slice()[1].input().len(), 2);

			assert_eq!(samples.as_slice()[0].expected().len(), 1);
			assert_eq!(samples.as_slice()[1].expected().len(), 1);

			assert_eq!(samples.as_slice()[0].input().data(), Array::from_vec(vec![1.0, 2.0]));
			assert_eq!(samples.as_slice()[1].input().data(), Array::from_vec(vec![4.0, 5.0]));

			assert_eq!(samples.as_slice()[0].expected().data(), Array::from_vec(vec![3.0]));
			assert_eq!(samples.as_slice()[1].expected().data(), Array::from_vec(vec![5.0]));
		}

		#[test]
		fn create_empty() {
			assert_eq!(
				SampleCollection::from_iter(vec![].into_iter()),
				Err(Error::empty_sample_collection())
			);
		}

		#[test]
		fn create_unmatching_input() {
			assert_eq!(
				SampleCollection::from_iter(vec![
					Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap(),
					Sample::new(vec![4.0]     , vec![5.0]).unwrap()
				].into_iter()),
				Err(Error::unmatching_sample_input_len(2, 1))
			);
			assert_eq!(
				SampleCollection::from_iter(vec![
					Sample::new(vec![1.0     ], vec![2.0]).unwrap(),
					Sample::new(vec![3.0, 4.0], vec![5.0]).unwrap()
				].into_iter()),
				Err(Error::unmatching_sample_input_len(1, 2))
			);
		}

		#[test]
		fn create_unmatching_expected() {
			assert_eq!(
				SampleCollection::from_iter(vec![
					Sample::new(vec![1.0, 2.0], vec![3.0, 4.0]).unwrap(),
					Sample::new(vec![5.0, 6.0], vec![7.0     ]).unwrap()
				].into_iter()),
				Err(Error::unmatching_sample_expected_len(2, 1))
			);
			assert_eq!(
				SampleCollection::from_iter(vec![
					Sample::new(vec![1.0, 2.0], vec![3.0     ]).unwrap(),
					Sample::new(vec![4.0, 5.0], vec![6.0, 7.0]).unwrap()
				].into_iter()),
				Err(Error::unmatching_sample_expected_len(1, 2))
			);
		}

		#[test]
		fn create_unmatching_both() {
			assert_eq!(
				SampleCollection::from_iter(vec![
					Sample::new(vec![0.0, 1.0], vec![2.0, 3.0]).unwrap(),
					Sample::new(vec![4.0     ], vec![5.0, 6.0]).unwrap(),
					Sample::new(vec![7.0, 8.0], vec![9.0     ]).unwrap()
				].into_iter()),
				Err(Error::unmatching_sample_input_len(2, 1))
			);
			assert_eq!(
				SampleCollection::from_iter(vec![
					Sample::new(vec![0.0, 1.0], vec![2.0, 3.0]).unwrap(),
					Sample::new(vec![4.0, 5.0], vec![6.0     ]).unwrap(),
					Sample::new(vec![7.0     ], vec![8.0, 9.0]).unwrap()
				].into_iter()),
				Err(Error::unmatching_sample_expected_len(2, 1))
			);
		}

		#[test]
		fn insert_ok() {
			let mut samples = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap()
			].into_iter()).unwrap();
			assert_eq!(samples.len(), 1);
			assert_eq!(samples.as_slice(), &[Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap()]);
			samples.insert(Sample::new(vec![4.0, 5.0], vec![6.0]).unwrap()).unwrap();
			assert_eq!(samples.len(), 2);
			assert_eq!(
				samples.as_slice(),
				&[
					Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap(),
					Sample::new(vec![4.0, 5.0], vec![6.0]).unwrap()
				]
			);
		}

		#[test]
		fn insert_unmatching_input() {
			let mut samples = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap()
			].into_iter()).unwrap();
			assert_eq!(
				samples.insert(Sample::new(vec![4.0], vec![5.0]).unwrap()),
				Err(Error::unmatching_sample_input_len(2, 1))
			);
			assert_eq!(
				samples.insert(Sample::new(vec![4.0, 5.0, 6.0], vec![7.0]).unwrap()),
				Err(Error::unmatching_sample_input_len(2, 3))
			);
		}

		#[test]
		fn insert_unmatching_expected() {
			let mut samples = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0], vec![2.0, 3.0]).unwrap()
			].into_iter()).unwrap();
			assert_eq!(
				samples.insert(Sample::new(vec![4.0], vec![5.0]).unwrap()),
				Err(Error::unmatching_sample_expected_len(2, 1))
			);
			assert_eq!(
				samples.insert(Sample::new(vec![4.0], vec![5.0, 6.0, 7.0]).unwrap()),
				Err(Error::unmatching_sample_expected_len(2, 3))
			);
		}

		#[test]
		fn len() {
			let samples = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap()
			].into_iter()).unwrap();
			assert_eq!(samples.len(), 1);
		}

		#[test]
		fn is_empty() {
			let samples = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap()
			].into_iter()).unwrap();
			assert_eq!(samples.is_empty(), false);
		}

		#[test]
		#[should_panic]
		fn is_empty_fail() {
			// Due to non-constructor construction invariants are broken.
			// Thus `is_empty` may fail spuriously.
			// Note that this kind of code is not possible outside of this module.
			let samples = SampleCollection{
				samples: vec![]
			};
			assert_eq!(samples.is_empty(), true);
		}

		#[test]
		fn input_len() {
			let samples_1 = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0], vec![2.0]).unwrap(),
				Sample::new(vec![3.0], vec![4.0]).unwrap(),
				Sample::new(vec![5.0], vec![6.0]).unwrap()
			].into_iter()).unwrap();
			let samples_2 = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap(),
				Sample::new(vec![4.0, 5.0], vec![6.0]).unwrap()
			].into_iter()).unwrap();
			let samples_3 = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0, 2.0, 3.0], vec![4.0]).unwrap()
			].into_iter()).unwrap();
			assert_eq!(samples_1.input_len(), 1);
			assert_eq!(samples_2.input_len(), 2);
			assert_eq!(samples_3.input_len(), 3);
		}

		#[test]
		fn expected_len() {
			let samples_1 = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0], vec![2.0]).unwrap(),
				Sample::new(vec![3.0], vec![4.0]).unwrap(),
				Sample::new(vec![5.0], vec![6.0]).unwrap()
			].into_iter()).unwrap();
			let samples_2 = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0], vec![2.0, 3.0]).unwrap(),
				Sample::new(vec![4.0], vec![5.0, 6.0]).unwrap()
			].into_iter()).unwrap();
			let samples_3 = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0], vec![2.0, 3.0, 4.0]).unwrap()
			].into_iter()).unwrap();
			assert_eq!(samples_1.expected_len(), 1);
			assert_eq!(samples_2.expected_len(), 2);
			assert_eq!(samples_3.expected_len(), 3);
		}
	}

	mod seq_scheduler {
		use super::*;

		#[test]
		fn next_sample() {
			let samples = samples![
				[1.0, 2.0] => 3.0,
				[42.0, 1337.0] => 0.0,
				[10.0, 1.0] => 0.1
			];
			let mut scheduler = SequentialSampleScheduler::new(samples.clone());
			assert_eq!(*scheduler.next_sample(), samples.as_slice()[0]);
			assert_eq!(*scheduler.next_sample(), samples.as_slice()[1]);
			assert_eq!(*scheduler.next_sample(), samples.as_slice()[2]);
			assert_eq!(*scheduler.next_sample(), samples.as_slice()[0]);
			assert_eq!(*scheduler.next_sample(), samples.as_slice()[1]);
			assert_eq!(*scheduler.next_sample(), samples.as_slice()[2]);
		}
	}

	mod rand_scheduler {
		use super::*;

		#[test]
		fn next_sample() {
			let samples = samples![
				[1.0, 2.0] => 3.0,
				[42.0, 1337.0] => 0.0,
				[10.0, 1.0] => 0.1
			];
			let mut scheduler = RandomSampleScheduler::new(samples.clone());
			assert!(samples.as_slice().contains(scheduler.next_sample()));
			assert!(samples.as_slice().contains(scheduler.next_sample()));
			assert!(samples.as_slice().contains(scheduler.next_sample()));
			assert!(samples.as_slice().contains(scheduler.next_sample()));
		}
	}

	mod samples_macro {
		use super::*;

		#[test]
		fn macro_eq() {
			let expansion = samples![
				[1.0, 2.0] => [3.0],
				[4.0, 5.0] => [5.0]
			];
			let expected = SampleCollection::from_iter(vec![
				Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap(),
				Sample::new(vec![4.0, 5.0], vec![5.0]).unwrap(),
			].into_iter()).unwrap();
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

	mod sample_macro {
		use super::*;

		#[test]
		fn macro_eq() {
			let expansion = sample!([1.0, 2.0] => [3.0]);
			let expected  = Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap();
			assert_eq!(expansion, expected);
		}

		#[test]
		fn macro_without_right_brackets() {
			let expansion = sample!([1.0, 2.0] => 3.0);
			let expected  = Sample::new(vec![1.0, 2.0], vec![3.0]).unwrap();
			assert_eq!(expansion, expected);
		}

		#[test]
		fn macro_without_left_brackets() {
			let expansion = sample!(1.0 => [2.0, 3.0]);
			let expected  = Sample::new(vec![1.0], vec![2.0, 3.0]).unwrap();
			assert_eq!(expansion, expected);
		}

		#[test]
		fn macro_without_brackets() {
			let expansion = sample!(1.0 => 2.0);
			let expected  = Sample::new(vec![1.0], vec![2.0]).unwrap();
			assert_eq!(expansion, expected);
		}

	}
}
