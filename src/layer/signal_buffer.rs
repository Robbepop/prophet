
use ndarray::prelude::*;
use errors::{Error, Result};

/// The `SignalBuffer` is a utility structure that stores an amount of signals equal
/// to a one-dimensional array with exactly one additional constant bias value of `1.0`.
/// 
/// This bias value cannot be overwritten but only read from.
/// 
/// This serves as a robustness abstraction to keep holding this invariant 
/// throughout the execution of the program.
/// 
#[derive(Debug, Clone, PartialEq)]
pub struct SignalBuffer(Array1<f32>);

impl SignalBuffer {
	/// Creates a new `SignalBuffer` with a variable length of `len` and an
	/// additional constant bias value of `1.0`.
	/// 
	/// So a call to `SignalBuffer::zeros(5)` actually constructs a buffer
	/// of length `6` with the last value set to `1.0`.
	/// 
	/// # Errors
	/// 
	/// Returns an error when trying to create a `SignalBuffer` with a length of zero.
	/// 
	pub fn zeros(len: usize) -> Result<SignalBuffer> {
		use std::iter;
		if len == 0 {
			return Err(Error::zero_sized_signal_buffer())
		}
		Ok(SignalBuffer(
			Array::from_iter(iter::repeat(0.0)
				.take(len)
				.chain(iter::once(1.0))
			)
		))
	}

	/// Creates a new `SignalBuffer` with its variable signals set
	/// to the given `input` and the last signal set to `1.0`.
	/// 
	/// # Errors
	/// 
	/// Returns an error when the length of `input` is zero.
	///  
	pub fn with_values(input: ArrayView1<f32>) -> Result<SignalBuffer> {
		if input.dim() == 0 {
			return Err(Error::zero_sized_signal_buffer())
		}
		let mut buf = SignalBuffer::zeros(input.dim())?;
		buf.assign(input)?;
		Ok(buf)
	}

	/// Creates a new `SignalBuffer` with its variable signals set
	/// to the first `len` values of the given `source` iterator.
	/// 
	/// # Errors
	/// 
	/// Returns an error ...
	/// 
	/// - if `source` generates less items than required by the given `len`.
	/// 
	/// - if `len` is equal to zero.
	pub fn from_iter<I>(len: usize, source: I) -> Result<SignalBuffer>
		where I: Iterator<Item=f32>
	{
		use std::iter;
		if len == 0 {
			return Err(Error::zero_sized_signal_buffer())
		}
		let result = SignalBuffer(Array::from_iter(
			source
				.take(len)
				.chain(iter::once(1.0))));
		if result.len() != len {
			return Err(Error::non_matching_number_of_signals(result.len(), len))
		}
		Ok(result)
	}

	/// Assigns the non-bias contents of this `SignalBuffer` to the contents
	/// of the given `values`.
	/// 
	/// # Errors
	/// 
	/// Returns an error if the length `values` does not match the length
	/// of this buffer without respect to its bias value.
	///  
	pub fn assign(&mut self, values: ArrayView1<f32>) -> Result<()> {
		if self.len() != values.dim() {
			return Err(Error::non_matching_assign_signals(values.dim(), self.len()))
		}
		Ok(self.0.assign(&values))
	}

	/// Returns the length of this `SignalBuffer` *without* respect to its bias value.
	#[inline]
	pub fn len(&self) -> usize {
		self.view().dim()
	}

	/// Returns the length of this `SignalBuffer` *with* respect to its bias value.
	#[inline]
	pub fn biased_len(&self) -> usize {
		self.biased_view().dim()
	}

	/// Returns a view to the contents of this `SignalBuffer` *with* respect to its bias value.
	#[inline]
	pub fn biased_view(&self) -> ArrayView1<f32> {
		self.0.view()
	}

	/// Returns a view to the contents of this `SignalBuffer` *without* respect to its bias value.
	#[inline]
	pub fn view(&self) -> ArrayView1<f32> {
		self.0.slice(s![..-1])
	}

	/// Returns a mutable view to the contents of this `SignalBuffer` *without* respect to its bias value.
	#[inline]
	pub fn view_mut(&mut self) -> ArrayViewMut1<f32> {
		self.0.slice_mut(s![..-1])
	}
}
