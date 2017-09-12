
use ndarray::prelude::*;
use errors::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct SignalBuffer(Array1<f32>);

impl SignalBuffer {
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

	pub fn with_values<'a, T>(input: T) -> Result<SignalBuffer>
		where T: Into<ArrayView1<'a, f32>>
	{
		let input = input.into();
		let mut buf = Array::zeros(input.dim() + 1);
		buf.assign(&input);
		Ok(SignalBuffer(buf))
	}

	pub fn from_iter<I>(len: usize, source: I) -> Result<SignalBuffer>
		where I: Iterator<Item=f32>
	{
		use std::iter;
		let result = SignalBuffer(Array::from_iter(
			source
				.take(len)
				.chain(iter::once(1.0))));
		if result.len() != len {
			return Err(Error::non_matching_number_of_signals(result.len(), len))
		}
		Ok(result)
	}

	pub fn assign(&mut self, values: ArrayView1<f32>) -> Result<()> {
		if self.len() != values.dim() {
			return Err(Error::non_matching_assign_signals(values.dim(), self.len()))
		}
		Ok(self.0.assign(&values))
	}

	#[inline]
	pub fn len(&self) -> usize {
		self.view().dim()
	}

	#[inline]
	pub fn biased_len(&self) -> usize {
		self.biased_view().dim()
	}

	#[inline]
	pub fn biased_view(&self) -> ArrayView1<f32> {
		self.0.view()
	}

	#[inline]
	pub fn view(&self) -> ArrayView1<f32> {
		self.0.slice(s![..-1])
	}

	#[inline]
	pub fn view_mut(&mut self) -> ArrayViewMut1<f32> {
		self.0.slice_mut(s![..-1])
	}
}
