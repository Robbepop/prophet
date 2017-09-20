
use ndarray::prelude::*;
use errors::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct ErrorSignalBuffer(Array1<f32>);

impl ErrorSignalBuffer {
	pub fn zeros(len: usize) -> Result<ErrorSignalBuffer> {
		if len == 0 {
			return Err(Error::zero_sized_gradient_buffer())
		}
		Ok(ErrorSignalBuffer(Array1::zeros(len + 1)))
	}

	pub fn with_values<'a, T>(input: T) -> Result<ErrorSignalBuffer>
		where T: Into<ArrayView1<'a, f32>>
	{
		let input = input.into();
		let mut buf = ErrorSignalBuffer::zeros(input.dim())?;
		buf.0.assign(&input);
		Ok(buf)
	}

	#[inline]
	pub fn reset_to_zeros(&mut self) {
		self.0.fill(0.0)
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
	pub fn view(&self) -> ArrayView1<f32> {
		self.0.slice(s![..-1])
	}

	#[inline]
	pub fn view_mut(&mut self) -> ArrayViewMut1<f32> {
		self.0.slice_mut(s![..-1])
	}

	#[inline]
	pub fn biased_view(&self) -> ArrayView1<f32> {
		self.0.view()
	}

	#[inline]
	pub fn biased_view_mut(&mut self) -> ArrayViewMut1<f32> {
		self.0.view_mut()
	}
}
