
use ndarray::prelude::*;
use errors::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct OutputBuffer(Array1<f32>);

impl OutputBuffer {
	pub fn new(len: usize) -> Result<OutputBuffer> {
		use std::iter;
		if len == 0 {
			return Err(Error::zero_sized_output_buffer())
		}
		Ok(OutputBuffer(
			Array::from_iter(iter::repeat(0.0)
				.take(len)
				.chain(iter::once(1.0))
			)
		))
	}

	pub fn from_slice(values: ArrayView1<f32>) -> Result<OutputBuffer> {
		let mut buf = Array::zeros(values.dim() + 1);
		buf.assign(&values);
		Ok(OutputBuffer(buf))
	}

	pub fn biased_view(&self) -> ArrayView1<f32> {
		self.0.view()
	}

	pub fn view(&self) -> ArrayView1<f32> {
		self.0.slice(s![..-1])
	}

	pub fn view_mut(&mut self) -> ArrayViewMut1<f32> {
		self.0.slice_mut(s![..-1])
	}
}
