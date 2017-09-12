
use ndarray::prelude::*;
use errors::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct GradientBuffer(Array1<f32>);

impl GradientBuffer {
	pub fn zeros(len: usize) -> Result<GradientBuffer> {
		if len == 0 {
			return Err(Error::zero_sized_gradient_buffer())
		}
		Ok(GradientBuffer(Array1::zeros(len)))
	}

	#[inline]
	pub fn reset_to_zeros(&mut self) {
		self.0.fill(0.0)
	}

	#[inline]
	pub fn view(&self) -> ArrayView1<f32> {
		self.0.view()
	}

	#[inline]
	pub fn view_mut(&mut self) -> ArrayViewMut1<f32> {
		self.0.view_mut()
	}
}
