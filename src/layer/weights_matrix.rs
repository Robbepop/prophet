
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use errors::{Error, Result};
use rand::distributions::Range;

#[derive(Debug, Clone, PartialEq)]
pub struct WeightsMatrix(Array2<f32>);

impl WeightsMatrix {
	pub fn zeros(inputs: usize, outputs: usize) -> Result<WeightsMatrix> {
		if inputs == 0 {
			return Err(Error::zero_inputs_weights_matrix())
		}
		if outputs == 0 {
			return Err(Error::zero_outputs_weights_matrix())
		}
		let biased_inputs = inputs + 1;
		let biased_shape  = (outputs,  biased_inputs);
		let total         =  outputs * biased_inputs;
		Ok(WeightsMatrix(
			Array::zeros(total).into_shape(biased_shape).unwrap()
		))
	}

	pub fn random(inputs: usize, outputs: usize) -> Result<WeightsMatrix> {
		if inputs == 0 {
			return Err(Error::zero_inputs_weights_matrix())
		}
		if outputs == 0 {
			return Err(Error::zero_outputs_weights_matrix())
		}
		let biased_inputs = inputs + 1;
		let biased_shape  = (outputs, biased_inputs);
		Ok(WeightsMatrix(
			Array2::random(biased_shape, Range::new(-1.0, 1.0))
		))
	}

	#[inline]
	pub fn inputs(&self) -> usize {
		self.0.cols()
	}

	#[inline]
	pub fn outputs(&self) -> usize {
		self.0.rows()
	}

	#[inline]
	pub fn view(&self) -> ArrayView2<f32> {
		self.0.view()
	}

	#[inline]
	pub fn view_mut(&mut self) -> ArrayViewMut2<f32> {
		self.0.view_mut()
	}
}
