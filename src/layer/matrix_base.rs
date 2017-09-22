
use ndarray::prelude::*;
use ndarray::iter::{Lanes, LanesMut};
use ndarray_rand::RandomExt;
use errors::{Error, Result};
use rand::distributions::Range;
use std::marker::PhantomData;

#[derive(Debug, Clone, PartialEq)]
pub struct MatrixBase<E>{
	data: Array2<f32>,
	marker: PhantomData<E>
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct WeightsMarker;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DeltaWeightsMarker;

pub type WeightsMatrix = MatrixBase<WeightsMarker>;
pub type DeltaWeightsMatrix = MatrixBase<DeltaWeightsMarker>;

impl DeltaWeightsMatrix {
	pub fn zeros(inputs: usize, outputs: usize) -> Result<DeltaWeightsMatrix> {
		if inputs == 0 {
			return Err(Error::zero_inputs_weights_matrix())
		}
		if outputs == 0 {
			return Err(Error::zero_outputs_weights_matrix())
		}
		let biased_inputs = inputs + 1;
		let biased_shape  = (outputs,  biased_inputs);
		let total         =  outputs * biased_inputs;
		Ok(DeltaWeightsMatrix{
			data: Array::zeros(total).into_shape(biased_shape).unwrap(),
			marker: PhantomData
		})
	}
}

impl WeightsMatrix {
	pub fn random(inputs: usize, outputs: usize) -> Result<WeightsMatrix> {
		if inputs == 0 {
			return Err(Error::zero_inputs_weights_matrix())
		}
		if outputs == 0 {
			return Err(Error::zero_outputs_weights_matrix())
		}
		let biased_inputs = inputs + 1;
		let biased_shape  = (outputs, biased_inputs);
		Ok(WeightsMatrix{
			data: Array2::random(biased_shape, Range::new(-1.0, 1.0)),
			marker: PhantomData
		})
	}

	pub fn apply_delta_weights(&mut self, deltas: &DeltaWeightsMatrix) {
		use std::ops::AddAssign;
		self.data.view_mut().add_assign(&deltas.view());
	}
}

impl<E> MatrixBase<E> {
	#[inline]
	pub fn inputs(&self) -> usize {
		self.data.cols() - 1
	}

	#[inline]
	pub fn biased_inputs(&self) -> usize {
		self.data.cols()
	}

	#[inline]
	pub fn outputs(&self) -> usize {
		self.data.rows()
	}

	#[inline]
	pub fn view(&self) -> ArrayView2<f32> {
		self.data.view()
	}

	#[inline]
	pub fn view_mut(&mut self) -> ArrayViewMut2<f32> {
		self.data.view_mut()
	}

	#[inline]
	pub fn genrows(&self) -> Lanes<f32, Ix1> {
		self.data.genrows()
	}

	#[inline]
	pub fn genrows_mut(&mut self) -> LanesMut<f32, Ix1> {
		self.data.genrows_mut()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn zeros_data() {
		let z = DeltaWeightsMatrix::zeros(2, 5).unwrap();
		let e = DeltaWeightsMatrix{
			data: Array::zeros((5, 3)),
			marker: PhantomData
		};
		assert_eq!(z, e);
	}

	#[test]
	fn zeros_failure() {
		assert_eq!(
			DeltaWeightsMatrix::zeros(0, 1),
			Err(Error::zero_inputs_weights_matrix()));
		assert_eq!(
			DeltaWeightsMatrix::zeros(1, 0),
			Err(Error::zero_outputs_weights_matrix()));
		assert_eq!(
			DeltaWeightsMatrix::zeros(0, 0),
			Err(Error::zero_inputs_weights_matrix()));
	}

	#[test]
	fn zeros_sizes() {
		let m = DeltaWeightsMatrix::zeros(2, 5).unwrap();
		assert_eq!(m.inputs(), 2);
		assert_eq!(m.biased_inputs(), 3);
		assert_eq!(m.outputs(), 5);
	}

	#[test]
	fn random_sizes() {
		let r = WeightsMatrix::random(2, 5).unwrap();
		assert_eq!(r.inputs(), 2);
		assert_eq!(r.biased_inputs(), 3);
		assert_eq!(r.outputs(), 5);
	}

	#[test]
	fn random_data() {
		let r = WeightsMatrix::random(2, 5).unwrap();
		let z = WeightsMatrix{
			data: Array::zeros((5, 3)),
			marker: PhantomData
		};
		assert!(r.view().all_close(&z.view(), 1.0));
	}

	#[test]
	fn apply_delta_weights() {
		let mut w = WeightsMatrix{
			data: Array::from_vec(vec![
				 1.0,  2.0,  3.0,
				10.0, 20.0, 30.0
			]).into_shape((2, 3)).unwrap(),
			marker: PhantomData
		};
		let d = DeltaWeightsMatrix{
			data: Array::from_vec(vec![
				0.1, 0.2, 0.3,
				1.0, 2.0, 3.0
			]).into_shape((2, 3)).unwrap(),
			marker: PhantomData
		};
		w.apply_delta_weights(&d);
		let expected_w = WeightsMatrix{
			data: Array::from_vec(vec![
				 1.1,  2.2,  3.3,
				11.0, 22.0, 33.0
			]).into_shape((2, 3)).unwrap(),
			marker: PhantomData
		};
		assert!(w.view().all_close(&expected_w.view(), 1e-10))
	}
}
