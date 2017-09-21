
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
