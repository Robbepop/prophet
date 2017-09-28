use ndarray::prelude::*;
use ndarray;

use errors::{Error, Result};

use std::marker::PhantomData;

#[derive(Debug)]
pub(crate) struct BufferBase<D, B>
	where D: ndarray::Data<Elem = f32>,
	      B: marker::Marker
{
	data: ArrayBase<D, Ix1>,
	marker: PhantomData<B>
}

pub(crate) type BufferView<'a, B> = BufferBase<ndarray::ViewRepr<&'a f32>, B>;
pub(crate) type BufferViewMut<'a, B> = BufferBase<ndarray::ViewRepr<&'a mut f32>, B>;
pub(crate) type Buffer<B> = BufferBase<ndarray::OwnedRepr<f32>, B>;

mod marker {
	pub(crate) trait Marker: Copy + Clone + PartialEq {}

	pub(crate) trait Biased: Marker {
		type Unbiased: Marker;
		const DEFAULT_BIAS_VALUE: f32;
	}
	pub(crate) trait Unbiased: Marker {}

	#[derive(Debug, Copy, Clone, PartialEq)]
	pub struct BiasedSignal;

	#[derive(Debug, Copy, Clone, PartialEq)]
	pub struct UnbiasedSignal;

	#[derive(Debug, Copy, Clone, PartialEq)]
	pub struct BiasedErrorSignal;

	#[derive(Debug, Copy, Clone, PartialEq)]
	pub struct UnbiasedErrorSignal;

	impl Biased for BiasedSignal {
		type Unbiased = UnbiasedSignal;
		const DEFAULT_BIAS_VALUE: f32 = 1.0;
	}

	impl Biased for BiasedErrorSignal {
		type Unbiased = UnbiasedErrorSignal;
		const DEFAULT_BIAS_VALUE: f32 = 0.0;
	}

	impl Unbiased for UnbiasedSignal {}
	impl Unbiased for UnbiasedErrorSignal {}

	impl Marker for BiasedSignal {}
	impl Marker for UnbiasedSignal {}
	impl Marker for BiasedErrorSignal {}
	impl Marker for UnbiasedErrorSignal {}
}

pub(crate) type BiasedSignalView<'a> = BufferView<'a, marker::BiasedSignal>;
pub(crate) type UnbiasedSignalView<'a> = BufferView<'a, marker::UnbiasedSignal>;
pub(crate) type BiasedErrorSignalView<'a> = BufferView<'a, marker::BiasedErrorSignal>;
pub(crate) type UnbiasedErrorSignalView<'a> = BufferView<'a, marker::UnbiasedErrorSignal>;

pub(crate) type BiasedSignalViewMut<'a> = BufferViewMut<'a, marker::BiasedSignal>;
pub(crate) type UnbiasedSignalViewMut<'a> = BufferViewMut<'a, marker::UnbiasedSignal>;
pub(crate) type BiasedErrorSignalViewMut<'a> = BufferViewMut<'a, marker::BiasedErrorSignal>;
pub(crate) type UnbiasedErrorSignalViewMut<'a> = BufferViewMut<'a, marker::UnbiasedErrorSignal>;

pub(crate) type BiasedSignalBuffer = Buffer<marker::BiasedSignal>;
pub(crate) type UnbiasedSignalBuffer = Buffer<marker::UnbiasedSignal>;
pub(crate) type BiasedErrorSignalBuffer = Buffer<marker::BiasedErrorSignal>;
pub(crate) type UnbiasedErrorSignalBuffer = Buffer<marker::UnbiasedErrorSignal>;

pub(crate) type Iter<'a> = ndarray::iter::Iter<'a, f32, Ix1>;
pub(crate) type IterMut<'a> = ndarray::iter::IterMut<'a, f32, Ix1>;

impl<D, B> PartialEq for BufferBase<D, B>
	where D: ndarray::Data<Elem = f32>,
	      B: marker::Marker
{
	fn eq(&self, rhs: &Self) -> bool {
		self.data == rhs.data
	}
}

impl<B> Clone for Buffer<B>
	where B: marker::Marker
{
	fn clone(&self) -> Buffer<B> {
		Buffer{
			data: self.data.clone(),
			marker: PhantomData
		}
	}
}

impl<B> Buffer<B>
	where B: marker::Marker
{
	#[inline]
	pub fn from_raw_parts(data: Array1<f32>) -> Result<Buffer<B>> {
		if data.dim() == 0 {
			return Err(Error::zero_sized_signal_buffer())
		}
		Ok(Buffer{data, marker: PhantomData})
	}
}

impl<B> Buffer<B>
	where B: marker::Biased
{
	/// Creates a new biased `SignalBuffer` with a variable length of `len` and an
	/// additional bias value.
	/// 
	/// So a call to `BiasedSignalBuffer::zeros(5)` actually constructs a buffer
	/// of length `6` with the last value set to `1.0`.
	/// 
	/// # Errors
	/// 
	/// Returns an error when trying to create a `BiasedSignalBuffer` with a length of zero.
	/// 
	#[inline]
	pub fn zeros_with_bias(len: usize) -> Result<Buffer<B>> {
		use std::iter;
		if len == 0 {
			return Err(Error::zero_sized_signal_buffer())
		}
		Ok(Buffer{
			data: Array::from_iter(iter::repeat(0.0)
				.take(len)
				.chain(iter::once(B::DEFAULT_BIAS_VALUE))
			),
			marker: PhantomData
		})
	}
}

impl<B> Buffer<B>
	where B: marker::Unbiased
{
	#[inline]
	pub fn zeros(len: usize) -> Result<Buffer<B>> {
		if len == 0 {
			return Err(Error::zero_sized_signal_buffer())
		}
		Ok(Buffer{
			data: Array::zeros(len),
			marker: PhantomData
		})
	}

	#[inline]
	pub fn reset_to_zeros(&mut self) {
		self.data.fill(0.0)
	}

	pub fn assign(&mut self, rhs: &BufferView<B>) -> Result<()> {
		if self.len() != rhs.len() {
			return Err(
				Error::unmatching_buffer_sizes(self.len(), rhs.len())
					.with_annotation("Occured in unbiased Buffer::assign method.")
			)
		}
		self.data.assign(&rhs.data());
		Ok(())
	}
}

impl<D, B> BufferBase<D, B>
	where D: ndarray::Data<Elem = f32>,
	      B: marker::Marker
{
	#[inline]
	pub fn len(&self) -> usize {
		self.data.dim()
	}

	#[inline]
	pub fn view(&self) -> BufferView<B> {
		BufferView{
			data: self.data.view(),
			marker: PhantomData
		}
	}

	#[inline]
	pub fn iter(&self) -> Iter {
		self.data.iter()
	}

	#[inline]
	pub fn data(&self) -> ArrayView1<f32> {
		self.data.view()
	}
}

impl<D, B> BufferBase<D, B>
	where D: ndarray::Data<Elem = f32>,
	      B: marker::Biased,
{
	#[inline]
	pub fn unbias(&self) -> BufferView<B::Unbiased> {
		BufferView{
			data: self.data.slice(s![..-1]),
			marker: PhantomData
		}
	}
}

impl<D, B> BufferBase<D, B>
	where D: ndarray::DataMut<Elem = f32>,
	      B: marker::Marker
{
	#[inline]
	pub fn view_mut(&mut self) -> BufferViewMut<B> {
		BufferViewMut{
			data: self.data.view_mut(),
			marker: PhantomData
		}
	}

	#[inline]
	pub fn iter_mut(&mut self) -> IterMut {
		self.data.iter_mut()
	}

	#[inline]
	pub fn data_mut(&mut self) -> ArrayViewMut1<f32> {
		self.data.view_mut()
	}
}

impl<D, B> BufferBase<D, B>
	where D: ndarray::DataMut<Elem = f32>,
	      B: marker::Biased
{
	#[inline]
	pub fn unbias_mut(&mut self) -> BufferViewMut<B::Unbiased> {
		BufferViewMut{
			data: self.data.slice_mut(s![..-1]),
			marker: PhantomData
		}
	}
}

impl<'a, D, B> IntoIterator for &'a BufferBase<D, B>
	where D: ndarray::Data<Elem = f32>,
	      B: marker::Marker
{
	type Item = &'a D::Elem;
	type IntoIter = Iter<'a>;

	#[inline]
	fn into_iter(self) -> Self::IntoIter {
		self.iter()
	}
}

impl<'a, D, B> IntoIterator for &'a mut BufferBase<D, B>
	where D: ndarray::DataMut<Elem = f32>,
	      B: marker::Marker
{
	type Item = &'a mut D::Elem;
	type IntoIter = IterMut<'a>;

	#[inline]
	fn into_iter(self) -> Self::IntoIter {
		self.iter_mut()
	}
}
