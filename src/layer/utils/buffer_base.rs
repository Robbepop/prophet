use ndarray::prelude::*;

use errors::{Error, Result};

use std::marker::PhantomData;

use std::fmt::Debug;
use ndarray::{Data, DataMut, ViewRepr, OwnedRepr};

#[derive(Debug, PartialEq)]
pub(crate) struct BufferBase<D, B>
	where D: Data<Elem = f32>
{
	data: ArrayBase<D, Ix1>,
	marker: PhantomData<B>
}

pub(crate) type BufferView<'a, B> = BufferBase<ViewRepr<&'a f32>, B>;
pub(crate) type BufferViewMut<'a, B> = BufferBase<ViewRepr<&'a mut f32>, B>;
pub(crate) type Buffer<B> = BufferBase<OwnedRepr<f32>, B>;

mod marker {
	pub(crate) trait Biased {
		type Unbiased;
		const DEFAULT_BIAS_VALUE: f32;
	}
	pub(crate) trait Unbiased {}

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
	impl Unbiased for UnbiasedSignal {}
	impl Biased for BiasedErrorSignal {
		type Unbiased = UnbiasedErrorSignal;
		const DEFAULT_BIAS_VALUE: f32 = 0.0;
	}
	impl Unbiased for UnbiasedErrorSignal {}
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

pub(crate) trait BiasedAccess<B>
	where B: marker::Biased
{
	fn biased_len(&self) -> usize;
	fn unbiased_len(&self) -> usize;
	fn biased_view(&self) -> BufferView<B>;
	fn unbiased_view(&self) -> BufferView<B::Unbiased>;
	fn data(&self) -> ArrayView1<f32>;
}

pub(crate) trait BiasedAccessMut<B>: BiasedAccess<B>
	where B: marker::Biased
{
	fn biased_view_mut(&mut self) -> BufferViewMut<B>;
	fn unbiased_view_mut(&mut self) -> BufferViewMut<B::Unbiased>;
	fn data_mut(&mut self) -> ArrayViewMut1<f32>;
}

pub(crate) trait UnbiasedAccess<B>
	where B: marker::Unbiased
{
	fn unbiased_len(&self) -> usize;
	fn unbiased_view(&self) -> BufferView<B>;
	fn data(&self) -> ArrayView1<f32>;
}

pub(crate) trait UnbiasedAccessMut<B>: UnbiasedAccess<B>
	where B: marker::Unbiased
{
	fn unbiased_view_mut(&mut self) -> BufferViewMut<B>;
	fn data_mut(&mut self) -> ArrayViewMut1<f32>;
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
	fn zeros_with_bias(len: usize) -> Result<Buffer<B>> {
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
	fn zeros(len: usize) -> Result<Buffer<B>> {
		if len == 0 {
			return Err(Error::zero_sized_signal_buffer())
		}
		Ok(Buffer{
			data: Array::zeros(len),
			marker: PhantomData
		})
	}
}

impl<D, B> BiasedAccess<B> for BufferBase<D, B>
	where D: Data<Elem = f32>,
	      B: marker::Biased
{
	#[inline]
	fn biased_len(&self) -> usize {
		self.data.dim()
	}

	#[inline]
	fn unbiased_len(&self) -> usize {
		self.data.dim() - 1
	}

	#[inline]
	fn biased_view(&self) -> BufferView<B> {
		BufferView{
			data: self.data.view(),
			marker: PhantomData
		}
	}

	#[inline]
	fn unbiased_view(&self) -> BufferView<B::Unbiased> {
		BufferView{
			data: self.data.slice(s![..-1]),
			marker: PhantomData
		}
	}

	#[inline]
	fn data(&self) -> ArrayView1<f32> {
		self.data.view()
	}
}

impl<D, B> BiasedAccessMut<B> for BufferBase<D, B>
	where D: DataMut<Elem = f32>,
	      B: marker::Biased
{
	#[inline]
	fn biased_view_mut(&mut self) -> BufferViewMut<B> {
		BufferViewMut{
			data: self.data.view_mut(),
			marker: PhantomData
		}
	}

	#[inline]
	fn unbiased_view_mut(&mut self) -> BufferViewMut<B::Unbiased> {
		BufferViewMut{
			data: self.data.slice_mut(s![..-1]),
			marker: PhantomData
		}
	}

	#[inline]
	fn data_mut(&mut self) -> ArrayViewMut1<f32> {
		self.data.view_mut()
	}
}

impl<D, B> UnbiasedAccess<B> for BufferBase<D, B>
	where D: Data<Elem = f32>,
	      B: marker::Unbiased
{
	#[inline]
	fn unbiased_len(&self) -> usize {
		self.data.dim()
	}

	#[inline]
	fn unbiased_view(&self) -> BufferView<B> {
		BufferView{
			data: self.data.view(),
			marker: PhantomData
		}
	}

	#[inline]
	fn data(&self) -> ArrayView1<f32> {
		self.data.view()
	}
}

impl<D, B> UnbiasedAccessMut<B> for BufferBase<D, B>
	where D: DataMut<Elem = f32>,
	      B: marker::Unbiased
{
	#[inline]
	fn unbiased_view_mut(&mut self) -> BufferViewMut<B> {
		BufferViewMut{
			data: self.data.view_mut(),
			marker: PhantomData
		}
	}

	#[inline]
	fn data_mut(&mut self) -> ArrayViewMut1<f32> {
		self.data.view_mut()
	}
}

// impl BiasedSignalBuffer {
// 	/// Creates a new `BiasedSignalBuffer` with its variable signals set
// 	/// to the given `input` and the last signal set to `1.0`.
// 	/// 
// 	/// # Errors
// 	/// 
// 	/// Returns an error when the length of `input` is zero.
// 	///  
// 	pub fn with_values(input: ArrayView1<f32>) -> Result<BiasedSignalBuffer> {
// 		if input.dim() == 0 {
// 			return Err(Error::zero_sized_signal_buffer())
// 		}
// 		let mut buf = BiasedSignalBuffer::zeros(input.dim())?;
// 		buf.assign(input)?;
// 		Ok(buf)
// 	}

// 	/// Creates a new `BiasedSignalBuffer` with its variable signals set
// 	/// to the first `len` values of the given `source` iterator.
// 	/// 
// 	/// # Errors
// 	/// 
// 	/// Returns an error ...
// 	/// 
// 	/// - if `source` generates less items than required by the given `len`.
// 	/// 
// 	/// - if `len` is equal to zero.
// 	pub fn from_iter<I>(len: usize, source: I) -> Result<BiasedSignalBuffer>
// 		where I: Iterator<Item=f32>
// 	{
// 		use std::iter;
// 		if len == 0 {
// 			return Err(Error::zero_sized_signal_buffer())
// 		}
// 		let result = BiasedSignalBuffer{
// 			data: Array::from_iter(source
// 				.take(len)
// 				.chain(iter::once(1.0))),
// 			marker: PhantomData
// 		};
// 		if result.len() != len {
// 			return Err(Error::non_matching_number_of_signals(result.len(), len))
// 		}
// 		Ok(result)
// 	}

// 	/// Assigns the non-bias contents of this `BiasedSignalBuffer` to the contents
// 	/// of the given `values`.
// 	/// 
// 	/// # Errors
// 	/// 
// 	/// Returns an error if the length `values` does not match the length
// 	/// of this buffer without respect to its bias value.
// 	///  
// 	pub fn assign<'a, T>(&mut self, values: T) -> Result<()>
// 		where T: Into<UnbiasedSignalView<'a>>
// 	{
// 		let values = values.into();
// 		if self.len() != values.len() {
// 			return Err(Error::non_matching_assign_signals(values.len(), self.len()))
// 		}
// 		Ok(self.data.assign(&values.view()))
// 	}
// }

// impl ErrorSignalBuffer {
// 	pub fn with_values<'a, T>(input: T) -> Result<ErrorSignalBuffer>
// 		where T: Into<UnbiasedSignalView<'a>>
// 	{
// 		let input = input.into();
// 		let mut buf = ErrorSignalBuffer::zeros(input.len())?;
// 		buf.data.assign(&input.view());
// 		Ok(buf)
// 	}

// 	#[inline]
// 	pub fn reset_to_zeros(&mut self) {
// 		self.data.fill(0.0)
// 	}
// }
