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

pub(crate) type SignalView<'a, B> = BufferBase<ViewRepr<&'a f32>, B>;
pub(crate) type SignalViewMut<'a, B> = BufferBase<ViewRepr<&'a mut f32>, B>;

pub(crate) type SignalBuffer<B> = BufferBase<OwnedRepr<f32>, B>;

mod marker {
	pub(crate) trait Biased {
		type Unbiased;
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
	}
	impl Unbiased for UnbiasedSignal {}
	impl Biased for BiasedErrorSignal {
		type Unbiased = UnbiasedErrorSignal;
	}
	impl Unbiased for UnbiasedErrorSignal {}
}

pub(crate) type BiasedSignalView<'a> = SignalView<'a, marker::BiasedSignal>;
pub(crate) type UnbiasedSignalView<'a> = SignalView<'a, marker::UnbiasedSignal>;
pub(crate) type BiasedErrorSignalView<'a> = SignalView<'a, marker::BiasedErrorSignal>;
pub(crate) type UnbiasedErrorSignalView<'a> = SignalView<'a, marker::UnbiasedErrorSignal>;

pub(crate) type BiasedSignalViewMut<'a> = SignalViewMut<'a, marker::BiasedSignal>;
pub(crate) type UnbiasedSignalViewMut<'a> = SignalViewMut<'a, marker::UnbiasedSignal>;
pub(crate) type BiasedErrorSignalViewMut<'a> = SignalViewMut<'a, marker::BiasedErrorSignal>;
pub(crate) type UnbiasedErrorSignalViewMut<'a> = SignalViewMut<'a, marker::UnbiasedErrorSignal>;

pub(crate) type BiasedSignalBuffer = SignalBuffer<marker::BiasedSignal>;
pub(crate) type UnbiasedSignalBuffer = SignalBuffer<marker::UnbiasedSignal>;
pub(crate) type BiasedErrorSignalBuffer = SignalBuffer<marker::BiasedErrorSignal>;
pub(crate) type UnbiasedErrorSignalBuffer = SignalBuffer<marker::UnbiasedErrorSignal>;

pub(crate) trait BiasedAccess<B>
	where B: marker::Biased
{
	fn biased_len(&self) -> usize;
	fn unbiased_len(&self) -> usize;
	fn biased_view(&self) -> SignalView<B>;
	fn unbiased_view(&self) -> SignalView<B::Unbiased>;
	fn data(&self) -> ArrayView1<f32>;
}

pub(crate) trait BiasedAccessMut<B>
	where B: marker::Biased
{
	fn biased_view_mut(&mut self) -> SignalViewMut<B>;
	fn unbiased_view_mut(&mut self) -> SignalViewMut<B::Unbiased>;
	fn data_mut(&mut self) -> ArrayViewMut1<f32>;
}

pub(crate) trait UnbiasedAccess<B> {
	fn unbiased_len(&self) -> usize;
	fn unbiased_view(&self) -> SignalView<B>;
	fn data(&self) -> ArrayView1<f32>;
}

pub(crate) trait UnbiasedAccessMut<B> {
	fn unbiased_view_mut(&mut self) -> SignalViewMut<B>;
	fn data_mut(&mut self) -> ArrayViewMut1<f32>;
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
	fn biased_view(&self) -> SignalView<B> {
		SignalView{
			data: self.data.view(),
			marker: PhantomData
		}
	}

	#[inline]
	fn unbiased_view(&self) -> SignalView<B::Unbiased> {
		SignalView{
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
	fn biased_view_mut(&mut self) -> SignalViewMut<B> {
		SignalViewMut{
			data: self.data.view_mut(),
			marker: PhantomData
		}
	}

	#[inline]
	fn unbiased_view_mut(&mut self) -> SignalViewMut<B::Unbiased> {
		SignalViewMut{
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
	fn unbiased_view(&self) -> SignalView<B> {
		SignalView{
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
	fn unbiased_view_mut(&mut self) -> SignalViewMut<B> {
		SignalViewMut{
			data: self.data.view_mut(),
			marker: PhantomData
		}
	}

	#[inline]
	fn data_mut(&mut self) -> ArrayViewMut1<f32> {
		self.data.view_mut()
	}
}

// pub type BiasedSignalBuffer = BufferBase<marker::Signal>;

// pub type ErrorSignalBuffer = BufferBase<marker::ErrorSignal>;

// impl BiasedSignalBuffer {
// 	/// Creates a new `BiasedSignalBuffer` with a variable length of `len` and an
// 	/// additional constant bias value of `1.0`.
// 	/// 
// 	/// So a call to `BiasedSignalBuffer::zeros(5)` actually constructs a buffer
// 	/// of length `6` with the last value set to `1.0`.
// 	/// 
// 	/// # Errors
// 	/// 
// 	/// Returns an error when trying to create a `BiasedSignalBuffer` with a length of zero.
// 	/// 
// 	pub fn zeros(len: usize) -> Result<BiasedSignalBuffer> {
// 		use std::iter;
// 		if len == 0 {
// 			return Err(Error::zero_sized_signal_buffer())
// 		}
// 		Ok(BiasedSignalBuffer{
// 			data: Array::from_iter(iter::repeat(0.0)
// 				.take(len)
// 				.chain(iter::once(1.0))
// 			),
// 			marker: PhantomData
// 		})
// 	}

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
// 	pub fn zeros(len: usize) -> Result<ErrorSignalBuffer> {
// 		if len == 0 {
// 			return Err(Error::zero_sized_gradient_buffer())
// 		}
// 		Ok(ErrorSignalBuffer{
// 			data: Array1::zeros(len + 1),
// 			marker: PhantomData
// 		})
// 	}

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

// impl<E> BufferBase<E> {
// 	/// Returns the length of this `BiasedSignalBuffer` *without* respect to its bias value.
// 	#[inline]
// 	pub fn len(&self) -> usize {
// 		self.view().dim()
// 	}

// 	/// Returns the length of this `BiasedSignalBuffer` *with* respect to its bias value.
// 	#[inline]
// 	pub fn biased_len(&self) -> usize {
// 		self.biased_view().dim()
// 	}

// 	/// Returns a view to the contents of this `BiasedSignalBuffer` *with* respect to its bias value.
// 	#[inline]
// 	pub fn biased_view(&self) -> ArrayView1<f32> {
// 		self.data.view()
// 	}

// 	/// Returns a view to the contents of this `BiasedSignalBuffer` *without* respect to its bias value.
// 	#[inline]
// 	pub fn view(&self) -> ArrayView1<f32> {
// 		self.data.slice(s![..-1])
// 	}

// 	/// Returns a mutable view to the contents of this `BiasedSignalBuffer` *without* respect to its bias value.
// 	#[inline]
// 	pub fn view_mut(&mut self) -> ArrayViewMut1<f32> {
// 		self.data.slice_mut(s![..-1])
// 	}
// }

// #[derive(Debug, Clone, PartialEq)]
// pub struct UnbiasedSignalView<'a>(ArrayView1<'a, f32>);

// impl<'a, T> From<T> for UnbiasedSignalView<'a>
// 	where T: Into<ArrayView1<'a, f32>>
// {
// 	fn from(view: T) -> Self {
// 		UnbiasedSignalView(view.into())
// 	}
// }

// impl<'a> From<&'a BiasedSignalBuffer> for UnbiasedSignalView<'a> {
// 	fn from(signal_buf: &'a BiasedSignalBuffer) -> Self {
// 		signal_buf.view().into()
// 	}
// }

// impl<'a> UnbiasedSignalView<'a> {
// 	/// Returns the length of this `UnbiasedSignalView`.
// 	#[inline]
// 	pub fn len(&self) -> usize {
// 		self.0.dim()
// 	}

// 	/// Returns a view to the contents of this `UnbiasedSignalView`.
// 	#[inline]
// 	pub fn view(&self) -> ArrayView1<f32> {
// 		self.0.view()
// 	}
// }
