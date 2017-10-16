use ndarray::prelude::*;
use ndarray;

use errors::{Error, Result};

use std::marker::PhantomData;

mod marker {
	pub trait Marker: Copy + Clone + PartialEq {}

	pub trait Biased: Marker {
		type Unbiased: Marker;
		const DEFAULT_BIAS_VALUE: f32;
	}

	pub trait Unbiased: Marker {}

	#[derive(Debug, Copy, Clone, PartialEq)]
	pub struct BiasedSignal;

	#[derive(Debug, Copy, Clone, PartialEq)]
	pub struct UnbiasedSignal;

	#[derive(Debug, Copy, Clone, PartialEq)]
	pub struct BiasedErrorSignal;

	#[derive(Debug, Copy, Clone, PartialEq)]
	pub struct UnbiasedErrorSignal;

	impl Marker for BiasedSignal {}
	impl Marker for UnbiasedSignal {}
	impl Marker for BiasedErrorSignal {}
	impl Marker for UnbiasedErrorSignal {}

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
}
use self::marker::{
	Marker,
	Biased,
	Unbiased
};

/// A generic 1-dimensional buffer that may represent owned or non-owned content
/// as well as buffers with or without an additional bias neuron value in the context of neural networks.
/// 
/// This may be used as a base type to represent input signals, output signals or error signals.
/// 
/// Note: This basically is just a very thin convenience wrapper around `ndarray`'s 1-dimensional
///       `Array` or `ArrayView`.
/// 
/// Note 2: It has yet to be proven if an abstraction for biased versus non-biased content is useful.
#[derive(Debug)]
pub struct BufferBase<D, B>
	where D: Data,
	      B: Marker
{
	data: ArrayBase<D, Ix1>,
	marker: PhantomData<B>
}

pub trait Data: ndarray::Data<Elem = f32> {}
pub trait DataMut: ndarray::DataMut<Elem = f32> {}

impl<T> Data for T where T: ndarray::Data<Elem = f32> {}
impl<T> DataMut for T where T: ndarray::DataMut<Elem = f32> {}

pub type ViewRepr<'a> = ndarray::ViewRepr<&'a f32>;
pub type ViewMutRepr<'a> = ndarray::ViewRepr<&'a mut f32>;
pub type OwnedRepr = ndarray::OwnedRepr<f32>;

pub type AnyBuffer<B> = BufferBase<OwnedRepr, B>;
pub type AnyView<'a, B> = BufferBase<ViewRepr<'a>, B>;
pub type AnyViewMut<'a, B> = BufferBase<ViewMutRepr<'a>, B>;

pub type BiasedSignalBufferBase<D> = BufferBase<D, marker::BiasedSignal>;
pub type UnbiasedSignalBufferBase<D> = BufferBase<D, marker::UnbiasedSignal>;
pub type BiasedErrorSignalBufferBase<D> = BufferBase<D, marker::BiasedErrorSignal>;
pub type UnbiasedErrorSignalBufferBase<D> = BufferBase<D, marker::UnbiasedErrorSignal>;

pub type BiasedSignalBuffer = BiasedSignalBufferBase<OwnedRepr>;
pub type BiasedSignalView<'a> = BiasedSignalBufferBase<ViewRepr<'a>>;
pub type BiasedSignalViewMut<'a> = BiasedSignalBufferBase<ViewMutRepr<'a>>;

pub type UnbiasedSignalBuffer = UnbiasedSignalBufferBase<OwnedRepr>;
pub type UnbiasedSignalView<'a> = UnbiasedSignalBufferBase<ViewRepr<'a>>;
pub type UnbiasedSignalViewMut<'a> = UnbiasedSignalBufferBase<ViewMutRepr<'a>>;

pub type BiasedErrorSignalBuffer = BiasedErrorSignalBufferBase<OwnedRepr>;
pub type BiasedErrorSignalView<'a> = BiasedErrorSignalBufferBase<ViewRepr<'a>>;
pub type BiasedErrorSignalViewMut<'a> = BiasedErrorSignalBufferBase<ViewMutRepr<'a>>;

pub type Iter<'a> = ndarray::iter::Iter<'a, f32, Ix1>;
pub type IterMut<'a> = ndarray::iter::IterMut<'a, f32, Ix1>;

impl<D, B> PartialEq for BufferBase<D, B>
	where D: Data,
	      B: Marker
{
	fn eq(&self, rhs: &Self) -> bool {
		self.data == rhs.data
	}
}

impl<B> Clone for AnyBuffer<B>
	where B: Marker
{
	fn clone(&self) -> Self {
		Self{
			data: self.data.clone(),
			marker: PhantomData
		}
	}
}

impl<D, B> BufferBase<D, B>
	where D: Data,
	      B: Biased
{
	/// Creates a biased owned `AnyBuffer` from an owned `ndarray::Array` or
	/// a biased non-owning `AnyView` or `AnyViewMut` from a non-owning `ndarray::ArrayView`
	/// or `ndarray::ArrayViewMut` respectively.
	/// 
	/// Note: No `From<...>` is implemented as a fast-fallback to this method since it would create
	///       an assymmetry with `from_raw` for an unbiased version of this method that has
	///       no additional safety checks to check for a user-provided matching bias value during runtime.
	/// 
	/// # Errors
	/// 
	/// - If the given array has a dimension of zero (`0`) or one (`1`).
	/// - If the value of the last element of the given array does not match the expected bias value.
	pub fn from_raw_with_bias<A>(data: A) -> Result<BufferBase<D, B>>
		where A: Into<ArrayBase<D, Ix1>>
	{
		let data = data.into();
		if data.dim() == 0 {
			return Err(Error::attempt_to_create_zero_sized_buffer())
		}
		if data.dim() == 1 {
			return Err(Error::too_few_values_provided_for_buffer_creation(2, data.dim()))
		}
		if data[data.dim() - 1] != B::DEFAULT_BIAS_VALUE {
			return Err(Error::unmatching_user_provided_bias_value(B::DEFAULT_BIAS_VALUE, data[data.dim() - 1]))
		}
		Ok(BufferBase{data, marker: PhantomData})
	}
}

impl<D, B> BufferBase<D, B>
	where D: Data,
	      B: Unbiased
{
	/// Creates an unbiased owned `AnyBuffer` from an owned `ndarray::Array` or 
	/// an unbiased non-owning `AnyView` or `AnyViewMut` from a non-owning `ndarray::ArrayView`
	/// or `ndarray::ArrayViewMut` respectively.
	/// 
	/// Note: No `From<...>` is implemented as a fast-fallback to this method since it would create
	///       an assymmetry with `from_raw_with_bias` for a biased version of this method that has
	///       safety checks to check for a user-provided matching bias value during runtime.
	/// 
	/// # Errors
	/// 
	/// - If the given `data` has a dimension (length) of zero (`0`).
	pub fn from_raw<A>(data: A) -> Result<BufferBase<D, B>>
		where A: Into<ArrayBase<D, Ix1>>
	{
		let data = data.into();
		if data.dim() == 0 {
			return Err(Error::attempt_to_create_zero_sized_buffer())
		}
		Ok(BufferBase{data, marker: PhantomData})
	}
}

impl<D, B> BufferBase<D, B>
	where D: Data,
	      B: Biased
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
	pub fn zeros_with_bias(len: usize) -> Result<AnyBuffer<B>> {
		use std::iter;
		if len == 0 {
			return Err(Error::attempt_to_create_zero_sized_buffer())
		}
		Ok(AnyBuffer{
			data: Array::from_iter(iter::repeat(0.0)
				.take(len)
				.chain(iter::once(B::DEFAULT_BIAS_VALUE))
			),
			marker: PhantomData
		})
	}
}

impl<D, B> BufferBase<D, B>
	where D: Data,
	      B: Unbiased
{
	/// Creates a new unbiased buffer with the given length and all values set to zero (`0`).
	/// 
	/// # Errors
	/// 
	/// - Returns an error upon trying to create a zero-length buffer.
	#[inline]
	pub fn zeros(len: usize) -> Result<UnbiasedSignalBuffer> {
		if len == 0 {
			return Err(Error::attempt_to_create_zero_sized_buffer())
		}
		Ok(UnbiasedSignalBuffer{
			data: Array::zeros(len),
			marker: PhantomData
		})
	}
}

impl<D> BiasedErrorSignalBufferBase<D>
	where D: DataMut
{
	/// Resets all values of this `ErrorSignalBuffer` to zero (`0`).
	#[inline]
	pub fn reset_to_zeros(&mut self) {
		self.data.fill(0.0)
	}
}

impl<D, B> BufferBase<D, B>
	where D: DataMut,
	      B: Unbiased
{
	pub fn assign(&mut self, rhs: &AnyView<B>) -> Result<()> {
		if self.dim() != rhs.dim() {
			return Err(
				Error::unmatching_buffer_sizes(self.dim(), rhs.dim())
					.with_annotation("Occured in unbiased Buffer::assign method.")
			)
		}
		self.data.assign(&rhs.data());
		Ok(())
	}
}

impl<D, B> BufferBase<D, B>
	where D: Data,
	      B: Marker
{
	#[inline]
	pub fn dim(&self) -> usize {
		self.data.dim()
	}

	#[inline]
	pub fn view(&self) -> AnyView<B> {
		AnyView{
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
	where D: DataMut,
	      B: Marker
{
	#[inline]
	pub fn view_mut(&mut self) -> AnyViewMut<B> {
		AnyViewMut{
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
	where D: Data,
	      B: Biased
{
	#[inline]
	pub fn unbias(&self) -> AnyView<B::Unbiased> {
		AnyView{
			data: self.data.slice(s![..-1]),
			marker: PhantomData
		}
	}
}

impl<D, B> BufferBase<D, B>
	where D: DataMut,
	      B: Biased
{
	#[inline]
	pub fn unbias_mut(&mut self) -> AnyViewMut<B::Unbiased> {
		AnyViewMut{
			data: self.data.slice_mut(s![..-1]),
			marker: PhantomData
		}
	}
}

impl<'a, B> AnyView<'a, B>
	where B: Marker
{
	#[inline]
	pub fn into_data(self) -> ArrayView1<'a, f32> {
		self.data
	}
}

impl<'a, B> AnyViewMut<'a, B>
	where B: Marker
{
	#[inline]
	pub fn into_data_mut(self) -> ArrayViewMut1<'a, f32> {
		self.data
	}
}

impl<'a, B> AnyView<'a, B>
	where B: Biased
{
	#[inline]
	pub fn into_unbiased(self) -> AnyView<'a, B::Unbiased> {
		let mut data = self.data;
		data.islice(s![..-1]);
		AnyView{data, marker: PhantomData}
	}
}

impl<'a, B> AnyViewMut<'a, B>
	where B: Biased
{
	#[inline]
	pub fn into_unbiased_mut(self) -> AnyViewMut<'a, B::Unbiased> {
		let mut data = self.data;
		data.islice(s![..-1]);
		AnyViewMut{data, marker: PhantomData}
	}
}

impl<'a, D, B> IntoIterator for &'a BufferBase<D, B>
	where D: Data,
	      B: Marker
{
	type Item = &'a D::Elem;
	type IntoIter = Iter<'a>;

	#[inline]
	fn into_iter(self) -> Self::IntoIter {
		self.iter()
	}
}

impl<'a, D, B> IntoIterator for &'a mut BufferBase<D, B>
	where D: DataMut,
	      B: Marker
{
	type Item = &'a mut D::Elem;
	type IntoIter = IterMut<'a>;

	#[inline]
	fn into_iter(self) -> Self::IntoIter {
		self.iter_mut()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	mod base {
		use super::*;

		#[test]
		#[ignore]
		fn partial_eq() {

		}

		#[test]
		fn from_raw_ok() {
			let vec = vec![1.0, 2.0, 3.0, 4.0];
			assert_eq!(
				AnyBuffer::<marker::UnbiasedSignal>::from_raw(vec.clone()),
				Ok(AnyBuffer{
					data: Array::from_vec(vec.clone()),
					marker: PhantomData
				})
			);
		}

		#[test]
		#[ignore]
		fn from_raw_fail() {
		}

		#[test]
		#[ignore]
		fn len() {
		}

		#[test]
		#[ignore]
		fn view() {
		}

		#[test]
		#[ignore]
		fn view_mut() {
		}

		#[test]
		#[ignore]
		fn iter() {
		}

		#[test]
		#[ignore]
		fn iter_mut() {
		}

		#[test]
		#[ignore]
		fn data() {
		}

		#[test]
		#[ignore]
		fn data_mut() {
		}

		#[test]
		#[ignore]
		fn into_data() {
		}

		#[test]
		#[ignore]
		fn into_data_mut() {
		}

		#[test]
		#[ignore]
		fn into_iter() {
		}
	}

	mod buffer {
		use super::*;

		#[test]
		#[ignore]
		fn clone() {
		}

		mod biased {
			use super::*;

			#[test]
			#[ignore]
			fn zeros_with_bias_ok() {
			}

			#[test]
			#[ignore]
			fn zeros_with_bias_fail() {
			}

			#[test]
			#[ignore]
			fn unbias() {
			}

			#[test]
			#[ignore]
			fn unbias_mut() {
			}

			#[test]
			#[ignore]
			fn into_unbias() {
			}

			#[test]
			#[ignore]
			fn into_unbias_mut() {
			}
		}

		mod unbiased {
			use super::*;

			#[test]
			#[ignore]
			fn zeros_ok() {
			}

			#[test]
			#[ignore]
			fn zeros_fail() {
			}

			#[test]
			#[ignore]
			fn assign_ok() {
			}

			#[test]
			#[ignore]
			fn assign_fail() {
			}
		}
	}

	mod error_signal {
		use super::*;

		#[test]
		#[ignore]
		fn reset_to_zeros() {
		}
	}

}
