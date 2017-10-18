use ndarray::prelude::*;
use ndarray;

use errors::{Error, Result};

use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;

mod marker {
	pub trait Marker: Copy + Clone + PartialEq {}

	pub trait Biased: Marker {
		type Unbiased: Unbiased;
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
// pub type UnbiasedErrorSignalBufferBase<D> = BufferBase<D, marker::UnbiasedErrorSignal>; // Never used!

pub type BiasedSignalBuffer = BiasedSignalBufferBase<OwnedRepr>;
pub type BiasedSignalView<'a> = BiasedSignalBufferBase<ViewRepr<'a>>;
pub type BiasedSignalViewMut<'a> = BiasedSignalBufferBase<ViewMutRepr<'a>>;

pub type UnbiasedSignalBuffer = UnbiasedSignalBufferBase<OwnedRepr>;
pub type UnbiasedSignalView<'a> = UnbiasedSignalBufferBase<ViewRepr<'a>>;
// pub type UnbiasedSignalViewMut<'a> = UnbiasedSignalBufferBase<ViewMutRepr<'a>>; // Never used!

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

impl<D, B> Debug for BufferBase<D, B>
	where D: Data,
	      B: Marker
{
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BufferBase")
            .field("data", &self.data)
            .field("marker", &self.marker)
            .finish()
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
		fn partial_eq_true() {
			fn assert_for_marker<B: Unbiased>() {
				let two_vals: Vec<f32> = vec![1.0, 2.0];
				let three_vals: Vec<f32> = vec![1.0, 2.0, 3.0];
				let three = AnyBuffer::<B>::from_raw(three_vals.clone()).unwrap();
				let same_as_three = AnyBuffer::<B>::from_raw(three_vals.clone()).unwrap();
				let just_two = AnyBuffer::<B>::from_raw(two_vals.clone()).unwrap();
				let same_as_just_two = AnyBuffer::<B>::from_raw(two_vals.clone()).unwrap();
				assert_eq!(three, same_as_three);
				assert_eq!(same_as_three, three);
				assert_eq!(just_two, same_as_just_two);
				assert_eq!(same_as_just_two, just_two);
				assert_ne!(three, just_two);
				assert_ne!(just_two, three);
			}
			assert_for_marker::<marker::UnbiasedSignal>();
			assert_for_marker::<marker::UnbiasedErrorSignal>();
		}

		#[test]
		#[ignore]
		fn clone() {
		}

		#[test]
		fn dim() {
			fn assert_for_marker<B: Marker>(len: usize) {
				assert_ne!(len, 0); // Test invariant.
				{
					let buf = AnyBuffer::<B>{
						data: Array::zeros(len),
						marker: PhantomData
					};
					assert_eq!(buf.dim(), len);
				}
				{
					let arr = Array::zeros(len);
					let buf = AnyView::<B>{
						data: arr.view(),
						marker: PhantomData
					};
					assert_eq!(buf.dim(), len);
				}
				{
					let mut arr = Array::zeros(len);
					let buf = AnyViewMut::<B>{
						data: arr.view_mut(),
						marker: PhantomData
					};
					assert_eq!(buf.dim(), len);
				}
			}
			for n in 1..10 {
				assert_for_marker::<marker::BiasedSignal>(n);
				assert_for_marker::<marker::BiasedErrorSignal>(n);
				assert_for_marker::<marker::UnbiasedSignal>(n);
				assert_for_marker::<marker::UnbiasedErrorSignal>(n);
			}
		}

		#[test]
		fn view() {
			fn assert_for_biased() {
				fn assert_for_marker<B: Biased>() {
					fn assert_for_buffer<B: Biased>(arr_with_bias: Array1<f32>) {
						let buf = AnyBuffer::<B>::from_raw_with_bias(arr_with_bias.clone()).unwrap();
						assert_eq!(
							buf.view(),
							AnyView::<B>{
								data: arr_with_bias.view(),
								marker: PhantomData
							}
						);
					}
					fn assert_for_view<B: Biased>(arr_with_bias: Array1<f32>) {
						let view = AnyView::<B>::from_raw_with_bias(&arr_with_bias).unwrap();
						assert_eq!(
							view.view(),
							AnyView::<B>{
								data: arr_with_bias.view(),
								marker: PhantomData
							}
						);
					}
					fn assert_for_view_mut<B: Biased>(arr_with_bias: Array1<f32>) {
						let mut cloned_arr_with_bias = arr_with_bias.clone();
						let view_mut = AnyViewMut::<B>::from_raw_with_bias(&mut cloned_arr_with_bias).unwrap();
						assert_eq!(
							view_mut.view(),
							AnyView::<B>{
								data: arr_with_bias.view(),
								marker: PhantomData
							}
						);
					}
					let arr_with_bias = Array1::<f32>::from_vec(
						vec![1.0, 2.0, B::DEFAULT_BIAS_VALUE]);
					assert_for_buffer::<B>(arr_with_bias.clone());
					assert_for_view::<B>(arr_with_bias.clone());
					assert_for_view_mut::<B>(arr_with_bias.clone());
				}
				assert_for_marker::<marker::BiasedSignal>();
				assert_for_marker::<marker::BiasedErrorSignal>();
			}
			fn assert_for_unbiased() {
				fn assert_for_marker<B: Unbiased>() {
					fn assert_for_buffer<B: Unbiased>(arr: Array1<f32>) {
						let buf = AnyBuffer::<B>::from_raw(arr.clone()).unwrap();
						assert_eq!(
							buf.view(),
							AnyView::<B>{
								data: arr.view(),
								marker: PhantomData
							}
						);
					}
					fn assert_for_view<B: Unbiased>(arr: Array1<f32>) {
						let view = AnyView::<B>::from_raw(&arr).unwrap();
						assert_eq!(
							view.view(),
							AnyView::<B>{
								data: arr.view(),
								marker: PhantomData
							}
						);
					}
					fn assert_for_view_mut<B: Unbiased>(arr: Array1<f32>) {
						let mut cloned_arr = arr.clone();
						let view_mut = AnyViewMut::<B>::from_raw(&mut cloned_arr).unwrap();
						assert_eq!(
							view_mut.view(),
							AnyView::<B>{
								data: arr.view(),
								marker: PhantomData
							}
						);
					}
					let arr = Array1::<f32>::from_vec(
						vec![1.0, 2.0, 3.0]);
					assert_for_buffer::<B>(arr.clone());
					assert_for_view::<B>(arr.clone());
					assert_for_view_mut::<B>(arr.clone());
				}
				assert_for_marker::<marker::UnbiasedSignal>();
				assert_for_marker::<marker::UnbiasedErrorSignal>();
			}
			assert_for_biased();
			assert_for_unbiased();
		}

		#[test]
		fn view_mut() {
			fn assert_for_biased() {
				fn assert_for_marker<B: Biased>() {
					fn assert_for_buffer<B: Biased>(arr_with_bias: Array1<f32>) {
						let mut arr_with_bias = arr_with_bias;
						let mut buf = AnyBuffer::<B>::from_raw_with_bias(arr_with_bias.clone()).unwrap();
						assert_eq!(
							buf.view_mut(),
							AnyViewMut::<B>{
								data: arr_with_bias.view_mut(),
								marker: PhantomData
							}
						);
					}
					fn assert_for_view_mut<B: Biased>(arr_with_bias: Array1<f32>) {
						let mut arr_with_bias = arr_with_bias;
						let mut cloned_arr_with_bias = arr_with_bias.clone();
						let mut view_mut = AnyViewMut::<B>::from_raw_with_bias(&mut cloned_arr_with_bias).unwrap();
						assert_eq!(
							view_mut.view_mut(),
							AnyViewMut::<B>{
								data: arr_with_bias.view_mut(),
								marker: PhantomData
							}
						);
					}
					let arr_with_bias = Array1::<f32>::from_vec(
						vec![1.0, 2.0, B::DEFAULT_BIAS_VALUE]);
					assert_for_buffer::<B>(arr_with_bias.clone());
					assert_for_view_mut::<B>(arr_with_bias.clone());
				}
				assert_for_marker::<marker::BiasedSignal>();
				assert_for_marker::<marker::BiasedErrorSignal>();
			}
			fn assert_for_unbiased() {
				fn assert_for_marker<B: Unbiased>() {
					fn assert_for_buffer<B: Unbiased>(arr: Array1<f32>) {
						let mut arr = arr;
						let mut buf = AnyBuffer::<B>::from_raw(arr.clone()).unwrap();
						assert_eq!(
							buf.view_mut(),
							AnyViewMut::<B>{
								data: arr.view_mut(),
								marker: PhantomData
							}
						);
					}
					fn assert_for_view_mut<B: Unbiased>(arr: Array1<f32>) {
						let mut arr = arr;
						let mut cloned_arr = arr.clone();
						let mut view_mut = AnyViewMut::<B>::from_raw(&mut cloned_arr).unwrap();
						assert_eq!(
							view_mut.view_mut(),
							AnyViewMut::<B>{
								data: arr.view_mut(),
								marker: PhantomData
							}
						);
					}
					let arr = Array1::<f32>::from_vec(
						vec![1.0, 2.0, 3.0]);
					assert_for_buffer::<B>(arr.clone());
					assert_for_view_mut::<B>(arr.clone());
				}
				assert_for_marker::<marker::UnbiasedSignal>();
				assert_for_marker::<marker::UnbiasedErrorSignal>();
			}
			assert_for_biased();
			assert_for_unbiased();
		}

		#[test]
		fn iter() {
			fn assert_for_biased<B: Biased>() {
				let buf = AnyBuffer::<B>::from_raw_with_bias(
					vec![1.0, 2.0, 42.0, 13.37, B::DEFAULT_BIAS_VALUE]).unwrap();
				let mut iter = buf.iter();
				assert_eq!(iter.next(), Some(&1.0));
				assert_eq!(iter.next(), Some(&2.0));
				assert_eq!(iter.next(), Some(&42.0));
				assert_eq!(iter.next(), Some(&13.37));
				assert_eq!(iter.next(), Some(&B::DEFAULT_BIAS_VALUE));
				assert_eq!(iter.next(), None);
			}
			fn assert_for_unbiased<B: Unbiased>() {
				let buf = AnyBuffer::<B>::from_raw(
					vec![1.0, 2.0, 42.0, 13.37, -1.0]).unwrap();
				let mut iter = buf.iter();
				assert_eq!(iter.next(), Some(&1.0));
				assert_eq!(iter.next(), Some(&2.0));
				assert_eq!(iter.next(), Some(&42.0));
				assert_eq!(iter.next(), Some(&13.37));
				assert_eq!(iter.next(), Some(&-1.0));
				assert_eq!(iter.next(), None);
			}
			assert_for_biased::<marker::BiasedSignal>();
			assert_for_biased::<marker::BiasedErrorSignal>();
			assert_for_unbiased::<marker::UnbiasedSignal>();
			assert_for_unbiased::<marker::UnbiasedErrorSignal>();
		}

		#[test]
		#[ignore]
		fn iter_mut() {
		}

		#[test]
		#[ignore]
		fn into_iter() {
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
	}

	mod biased {
		use super::*;

		#[test]
		fn zeros_with_bias_ok() {
			fn assert_for_marker_and_len<B: Biased>(len: usize) {
				use std::iter;
				assert!(len != 0); // This test should only check for valid results!
				assert_eq!(
					AnyBuffer::<B>::zeros_with_bias(len),
					Ok(AnyBuffer{
						data: Array::from_iter(
							iter::repeat(0.0).take(len).chain(iter::once(B::DEFAULT_BIAS_VALUE))),
						marker: PhantomData
					})
				)
			}
			for n in 1..10 {
				assert_for_marker_and_len::<marker::BiasedSignal>(n);
				assert_for_marker_and_len::<marker::BiasedErrorSignal>(n);
			}
		}

		#[test]
		fn zeros_with_bias_fail() {
			fn assert_for_marker_and_len<B: Biased>() {
				assert_eq!(
					AnyBuffer::<B>::zeros_with_bias(0),
					Err(Error::attempt_to_create_zero_sized_buffer())
				)
			}
			assert_for_marker_and_len::<marker::BiasedSignal>();
			assert_for_marker_and_len::<marker::BiasedErrorSignal>();
		}

		#[test]
		fn from_raw_with_bias_ok() {
			fn assert_for_marker<B: Biased>() {
				{
					let vec = vec![1.0, 2.0, B::DEFAULT_BIAS_VALUE];
					let actual   = AnyBuffer::<B>::from_raw_with_bias(vec.clone());
					let expected = Ok(AnyBuffer{
						data: Array::from_vec(vec.clone()),
						marker: PhantomData
					});
					assert_eq!(actual, expected);
				}{
					let slice = &[1.0, 2.0, B::DEFAULT_BIAS_VALUE];
					let actual   = AnyView::<B>::from_raw_with_bias(slice);
					let expected = Ok(AnyView{
						data: slice.into(),
						marker: PhantomData
					});
					assert_eq!(actual, expected);
				}{
					let actual_vals = &mut [1.0, 2.0, B::DEFAULT_BIAS_VALUE];
					// Requires different slices due to borrow checker!
					let expected_vals = &mut [1.0, 2.0, B::DEFAULT_BIAS_VALUE];
					let actual   = AnyViewMut::<B>::from_raw_with_bias(actual_vals);
					let expected = Ok(AnyViewMut{
						data: expected_vals.into(),
						marker: PhantomData
					});
					assert_eq!(actual, expected);
				}
			}
			assert_for_marker::<marker::BiasedSignal>();
			assert_for_marker::<marker::BiasedErrorSignal>();
		}

		#[test]
		fn from_raw_with_bias_fail() {
			fn assert_for_marker<B: Biased>() {
				{
					let actual   = AnyBuffer::<B>::from_raw_with_bias(vec![]);
					let expected = Err(Error::attempt_to_create_zero_sized_buffer());
					assert_eq!(actual, expected);
				}
				{
					let actual   = AnyBuffer::<B>::from_raw_with_bias(vec![42.0]);
					let expected = Err(Error::too_few_values_provided_for_buffer_creation(2, 1));
					assert_eq!(actual, expected);
				}
				{
					let actual   = AnyBuffer::<B>::from_raw_with_bias(vec![1.0, 42.0]);
					let expected = Err(Error::unmatching_user_provided_bias_value(B::DEFAULT_BIAS_VALUE, 42.0));
					assert_eq!(actual, expected);
				}
			}
			assert_for_marker::<marker::BiasedSignal>();
			assert_for_marker::<marker::BiasedErrorSignal>();
		}

		#[test]
		fn unbias() {
			fn assert_for_marker<B: Biased>() {
				let biased_values = vec![42.0, 1337.0, B::DEFAULT_BIAS_VALUE];
				let unbiased_values = vec![42.0, 1337.0];
				let biased_buf = AnyBuffer::<B>::from_raw_with_bias(biased_values).unwrap();
				let unbiased_buf = AnyView::<B::Unbiased>::from_raw(&unbiased_values).unwrap();
				assert_eq!(
					biased_buf.unbias(),
					unbiased_buf
				);
			}
			assert_for_marker::<marker::BiasedSignal>();
			assert_for_marker::<marker::BiasedErrorSignal>();
		}

		#[test]
		fn unbias_mut() {
			fn assert_for_marker<B: Biased>() {
				let biased_values = vec![42.0, 1337.0, B::DEFAULT_BIAS_VALUE];
				let mut unbiased_values = vec![42.0, 1337.0];
				let mut biased_buf = AnyBuffer::<B>::from_raw_with_bias(biased_values).unwrap();
				let unbiased_buf = AnyViewMut::<B::Unbiased>::from_raw(&mut unbiased_values).unwrap();
				assert_eq!(
					biased_buf.unbias_mut(),
					unbiased_buf
				);
			}
			assert_for_marker::<marker::BiasedSignal>();
			assert_for_marker::<marker::BiasedErrorSignal>();
		}

		#[test]
		fn into_unbias() {
			fn assert_for_marker<B: Biased>() {
				let biased_values = vec![42.0, 1337.0, B::DEFAULT_BIAS_VALUE];
				let unbiased_values = vec![42.0, 1337.0];
				let biased_view = AnyView::<B>::from_raw_with_bias(&biased_values).unwrap();
				let unbiased_view = AnyView::<B::Unbiased>::from_raw(&unbiased_values).unwrap();
				assert_eq!(
					biased_view.into_unbiased(),
					unbiased_view
				);
			}
			assert_for_marker::<marker::BiasedSignal>();
			assert_for_marker::<marker::BiasedErrorSignal>();
		}

		#[test]
		fn into_unbias_mut() {
			fn assert_for_marker<B: Biased>() {
				let mut biased_values = vec![42.0, 1337.0, B::DEFAULT_BIAS_VALUE];
				let mut unbiased_values = vec![42.0, 1337.0];
				let biased_view = AnyViewMut::<B>::from_raw_with_bias(
					&mut biased_values).unwrap();
				let unbiased_view = AnyViewMut::<B::Unbiased>::from_raw(
					&mut unbiased_values).unwrap();
				assert_eq!(
					biased_view.into_unbiased_mut(),
					unbiased_view
				);
			}
			assert_for_marker::<marker::BiasedSignal>();
			assert_for_marker::<marker::BiasedErrorSignal>();
		}
	}

	mod unbiased {
		use super::*;

		#[test]
		fn zeros_ok() {
			fn assert_for_marker_and_len<B: Unbiased>(len: usize) {
				assert!(len != 0); // This test should only check for valid results!
				assert_eq!(
					AnyBuffer::<B>::zeros(len),
					Ok(AnyBuffer{
						data: Array::zeros(len),
						marker: PhantomData
					})
				)
			}
			for n in 1..10 {
				assert_for_marker_and_len::<marker::UnbiasedSignal>(n);
				assert_for_marker_and_len::<marker::UnbiasedErrorSignal>(n);
			}
		}

		#[test]
		fn zeros_fail() {
			fn assert_for_marker_and_len<B: Unbiased>() {
				assert_eq!(
					AnyBuffer::<B>::zeros(0),
					Err(Error::attempt_to_create_zero_sized_buffer())
				)
			}
			assert_for_marker_and_len::<marker::UnbiasedSignal>();
			assert_for_marker_and_len::<marker::UnbiasedErrorSignal>();
		}

		#[test]
		fn from_raw_ok() {
			fn assert_for_marker<B: Unbiased>() {
				{
					let vec = vec![1.0, 2.0, 3.0];
					let actual   = AnyBuffer::<B>::from_raw(vec.clone());
					let expected = Ok(AnyBuffer{
						data: Array::from_vec(vec.clone()),
						marker: PhantomData
					});
					assert_eq!(actual, expected);
				}{
					let slice = &[1.0, 2.0, 3.0];
					let actual   = AnyView::<B>::from_raw(slice);
					let expected = Ok(AnyView{
						data: slice.into(),
						marker: PhantomData
					});
					assert_eq!(actual, expected);
				}{
					let actual_vals = &mut [1.0, 2.0, 3.0];
					// Requires different slices due to borrow checker!
					let expected_vals = &mut [1.0, 2.0, 3.0];
					let actual   = AnyViewMut::<B>::from_raw(actual_vals);
					let expected = Ok(AnyViewMut{
						data: expected_vals.into(),
						marker: PhantomData
					});
					assert_eq!(actual, expected);
				}
			}
			assert_for_marker::<marker::UnbiasedSignal>();
			assert_for_marker::<marker::UnbiasedErrorSignal>();
		}

		#[test]
		fn from_raw_fail() {
			fn assert_for_marker<B: Unbiased>() {
				{
					let vec      = vec![];
					let actual   = AnyBuffer::<B>::from_raw(vec.clone());
					let expected = Err(Error::attempt_to_create_zero_sized_buffer());
					assert_eq!(actual, expected);
				}
			}
			assert_for_marker::<marker::UnbiasedSignal>();
			assert_for_marker::<marker::UnbiasedErrorSignal>();
		}

		#[test]
		fn assign_ok() {
			fn assert_for_marker<B: Unbiased>() {
				let mut buf_before = AnyBuffer::<B>::from_raw(
					vec![42.0, 1337.0, 77.7]).unwrap();
				let     buf_assign = AnyView::<B>::from_raw(
					&[11.1, 22.2, 33.3]).unwrap();
				buf_before.assign(&buf_assign).unwrap();
				assert_eq!(
					buf_before.view(),
					buf_assign
				);
			}
			assert_for_marker::<marker::UnbiasedSignal>();
			assert_for_marker::<marker::UnbiasedErrorSignal>();
		}

		#[test]
		fn assign_fail() {
			fn assert_for_marker<B: Unbiased>() {
				let buf_before = AnyBuffer::<B>::from_raw(
					vec![42.0, 1337.0, 77.7]).unwrap();
				let     buf_less = AnyView::<B>::from_raw(
					&[11.1, 22.2]).unwrap();
				let     buf_more = AnyView::<B>::from_raw(
					&[11.1, 22.2, 33.3, 44.4]).unwrap();
				assert_ne!(buf_before.dim(), buf_less.dim()); // Test invariant!
				assert_ne!(buf_before.dim(), buf_more.dim()); // Test invariant!
				assert_eq!(
					buf_before.clone().assign(&buf_less),
					Err(Error::unmatching_buffer_sizes(buf_before.dim(), buf_less.dim())
						.with_annotation("Occured in unbiased Buffer::assign method."))
				);
				assert_eq!(
					buf_before.clone().assign(&buf_more),
					Err(Error::unmatching_buffer_sizes(buf_before.dim(), buf_more.dim())
						.with_annotation("Occured in unbiased Buffer::assign method."))
				);
			}
			assert_for_marker::<marker::UnbiasedSignal>();
			assert_for_marker::<marker::UnbiasedErrorSignal>();
		}
	}

	mod error_signal {
		use super::*;

		#[test]
		fn reset_to_zeros() {
			let mut biased_error_signal = BiasedErrorSignalBuffer::from_raw_with_bias(
				vec![1.0, 2.0, 0.0]).unwrap();
			biased_error_signal.reset_to_zeros();
			let zeros = BiasedErrorSignalBuffer::from_raw_with_bias(vec![0.0, 0.0, 0.0]).unwrap();
			assert_eq!(biased_error_signal, zeros);
		}
	}
}
