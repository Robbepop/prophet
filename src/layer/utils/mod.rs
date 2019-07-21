mod buffer_base;
mod matrix_base;

pub(crate) use self::buffer_base::{
	BiasedErrorSignalBuffer,
	// UnbiasedErrorSignalBuffer,

	// Iter,
	// IterMut
	BiasedErrorSignalView,
	// UnbiasedSignalViewMut,
	BiasedErrorSignalViewMut,
	// UnbiasedErrorSignalViewMut,
	BiasedSignalBuffer,
	BiasedSignalView,
	// UnbiasedErrorSignalView,
	BiasedSignalViewMut,
	UnbiasedSignalBuffer,
	UnbiasedSignalView,
};
pub(crate) use self::matrix_base::{DeltaWeightsMatrix, WeightsMatrix};

pub mod prelude {
	#[doc(no_inline)]
	pub(crate) use super::{
		// UnbiasedSignalBuffer,
		BiasedErrorSignalBuffer,
		// UnbiasedErrorSignalBuffer,

		// Iter,
		// IterMut
		BiasedErrorSignalView,
		// UnbiasedSignalViewMut,
		BiasedErrorSignalViewMut,
		// UnbiasedErrorSignalViewMut,
		BiasedSignalBuffer,
		BiasedSignalView,
		// UnbiasedErrorSignalView,
		BiasedSignalViewMut,
		DeltaWeightsMatrix,

		UnbiasedSignalView,
		WeightsMatrix,
	};
}
