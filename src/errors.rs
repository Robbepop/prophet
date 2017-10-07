//! Errors that may happen while using this crate and its `Result` type are defined here.

use std::fmt;
use std::error;
use std::result;

/// Kinds of errors that may occure while using this crate.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ErrorKind {
	/// Occures when invalid sample input sizes are recognized.
	UnmatchingInputSampleSize,

	/// Occures when invalid sample target sizes are recognized.
	UnmatchingTargetSampleSize,

	/// Occures when the learning rate is not within the valid
	/// range of `(0,1)`.
	InvalidLearnRate(f32),

	/// Occures when the learning momentum is not within the
	/// valid range of `(0,1)`.
	InvalidLearnMomentum(f32),

	/// Occures when the specified average net error
	/// criterion is invalid.
	InvalidRecentMSECriterion,

	/// Occures when trying to create a `LayerSize` that 
	/// represents zero (0) neurons.
	ZeroLayerSize,

	/// Occures when trying to create an `OutputBuffer`
	/// with zero non-bias neuron values.
	ZeroSizedSignalBuffer,

	/// Occures when trying to assign a non-matching number of input
	/// signals to a buffer.
	NonMatchingNumberOfSignals,

	/// Occures when trying to create a `GradientBuffer`
	/// representing zero values.
	ZeroSizedGradientBuffer,

	/// Occures upon creating a `WeightsMatrix`
	/// for `0` (zero) inputs.
	ZeroInputsWeightsMatrix,

	/// Occures upon creating a `WeightsMatrix`
	/// for `0` (zero) outputs.
	ZeroOutputsWeightsMatrix,

	/// Occures when doing some generic operation (e.g. assigning) with
	/// two buffers of unequal sizes.
	UnmatchingBufferSizes{
		/// The signal length of the left-hand-side buffer.
		lhs_size: usize,
		/// The signal length of the right-hand-side buffer.
		rhs_size: usize
	},

	/// Occures when trying to create a `SampleCollection` from an empty set of samples.
	EmptySampleCollection,

	/// Occures when trying to create a `SampleCollection` from a set of samples with unmatching input lengths.
	UnmatchingSampleInputLength{
		/// The required sample signal input length.
		required_len: usize,
		/// The actual and errorneous sample signal input length.
		actual_len: usize
	},

	/// Occures when trying to create a `SampleCollection` from a set of samples with unmatching input lengths.
	UnmatchingSampleExpectedLength{
		/// The required sample signal expected length.
		required_len: usize,
		/// The actual and errorneous sample signal expected length.
		actual_len: usize
	}

}

/// The error class used in `Prophet`.
#[derive(Debug, Clone, PartialEq)]
pub struct Error{
	kind      : ErrorKind,
	message   : String,
	annotation: Option<String>
}

impl Error {
	/// Returns a reference to the kind of this `Error`.
	#[inline]
	pub fn kind(&self) -> &ErrorKind {
		&self.kind
	}

	/// Returns the error message description of this `Error`.
	#[inline]
	pub fn message(&self) -> &str {
		self.message.as_str()
	}

	/// Returns the optional annotation of this `Error`.
	#[inline]
	pub fn annotation(&self) -> Option<&str> {
		match self.annotation {
			Some(ref ann) => Some(ann.as_str()),
			None          => None
		}
	}

	// /// Returns a new `Error` for the given kind and with the given message.
	// /// 
	// /// Note: Instances created with this method won't have an annotation.
	// ///       Use `with_annotation(..)` to add one if needed.
	// #[inline]
	// pub(crate) fn new(kind: ErrorKind, message: String) -> Error {
	// 	Error{kind, message, annotation: None}
	// }

	/// Consumes this error and returns itself with the given annotation added to it.
	/// 
	/// Note: This will replace an already existing annotation.
	#[inline]
	pub(crate) fn with_annotation<A>(mut self, annotation: A) -> Error
		where A: Into<String>
	{
		self.annotation = Some(annotation.into());
		self
	}
}

impl Error {
	/// Creates a new `ZeroLayerSize` error.
	pub(crate) fn zero_layer_size() -> Error {
		Error{
			kind: ErrorKind::ZeroLayerSize,
			message: String::from("Cannot create a layer size representing zero (0) neurons."),
			annotation: None
		}
	}

	/// Creates a new `InvalidLearnRate` error with the given invalid learning rate.
	pub(crate) fn invalid_learn_rate(lr: f32) -> Error {
		Error{
			kind: ErrorKind::InvalidLearnRate(lr),
			message: format!("Tried to create an invalid learning rate of {:?}. Valid learning rates must be between `0.0` and `1.0`.", lr),
			annotation: None
		}
	}

	/// Creates a new `InvalidLearnMomentum` error with the given invalid learning momentum.
	pub(crate) fn invalid_learn_momentum(lm: f32) -> Error {
		Error{
			kind: ErrorKind::InvalidLearnMomentum(lm),
			message: format!("Tried to create an invalid learning momentum of {:?}. Valid learning momentums must be between `0.0` and `1.0`.", lm),
			annotation: None
		}
	}

	/// Creates a new `InvalidLearnMomentum` error with the given invalid learning momentum.
	pub(crate) fn invalid_recent_mse(recent_mse: f64) -> Error {
		Error{
			kind: ErrorKind::InvalidRecentMSECriterion,
			message: format!("Tried to create an invalid `RecentMSE` criterion of {:?}. Only strictly positive values are allowed.", recent_mse),
			annotation: None
		}
	}

	/// Creates a new `InvalidLearnMomentum` error with the given invalid learning momentum.
	pub(crate) fn unmatching_input_sample_size(actual: usize, req: usize) -> Error {
		Error{
			kind: ErrorKind::UnmatchingInputSampleSize,
			message: format!("Tried to create an input sample with {:?} neurons while {:?} are required.", actual, req),
			annotation: None
		}
	}

	/// Creates a new `InvalidLearnMomentum` error with the given invalid learning momentum.
	pub(crate) fn unmatching_target_sample_size(actual: usize, req: usize) -> Error {
		Error{
			kind: ErrorKind::UnmatchingTargetSampleSize,
			message: format!("Tried to create an target sample with {:?} neurons while {:?} are required.", actual, req),
			annotation: None
		}
	}

	/// Creates a new `ZeroSizedOutputBuffer` error.
	pub(crate) fn zero_sized_signal_buffer() -> Error {
		Error{
			kind: ErrorKind::ZeroSizedSignalBuffer,
			message: "Tried to create an OutputBuffer representing zero non-bias values.".to_owned(),
			annotation: None
		}
	}

	/// Creates a new `ZeroInputsWeightsMatrix` error.
	pub(crate) fn zero_inputs_weights_matrix() -> Error {
		Error{
			kind: ErrorKind::ZeroInputsWeightsMatrix,
			message: "Tried to create a WeightsMatrix for zero inputs. Must be at least one!".to_owned(),
			annotation: None
		}
	}

	/// Creates a new `ZeroInputsWeightsMatrix` error.
	pub(crate) fn zero_outputs_weights_matrix() -> Error {
		Error{
			kind: ErrorKind::ZeroOutputsWeightsMatrix,
			message: "Tried to create a WeightsMatrix for zero outputs. Must be at least one!".to_owned(),
			annotation: None
		}
	}

	/// Creates a new `UnmatchingBufferSizes` error.
	pub(crate) fn unmatching_buffer_sizes(lhs_size: usize, rhs_size: usize) -> Error {
		Error{
			kind: ErrorKind::UnmatchingBufferSizes{lhs_size, rhs_size},
			message: format!(
				"Tried to operate on buffers with non-matching sizes of {:?} and {:?} elements.",
				lhs_size,
				rhs_size
			),
			annotation: None
		}
	}

	/// Creates a new `EmptySampleCollection` error.
	pub(crate) fn empty_sample_collection() -> Error {
		Error{
			kind: ErrorKind::EmptySampleCollection,
			message: "Tried to create a SampleCollection from an empty set of samples!".to_owned(),
			annotation: None
		}
	}

	/// Creates a new `UnmatchingSampleInputLength` error.
	pub(crate) fn unmatching_sample_input_len(required_len: usize, actual_len: usize) -> Error {
		Error{
			kind: ErrorKind::UnmatchingSampleInputLength{required_len, actual_len},
			message: "Tried to create a SampleCollection from an set of samples with unmatching input lengths!".to_owned(),
			annotation: None
		}
	}

	/// Creates a new `UnmatchingSampleExpectedLength` error.
	pub(crate) fn unmatching_sample_expected_len(required_len: usize, actual_len: usize) -> Error {
		Error{
			kind: ErrorKind::UnmatchingSampleExpectedLength{required_len, actual_len},
			message: "Tried to create a SampleCollection from an set of samples with unmatching expected lengths!".to_owned(),
			annotation: None
		}
	}

}

impl<T> Into<Result<T>> for Error {
	fn into(self) -> Result<T> {
		Err(self)
	}
}

impl fmt::Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		<Self as fmt::Debug>::fmt(self, f)
	}
}

impl error::Error for Error {
	fn description(&self) -> &str {
		self.message.as_str()
	}
}

/// Result type for procedures of this crate.
pub type Result<T> = result::Result<T, Error>;
