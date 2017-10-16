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

	/// Occures upon trying to create a zero sized buffer.
	AttemptToCreateZeroSizedBuffer,

	/// Occures when the user provides too few values to create a new buffer.
	TooFewValueProvidedForBufferCreation{
		/// The expected minimum size of the user provided value array.
		expected_min: usize,
		/// The actual size of the user provided value array.
		actual: usize
	},

	/// Occures when a user provided bias value does not match the expected value.
	UnmatchingUserProvidedBiasValue{
		/// The expected bias value.
		expected: f32,
		/// The actual bias value.
		actual: f32
	},

	/// Occures when doing some generic operation (e.g. assigning) with
	/// two buffers of unequal sizes.
	UnmatchingBufferSizes{
		/// The signal length of the left-hand-side buffer.
		left: usize,
		/// The signal length of the right-hand-side buffer.
		right: usize
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
	},

	/// Occures when trying to create a `BelowRecentMSE` condition with an invalid target value.
	InvalidBelowRecentMSEConditionTarget(f32),

	/// Occures when trying to create a `BelowRecentMSE` condition with an invalid momentum.
	InvalidBelowRecentMSEConditionMomentum(f32),

	/// Occures when trying to instantiate a `MeanSquaredError` with a negative value.
	MSEInvalidNegativeValue(f32),

	/// Occures when trying to calculate a `MeanSquaredError` with an empty actual buffer.
	MSEInvalidEmptyActualBuffer,

	/// Occures when trying to calculate a `MeanSquaredError` with an empty expected buffer.
	MSEInvalidEmptyExpectedBuffer,

	/// Occures when trying to calculate a `MeanSquaredError` actual and expected buffers of unmatching sizes.
	MSEUnmatchingActualAndExpectedBuffers{
		/// The size of the buffer storing the actual values.
		actual_len: usize,
		/// The size of the buffer storing the expected values.
		expected_len: usize
	}
}

// Error kinds:
// 
// - AttemptToCreateZeroSizedBuffer
// - TooFewValueProvidedForBufferCreation{expected_min, actual}
// - UnmatchingUserProvidedBiasValue{expected, actual}
// - UnmatchingBufferSizes{left, right}

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

	/// Creates a new `AttemptToCreateZeroSizedBuffer` error.
	pub(crate) fn attempt_to_create_zero_sized_buffer() -> Error {
		Error{
			kind: ErrorKind::AttemptToCreateZeroSizedBuffer,
			message: "Attempted to create a buffer with a length (dimension) of zero (1).".to_owned(),
			annotation: None
		}
	}

	/// Creates a new `TooFewValueProvidedForBufferCreation` error.
	pub(crate) fn too_few_values_provided_for_buffer_creation(expected_min: usize, actual: usize) -> Error {
		assert!(actual < expected_min);
		Error{
			kind: ErrorKind::TooFewValueProvidedForBufferCreation{expected_min, actual},
			message: format!(
				"Expected at least {:?} user provided array elements for buffer creation
				 but found only {:?} instead.",
				 expected_min,
				 actual
			),
			annotation: None
		}
	}

	/// Creates a new `UnmatchingUserProvidedBiasValue` error.
	pub(crate) fn unmatching_user_provided_bias_value(expected: f32, actual: f32) -> Error {
		assert!(expected != actual);
		Error{
			kind: ErrorKind::UnmatchingUserProvidedBiasValue{expected, actual},
			message: format!(
				"Expected a bias value of {:?} as the last value of the given data but found a bias value of
			     {:?} instead.", expected, actual
			),
			annotation: None
		}
	}

	/// Creates a new `UnmatchingBufferSizes` error.
	pub(crate) fn unmatching_buffer_sizes(lhs_size: usize, rhs_size: usize) -> Error {
		Error{
			kind: ErrorKind::UnmatchingBufferSizes{left: lhs_size, right: rhs_size},
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

	/// Creates a new `InvalidBelowRecentMSEConditionTarget` error.
	pub(crate) fn invalid_below_recent_mse_target(invalid_target: f32) -> Error {
		Error{
			kind: ErrorKind::InvalidBelowRecentMSEConditionTarget(invalid_target),
			message: format!("Tried to create a BelowRecentMSE condition with an invalid target of {:?}!", invalid_target),
			annotation: None
		}
	}

	/// Creates a new `InvalidBelowRecentMSEConditionMomentum` error.
	pub(crate) fn invalid_below_recent_mse_momentum(invalid_momentum: f32) -> Error {
		Error{
			kind: ErrorKind::InvalidBelowRecentMSEConditionMomentum(invalid_momentum),
			message: format!("Tried to create a BelowRecentMSE condition with an invalid momentum of {:?}!", invalid_momentum),
			annotation: None
		}
	}

	/// Creates a new `MSEInvalidNegativeValue` error.
	pub(crate) fn mse_invalid_negative_value(value: f32) -> Error {
		Error{
			kind: ErrorKind::MSEInvalidNegativeValue(value),
			message: format!("Tried to instantiate a MeanSquaredError with a negative value of {:?}!", value),
			annotation: None
		}
	}

	/// Creates a new `MSEInvalidEmptyActualBuffer` error.
	pub(crate) fn mse_invalid_empty_actual_buffer() -> Error {
		Error{
			kind: ErrorKind::MSEInvalidEmptyActualBuffer,
			message: "Tried to calculate a MeanSquaredError with an empty actual buffer!".to_owned(),
			annotation: None
		}
	}

	/// Creates a new `MSEInvalidEmptyExpectedBuffer` error.
	pub(crate) fn mse_invalid_empty_expected_buffer() -> Error {
		Error{
			kind: ErrorKind::MSEInvalidEmptyExpectedBuffer,
			message: "Tried to calculate a MeanSquaredError with an empty expected buffer!".to_owned(),
			annotation: None
		}
	}

	/// Creates a new `MSEUnmatchingActualAndExpectedBuffers` error.
	pub(crate) fn mse_unmatching_actual_and_empty_buffers(actual_len: usize, expected_len: usize) -> Error {
		Error{
			kind: ErrorKind::MSEUnmatchingActualAndExpectedBuffers{actual_len, expected_len},
			message: format!(
				"Tried to calculate a MeanSquaredError with unmatching buffer sizes of {:?} (actual) and {:?} (expected).",
				actual_len, expected_len
			),
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
