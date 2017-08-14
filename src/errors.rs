//! Errors that may happen while using this crate and its `Result` type are defined here.

use std::fmt;
use std::error;
use std::result;

/// Kinds of errors that may occure while using this crate.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ErrorKind {
	/// Occures when invalid sample input sizes are recognized.
	UnmatchingInputSampleSize,

	/// Occures when invalid sample target sizes are recognized.
	UnmatchingTargetSampleSize,

	/// Occures when the learning rate is not within the valid
	/// range of `(0,1)`.
	InvalidLearnRate,

	/// Occures when the learning momentum is not within the
	/// valid range of `(0,1)`.
	InvalidLearnMomentum,

	/// Occures when the specified average net error
	/// criterion is invalid.
	InvalidRecentMSE,

	/// Occures when the specified mean squared error
	/// criterion is invalid.
	InvalidLatestMSE,

	/// Occures when trying to create a `LayerSize` that 
	/// represents zero (0) neurons.
	ZeroLayerSize
}

/// The error class used in `Prophet`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

	/// Returns a new `Error` for the given kind and with the given message.
	/// 
	/// Note: Instances created with this method won't have an annotation.
	///       Use `with_annotation(..)` to add one if needed.
	#[inline]
	pub(crate) fn new(kind: ErrorKind, message: String) -> Error {
		Error{kind, message, annotation: None}
	}

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
			message: format!("Cannot create a layer size representing zero (0) neurons."),
			annotation: None
		}
	}

	/// Creates a new `InvalidLearnRate` error with the given invalid learning rate.
	pub(crate) fn invalid_learn_rate(lr: f64) -> Error {
		Error{
			kind: ErrorKind::InvalidLearnRate,
			message: format!("Tried to create an invalid learning rate of {:?}. Valid learning rates must be between `0.0` and `1.0`.", lr),
			annotation: None
		}
	}

	/// Creates a new `InvalidLearnMomentum` error with the given invalid learning momentum.
	pub(crate) fn invalid_learn_momentum(lm: f64) -> Error {
		Error{
			kind: ErrorKind::InvalidLearnMomentum,
			message: format!("Tried to create an invalid learning momentum of {:?}. Valid learning momentums must be between `0.0` and `1.0`.", lm),
			annotation: None
		}
	}

	/// Creates a new `InvalidLearnMomentum` error with the given invalid learning momentum.
	pub(crate) fn invalid_recent_mse(recent_mse: f64) -> Error {
		Error{
			kind: ErrorKind::InvalidRecentMSE,
			message: format!("Tried to create an invalid `RecentMSE` criterion of {:?}. Only strictly positive values are allowed.", recent_mse),
			annotation: None
		}
	}

	/// Creates a new `InvalidLearnMomentum` error with the given invalid learning momentum.
	pub(crate) fn invalid_latest_mse(latest_mse: f64) -> Error {
		Error{
			kind: ErrorKind::InvalidLatestMSE,
			message: format!("Tried to create an invalid `LatestMSE` criterion of {:?}. Only strictly positive values are allowed.", latest_mse),
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
