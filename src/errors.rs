//! Errors that may happen while using this crate and its `Result` type are defined here.

/// Kinds of errors that may occure while using this crate.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ErrorKind {
	/// Occures when invalid sample input sizes are recognized.
	InvalidSampleInputSize,

	/// Occures when invalid sample target sizes are recognized.
	InvalidSampleTargetSize,

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
}

/// Result type for procedures of this crate.
pub type Result<T> = ::std::result::Result<T, ErrorKind>;
