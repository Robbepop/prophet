use std::time::{Duration};
use std::fmt::Debug;

use errors::{Result, Error};
// use utils::{LearnRate, LearnMomentum};

/// Provides an interface for training stats during the training process
/// that can be used and queried by halting conditions to check whether their
/// halting requirements are met.
pub trait TrainingState {
	// TODO: API
}

/// Any `HaltingCondition` must implement this trait.
/// 
/// This is used to query a `TrainingState` and check whether the requirements
/// of the given `HaltingCondition` are met.
/// 
/// With this trait users can implement their own `HaltCondition`s.
pub trait HaltCondition: Debug + Clone {
	/// Returns `true` if the halting condition requirements are met by
	/// the given `TrainingStats` implementation `stats`.
	fn is_satisfied(&self, stats: &TrainingState) -> bool;
}

/// Stop when both inner `HaltCondition`s meet the requirements.
/// 
/// Users need to care that the requirements of `lhs` and `rhs`
/// are not mutually exclusive. Otherwise, the learning process will never stop.
#[derive(Debug, Clone, PartialEq)]
pub struct Conjunction<L, R>
	where L: HaltCondition,
	      R: HaltCondition
{
	/// The left-hand-side halting condition.
	lhs: Box<L>,
	/// The right-hand-side halting condition.
	rhs: Box<R>
}

/// Stop when any of the inner `HaltCondition`s meet the requirements.
#[derive(Debug, Clone, PartialEq)]
pub struct Disjunction<L, R>
	where L: HaltCondition,
	      R: HaltCondition
{
	/// The left-hand-side halting condition.
	lhs: Box<L>,
	/// The right-hand-side halting condition.
	rhs: Box<R>
}

/// Stop after the given timed duration.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeOut(Duration);

/// Stop after the given amount of epochs.
/// 
/// By default an epoch is as large as the number of samples in the given sample set,
/// however, this default value can be adjusted by the user during the setup process of
/// a training instance.
#[derive(Debug, Clone, PartialEq)]
pub struct Epochs(usize);

/// Stop as soon as the recent mean squared error (RMSE) drops below the given `target` value.
/// 
/// The given `momentum` ranges from `(0, 1)` and regulates how strongly the RMSE depends on earlier iterations.
/// A `momentum` near `0` has near to no influence by earlier iterations while a `momentum` near `1`
/// is influenced heavily by earlier iterations.
/// 
/// Note: Given a momentum `m` the RMSE in the `i+1`th iteration (`rmse_(i+1)`) is calculated by
///       the following formula. `mse_(n)` stands for the mean squared error of the `n`th iteration.
/// 
/// - `rmse_(i+1) := 0.05 * mse_(i+1) + 0.95 * rmse_i`
#[derive(Debug, Clone, PartialEq)]
pub struct RecentMSE{
	/// The given `momentum` ranges from `(0, 1)` and regulates how strongly the RMSE depends on earlier iterations.
	/// A `momentum` near `0` has near to no influence by earlier iterations while a `momentum` near `1`
	/// is influenced heavily by earlier iterations.
	momentum: f64,
	/// This represents the target recent mean squared error.
	/// The training process will stop once the training reaches this target value.
	/// 
	/// Normally this value should be near zero (`0`), e.g. `0.03`.
	/// 
	/// The lower this value is, the better are the results of the resulting neural net once
	/// the training has finished. However, trying to reach a very low `taget` value can be very time
	/// consuming and sometimes even impossible.
	target: f64
}

impl<L, R> Conjunction<L, R>
	where L: HaltCondition,
	      R: HaltCondition
{
	/// Creates a new `HaltCondition` that stops the training process whenever
	/// all of the given inner `HaltCondition`s `lhs` and `rhs` meet their requirements.
	pub fn all(lhs: L, rhs: R) -> Self {
		Conjunction{
			lhs: Box::new(lhs),
			rhs: Box::new(rhs)
		}
	}
}

impl<L, R> HaltCondition for Conjunction<L, R>
	where L: HaltCondition,
	      R: HaltCondition
{
	fn is_satisfied(&self, stats: &TrainingState) -> bool {
		self.lhs.is_satisfied(stats) && self.rhs.is_satisfied(stats)
	}
}

impl<L, R> Disjunction<L, R>
	where L: HaltCondition,
	      R: HaltCondition
{
	/// Creates a new `HaltCondition` that stops the training process whenever
	/// any of the given inner `HaltCondition`s `lhs` and `rhs` meet their requirements.
	pub fn any(lhs: L, rhs: R) -> Self {
		Disjunction{
			lhs: Box::new(lhs),
			rhs: Box::new(rhs)
		}
	}
}

impl<L, R> HaltCondition for Disjunction<L, R>
	where L: HaltCondition,
	      R: HaltCondition
{
	fn is_satisfied(&self, stats: &TrainingState) -> bool {
		self.lhs.is_satisfied(stats) || self.rhs.is_satisfied(stats)
	}
}

impl RecentMSE {
	/// Creates a new `RecentMSE` `HaltCondition` with the given `momentum`
	/// and `target` value.
	/// 
	/// # Errors
	/// 
	/// - If `momentum` is not within the range `(0, 1)`.
	/// - If `target` is not strictly positive.
	pub fn new(momentum: f64, target: f64) -> Result<RecentMSE> {
		if !(0.0 < momentum && momentum < 1.0) {
			// Error! Momentum invalid.
		}
		if !(0.0 < target) {
			// Error! Target invalid.
		}
		Ok(RecentMSE{momentum, target})
	}
}

impl HaltCondition for TimeOut {
	fn is_satisfied(&self, _stats: &TrainingState) -> bool {
		unimplemented!()
	}
}

impl HaltCondition for Epochs {
	fn is_satisfied(&self, _stats: &TrainingState) -> bool {
		unimplemented!()
	}
}

impl HaltCondition for RecentMSE {
	fn is_satisfied(&self, _stats: &TrainingState) -> bool {
		unimplemented!()
	}
}
