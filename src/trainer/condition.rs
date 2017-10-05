//! Provides conditions that can be used by training utilities such as halting condition
//! or conditions for constraining logging to query and evaluate a training state during
//! training process.
//! 
//! Condition types can be easily extended by the user.

use std::time;
use std::fmt::Debug;

use errors::{Result};

/// Provides an interface for training stats during the training process
/// that can be used and queried by halting conditions to check whether their
/// halting requirements are met.
pub trait TrainingState {
	/// Returns the point in time when the training has started.
	fn time_started(&self) -> time::Instant;

	/// Returns the elapsed time since the start of training.
	#[inline]
	fn duration_elapsed(&self) -> time::Duration {
		time::Instant::now().duration_since(self.time_started())
	}

	/// Returns the number of predict iterations so far.
	/// 
	/// Note: This is highly correlated with `epochs()`.
	fn iterations(&self) -> usize;

	/// Returns the number of epochs so far.
	/// 
	/// Note: This is highly correlated with `iterations()`.
	fn epochs_passed(&self) -> usize;

	/// Returns the latest mean-squared-error (MSE) of the training.
	fn latest_mse(&self) -> f64;
}

/// This is used to query a `TrainingState` and check whether the requirements
/// of the given `HaltingCondition` are met.
/// 
/// With this trait users can implement their own `TrainCondition`s.
pub trait TrainCondition: Debug {
	/// Returns `true` if the halting condition requirements are met by
	/// the given `TrainingStats` implementation `stats`.
	fn evaluate(&mut self, stats: &TrainingState) -> bool;
}

/// Always evaluate to `true` for any given `TrainCondition`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Always;

/// Never evaluate to `true` for any given `TrainCondition`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Never;

/// Evaluates to `true` if its inner `TrainCondition`
/// evaluates to `false` for the given `TrainingState`.
#[derive(Debug, Clone, PartialEq)]
pub struct Not<C>
	where C: TrainCondition
{
	inner: Box<C>
}

/// Evaluates to `true` if both inner conditions evaluate to `true` for the given `TrainingState`.
/// 
/// Users need to care themselves that the requirements of `lhs` and `rhs`
/// are not mutually exclusive to prevent this condition to be a contradiction.
#[derive(Debug, Clone, PartialEq)]
pub struct Conjunction<L, R>
	where L: TrainCondition,
	      R: TrainCondition
{
	/// The left-hand-side halting condition.
	lhs: Box<L>,
	/// The right-hand-side halting condition.
	rhs: Box<R>
}

/// Evaluates to `true` if any inner condition evaluates to `true` for the given `TrainingState`.
/// 
/// Users need to care themselves that the requirements of `lhs` and `rhs` do not form a tautology.
#[derive(Debug, Clone, PartialEq)]
pub struct Disjunction<L, R>
	where L: TrainCondition,
	      R: TrainCondition
{
	/// The left-hand-side halting condition.
	lhs: Box<L>,
	/// The right-hand-side halting condition.
	rhs: Box<R>
}

/// Evaluates to `true` if the duration passed since the start of the training process of the
/// given `TrainingState` exceeds its given duration.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TimeElapsed(time::Duration);

/// Evaluates to `true` if the given `TrainingState` exceeds the given amount of `epochs` of this condition.
/// 
/// By default an epoch is as large as the number of samples in the given sample set,
/// however, this default value can be adjusted by the user during the setup process of
/// a training instance.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct EpochsPassed(usize);

/// Evaluates to `true` as soon as the recent mean squared error (RMSE) drops below the given `target` value
/// for the first time during the training process.
/// 
/// The given `momentum` ranges from `(0, 1)` and regulates how strongly the RMSE depends on earlier iterations.
/// A `momentum` near `0.0` has near to no influence by earlier iterations while a `momentum` near `1.0`
/// is influenced heavily by earlier iterations.
/// 
/// Note: Given a momentum `m` the RMSE in the `i+1`th iteration (`rmse_(i+1)`) is calculated by
///       the following formula. `mse_(n)` stands for the mean squared error of the `n`th iteration.
/// 
/// - `rmse_(i+1) := (1.0 - m) * mse_(i+1) + m * rmse_i`
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BelowRecentMSE{
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
	target: f64,
	/// This is the current recent mean squared error calculated so far in the training process.
	rmse: f64
}

/// Evaluates to `true` once every given amount of time passed.
/// 
/// This is a special kind of training condition since its evaluation is not at all dependend on the
/// training state. Also it is not static once it changes its evaluation in constrast to other conditions
/// such as `EpochsPassed` or `TimeElapsed`.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TimeInterval{
	/// The duration between the interval at which this condition evalutes to `true` once.
	time_step: time::Duration,
	/// The latest point in time where this condition evaluated to `true`.
	latest: time::Instant
}

impl TrainCondition for Always {
	#[inline]
	fn evaluate(&mut self, _stats: &TrainingState) -> bool { true }
}

impl TrainCondition for Never {
	#[inline]
	fn evaluate(&mut self, _stats: &TrainingState) -> bool { false }
}

impl<C> Not<C>
	where C: TrainCondition
{
	/// Creates a new `TrainCondition` that represents a logical-not.
	pub fn not(inner: C) -> Self {
		Not{inner: Box::new(inner)}
	}
}

impl<C> TrainCondition for Not<C>
	where C: TrainCondition
{
	#[inline]
	fn evaluate(&mut self, stats: &TrainingState) -> bool {
		!self.inner.evaluate(stats)
	}
}

impl<L, R> Conjunction<L, R>
	where L: TrainCondition,
	      R: TrainCondition
{
	/// Creates a new `TrainCondition` that represents a logical-and.
	pub fn all(lhs: L, rhs: R) -> Self {
		Conjunction{
			lhs: Box::new(lhs),
			rhs: Box::new(rhs)
		}
	}
}

impl<L, R> TrainCondition for Conjunction<L, R>
	where L: TrainCondition,
	      R: TrainCondition
{
	#[inline]
	fn evaluate(&mut self, stats: &TrainingState) -> bool {
		self.lhs.evaluate(stats) && self.rhs.evaluate(stats)
	}
}

impl<L, R> Disjunction<L, R>
	where L: TrainCondition,
	      R: TrainCondition
{
	/// Creates a new `TrainCondition` that represents a logical-or.
	pub fn any(lhs: L, rhs: R) -> Self {
		Disjunction{
			lhs: Box::new(lhs),
			rhs: Box::new(rhs)
		}
	}
}

impl<L, R> TrainCondition for Disjunction<L, R>
	where L: TrainCondition,
	      R: TrainCondition
{
	#[inline]
	fn evaluate(&mut self, stats: &TrainingState) -> bool {
		self.lhs.evaluate(stats) || self.rhs.evaluate(stats)
	}
}

impl BelowRecentMSE {
	/// Creates a new `BelowRecentMSE` `TrainCondition` with the given `momentum`
	/// and `target` value.
	/// 
	/// # Errors
	/// 
	/// - If `momentum` is not within the range `(0, 1)`.
	/// - If `target` is not strictly positive.
	pub fn new(momentum: f64, target: f64) -> Result<BelowRecentMSE> {
		if !(0.0 < momentum && momentum < 1.0) {
			// Error! Momentum invalid.
		}
		if !(0.0 < target) {
			// Error! Target invalid.
		}
		Ok(BelowRecentMSE{momentum, target, rmse: 1.0})
	}
}

impl TimeInterval {
	/// Creates a new `TimeInterval` `TrainCondition` with the given `time_step` duration
	/// that evaluates to `true` every time the duration `time_step` has passed.
	/// 
	/// Note: This condition is especially useful for logging purposes where a user want to
	///       log the training state once every given amount of time.
	pub fn once_in(time_step: time::Duration) -> TimeInterval {
		TimeInterval{time_step, latest: time::Instant::now()}
	}
}

impl TrainCondition for TimeElapsed {
	#[inline]
	fn evaluate(&mut self, stats: &TrainingState) -> bool {
		stats.duration_elapsed() < self.0
	}
}

impl TrainCondition for EpochsPassed {
	#[inline]
	fn evaluate(&mut self, stats: &TrainingState) -> bool {
		stats.epochs() >= self.0
	}
}

impl TrainCondition for BelowRecentMSE {
	#[inline]
	fn evaluate(&mut self, stats: &TrainingState) -> bool {
		self.rmse = (1.0 - self.momentum) * stats.latest_mse() + self.momentum * self.rmse;
		self.rmse <= self.target
	}
}

impl TrainCondition for TimeInterval {
	#[inline]
	fn evaluate(&mut self, _stats: &TrainingState) -> bool {
		if time::Instant::now().duration_since(self.latest) >= self.time_step {
			self.latest = time::Instant::now();
			true
		}
		else {
			false
		}
	}
}
