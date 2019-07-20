//! Provides conditions that can be used by training utilities such as halting condition
//! or conditions for constraining logging to query and evaluate a training state during
//! training process.
//! 
//! Condition types can be easily extended by the user.

use std::time;
use std::fmt::Debug;

use crate::errors::{Error, Result};
use crate::trainer::MeanSquaredError;

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
	fn latest_mse(&self) -> MeanSquaredError;
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
	inner: C
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
	lhs: L,
	/// The right-hand-side halting condition.
	rhs: R
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
	lhs: L,
	/// The right-hand-side halting condition.
	rhs: R
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
	momentum: f32,
	/// This represents the target recent mean squared error.
	/// The training process will stop once the training reaches this target value.
	/// 
	/// Normally this value should be near zero (`0`), e.g. `0.03`.
	/// 
	/// The lower this value is, the better are the results of the resulting neural net once
	/// the training has finished. However, trying to reach a very low `taget` value can be very time
	/// consuming and sometimes even impossible.
	target: f32,
	/// This is the current recent mean squared error calculated so far in the training process.
	rmse: f32
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
	pub fn new(inner: C) -> Self {
		Not{inner: inner}
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
	pub fn new(lhs: L, rhs: R) -> Self {
		Conjunction{lhs, rhs}
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
	pub fn new(lhs: L, rhs: R) -> Self {
		Disjunction{lhs, rhs}
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
	/// - If `momentum` is not within the range `[0, 1)`.
	/// - If `target` is not strictly positive.
	pub fn new(momentum: f32, target: f32) -> Result<BelowRecentMSE> {
		if !(0.0 <= momentum && momentum < 1.0) {
			return Err(Error::invalid_below_recent_mse_momentum(momentum))
		}
		if !(0.0 < target) {
			return Err(Error::invalid_below_recent_mse_target(target))
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
	pub fn new(time_step: time::Duration) -> TimeInterval {
		TimeInterval{time_step, latest: time::Instant::now()}
	}
}

impl TrainCondition for TimeElapsed {
	#[inline]
	fn evaluate(&mut self, stats: &TrainingState) -> bool {
		stats.duration_elapsed() >= self.0
	}
}

impl TrainCondition for EpochsPassed {
	#[inline]
	fn evaluate(&mut self, stats: &TrainingState) -> bool {
		stats.epochs_passed() >= self.0
	}
}

impl TrainCondition for BelowRecentMSE {
	#[inline]
	fn evaluate(&mut self, stats: &TrainingState) -> bool {
		self.rmse = (1.0 - self.momentum) * stats.latest_mse().to_f32() + self.momentum * self.rmse;
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

#[cfg(test)]
mod tests {
	use super::*;

	#[derive(Debug, Copy, Clone, PartialEq)]
	struct DummyContext {
		time_started: time::Instant,
		iterations: usize,
		epochs_passed: usize,
		latest_mse: MeanSquaredError
	}

	impl TrainingState for DummyContext {
		#[inline]
		fn time_started(&self) -> time::Instant {
			self.time_started
		}

		fn iterations(&self) -> usize {
			self.iterations
		}

		fn epochs_passed(&self) -> usize {
			self.epochs_passed
		}

		fn latest_mse(&self) -> MeanSquaredError {
			self.latest_mse
		}
	}

	fn dummy_state() -> DummyContext {
		DummyContext{
			time_started: time::Instant::now(),
			iterations: 42,
			epochs_passed: 1337,
			latest_mse: MeanSquaredError::new(7.77).unwrap()
		}
	}

	mod always {
		use super::*;

		#[test]
		fn simple_eval() {
			assert_eq!(Always.evaluate(&dummy_state()), true)
		}
	}

	mod never {
		use super::*;

		#[test]
		fn simple_eval() {
			assert_eq!(Never.evaluate(&dummy_state()), false)
		}
	}

	mod not {
		use super::*;

		#[test]
		fn not_true() {
			assert_eq!(Not::new(Always).evaluate(&dummy_state()), false)
		}

		#[test]
		fn not_false() {
			assert_eq!(Not::new(Never).evaluate(&dummy_state()), true)
		}

		#[test]
		fn involution() {
			assert_eq!(
				Not::new(Not::new(Always)).evaluate(&dummy_state()),
				Always.evaluate(&dummy_state())
			)
		}
	}

	mod conjunction {
		use super::*;

		#[test]
		fn true_and_true() {
			assert_eq!(Conjunction::new(Always, Always).evaluate(&dummy_state()), true)
		}

		#[test]
		fn true_and_false() {
			assert_eq!(Conjunction::new(Always, Never).evaluate(&dummy_state()), false)
		}

		#[test]
		fn false_and_true() {
			assert_eq!(Conjunction::new(Never, Always).evaluate(&dummy_state()), false)
		}

		#[test]
		fn false_and_false() {
			assert_eq!(Conjunction::new(Never, Never).evaluate(&dummy_state()), false)
		}
	}

	mod disjunction {
		use super::*;

		#[test]
		fn true_or_true() {
			assert_eq!(Disjunction::new(Always, Always).evaluate(&dummy_state()), true)
		}

		#[test]
		fn true_or_false() {
			assert_eq!(Disjunction::new(Always, Never).evaluate(&dummy_state()), true)
		}

		#[test]
		fn false_or_true() {
			assert_eq!(Disjunction::new(Never, Always).evaluate(&dummy_state()), true)
		}

		#[test]
		fn false_or_false() {
			assert_eq!(Disjunction::new(Never, Never).evaluate(&dummy_state()), false)
		}
	}

	mod time_elapsed {
		use super::*;

		fn time_elapsed_ctx() -> DummyContext {
			let mut state = dummy_state();
			state.time_started = time::Instant::now();
			state
		}

		#[test]
		fn eval_true() {
			assert_eq!(TimeElapsed(time::Duration::from_secs(0)).evaluate(&time_elapsed_ctx()), true)
		}

		#[test]
		fn eval_false() {
			assert_eq!(TimeElapsed(time::Duration::from_secs(1000)).evaluate(&time_elapsed_ctx()), false)
		}

		#[test]
		fn before_and_after_elapsed() {
			// This test breaks on CI if their `Instant::now()` (what `ctx.time_started` is),
			// comes from a system clock that's not  up to date. In fact, on Appveyor the time
			// now seems to start from 0s(?) judging from the fact that we can observe an
			// `Instant { t: 303.4676818s }` ...
			let     dur = time::Duration::from_secs(10);
			let mut ctx = time_elapsed_ctx();
			let mut cond = TimeElapsed(dur);
			assert_eq!(cond.evaluate(&ctx), false);
			ctx.time_started -= dur;
			assert_eq!(cond.evaluate(&ctx), true);
		}
	}

	mod epochs_passed {
		use super::*;

		fn epochs_passed_ctx(epochs_passed: usize) -> DummyContext {
			let mut state = dummy_state();
			state.epochs_passed = epochs_passed;
			state
		}

		#[test]
		fn eval_true() {
			assert_eq!(EpochsPassed(0).evaluate(&epochs_passed_ctx(42)), true)
		}

		#[test]
		fn eval_barely_true() {
			assert_eq!(EpochsPassed(42).evaluate(&epochs_passed_ctx(42)), true)
		}

		#[test]
		fn eval_false() {
			assert_eq!(EpochsPassed(1337).evaluate(&epochs_passed_ctx(42)), false)
		}

		#[test]
		fn eval_barely_false() {
			assert_eq!(EpochsPassed(43).evaluate(&epochs_passed_ctx(42)), false)
		}

		#[test]
		fn before_and_after_passed() {
			let mut ctx    = epochs_passed_ctx(41);
			let mut cond   = EpochsPassed(42);
			assert_eq!(cond.evaluate(&ctx), false);
			ctx.epochs_passed += 1;
			assert_eq!(cond.evaluate(&ctx), true);
			ctx.epochs_passed += 1;
			assert_eq!(cond.evaluate(&ctx), true);
		}
	}

	mod time_interval {
		use super::*;

		#[test]
		fn eval_true() {
			let mut cond = TimeInterval::new(time::Duration::from_secs(0));
			assert_eq!(cond.evaluate(&dummy_state()), true);
			assert_eq!(cond.evaluate(&dummy_state()), true);
		}

		#[test]
		fn eval_false() {
			let mut cond = TimeInterval::new(time::Duration::from_secs(42));
			assert_eq!(cond.evaluate(&dummy_state()), false);
			assert_eq!(cond.evaluate(&dummy_state()), false);
		}

		#[test]
		fn interval() {
			// Similar to `before_and_after_elapsed()`, the clock on CI might start from 0s,
			// we have to pick a really small duration for this test not to panic.
			let dur_in_s = 10;
			let     dur  = time::Duration::from_secs(dur_in_s);
			let half_dur = time::Duration::from_secs(dur_in_s / 2);
			let mut cond = TimeInterval::new(dur);
			assert_eq!(cond.evaluate(&dummy_state()), false);
			cond.latest -= dur;
			assert_eq!(cond.evaluate(&dummy_state()), true);
			assert_eq!(cond.evaluate(&dummy_state()), false);
			cond.latest -= half_dur;
			assert_eq!(cond.evaluate(&dummy_state()), false);
			cond.latest -= half_dur;
			assert_eq!(cond.evaluate(&dummy_state()), true);
			assert_eq!(cond.evaluate(&dummy_state()), false);
		}
	}

	mod below_recent_mse {
		use super::*;

		#[test]
		fn construct_invalid_target() {
			assert_eq!(
				BelowRecentMSE::new(0.5, 0.0),
				Err(Error::invalid_below_recent_mse_target(0.0))
			);
			assert_eq!(
				BelowRecentMSE::new(0.5, -1.0),
				Err(Error::invalid_below_recent_mse_target(-1.0))
			);
		}

		#[test]
		fn construct_invalid_momentum() {
			let eps = 1e-4;

			assert_eq!(
				BelowRecentMSE::new(-1.0, 0.5),
				Err(Error::invalid_below_recent_mse_momentum(-1.0))
			);
			assert_eq!(
				BelowRecentMSE::new(0.0-eps, 0.5),
				Err(Error::invalid_below_recent_mse_momentum(0.0-eps))
			);
			assert_eq!(
				BelowRecentMSE::new(1.0, 0.5),
				Err(Error::invalid_below_recent_mse_momentum(1.0))
			);
			assert_eq!(
				BelowRecentMSE::new(1.0+eps, 0.5),
				Err(Error::invalid_below_recent_mse_momentum(1.0+eps))
			);
			assert_eq!(
				BelowRecentMSE::new(7.7, 0.5),
				Err(Error::invalid_below_recent_mse_momentum(7.7))
			);
		}

		#[test]
		fn construct_ok() {
			let eps = 1e-4;

			assert_eq!(
				BelowRecentMSE::new(0.0, 0.0 + eps),
				Ok(BelowRecentMSE{momentum: 0.0, target: 0.0 + eps, rmse: 1.0})
			);
			assert_eq!(
				BelowRecentMSE::new(0.0, 1.0),
				Ok(BelowRecentMSE{momentum: 0.0, target: 1.0, rmse: 1.0})
			);
			assert_eq!(
				BelowRecentMSE::new(0.5, 0.5),
				Ok(BelowRecentMSE{momentum: 0.5, target: 0.5, rmse: 1.0})
			);
			assert_eq!(
				BelowRecentMSE::new(1.0-eps, 0.5),
				Ok(BelowRecentMSE{momentum: 1.0-eps, target: 0.5, rmse: 1.0})
			);
		}

		fn latest_mse_context<MSE: Into<MeanSquaredError>>(latest_mse: MSE) -> DummyContext {
			let mut ctx = dummy_state();
			ctx.latest_mse = latest_mse.into();
			ctx
		}

		#[test]
		fn eval_true() {
			let mut cond = BelowRecentMSE::new(0.0, 0.5).unwrap();
			assert_eq!(cond.evaluate(&latest_mse_context(0.05)), true);
			assert_eq!(cond.evaluate(&latest_mse_context(0.49)), true);
			assert_eq!(cond.evaluate(&latest_mse_context(0.50)), true);
		}

		#[test]
		fn eval_false_to_true_high_momentum() {
			let mut cond = BelowRecentMSE::new(0.5, 0.5).unwrap();
			assert_eq!(cond.evaluate(&latest_mse_context(0.05)), false);
			assert_eq!(cond.evaluate(&latest_mse_context(0.05)), true);
		}

		#[test]
		fn eval_false() {
			let mut cond = BelowRecentMSE::new(0.0, 0.5).unwrap();
			assert_eq!(cond.evaluate(&latest_mse_context(0.51)), false);
			assert_eq!(cond.evaluate(&latest_mse_context(1.00)), false);
		}

		#[test]
		fn eval_true_to_false_high_momentum() {
			let mut cond = BelowRecentMSE::new(0.5, 0.75).unwrap();
			assert_eq!(cond.evaluate(&latest_mse_context(0.2)), true);
			assert_eq!(cond.evaluate(&latest_mse_context(1.0)), false);
		}
	}
}
