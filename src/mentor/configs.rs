
use std::time::{Duration};

use crate::errors::{Result, Error};
use crate::utils::{LearnRate, LearnMomentum};

/// Cirterias after which the learning process holds.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Criterion {
	/// Stop after the given duration of time.
	TimeOut(Duration),

	/// Stop after the given amount of learning iterations.
	Iterations(u64),

	/// Stop as soon as the recent mean squared error
	/// drops below the given value.
	RecentMSE(f64),
}

impl Criterion {
	/// Checks if this criterion is valid.
	pub fn check_validity(&self) -> Result<()> {
		use self::Criterion::*;
		match *self {
			TimeOut(_)    |
			Iterations(_) => Ok(()),
			RecentMSE(recent_mse) => {
				if recent_mse > 0.0 {
					Ok(())
				} else {
					Err(Error::invalid_recent_mse(recent_mse))
				}
			}
		}
	}
}

/// Learning rate configuration.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LearnRateConfig {
	/// Automatically adapt learn rate during learning.
	Adapt,

	/// Use the given fixed learn rate.
	Fixed(LearnRate),
}

/// Learning momentum configuration.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LearnMomentumConfig {
	/// Automatically adapt learn momentum during learning.
	Adapt,

	/// Use the given fixed learn momentum.
	Fixed(LearnMomentum),
}

/// Logging interval for logging stats during the learning process.
/// 
/// Default logging configuration is to never log anything.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LogConfig {
	/// Never log anything.
	Never,

	/// Log in intervals based on the given duration.
	TimeSteps(Duration),

	/// Log every given number of training iterations.
	Iterations(u64)
}

impl Default for LogConfig {
	fn default() -> Self {
		LogConfig::Never
	}
}

/// Sample scheduling strategy while learning.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Scheduling {
	/// Pick samples randomly.
	///
	/// This usually is a good approach to defeat sample-pattern learning.
	Random,

	/// Pick samples in order.
	///
	/// This maybe useful for testing purposes.
	Iterative,
}
