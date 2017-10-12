
use std::time::{Duration};

use errors::ErrorKind::{InvalidLatestMSE, InvalidRecentMSE};
use errors::Result;
use traits::{LearnRate, LearnMomentum};

/// Cirterias after which the learning process holds.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Criterion {
	/// Stop after the given duration of time.
	TimeOut(Duration),

	/// Stop after the given amount of learning iterations.
	Iterations(u64),

	/// Stop when the latest mean square error drops below the given value.
	LatestMSE(f64),

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
			LatestMSE(mse) => {
				if mse > 0.0 && mse < 1.0 {
					Ok(())
				} else {
					Err(InvalidLatestMSE)
				}
			}
			RecentMSE(recent) => {
				if recent > 0.0 && recent < 1.0 {
					Ok(())
				} else {
					Err(InvalidRecentMSE)
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
