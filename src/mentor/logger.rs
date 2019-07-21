use std::time::{Duration, SystemTime};

use crate::mentor::configs::LogConfig;

/// Status during the learning process.
#[derive(Debug, Copy, Clone)]
pub struct Stats {
	/// Number of samples learned so far.
	pub iterations: u64,

	/// Time passed since beginning of the training.
	pub elapsed_time: Duration,

	/// The latest mean squared error.
	pub latest_mse: f64,

	/// The recent mean squared error.
	pub recent_mse: f64,
}

/// Logger facility for stats logging during the learning process.
#[derive(Debug, Clone)]
pub enum Logger {
	Never,
	TimeSteps {
		last_log: SystemTime,
		interval: Duration,
	},
	Iterations(u64),
}

impl From<LogConfig> for Logger {
	fn from(config: LogConfig) -> Self {
		use self::LogConfig::*;
		match config {
			Never => Logger::Never,
			TimeSteps(duration) => Logger::TimeSteps {
				last_log: SystemTime::now(),
				interval: duration,
			},
			Iterations(interval) => Logger::Iterations(interval),
		}
	}
}

impl Logger {
	fn log(stats: Stats) {
		info!("{:?}\n", stats);
		println!("{:?}", stats)
	}

	pub fn try_log(&mut self, stats: Stats) {
		use self::Logger::*;
		match *self {
			TimeSteps {
				ref mut last_log,
				interval,
			} => {
				if last_log.elapsed().expect("expected valid duration") >= interval {
					Self::log(stats);
					*last_log = SystemTime::now();
				}
			}
			Iterations(interval) => {
				if stats.iterations % interval == 0 {
					Self::log(stats)
				}
			}
			_ => {
				// nothing to do here!
			}
		}
	}
}
