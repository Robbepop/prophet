use ndarray::prelude::*;

use rand::{Rng, ThreadRng, thread_rng};

use mentor::configs::Scheduling;

// /// Mentors are objects that train a given disciple structure
// /// resulting in a prophet structure that can be used to predict
// /// data.
// /// The static type of the trainable `Disciple` and the resuting `Prophet`
// /// has to be known.
// ///
// /// Mentors define different criteria under which a disciple is
// /// meant to be fully (or well-enough) trained to become a prophet.
// ///
// /// A naive implementation is the `AvgNetErrorMentor` that simply
// /// trains its disciple until the average net error decreases below
// /// a given value. For this the mentor requires some sample training pieces.
// trait Mentor {
// 	type D: Disciple;
// 	type P: Prophet;

/// A sample used to train a disciple during supervised learning.
#[derive(Debug, Clone)]
pub struct Sample {
	/// The input parameter of this `Sample`.
	pub input: Array1<f32>,

	/// The expected target values of this `Sample`.
	pub target: Array1<f32>,
}

impl<A1, A2> From<(A1, A2)> for Sample
    where A1: Into<Vec<f32>>,
          A2: Into<Vec<f32>>
{
	fn from(from: (A1, A2)) -> Sample {
		Sample {
			input : Array1::from_vec(from.0.into()),
			target: Array1::from_vec(from.1.into()),
		}
	}
}

/// A sample view used to train a disciple during supervised learning.
/// 
/// Views are non-owning.
#[derive(Debug, Clone)]
pub struct SampleView<'a> {
	/// The input parameter of this `SampleView`.
	pub input: ArrayView1<'a, f32>,

	/// The expected target values of this `SampleView`.
	pub target: ArrayView1<'a, f32>,
}

impl<'a> From<&'a Sample> for SampleView<'a> {
	fn from(from: &'a Sample) -> SampleView<'a> {
		SampleView {
			input: from.input.view(),
			target: from.target.view(),
		}
	}
}


/// Creates a vector of samples.
/// 
/// Given the following definitions
/// 
/// ```rust,no_run
/// # #[allow(unused_variables)]
/// let t =  1.0;
/// # #[allow(unused_variables)]
/// let f = -1.0;
/// ```
/// ... this macro invokation ...
/// 
/// ```rust
/// # #[macro_use]
/// # extern crate prophet;
/// # use prophet::prelude::*;
/// # fn main() {
/// # let t =  1.0;
/// # let f = -1.0;
/// # #[allow(unused_variables)]
/// let samples = samples![
/// 	[f, f] => [f],
/// 	[t, f] => [t],
/// 	[f, t] => [t],
/// 	[t, t] => [f]
/// ];
/// # }
/// ```
/// 
/// ... will expand to this
/// 
/// ```rust,no_run
/// # extern crate prophet;
/// # use prophet::prelude::*;
/// # fn main() {
/// # let t =  1.0;
/// # let f = -1.0;
/// # #[allow(unused_variables)]
/// let samples = vec![
/// 	Sample::from((vec![f, f], vec![f])),
/// 	Sample::from((vec![t, f], vec![t])),
/// 	Sample::from((vec![f, t], vec![t])),
/// 	Sample::from((vec![t, t], vec![f])),
/// ];
/// # }
/// ```
#[macro_export]
macro_rules! samples {
	[
		$(
			[ $($i:expr),+ ] => [ $($e:expr),+ ]
		),+
	] => {
		vec![$(
			Sample::from(
				(
					vec![$($i),+],
					vec![$($e),+]
				)
			)
		),+]
	};
}

/// A scheduler for indices with a scheduling strategy.
///
/// Used by `SampleScheduler` to pick samples with different scheduling strategies.
#[derive(Clone)]
enum Scheduler {
	/// Samples randomly.
	Random(ThreadRng),

	/// Samples iteratively.
	Iterative(u64),
}

impl ::std::fmt::Debug for Scheduler {
	fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
		use self::Scheduler::*;
		match self {
			&Random(_)    => write!(f, "Scheduler::Random(_)"),
			&Iterative(x) => write!(f, "Scheduler::Iterative({})", x),
		}
	}
}

impl Scheduler {
	/// Creates a new `Scheduler` from a given scheduling strategy.
	fn from_kind(kind: Scheduling) -> Self {
		use mentor::configs::Scheduling::*;
		match kind {
			Random    => Scheduler::Random(thread_rng()),
			Iterative => Scheduler::Iterative(0),
		}
	}

	/// Returns the next scheduled index.
	///
	/// The returned index is then used by the `SampleScheduler`
	/// to pick the associated sample.
	fn next(&mut self, num_samples: usize) -> usize {
		use self::Scheduler::*;
		match self {
			&mut Random(ref mut rng) => {
				rng.gen_range(0, num_samples)
			},
			&mut Iterative(ref mut cur) => {
				let next = *cur as usize % num_samples;
				*cur += 1;
				next
			}
		}
	}
}

/// Organizes the scheduling of samples with different strategies.
#[derive(Debug, Clone)]
pub struct SampleScheduler {
	samples  : Vec<Sample>,
	scheduler: Scheduler,
}

impl SampleScheduler {
	/// Creates a new `SampleScheduler` from given samples and a scheduling strategy.
	pub fn from_samples(kind: Scheduling, samples: Vec<Sample>) -> Self {
		SampleScheduler {
			samples: samples,
			scheduler: Scheduler::from_kind(kind),
		}
	}

	/// Returns the next sample.
	pub fn next(&mut self) -> SampleView {
		let len_samples = self.samples.len();
		let id = self.scheduler.next(len_samples);
		(&self.samples[id]).into()
	}
}
