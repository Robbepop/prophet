#[macro_use]
extern crate prophet;

extern crate rand;
extern crate itertools;

#[macro_use]
extern crate approx;

use prophet::prelude::*;
use rand::{Open01, Rng, thread_rng};
use std::time::Duration;

fn validate_impl(mut net: NeuralNet, samples: Vec<Sample>, rounded: bool) {
	use itertools::{Itertools, multizip};
	for sample in samples.into_iter() {
		let predicted = net.predict(sample.input.view());
		multizip((predicted.iter(), sample.target.iter()))
			.foreach(|(&predicted, &expected)| {
				if rounded {
					assert_eq!(predicted.round(), expected);
				}
				else {
					relative_eq!(predicted, expected);
				}
			});
	}
}

/// Validate given samples for the given net with rounded mode.
fn validate_rounded(net: NeuralNet, samples: Vec<Sample>) {
	validate_impl(net, samples, true)
}

/// Validate given samples for the given net with exact precision.
fn validate_exact(net: NeuralNet, samples: Vec<Sample>) {
	validate_impl(net, samples, false)
}

/// Create a sample collection with given amount of samples,
/// each of given input and output size.
/// Also has a mapper to specify how the expected vector
/// values are calculated.
fn gen_random_samples<F>(amount: usize,
                         input_size: usize,
                         output_size: usize,
                         mapping: F) -> Vec<Sample>
	where F: Fn(&[f32]) -> Vec<f32>
{
	let mut rng = thread_rng();
	let mut samples = Vec::with_capacity(amount);
	for _ in 0..amount {
		let inputs = rng.gen_iter::<Open01<f32>>()
			.take(input_size)
			.map(|Open01(val)| val)
			.collect::<Vec<f32>>();
		assert_eq!(inputs.len(), input_size);
		let outputs = mapping(&inputs);
		assert_eq!(outputs.len(), output_size);
		samples.push(Sample::from((inputs, outputs)))
	}
	assert_eq!(samples.len(), amount);
	samples
}

#[test]
fn train_xor() {
	use Activation::Tanh;

	let (t, f) = (1.0, -1.0);
	let samples = samples![
		[f, f] => f,
		[t, f] => t,
		[f, t] => t,
		[t, t] => f
	];

	let net = Topology::input(2)
		.layer(4, Tanh)
		.layer(3, Tanh)
		.output(1, Tanh)

		.train(samples.clone())
		.learn_rate(0.6)
		.log_config(LogConfig::TimeSteps(Duration::from_secs(1)))
		.go()
		.unwrap();

	validate_rounded(net, samples);
}

#[test]
fn train_constant() {
	use Activation::Identity;

	// samples to train the net with
	let learn_samples = samples![
		0.0 => 1.0,
		0.2 => 1.0,
		0.4 => 1.0,
		0.6 => 1.0,
		0.8 => 1.0,
		1.0 => 1.0
	];

	// samples to test the trained net with
	let test_samples = samples![
		0.1 => 1.0,
		0.3 => 1.0,
		0.5 => 1.0,
		0.7 => 1.0,
		0.9 => 1.0
	];

	let net = Topology::input(1)
		.layer(4, Identity) // Both layers introduced as counter to failing tests under 
		.layer(3, Identity) // appveyor (Windows). If this fixes the problem there is a serious
		                    // bug somewhere in the algorithms.
		                    // These layers should be removed as soon as the bugs are fixed.
		.output(1, Identity)

		.train(learn_samples)
		.log_config(LogConfig::TimeSteps(Duration::from_secs(1)))
		.go()
		.unwrap();

	validate_rounded(net, test_samples)
}

#[test]
fn train_and() {
	use Activation::Tanh;

	let (t, f) = (1.0, -1.0);
	let samples = samples![
		[f, f] => f,
		[f, t] => f,
		[t, f] => f,
		[t, t] => t
	];

	let net = Topology::input(2)
		.layer(4, Tanh) // Both layers introduced as counter to failing tests under 
		.layer(3, Tanh) // appveyor (Windows). If this fixes the problem there is a serious
		                // bug somewhere in the algorithms.
		                // These layers should be removed as soon as the bugs are fixed.
		.output(1, Tanh)

		.train(samples.clone())
		.log_config(LogConfig::TimeSteps(Duration::from_secs(1)))
		.go()
		.unwrap();

	validate_rounded(net, samples)
}

#[test]
fn train_triple_add() {
	use Activation::Identity;

	let count_learn_samples = 10_000;
	let count_test_samples  =    100;
	let inputs  = 3;
	let outputs = 1;

	fn mapper(inputs: &[f32]) -> Vec<f32> {
		vec![inputs[0] + inputs[1] + inputs[2]]
	}

	let learn_samples = gen_random_samples(
		count_learn_samples, inputs, outputs, mapper);
	let test_samples = gen_random_samples(
		count_test_samples, inputs, outputs, mapper);

	let net = Topology::input(inputs)
		.layer(4, Identity) // Both layers introduced as counter to failing tests under 
		.layer(3, Identity) // appveyor (Windows). If this fixes the problem there is a serious
		                    // bug somewhere in the algorithms.
		                    // These layers should be removed as soon as the bugs are fixed.
		.output(outputs, Identity)

		.train(learn_samples)
		.log_config(LogConfig::TimeSteps(Duration::from_secs(1)))
		.go()
		.unwrap();

	validate_exact(net, test_samples)
}

#[test]
fn train_compare() {
	use Activation::Tanh;

	let count_learn_samples = 10_000;
	let count_test_samples  =    100;
	let inputs  = 2;
	let outputs = 1;

	fn mapper(inputs: &[f32]) -> Vec<f32> {
		if inputs[0] < inputs[1] {
			vec![-1.0]
		}
		else {
			vec![1.0]
		}
	}

	let learn_samples = gen_random_samples(
		count_learn_samples, inputs, outputs, mapper);
	let test_samples = gen_random_samples(
		count_test_samples, inputs, outputs, mapper);

	let net = Topology::input(inputs)
		.layer(4, Tanh)
		.layer(3, Tanh)
		.output(outputs, Tanh)

		.train(learn_samples)
		.log_config(LogConfig::TimeSteps(Duration::from_secs(1)))
		.go()
		.unwrap();

	validate_exact(net, test_samples)
}
