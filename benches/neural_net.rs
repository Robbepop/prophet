#![feature(test)]

extern crate prophet;
extern crate test;

#[cfg(test)]
mod neural_net {
	use prophet::prelude::*;
	use prophet::internal::*;

	use test::{
		Bencher,
		black_box
	};

	fn create_giant_net() -> NeuralNet {
		use self::Activation::Tanh;
		NeuralNet::from_topology(
			Topology::input(2)
				.layers(&[
					(1000, Tanh),
					(1000, Tanh),
					(1000, Tanh),
					(1000, Tanh),
					(1000, Tanh),
					(1000, Tanh),
					(1000, Tanh),
					(1000, Tanh),
					(1000, Tanh),
					(1000, Tanh)
				])
				.output(1, Tanh))
	}

	#[bench]
	fn predict(bencher: &mut Bencher) {
		let mut net = create_giant_net();
		let (t, f)  = (1.0, -1.0);
		bencher.iter(|| {
			black_box(net.predict(&[f, f]));
			black_box(net.predict(&[f, t]));
			black_box(net.predict(&[t, f]));
			black_box(net.predict(&[t, t]));
		});
	}

	#[bench]
	fn update_gradients(bencher: &mut Bencher) {
		let mut net = create_giant_net();
		bencher.iter(|| {
			net.update_gradients(&[1.0]);
		});
	}

	#[bench]
	fn update_weights(bencher: &mut Bencher) {
		let mut net = create_giant_net();
		bencher.iter(|| {
			net.update_weights(&[1.0], LearnRate::default(), LearnMomentum::default());
		});
	}
}
