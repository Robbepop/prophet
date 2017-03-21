#![feature(test)]

extern crate prophet;
extern crate test;

mod neural_net {
	use prophet::prelude::*;

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
}