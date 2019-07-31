use super::*;

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
                (1000, Tanh),
            ])
            .output(1, Tanh),
    )
}

#[bench]
fn construct(bencher: &mut Bencher) {
    bencher.iter(|| {
        black_box(create_giant_net());
    });
}

#[bench]
fn predict(bencher: &mut Bencher) {
    let mut net = create_giant_net();
    let (t, f) = (1.0, -1.0);
    bencher.iter(|| {
        black_box(net.predict(&[f, f]));
        black_box(net.predict(&[f, t]));
        black_box(net.predict(&[t, f]));
        black_box(net.predict(&[t, t]));
    });
}

#[bench]
fn update_gradients(bencher: &mut Bencher) {
    use traits::UpdateGradients;
    let mut net = create_giant_net();
    bencher.iter(|| {
        net.update_gradients(&[1.0]);
    });
}

#[bench]
fn update_weights(bencher: &mut Bencher) {
    use traits::UpdateWeights;
    use utils::{
        LearnMomentum,
        LearnRate,
    };
    let mut net = create_giant_net();
    bencher.iter(|| {
        net.update_weights(&[1.0, 1.0], LearnRate::default(), LearnMomentum::default());
    });
}
