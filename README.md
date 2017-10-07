
PROPHET - Neural Network Library
================================

|       Linux       |       Windows       |       Codecov        |      Coveralls      |          Docs         |     Crates.io         |
|:-----------------:|:-------------------:|:--------------------:|:-------------------:|:---------------------:|:---------------------:|
| [![travis][1]][2] | [![appveyor][3]][4] | [![codecov][14]][15] | [![coverage][5]][6] | [![docs][11]][12]     | [![crates.io][9]][10] |

A simple neural net implementation written in Rust with a focus on cache-efficiency and sequential performance.

Currently only supports supervised learning with fully connected layers.

## How to

The preferred way to set-up prophet is via cargo.
Download with cargo or directly via [crates.io](https://crates.io/crates/prophet).

Compile prophet with

```
cargo build
```

Run the test suite with

```
cargo test --release
```

Note: It is recommended to use `--release` for testing since optimizations are insanely effective for prophet.

For additional information while running some long tests use

```
cargo test --release --verbose -- --nocapture
```

Run performance test with

```
cargo bench --features benches
```

## Example: XOR Training

Define your XOR samples with ...

```rust
let (t, f) = (1.0, -1.0); // 1.0 stands for true (⊤) and -1.0 stands for false (⊥).
let samples = samples![
    [f, f] => f, // ⊥ ⊕ ⊥ ⇒ ⊥
    [t, f] => t, // ⊤ ⊕ ⊥ ⇒ ⊤
    [f, t] => t, // ⊥ ⊕ ⊤ ⇒ ⊤
    [t, t] => f  // ⊤ ⊕ ⊤ ⇒ ⊥
];
```

Then define your neural net topology and start training it ...

```rust
let net = Topology::input(2)           // 2 input neurons
    .fully_connect(2).activation(Tanh) // a hidden layer with 2 neurons
    .fully_connect(1).activation(Tanh) // 1 output neuron

    .train()
    .learn_rate(0.6)     // Use a fixed learn rate of 0.6.
    .learn_momentum(0.5) // Use a fixed learn momentum of 0.5.
    .log_when(TimeInterval::once_in(Duration::from_secs(1))) // Log once every second.
    .stop_when(BelowRecentMSE::new(0.9, 0.05));              // Stop when the recent MSE drops below 0.05.

    .start()   // Start the training process.
    .unwrap(); // Receive the trained neural network.
```

Now you may want to check if learning was successful ...

```rust
for sample in samples {
    assert!(net.predict(sample.input())
        .all_close_to(sample.expected()))
}
```

## Planned Features

- Convolutional layers: Foundations have been layed out already!
- Parallel computation via GPGPU support using Vulkano or similar.
- More flexible learning methods.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Dual licence: [![badge][7]](LICENSE-MIT) [![badge][8]](LICENSE-APACHE)

## Release Notes (YYYY/MM/DD)

### 0.4.1 (2017/08/27)

- Fixed long-standing undeterministic [bug](https://github.com/Robbepop/prophet/issues/2).
- Reverted `ChaChaRng` usage in `NeuralLayer::random` - it is much faster and `ChaChaRng`'s safety is not needed.

### 0.4.0 (2017/08/09)

- Updated `ndarray` dependency version from `0.9` to `0.10`
- Updated `serde` dependency version from `0.9` to `1.0`
- Enabled `serde` feature by default.
- `NeuralLayer::random` now uses `ChaChaRng` internally instead of `weak_rng`
- Devel:
	- travisCI now using new trusty environment
	- travisCI now uploads code coverage to coveralls and codecov.io
	- travisCI no longer requires `sudo`

[1]: https://travis-ci.org/Robbepop/prophet.svg?branch=master
[2]: https://travis-ci.org/Robbepop/prophet
[3]: https://ci.appveyor.com/api/projects/status/2ckrux25wpa5eseh/branch/master?svg=true
[4]: https://ci.appveyor.com/project/Robbepop/prophet/branch/master
[5]: https://coveralls.io/repos/github/Robbepop/prophet/badge.svg?branch=next
[6]: https://coveralls.io/github/Robbepop/prophet?branch=next
[7]: https://img.shields.io/badge/license-MIT-blue.svg
[8]: https://img.shields.io/badge/license-APACHE-orange.svg
[9]: https://img.shields.io/crates/v/prophet.svg
[10]: https://crates.io/crates/prophet
[11]: https://docs.rs/prophet/badge.svg
[12]: https://docs.rs/prophet
[14]: https://codecov.io/gh/robbepop/prophet/branch/next/graph/badge.svg
[15]: https://codecov.io/gh/Robbepop/prophet/branch/next
