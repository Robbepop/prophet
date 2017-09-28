
PROPHET - Neural Network Library
================================

|       Linux       |       Windows       |       Coverage      |          Docs         |     Crates.io         |       Licence      |
|:-----------------:|:-------------------:|:-------------------:|:---------------------:|:---------------------:|:------------------:|
| [![travis][1]][2] | [![appveyor][3]][4] | [![coverage][5]][6] | [![docs][11]][12 ]    | [![crates.io][9]][10] | [![licence][7]][8] |


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
let (t, f) = (1.0, -1.0); // In our samples 1.0 stands for true and -1.0 stands for false.
let samples = samples![
	[f, f] => f, // false XOR false <=> false
	[t, f] => t, // true  XOR false <=> true
	[f, t] => t, // false XOR true  <=> true
	[t, t] => f  // true  XOR true  <=> false
];
```

Then define your neural net topology and start training it ...

```rust
let net =
	Topology::input(2) // 2 input neurons
	.layer(2, Tanh)    // a hidden layer with 2 neurons
	.output(1, Tanh)   // 1 output neuron

	.train(samples.clone())
	.learn_rate(0.6)     // use learn rate of 0.6
	.learn_momentum(0.5) // use learn momentum of 0.5
	.log_config(
		// Log the current state every second
		LogConfig::TimeSteps(Duration::from_secs(1)))
	.go()
	.unwrap();
```

Now you may want to check if learning was successful ...

```rust
for sample in samples {
	let predicted = net.predict(sample.input.view());
	multizip((predicted.iter(), sample.target.iter())).foreach(
		|(&predicted, &expected)| {
			(predicted.round() - expected).abs() < 0.05
		}
	);
}
```

## Planned Features

- Convolutional layers: Foundations have been layed out already!
- Parallel computation via GPGPU support using Vulkano or similar.
- More flexible learning methods.

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
[5]: https://coveralls.io/repos/github/Robbepop/prophet/badge.svg?branch=master
[6]: https://coveralls.io/github/Robbepop/prophet?branch=master
[7]: https://img.shields.io/badge/license-MIT-blue.svg
[8]: https://github.com/Robbepop/prophet/blob/master/LICENSE
[9]: https://img.shields.io/crates/v/prophet.svg
[10]: https://crates.io/crates/prophet
[11]: https://docs.rs/prophet/badge.svg
[12]: https://docs.rs/prophet
