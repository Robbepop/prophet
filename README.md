
PROPHET - Neural Network Library
================================

|       Linux       |       Windows       |       Codecov        |       Coverage       |        Docs        |       Crates.io       |
|:-----------------:|:-------------------:|:--------------------:|:--------------------:|:------------------:|:---------------------:|
| [![travis][1]][2] | [![appveyor][3]][4] | [![codecov][13]][14] | [![coveralls][5]][6] | [![docs][11]][12 ] | [![crates.io][9]][10] |

A simple neural net implementation written in Rust with a focus on cache-efficiency and sequential performance.

Currently only supports supervised learning with fully connected layers.

## How to use

The preferred way to receive prophet is via cargo or github.

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

## Planned Features

- Convolutional Layers: Foundations have been layed out already!
- GPGPU Support by Vulkano
- Even more flexible learning methods

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Dual licence: [![badge][7]](LICENSE-MIT) [![badge][8]](LICENSE-APACHE)

## Release Notes (YYYY/MM/DD)

### 0.4.2 (2017/10/13)

- Relicensed the library under the dual license model where the user can choose between MIT or APACHE version 2.0.
- Improved performance of learning algorithms by up to 27%*. (*Tested on my local machine.)
- Updated ndarray from 0.10.10 to 0.10.11 and itertools from 0.6.5 to 0.7.0.
- Relaxed dependency version constraints for rand, num, log and ndarray.
- Usability: Added a HOW TO USE section to the README.
- Dev
	- Added some unit tests for `NeuralNet` components for improved stability and maintainability.

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
[8]: https://img.shields.io/badge/license-APACHE-orange.svg
[9]: https://img.shields.io/crates/v/prophet.svg
[10]: https://crates.io/crates/prophet
[11]: https://docs.rs/prophet/badge.svg
[12]: https://docs.rs/prophet
[13]: https://codecov.io/gh/robbepop/prophet/branch/next/graph/badge.svg
[14]: https://codecov.io/gh/Robbepop/prophet/branch/next