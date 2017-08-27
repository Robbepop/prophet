
PROPHET - Neural Network Library
================================

|        Linux        |       Windows       |       Coverage       |          Docs         |     Crates.io      |       Licence      |
|:-------------------:|:-------------------:|:--------------------:|:---------------------:|:------------------:|:------------------:|
| [![travisCI][1]][2] | [![appveyor][3]][4] | [![coveralls][5]][6] | [![licence][11]][12 ] | [![chat][9]][10]   | [![licence][7]][8] |


A simple neural net implementation written in Rust with a focus on cache-efficiency and sequential performance.

Currently only supports supervised learning with fully connected layers.

## Planned Features

- Convolutional Layers: Foundations have been layed out already!
- GPGPU Support by Vulkano
- Even more flexible learning methods

## Release Notes (YYYY/MM/DD)

### 0.4.1 (2017/08/27)

- Fixed long-standing undeterministic issue #2.

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
[8]: ./LICENCE
[9]: https://img.shields.io/crates/v/prophet.svg
[10]: https://crates.io/crates/prophet
[11]: https://docs.rs/prophet/badge.svg
[12]: https://docs.rs/prophet
