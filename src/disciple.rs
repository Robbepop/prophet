//! Provides operations, data structures and error definitions for Disciple objects
//! which form the basis for topologies of neural networks.

use std::slice::Iter;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::error::Error;

/// Errors that can happen during operations performed on Disciple objects.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum DiscipleError {
	/// Error indicating that too few layers were given during Disciple construction.
	TooFewLayers(usize),
	/// Error indicating that too few input neurons were specified during Disciple construction.
	TooFewInputNeurons(usize),
	/// Error indicating that too few output neurons were specified during Disciple construction.
	TooFewOutputNeurons(usize)
}

/// Result type for disciple operations.
pub type DiscipleResult<T> = Result<T, DiscipleError>;

impl Display for DiscipleError {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		use self::DiscipleError::*;
		match *self {
			TooFewLayers(count_layers) => write!(f,
				"Too few layers specified! Only {} were given.", count_layers),
			TooFewInputNeurons(count_inputs) => write!(f,
				"Too few input neurons specified! Only {} were given.", count_inputs),
			TooFewOutputNeurons(count_outputs) => write!(f,
				"Too few output neurons specified! Only {} were given.", count_outputs)
		}
	}
}

impl Error for DiscipleError {
	fn description(&self) -> &str {
		"Error during construction of a Disciple object."
	}

	fn cause(&self) -> Option<&Error> {
		None
	}
}

/// Disciples represent a topological structure for a neural network.
/// They can be trained to become fully qualified Prophet's that may predict data.
#[derive(Debug, Clone, PartialEq)]
pub struct Disciple {
	layer_sizes: Vec<usize>
}

impl Disciple {
	/// Creates a new Disciple instance with a topology defined
	/// as the given vector of layer sizes.
	pub fn from_vec(layer_sizes: Vec<usize>) -> DiscipleResult<Self> {
		use self::DiscipleError::*;
		let count_layers = layer_sizes.len();
		match count_layers {
			0...1 => Err(TooFewLayers(count_layers)),
			_     => {
				let &count_inputs  = layer_sizes.first().unwrap_or(&0);
				let &count_outputs = layer_sizes.last().unwrap_or(&0);
				if      count_inputs  == 0 { Err(TooFewInputNeurons(count_inputs))   }
				else if count_outputs == 0 { Err(TooFewOutputNeurons(count_outputs)) }
				else { Ok(Disciple{ layer_sizes: layer_sizes }) }
			}
		}
	}

	/// Iterates over the layer sizes of this Disciple's topology definition.
	pub fn iter_layer_sizes<'a>(&'a self) -> Iter<'a, usize> {
		self.layer_sizes.iter()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn from_vec_result() {
		use super::DiscipleError::*;
		assert_eq!(Disciple::from_vec(Vec::new()), Err(TooFewLayers(0)));
		assert_eq!(Disciple::from_vec(vec!(1; 1)), Err(TooFewLayers(1)));
		assert_eq!(Disciple::from_vec(vec!(0, 1)), Err(TooFewInputNeurons(0)));
		assert_eq!(Disciple::from_vec(vec!(1, 0)), Err(TooFewOutputNeurons(0)));
		assert_eq!(Disciple::from_vec(vec!(1, 1)), Ok(Disciple{ layer_sizes: vec!(1, 1) }));
	}
}
