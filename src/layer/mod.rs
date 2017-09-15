mod signal_buffer;
mod gradient_buffer;
mod weights_matrix;
mod traits;

mod fully_connected_layer;
mod activation_layer;

pub use self::signal_buffer::SignalBuffer;
pub use self::gradient_buffer::GradientBuffer;
pub use self::weights_matrix::WeightsMatrix;

pub use self::fully_connected_layer::FullyConnectedLayer;
pub use self::activation_layer::ActivationLayer;

pub(crate) use self::traits::{
	ProcessSignal,
	HasOutputSignal,
	CalculateErrorGradients,
	HasGradientBuffer,
	PropagateGradients,
	ApplyGradients
};
