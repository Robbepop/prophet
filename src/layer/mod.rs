mod signal_buffer;
mod error_signal_buffer;
mod matrix_base;
mod traits;
mod layer;

mod fully_connected_layer;
mod activation_layer;

// pub use self::signal_buffer::SignalBuffer;
// pub use self::error_signal_buffer::ErrorSignalBuffer;
// pub use self::matrix_base::WeightsMatrix;

pub use self::fully_connected_layer::FullyConnectedLayer;
pub use self::activation_layer::ActivationLayer;

// pub(crate) use self::traits::{
// 	ProcessInputSignal,
// 	HasOutputSignal,
// 	CalculateOutputErrorSignal,
// 	HasErrorSignal,
// 	PropagateErrorSignal,
// 	ApplyErrorSignalCorrection
// };

