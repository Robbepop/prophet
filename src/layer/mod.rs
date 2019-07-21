mod any_layer;
mod traits;
pub(crate) mod utils;

mod activation_layer;
mod container_layer;
mod fully_connected_layer;

pub(crate) use self::activation_layer::ActivationLayer;
pub(crate) use self::any_layer::AnyLayer;
pub(crate) use self::container_layer::ContainerLayer;
pub(crate) use self::fully_connected_layer::FullyConnectedLayer;

pub(crate) use self::traits::{
	ApplyErrorSignalCorrection, CalculateOutputErrorSignal, HasOutputSignal, ProcessInputSignal,
};
