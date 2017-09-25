pub(crate) mod utils;
mod traits;
mod layer;

mod fully_connected_layer;
mod activation_layer;
mod container_layer;

pub(crate) use self::fully_connected_layer::FullyConnectedLayer;
pub(crate) use self::activation_layer::ActivationLayer;
pub(crate) use self::container_layer::ContainerLayer;
