mod any_layer;
mod traits;
pub(crate) mod utils;

mod activation_layer;
mod container_layer;
mod fully_connected_layer;

pub(crate) use self::{
    activation_layer::ActivationLayer,
    any_layer::AnyLayer,
    container_layer::ContainerLayer,
    fully_connected_layer::FullyConnectedLayer,
};

pub(crate) use self::traits::{
    ApplyErrorSignalCorrection,
    CalculateOutputErrorSignal,
    HasOutputSignal,
    ProcessInputSignal,
};
