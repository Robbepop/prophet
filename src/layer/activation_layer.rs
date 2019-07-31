use crate::{
    activation::Activation,
    errors::Result,
    layer::{
        traits::prelude::*,
        utils::prelude::*,
    },
    topology_v4::{
        self,
        LayerSize,
    },
    utils::{
        LearnMomentum,
        LearnRate,
    },
};

/// Activation layers simply apply their activation function onto incoming signals.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationLayer {
    /// These are only required for a correct implementation of the back propagation
    /// algorithm where the derived activation function is applied to the net value
    /// instead of the output signal, so with the current design we have to store both.
    ///
    /// Maybe this situation could be improved in the future by using references
    /// or shared ownership of the input with the previous layer ... but then again
    /// we had to know our previous layer.
    inputs: BiasedSignalBuffer,
    /// The outputs of this activation layer.
    ///
    /// This is basically equivalent to the input transformed with the activation
    /// of this layer.
    outputs: BiasedSignalBuffer,
    /// The buffer for the back propagated error signal.
    error_signal: BiasedErrorSignalBuffer,
    /// The activation function of this `ActivationLayer`.
    act: Activation,
}

impl ActivationLayer {
    /// Creates a new `ActivationLayer` with the given number of inputs
    /// and outputs (not respecting the bias input and output) and given
    /// an activation function.
    pub fn new<L>(len: L, act: Activation) -> Result<Self>
    where
        L: Into<LayerSize>,
    {
        let len = len.into();
        Ok(ActivationLayer {
            inputs: BiasedSignalBuffer::zeros_with_bias(len.to_usize())?,
            outputs: BiasedSignalBuffer::zeros_with_bias(len.to_usize())?,
            error_signal: BiasedErrorSignalBuffer::zeros_with_bias(len.to_usize())?,
            act,
        })
    }

    /// Creates a new `ActivationLayer` from the given topology based abstract activation layer.
    pub fn from_top_layer(
        top_layer: topology_v4::ActivationLayer,
    ) -> Result<ActivationLayer> {
        use crate::topology_v4::Layer;
        ActivationLayer::new(top_layer.input_len(), top_layer.activation_fn())
    }
}

impl From<topology_v4::ActivationLayer> for ActivationLayer {
    fn from(top_layer: topology_v4::ActivationLayer) -> ActivationLayer {
        ActivationLayer::from_top_layer(top_layer)
            .expect("Expected a well-formed abstracted topology ActivationLayer.")
    }
}

impl ProcessInputSignal for ActivationLayer {
    fn process_input_signal(&mut self, input_signal: BiasedSignalView) {
        if self.inputs() != input_signal.dim() {
            panic!("Error: unmatching signals to layer size") // TODO: Replace this with error.
        }
        let act = self.act; // Required since borrow-checker doesn't allow
                            // using `self.act` within method-call context.
        use ndarray::Zip;
        Zip::from(&mut self.inputs.unbias_mut().data_mut())
            .and(&mut self.outputs.unbias_mut().data_mut())
            .and(&input_signal.unbias().data())
            .apply(|s_i, s_o, &p_i| {
                *s_i = p_i; // Note: Required for correct back propagation!
                *s_o = act.base(p_i);
            });
    }
}

impl CalculateOutputErrorSignal for ActivationLayer {
    fn calculate_output_error_signal(&mut self, target_signals: UnbiasedSignalView) {
        if self.outputs() != target_signals.dim() {
            // Note: Target signals do not respect bias values.
            //       We could model this in a way that `target_signals` are simply one element shorter.
            //       Or they have also `1.0` as their last element which eliminates
            //       the resulting error signal's last element to `0.0`.
            panic!("Error: unmatching length of output signal and target signal") // TODO: Replace this with error.
        }
        use ndarray::Zip;
        Zip::from(&mut self.error_signal.unbias_mut().data_mut())
            .and(&self.outputs.unbias().data())
            .and(&target_signals.data())
            .apply(|e, &o, &t| *e = t - o);
    }
}

impl PropagateErrorSignal for ActivationLayer {
    fn propagate_error_signal<P>(&mut self, propagated: &mut P)
    where
        P: HasErrorSignal,
    {
        if self.inputs() != propagated.error_signal().dim() {
            panic!("Error: unmatching signals to layer size") // TODO: Replace this with error.
        }
        use ndarray::Zip;
        let act = self.act;
        // Calculate the gradients and multiply them to the current
        // error signal and propagate the result to the next layer.
        Zip::from(&mut propagated.error_signal_mut().data_mut())
            .and(&self.inputs.data())
            .and(&self.error_signal.data())
            .apply(|o_e, &s_n, &s_e| *o_e += s_e * act.derived(s_n));
        // We need to set the error signals for `ActivationLayer`s to zero
        // since it would corrupt error signal propagation without applying
        // error signal correction afterwards which is a known optimization
        // and thus used often.
        // Note: This applies only to `ActivationLayer`s!
        self.error_signal_mut().reset_to_zeros()
    }
}

impl ApplyErrorSignalCorrection for ActivationLayer {
    fn apply_error_signal_correction(
        &mut self,
        _signal: BiasedSignalView,
        _lr: LearnRate,
        _lm: LearnMomentum,
    ) {
        // Nothing to do here since there are no weights that could be updated!
    }
}

impl HasOutputSignal for ActivationLayer {
    fn output_signal(&self) -> BiasedSignalView {
        self.outputs.view()
    }

    fn output_signal_mut(&mut self) -> BiasedSignalViewMut {
        self.outputs.view_mut()
    }
}

impl HasErrorSignal for ActivationLayer {
    fn error_signal(&self) -> BiasedErrorSignalView {
        self.error_signal.view()
    }

    fn error_signal_mut(&mut self) -> BiasedErrorSignalViewMut {
        self.error_signal.view_mut()
    }
}

impl SizedLayer for ActivationLayer {
    fn inputs(&self) -> usize {
        self.inputs.dim()
    }

    fn outputs(&self) -> usize {
        self.outputs.dim()
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    #[ignore]
    fn new() {
        // Cannot fail anymore with the use of LayerSize.
    }

    #[test]
    #[ignore]
    fn from_top_layer() {}

    #[test]
    #[ignore]
    fn from() {}

    #[test]
    #[ignore]
    fn inputs() {}

    #[test]
    #[ignore]
    fn outputs() {}

    #[test]
    #[ignore]
    fn output_signal() {}

    #[test]
    #[ignore]
    fn error_signal() {}

    #[test]
    #[ignore]
    fn process_input_signal() {}

    #[test]
    #[ignore]
    fn calculate_output_error_signal() {}

    #[test]
    #[ignore]
    fn propagate_error_signal() {}

    #[test]
    #[ignore]
    fn apply_error_signal_correction() {}
}
