use crate::{
    layer::utils::prelude::*,
    utils::{
        LearnMomentum,
        LearnRate,
    },
};

pub(crate) trait ProcessInputSignal {
    fn process_input_signal(&mut self, signal: BiasedSignalView);
}

pub(crate) trait HasOutputSignal {
    fn output_signal(&self) -> BiasedSignalView;
    fn output_signal_mut(&mut self) -> BiasedSignalViewMut;
}

pub(crate) trait CalculateOutputErrorSignal {
    fn calculate_output_error_signal(&mut self, target_signals: UnbiasedSignalView);
}

pub(crate) trait HasErrorSignal {
    fn error_signal(&self) -> BiasedErrorSignalView;
    fn error_signal_mut(&mut self) -> BiasedErrorSignalViewMut;
}

pub(crate) trait PropagateErrorSignal {
    fn propagate_error_signal<P>(&mut self, propagated: &mut P)
    where
        P: HasErrorSignal;
}

pub(crate) trait ApplyErrorSignalCorrection {
    fn apply_error_signal_correction(
        &mut self,
        signal: BiasedSignalView,
        lr: LearnRate,
        lm: LearnMomentum,
    );
}

pub(crate) trait SizedLayer {
    fn inputs(&self) -> usize;
    fn outputs(&self) -> usize;
}

pub mod prelude {
    #[doc(no_inline)]
    pub(crate) use super::{
        ApplyErrorSignalCorrection,
        CalculateOutputErrorSignal,
        HasErrorSignal,
        HasOutputSignal,
        ProcessInputSignal,
        PropagateErrorSignal,
        SizedLayer,
    };
}
