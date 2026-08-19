use crate::ensemble::{BooleanEnsemble, EnsembleConfig, EnsembleError, RegressionHead};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BRegressorError {
    InvalidConfig,
    InvalidInput,
    NonFiniteTarget,
    PendingPrediction,
    NoPendingPrediction,
    Build,
    Expert,
}

impl From<EnsembleError> for BRegressorError {
    fn from(error: EnsembleError) -> Self {
        match error {
            EnsembleError::InvalidConfig => Self::InvalidConfig,
            EnsembleError::InvalidInput => Self::InvalidInput,
            EnsembleError::PendingPrediction => Self::PendingPrediction,
            EnsembleError::NoPendingPrediction => Self::NoPendingPrediction,
            EnsembleError::Build => Self::Build,
            EnsembleError::Expert => Self::Expert,
        }
    }
}

/// Online regression over an ensemble of batch-learned conjunction experts.
#[derive(Debug)]
pub struct BRegressor {
    ensemble: BooleanEnsemble<RegressionHead>,
}

impl BRegressor {
    pub(crate) fn new(
        source_feature_count: usize,
        config: EnsembleConfig,
    ) -> Result<Self, BRegressorError> {
        Ok(Self {
            ensemble: BooleanEnsemble::new(
                source_feature_count,
                RegressionHead::new(config.max_functions),
                config,
                0,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_hyperparameters(
        source_feature_count: usize,
        learning_rate: f32,
        l2: f32,
        batch_size: usize,
        sgd_steps: usize,
        max_conjunctions: usize,
        max_conjunction_length: usize,
        max_functions: usize,
        max_experts: usize,
        stale_layers: usize,
    ) -> Result<Self, BRegressorError> {
        Self::new(
            source_feature_count,
            EnsembleConfig {
                learning_rate,
                l2,
                sgd_steps,
                batch_size,
                max_conjunctions,
                max_conjunction_length,
                max_functions,
                max_experts,
                stale_layers,
            },
        )
    }

    #[must_use]
    pub fn intercept(&self) -> f32 {
        self.ensemble.head.intercept
    }

    #[must_use]
    pub fn n_observed(&self) -> usize {
        self.ensemble.n_observed
    }

    #[must_use]
    pub fn function_count(&self) -> usize {
        self.ensemble.head.active
    }

    #[must_use]
    pub fn weight(&self, index: usize) -> Option<f32> {
        if index >= self.ensemble.head.active {
            return None;
        }
        Some(self.ensemble.head.weights[index])
    }

    pub fn predict(&mut self, features: &[bool]) -> Result<f32, BRegressorError> {
        self.ensemble.begin_predict(features)?;
        let count = self.ensemble.head.active;
        Ok(self
            .ensemble
            .head
            .predict(&self.ensemble.workspace.pending_function_values[..count]))
    }

    pub fn update(&mut self, target: f32) -> Result<(), BRegressorError> {
        if !target.is_finite() {
            return Err(BRegressorError::NonFiniteTarget);
        }
        self.ensemble.update(target).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::BRegressor;

    fn model(batch_size: usize, max_functions: usize) -> BRegressor {
        BRegressor::with_hyperparameters(2, 0.1, 0.0, batch_size, 1, 8, 7, max_functions, 64, 2)
            .unwrap()
    }

    fn weights(model: &mut BRegressor) -> &mut [f32] {
        &mut model.ensemble.head.weights[..model.ensemble.head.active]
    }

    fn step(model: &mut BRegressor, features: &[bool], target: f32) {
        model.predict(features).unwrap();
        model.update(target).unwrap();
    }

    #[test]
    fn batch_fill_excludes_new_function_from_sgd() {
        let mut model = model(2, 8);
        step(&mut model, &[false, false], -1.0);
        step(&mut model, &[true, false], 1.0);
        assert_eq!(model.function_count(), 1);
        assert_eq!(model.weight(0), Some(0.0));
    }

    #[test]
    fn ensemble_appends_every_batch() {
        let mut model = model(1, 3);
        step(&mut model, &[false, true], 1.0);
        step(&mut model, &[true, false], -1.0);
        step(&mut model, &[false, false], 0.0);
        step(&mut model, &[true, true], 1.0);
        assert_eq!(model.function_count(), 3);
    }

    #[test]
    fn prunes_smallest_abs_weight() {
        let mut model = model(1, 2);
        step(&mut model, &[false, true], 1.0);
        step(&mut model, &[true, false], -1.0);
        weights(&mut model)[0] = 2.0;
        step(&mut model, &[false, false], 0.0);
        assert_eq!(model.function_count(), 2);
        assert!(model.weight(0).unwrap().abs() >= model.weight(1).unwrap().abs());
    }

    #[test]
    fn prune_tiebreak_oldest_at_equal_abs_weight() {
        let mut model = model(1, 2);
        step(&mut model, &[false, true], 1.0);
        step(&mut model, &[true, false], -1.0);
        weights(&mut model).fill(0.0);
        step(&mut model, &[false, false], 0.0);
        assert_eq!(model.function_count(), 2);
    }

    #[test]
    fn update_requires_preceding_predict() {
        let mut model = model(2, 8);
        assert_eq!(
            model.update(-1.0),
            Err(super::BRegressorError::NoPendingPrediction)
        );
        model.predict(&[false, false]).unwrap();
        assert_eq!(
            model.predict(&[true, false]),
            Err(super::BRegressorError::PendingPrediction)
        );
        model.update(-1.0).unwrap();
    }

}
