use crate::ensemble::{BooleanEnsemble, ClassificationHead, EnsembleConfig, EnsembleError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BClassifierError {
    InvalidConfig,
    InvalidInput,
    InvalidTarget,
    PendingPrediction,
    NoPendingPrediction,
    Build,
    Compact,
}

impl From<EnsembleError> for BClassifierError {
    fn from(error: EnsembleError) -> Self {
        match error {
            EnsembleError::InvalidConfig => Self::InvalidConfig,
            EnsembleError::InvalidInput => Self::InvalidInput,
            EnsembleError::PendingPrediction => Self::PendingPrediction,
            EnsembleError::NoPendingPrediction => Self::NoPendingPrediction,
            EnsembleError::Build => Self::Build,
            EnsembleError::Compact => Self::Compact,
        }
    }
}

/// Online multiclass classification over batch-learned boolean functions.
#[derive(Debug)]
pub struct BClassifier {
    ensemble: BooleanEnsemble<ClassificationHead>,
}

impl BClassifier {
    pub(crate) fn new(
        source_feature_count: usize,
        n_classes: usize,
        config: EnsembleConfig,
    ) -> Result<Self, BClassifierError> {
        if n_classes < 2 {
            return Err(BClassifierError::InvalidConfig);
        }
        Ok(Self {
            ensemble: BooleanEnsemble::new(
                source_feature_count,
                ClassificationHead::new(n_classes, config.max_functions),
                config,
                n_classes,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_hyperparameters(
        source_feature_count: usize,
        n_classes: usize,
        learning_rate: f32,
        l2: f32,
        batch_size: usize,
        sgd_steps: usize,
        parent_top_k: usize,
        max_functions: usize,
        max_expert_nodes: usize,
        l_pat: usize,
    ) -> Result<Self, BClassifierError> {
        Self::new(
            source_feature_count,
            n_classes,
            EnsembleConfig {
                learning_rate,
                l2,
                sgd_steps,
                batch_size,
                parent_top_k,
                max_functions,
                max_expert_nodes,
                l_pat,
            },
        )
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
    pub fn intercept(&self, class_index: usize) -> Option<f32> {
        self.ensemble.head.intercepts.get(class_index).copied()
    }

    #[must_use]
    pub fn weight(&self, function_index: usize, class_index: usize) -> Option<f32> {
        if function_index >= self.ensemble.head.active
            || class_index >= self.ensemble.head.n_classes
        {
            return None;
        }
        Some(self.ensemble.head.expert_weights(function_index)[class_index])
    }

    pub fn predict(&mut self, features: &[bool]) -> Result<usize, BClassifierError> {
        self.ensemble.begin_predict(features)?;
        let count = self.ensemble.head.active;
        Ok(self.ensemble.head.predict_with_scratch(
            &self.ensemble.workspace.pending_function_values[..count],
            &mut self.ensemble.workspace.logits,
        ))
    }

    pub fn update(&mut self, target: usize) -> Result<(), BClassifierError> {
        if target >= self.ensemble.head.n_classes {
            return Err(BClassifierError::InvalidTarget);
        }
        self.ensemble.update(target).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::BClassifier;

    fn model(batch_size: usize, max_functions: usize) -> BClassifier {
        BClassifier::with_hyperparameters(2, 3, 0.1, 0.0, batch_size, 1, 8, max_functions, 64, 2)
            .unwrap()
    }

    fn step(model: &mut BClassifier, features: &[bool], target: usize) {
        model.predict(features).unwrap();
        model.update(target).unwrap();
    }

    #[test]
    fn batch_fill_excludes_new_function_from_sgd() {
        let mut model = model(2, 8);
        step(&mut model, &[false, false], 0);
        step(&mut model, &[true, false], 1);
        assert_eq!(model.function_count(), 1);
        assert_eq!(model.weight(0, 0), Some(0.0));
    }

    #[test]
    fn ensemble_appends_every_batch() {
        let mut model = model(1, 3);
        step(&mut model, &[false, true], 0);
        step(&mut model, &[true, false], 1);
        step(&mut model, &[false, false], 2);
        step(&mut model, &[true, true], 0);
        assert_eq!(model.function_count(), 3);
    }
}
