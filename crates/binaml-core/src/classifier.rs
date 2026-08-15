use crate::ensemble::{BooleanEnsemble, ClassificationHead, EnsembleConfig, EnsembleError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BClassifierError {
    InvalidConfig,
    InvalidInput,
    InvalidTarget,
    Build,
    Compact,
}

impl From<EnsembleError> for BClassifierError {
    fn from(error: EnsembleError) -> Self {
        match error {
            EnsembleError::InvalidConfig => Self::InvalidConfig,
            EnsembleError::InvalidInput => Self::InvalidInput,
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
                ClassificationHead::new(n_classes),
                config,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_hyperparameters(
        source_feature_count: usize,
        n_classes: usize,
        learning_rate: f64,
        l2: f64,
        batch_size: usize,
        sgd_steps: usize,
        parent_top_k: usize,
        max_layers: usize,
        max_functions: usize,
    ) -> Result<Self, BClassifierError> {
        Self::new(
            source_feature_count,
            n_classes,
            EnsembleConfig {
                learning_rate,
                l2,
                sgd_steps,
                batch_size,
                max_layers,
                parent_top_k,
                max_functions,
            },
        )
    }

    #[must_use]
    pub fn n_observed(&self) -> usize {
        self.ensemble.n_observed
    }

    #[must_use]
    pub fn function_count(&self) -> usize {
        self.ensemble.functions.len()
    }

    #[must_use]
    pub fn intercept(&self, class_index: usize) -> Option<f64> {
        self.ensemble.head.intercepts.get(class_index).copied()
    }

    #[must_use]
    pub fn weight(&self, function_index: usize, class_index: usize) -> Option<f64> {
        self.ensemble
            .head
            .weights
            .get(function_index)
            .and_then(|class_weights| class_weights.get(class_index).copied())
    }

    pub fn predict(&self, features: &[bool]) -> Result<usize, BClassifierError> {
        self.ensemble.validate_features(features)?;
        Ok(self
            .ensemble
            .head
            .predict(&self.ensemble.function_values(features)))
    }

    pub fn observe(&mut self, features: &[bool], target: usize) -> Result<(), BClassifierError> {
        if target >= self.ensemble.head.n_classes {
            return Err(BClassifierError::InvalidTarget);
        }
        self.ensemble.observe(features, target).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::BClassifier;

    fn model(batch_size: usize, max_functions: usize) -> BClassifier {
        BClassifier::with_hyperparameters(2, 3, 0.1, 0.0, batch_size, 1, 2, 1, max_functions).unwrap()
    }

    #[test]
    fn batch_fill_excludes_new_function_from_sgd() {
        let mut model = model(2, 8);
        model.observe(&[false, false], 0).unwrap();
        model.observe(&[true, false], 1).unwrap();
        assert_eq!(model.function_count(), 1);
        assert_eq!(model.weight(0, 0), Some(0.0));
    }

    #[test]
    fn ensemble_appends_every_batch() {
        let mut model = model(1, 3);
        model.observe(&[false, true], 0).unwrap();
        model.observe(&[true, false], 1).unwrap();
        model.observe(&[false, false], 2).unwrap();
        model.observe(&[true, true], 0).unwrap();
        assert_eq!(model.function_count(), 3);
    }
}
