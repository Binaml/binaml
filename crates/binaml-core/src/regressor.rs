use crate::{
    compact, FunctionBuildConfig, FunctionBuilder, FunctionBuildError, FunctionGraph, SignBatch,
};
use crate::function_compact::CompactError;

#[derive(Debug, Clone, Copy)]
struct RegressorConfig {
    learning_rate: f64,
    l2: f64,
    sgd_steps: usize,
    batch_size: usize,
    max_layers: usize,
    parent_top_k: usize,
    max_functions: usize,
}

impl RegressorConfig {
    fn build_config(&self) -> FunctionBuildConfig {
        FunctionBuildConfig {
            batch_size: self.batch_size,
            parent_top_k: self.parent_top_k,
            max_layers: self.max_layers,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BRegressorError {
    InvalidConfig,
    InvalidInput,
    NonFiniteTarget,
    Build,
    Compact,
}

impl From<FunctionBuildError> for BRegressorError {
    fn from(_: FunctionBuildError) -> Self {
        Self::Build
    }
}

impl From<CompactError> for BRegressorError {
    fn from(_: CompactError) -> Self {
        Self::Compact
    }
}

/// Online regression over an ensemble of batch-learned boolean functions.
#[derive(Debug)]
pub struct BRegressor {
    config: RegressorConfig,
    source_feature_count: usize,
    intercept: f64,
    functions: Vec<FunctionGraph>,
    weights: Vec<f64>,
    feature_batch_features: Vec<Vec<bool>>,
    feature_batch_signs: Vec<bool>,
    n_observed: usize,
}

impl BRegressor {
    pub(crate) fn new(
        source_feature_count: usize,
        config: RegressorConfig,
    ) -> Result<Self, BRegressorError> {
        if source_feature_count == 0
            || config.max_functions == 0
            || config.batch_size == 0
            || config.batch_size > crate::FeatureCounter::MAX_BATCH_SIZE
            || !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || !config.l2.is_finite()
            || config.l2 < 0.0
            || config.sgd_steps == 0
            || config.parent_top_k == 0
            || config.max_layers == 0
        {
            return Err(BRegressorError::InvalidConfig);
        }
        Ok(Self {
            config,
            source_feature_count,
            intercept: 0.0,
            functions: Vec::new(),
            weights: Vec::new(),
            feature_batch_features: Vec::with_capacity(config.batch_size),
            feature_batch_signs: Vec::with_capacity(config.batch_size),
            n_observed: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_hyperparameters(
        source_feature_count: usize,
        learning_rate: f64,
        l2: f64,
        batch_size: usize,
        sgd_steps: usize,
        parent_top_k: usize,
        max_layers: usize,
        max_functions: usize,
    ) -> Result<Self, BRegressorError> {
        Self::new(
            source_feature_count,
            RegressorConfig {
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
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    #[must_use]
    pub fn n_observed(&self) -> usize {
        self.n_observed
    }

    #[must_use]
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    #[must_use]
    pub fn weight(&self, index: usize) -> Option<f64> {
        self.weights.get(index).copied()
    }

    pub fn predict(&self, features: &[bool]) -> Result<f64, BRegressorError> {
        self.validate_features(features)?;
        Ok(self.prediction_from_values(&self.evaluate_functions(features)))
    }

    pub fn observe(
        &mut self,
        features: &[bool],
        target: f64,
    ) -> Result<(), BRegressorError> {
        self.validate_features(features)?;
        if !target.is_finite() {
            return Err(BRegressorError::NonFiniteTarget);
        }
        self.n_observed += 1;

        let function_values = self.evaluate_functions(features);
        let sign = target - self.prediction_from_values(&function_values) >= 0.0;
        self.feature_batch_features.push(features.to_vec());
        self.feature_batch_signs.push(sign);

        for _ in 0..self.config.sgd_steps {
            self.update_single_sample(target, &function_values);
        }

        if self.feature_batch_features.len() == self.config.batch_size {
            self.finish_batch()?;
            self.feature_batch_features.clear();
            self.feature_batch_signs.clear();
        }
        Ok(())
    }

    fn validate_features(&self, features: &[bool]) -> Result<(), BRegressorError> {
        (features.len() == self.source_feature_count)
            .then_some(())
            .ok_or(BRegressorError::InvalidInput)
    }

    fn evaluate_functions(&self, features: &[bool]) -> Vec<bool> {
        self.functions
            .iter()
            .map(|function| function.evaluate(features))
            .collect()
    }

    fn prediction_from_values(&self, function_values: &[bool]) -> f64 {
        self.intercept
            + function_values
                .iter()
                .zip(&self.weights)
                .map(|(value, weight)| weight * (2.0 * f64::from(*value) - 1.0))
                .sum::<f64>()
    }

    fn update_single_sample(&mut self, target: f64, function_values: &[bool]) {
        let prediction = self.prediction_from_values(function_values);
        let error = target - prediction;
        let rate = self.config.learning_rate;
        let decay = 1.0 - rate * self.config.l2;
        for weight in &mut self.weights {
            *weight *= decay;
        }
        self.intercept += rate * error;
        for (weight, value) in self.weights.iter_mut().zip(function_values) {
            let centered = 2.0 * f64::from(*value) - 1.0;
            *weight += rate * error * centered;
        }
    }

    fn finish_batch(&mut self) -> Result<(), BRegressorError> {
        let columns: Vec<Vec<bool>> = (0..self.source_feature_count)
            .map(|index| {
                self.feature_batch_features
                    .iter()
                    .map(|row| row[index])
                    .collect()
            })
            .collect();
        let column_refs: Vec<&[bool]> = columns.iter().map(Vec::as_slice).collect();
        let batch = SignBatch {
            feature_columns: &column_refs,
            signs: &self.feature_batch_signs,
        };
        let (ephemeral, output) = FunctionBuilder::build(batch, self.config.build_config())?;
        let graph = compact(ephemeral, output)?;
        self.functions.push(graph);
        self.weights.push(0.0);
        if self.functions.len() > self.config.max_functions {
            let index = self
                .weights
                .iter()
                .enumerate()
                .min_by(|(left_index, left_weight), (right_index, right_weight)| {
                    left_weight
                        .abs()
                        .total_cmp(&right_weight.abs())
                        .then_with(|| left_index.cmp(right_index))
                })
                .map(|(index, _)| index)
                .expect("non-empty ensemble");
            self.functions.remove(index);
            self.weights.remove(index);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::BRegressor;

    fn model(batch_size: usize, max_functions: usize) -> BRegressor {
        BRegressor::with_hyperparameters(2, 0.1, 0.0, batch_size, 1, 2, 1, max_functions).unwrap()
    }

    #[test]
    fn batch_fill_excludes_new_function_from_sgd() {
        let mut model = model(2, 8);
        model.observe(&[false, false], -1.0).unwrap();
        model.observe(&[true, false], 1.0).unwrap();
        assert_eq!(model.function_count(), 1);
        assert_eq!(model.weight(0), Some(0.0));
    }

    #[test]
    fn ensemble_appends_every_batch() {
        let mut model = model(1, 3);
        model.observe(&[false, true], 1.0).unwrap();
        model.observe(&[true, false], -1.0).unwrap();
        model.observe(&[false, false], 0.0).unwrap();
        model.observe(&[true, true], 1.0).unwrap();
        assert_eq!(model.function_count(), 3);
    }

    #[test]
    fn prunes_smallest_abs_weight() {
        let mut model = model(1, 2);
        model.observe(&[false, true], 1.0).unwrap();
        model.observe(&[true, false], -1.0).unwrap();
        model.weights[0] = 2.0;
        model.observe(&[false, false], 0.0).unwrap();
        assert_eq!(model.function_count(), 2);
        assert!(model.weight(0).unwrap().abs() >= model.weight(1).unwrap().abs());
    }

    #[test]
    fn prune_tiebreak_oldest_at_equal_abs_weight() {
        let mut model = model(1, 2);
        model.observe(&[false, true], 1.0).unwrap();
        model.observe(&[true, false], -1.0).unwrap();
        model.weights = vec![0.0, 0.0];
        model.observe(&[false, false], 0.0).unwrap();
        assert_eq!(model.function_count(), 2);
    }
}
