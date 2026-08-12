use crate::{
    FeatureId, FeatureLearner, FeatureLearningConfig, FeatureLearningError, FeatureStore, SignBatch,
};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug)]
struct ReplayLayout {
    references: Vec<FeatureId>,
    nodes: Vec<FeatureNode>,
}

#[derive(Debug)]
enum FeatureNode {
    Source(usize),
    Composed {
        first: usize,
        second: usize,
        truth_table: u8,
    },
}

impl ReplayLayout {
    fn new(store: &FeatureStore) -> Result<Self, BRegressorError> {
        let mut references = Vec::new();
        let mut slots = HashMap::new();
        for layer in 0..=store.learned_layer_count() {
            for reference in store.live_refs_in_layer(layer) {
                slots.insert(reference, references.len());
                references.push(reference);
            }
        }
        let nodes = references
            .iter()
            .map(|reference| {
                if reference.layer == 0 {
                    return Ok::<FeatureNode, BRegressorError>(FeatureNode::Source(
                        reference.index,
                    ));
                }
                let feature = store.get(*reference).ok_or(BRegressorError::InvalidInput)?;
                Ok(FeatureNode::Composed {
                    first: *slots
                        .get(&feature.inputs[0])
                        .ok_or(BRegressorError::InvalidInput)?,
                    second: *slots
                        .get(&feature.inputs[1])
                        .ok_or(BRegressorError::InvalidInput)?,
                    truth_table: feature.truth_table,
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { references, nodes })
    }

    fn evaluate(&self, features: &[bool]) -> Result<Vec<u8>, BRegressorError> {
        let mut values = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let value = match *node {
                FeatureNode::Source(index) => {
                    u8::from(*features.get(index).ok_or(BRegressorError::InvalidInput)?)
                }
                FeatureNode::Composed {
                    first,
                    second,
                    truth_table,
                } => u8::from(
                    truth_table & (1_u8 << ((values[first] << 1) | values[second])) != 0_u8,
                ),
            };
            values.push(value);
        }
        Ok(values)
    }
}

#[derive(Debug)]
struct ReplayCache {
    layout: ReplayLayout,
    rows: VecDeque<Vec<u8>>,
}

impl ReplayCache {
    fn new(store: &FeatureStore, capacity: usize) -> Result<Self, BRegressorError> {
        Ok(Self {
            layout: ReplayLayout::new(store)?,
            rows: VecDeque::with_capacity(capacity),
        })
    }

    fn push(&mut self, features: &[bool]) -> Result<(), BRegressorError> {
        self.rows.push_back(self.layout.evaluate(features)?);
        Ok(())
    }
}

/// Configuration for [`BRegressor`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct RegressorConfig {
    pub learning_rate: f64,
    pub l2: f64,
    pub sgd_steps: usize,
    pub feature_learning: FeatureLearningConfig,
}

/// Why a streaming feature-regression operation could not be completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BRegressorError {
    InvalidConfig,
    InvalidInput,
    NonFiniteTarget,
    Learning,
}

impl From<FeatureLearningError> for BRegressorError {
    fn from(_: FeatureLearningError) -> Self {
        Self::Learning
    }
}

/// Online linear regression over input bits and composed boolean features.
#[derive(Debug)]
pub struct BRegressor {
    config: RegressorConfig,
    learner: FeatureLearner,
    weights: HashMap<FeatureId, f64>,
    intercept: f64,
    replay_features: VecDeque<Vec<bool>>,
    replay_targets: VecDeque<f64>,
    replay_cache: ReplayCache,
    n_observed: usize,
}

impl BRegressor {
    pub(crate) fn new(
        source_feature_count: usize,
        config: RegressorConfig,
    ) -> Result<Self, BRegressorError> {
        if source_feature_count == 0
            || !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || !config.l2.is_finite()
            || config.l2 < 0.0
            || config.sgd_steps == 0
        {
            return Err(BRegressorError::InvalidConfig);
        }
        let learner = FeatureLearner::new(source_feature_count, config.feature_learning)?;
        let replay_cache = ReplayCache::new(learner.store(), config.feature_learning.batch_size)?;
        let weights = (0..source_feature_count)
            .map(|index| (FeatureId { layer: 0, index }, 0.0))
            .collect();
        Ok(Self {
            config,
            learner,
            weights,
            intercept: 0.0,
            replay_features: VecDeque::with_capacity(config.feature_learning.batch_size),
            replay_targets: VecDeque::with_capacity(config.feature_learning.batch_size),
            replay_cache,
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
        features_per_layer: usize,
        candidate_capacity: usize,
        max_layers: usize,
    ) -> Result<Self, BRegressorError> {
        Self::new(
            source_feature_count,
            RegressorConfig {
                learning_rate,
                l2,
                sgd_steps,
                feature_learning: FeatureLearningConfig {
                    batch_size,
                    parent_top_k,
                    features_per_layer,
                    candidate_capacity,
                    max_layers,
                },
            },
        )
    }

    #[must_use]
    pub(crate) fn store(&self) -> &FeatureStore {
        self.learner.store()
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
    #[allow(dead_code)]
    pub(crate) fn weight(&self, reference: FeatureId) -> Option<f64> {
        self.weights.get(&reference).copied()
    }

    pub fn predict(&self, features: &[bool]) -> Result<f64, BRegressorError> {
        self.validate_features(features)?;
        let values = self.replay_cache.layout.evaluate(features)?;
        Ok(self.prediction_from_values(&values))
    }

    /// Updates the linear model from a rolling replay batch.
    pub fn observe(&mut self, features: &[bool], target: f64) -> Result<(), BRegressorError> {
        self.validate_features(features)?;
        if !target.is_finite() {
            return Err(BRegressorError::NonFiniteTarget);
        }
        if self.replay_features.len() == self.config.feature_learning.batch_size {
            self.replay_features.pop_front();
            self.replay_targets.pop_front();
            self.replay_cache.rows.pop_front();
        }
        self.replay_cache.push(features)?;
        self.replay_features.push_back(features.to_vec());
        self.replay_targets.push_back(target);
        self.n_observed += 1;

        let feature_signs = self
            .n_observed
            .is_multiple_of(self.config.feature_learning.batch_size)
            .then(|| self.replay_residual_signs())
            .transpose()?;
        for _ in 0..self.config.sgd_steps {
            self.update_replay_batch();
        }
        if let Some(signs) = feature_signs {
            self.learn_replay_batch(&signs)?;
        }
        Ok(())
    }

    fn validate_features(&self, features: &[bool]) -> Result<(), BRegressorError> {
        (features.len() == self.store().source_feature_count())
            .then_some(())
            .ok_or(BRegressorError::InvalidInput)
    }

    fn model_value(&self, reference: FeatureId, value: bool) -> f64 {
        let value = 2.0 * f64::from(value) - 1.0;
        self.weights.get(&reference).copied().unwrap_or(0.0) * value
    }

    fn prediction_from_values(&self, values: &[u8]) -> f64 {
        self.intercept
            + self
                .replay_cache
                .layout
                .references
                .iter()
                .zip(values)
                .map(|(reference, value)| self.model_value(*reference, *value != 0))
                .sum::<f64>()
    }

    fn replay_residual_signs(&self) -> Result<Vec<bool>, BRegressorError> {
        self.replay_targets
            .iter()
            .zip(&self.replay_cache.rows)
            .map(|(target, values)| Ok(*target - self.prediction_from_values(values) >= 0.0))
            .collect()
    }

    fn update_replay_batch(&mut self) {
        let mut intercept_gradient = 0.0;
        let mut weight_gradients = HashMap::new();
        for (target, values) in self.replay_targets.iter().zip(&self.replay_cache.rows) {
            let prediction = self.prediction_from_values(values);
            let error = target - prediction;
            intercept_gradient += error;
            for (reference, value) in self.replay_cache.layout.references.iter().zip(values) {
                let value = 2.0 * f64::from(*value) - 1.0;
                *weight_gradients.entry(*reference).or_insert(0.0) += error * value;
            }
        }
        let batch_size = self.replay_features.len() as f64;
        let rate = self.config.learning_rate;
        let decay = 1.0 - rate * self.config.l2;
        for weight in self.weights.values_mut() {
            *weight *= decay;
        }
        self.intercept += rate * intercept_gradient / batch_size;
        for (reference, gradient) in weight_gradients {
            let weight = self.weights.entry(reference).or_insert(0.0);
            *weight += rate * gradient / batch_size;
        }
    }

    fn learn_replay_batch(&mut self, signs: &[bool]) -> Result<(), BRegressorError> {
        let source_count = self.store().source_feature_count();
        let columns: Vec<Vec<bool>> = (0..source_count)
            .map(|index| self.replay_features.iter().map(|row| row[index]).collect())
            .collect();
        let column_refs: Vec<&[bool]> = columns.iter().map(Vec::as_slice).collect();
        self.learner.observe_batch(SignBatch {
            feature_columns: &column_refs,
            signs,
        })?;
        self.sync_weights();
        let layout = ReplayLayout::new(self.store())?;
        self.replay_cache.layout = layout;
        self.replay_cache.rows = self
            .replay_features
            .iter()
            .map(|features| self.replay_cache.layout.evaluate(features))
            .collect::<Result<_, _>>()?;
        Ok(())
    }

    fn sync_weights(&mut self) {
        let live: HashSet<_> = (0..=self.store().learned_layer_count())
            .flat_map(|layer| self.store().live_refs_in_layer(layer))
            .collect();
        self.weights.retain(|reference, _| live.contains(reference));
        for reference in live {
            self.weights.entry(reference).or_insert(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BRegressor, RegressorConfig};
    use crate::{FeatureId, FeatureLearningConfig};

    fn config(batch_size: usize) -> RegressorConfig {
        RegressorConfig {
            learning_rate: 0.1,
            l2: 0.2,
            sgd_steps: 1,
            feature_learning: FeatureLearningConfig {
                batch_size,
                parent_top_k: 2,
                features_per_layer: 2,
                candidate_capacity: 4,
                max_layers: 1,
            },
        }
    }

    #[test]
    fn updates_source_weights_with_replay_batch_gradient_and_weight_decay() {
        let mut model = BRegressor::new(1, config(2)).unwrap();
        model.observe(&[true], 2.0).unwrap();
        model.observe(&[false], 0.2).unwrap();

        assert!(
            (model.weight(FeatureId { layer: 0, index: 0 }).unwrap() - 0.266).abs() < f64::EPSILON
        );
        assert!((model.intercept() - 0.29).abs() < f64::EPSILON);
        assert_eq!(model.n_observed(), 2);
    }

    #[test]
    fn residual_signs_train_and_promote_a_feature() {
        let mut model = BRegressor::new(2, config(2)).unwrap();
        model.observe(&[false, false], -1.0).unwrap();
        model.observe(&[true, false], 1.0).unwrap();
        model.observe(&[false, false], -1.0).unwrap();
        model.observe(&[true, false], 1.0).unwrap();

        let learned = model.store().live_refs_in_layer(1);
        assert_eq!(learned.len(), 1);
        assert_eq!(model.weight(learned[0]), Some(0.0));

        model.observe(&[false, false], -1.0).unwrap();
        assert_eq!(model.n_observed(), 5);
    }

    #[test]
    fn centers_binary_feature_contributions() {
        let mut config = config(1);
        config.l2 = 0.0;
        let mut model = BRegressor::new(1, config).unwrap();

        model.observe(&[false], 1.0).unwrap();

        assert!(
            (model.weight(FeatureId { layer: 0, index: 0 }).unwrap() + 0.1).abs() < f64::EPSILON
        );
        assert!((model.intercept() - 0.1).abs() < f64::EPSILON);
    }
}
