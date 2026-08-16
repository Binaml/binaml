use crate::function_compact::CompactError;
use crate::{
    compact, FunctionBuildConfig, FunctionBuildError, FunctionBuilder, FunctionGraph, SignBatch,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct EnsembleConfig {
    pub learning_rate: f64,
    pub l2: f64,
    pub sgd_steps: usize,
    pub batch_size: usize,
    pub max_layers: usize,
    pub parent_top_k: usize,
    pub max_functions: usize,
}

impl EnsembleConfig {
    pub fn validate(&self, source_feature_count: usize) -> Result<(), EnsembleError> {
        if source_feature_count == 0
            || self.max_functions == 0
            || self.batch_size == 0
            || self.batch_size > crate::FeatureCounter::MAX_BATCH_SIZE
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.l2.is_finite()
            || self.l2 < 0.0
            || self.sgd_steps == 0
            || self.parent_top_k == 0
            || self.max_layers == 0
        {
            return Err(EnsembleError::InvalidConfig);
        }
        Ok(())
    }

    fn build_config(&self) -> FunctionBuildConfig {
        FunctionBuildConfig {
            batch_size: self.batch_size,
            parent_top_k: self.parent_top_k,
            max_layers: self.max_layers,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EnsembleError {
    InvalidConfig,
    InvalidInput,
    PendingPrediction,
    NoPendingPrediction,
    Build,
    Compact,
}

impl From<FunctionBuildError> for EnsembleError {
    fn from(_: FunctionBuildError) -> Self {
        Self::Build
    }
}

impl From<CompactError> for EnsembleError {
    fn from(_: CompactError) -> Self {
        Self::Compact
    }
}

pub(crate) trait EnsembleHead: std::fmt::Debug {
    type Target: Copy;

    fn validate_target(&self, target: Self::Target) -> Result<(), EnsembleError>;
    fn batch_sign(&self, target: Self::Target, function_values: &[bool]) -> bool;
    fn update(&mut self, target: Self::Target, function_values: &[bool], rate: f64, l2: f64);
    fn append_function(&mut self);
    fn remove_function(&mut self, index: usize);
    fn prune_index(&self) -> usize;
}

#[derive(Debug)]
struct PendingStep {
    features: Vec<bool>,
    function_values: Vec<bool>,
}

#[derive(Debug)]
pub(crate) struct BooleanEnsemble<H: EnsembleHead> {
    pub config: EnsembleConfig,
    pub source_feature_count: usize,
    pub head: H,
    pub functions: Vec<FunctionGraph>,
    pub feature_batch_features: Vec<Vec<bool>>,
    pub feature_batch_signs: Vec<bool>,
    pub n_observed: usize,
    pending: Option<PendingStep>,
}

impl<H: EnsembleHead> BooleanEnsemble<H> {
    pub fn new(
        source_feature_count: usize,
        head: H,
        config: EnsembleConfig,
    ) -> Result<Self, EnsembleError> {
        config.validate(source_feature_count)?;
        Ok(Self {
            config,
            source_feature_count,
            head,
            functions: Vec::new(),
            feature_batch_features: Vec::with_capacity(config.batch_size),
            feature_batch_signs: Vec::with_capacity(config.batch_size),
            n_observed: 0,
            pending: None,
        })
    }

    pub fn validate_features(&self, features: &[bool]) -> Result<(), EnsembleError> {
        (features.len() == self.source_feature_count)
            .then_some(())
            .ok_or(EnsembleError::InvalidInput)
    }

    pub fn function_values(&self, features: &[bool]) -> Vec<bool> {
        self.functions
            .iter()
            .map(|function| function.evaluate(features))
            .collect()
    }

    pub fn begin_predict(&mut self, features: &[bool]) -> Result<Vec<bool>, EnsembleError> {
        if self.pending.is_some() {
            return Err(EnsembleError::PendingPrediction);
        }
        self.validate_features(features)?;
        let function_values = self.function_values(features);
        self.pending = Some(PendingStep {
            features: features.to_vec(),
            function_values: function_values.clone(),
        });
        Ok(function_values)
    }

    pub fn update(&mut self, target: H::Target) -> Result<(), EnsembleError> {
        self.head.validate_target(target)?;
        let pending = self
            .pending
            .take()
            .ok_or(EnsembleError::NoPendingPrediction)?;
        self.n_observed += 1;

        let sign = self.head.batch_sign(target, &pending.function_values);
        self.feature_batch_features.push(pending.features);
        self.feature_batch_signs.push(sign);

        for _ in 0..self.config.sgd_steps {
            self.head.update(
                target,
                &pending.function_values,
                self.config.learning_rate,
                self.config.l2,
            );
        }

        if self.feature_batch_features.len() == self.config.batch_size {
            self.finish_batch()?;
            self.feature_batch_features.clear();
            self.feature_batch_signs.clear();
        }
        Ok(())
    }

    fn finish_batch(&mut self) -> Result<(), EnsembleError> {
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
        self.head.append_function();
        if self.functions.len() > self.config.max_functions {
            let index = self.head.prune_index();
            self.functions.remove(index);
            self.head.remove_function(index);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct RegressionHead {
    pub intercept: f64,
    pub weights: Vec<f64>,
}

impl RegressionHead {
    pub fn new() -> Self {
        Self {
            intercept: 0.0,
            weights: Vec::new(),
        }
    }

    pub fn predict(&self, function_values: &[bool]) -> f64 {
        self.intercept
            + function_values
                .iter()
                .zip(&self.weights)
                .map(|(value, weight)| weight * (2.0 * f64::from(*value) - 1.0))
                .sum::<f64>()
    }
}

impl EnsembleHead for RegressionHead {
    type Target = f64;

    fn validate_target(&self, target: Self::Target) -> Result<(), EnsembleError> {
        if target.is_finite() {
            Ok(())
        } else {
            Err(EnsembleError::InvalidInput)
        }
    }

    fn batch_sign(&self, target: Self::Target, function_values: &[bool]) -> bool {
        target - self.predict(function_values) >= 0.0
    }

    fn update(&mut self, target: Self::Target, function_values: &[bool], rate: f64, l2: f64) {
        let prediction = self.predict(function_values);
        let error = target - prediction;
        let decay = 1.0 - rate * l2;
        for weight in &mut self.weights {
            *weight *= decay;
        }
        self.intercept += rate * error;
        for (weight, value) in self.weights.iter_mut().zip(function_values) {
            let centered = 2.0 * f64::from(*value) - 1.0;
            *weight += rate * error * centered;
        }
    }

    fn append_function(&mut self) {
        self.weights.push(0.0);
    }

    fn remove_function(&mut self, index: usize) {
        self.weights.remove(index);
    }

    fn prune_index(&self) -> usize {
        self.weights
            .iter()
            .enumerate()
            .min_by(|(left_index, left_weight), (right_index, right_weight)| {
                left_weight
                    .abs()
                    .total_cmp(&right_weight.abs())
                    .then_with(|| left_index.cmp(right_index))
            })
            .map(|(index, _)| index)
            .expect("non-empty ensemble")
    }
}

#[derive(Debug)]
pub(crate) struct ClassificationHead {
    pub n_classes: usize,
    pub intercepts: Vec<f64>,
    pub weights: Vec<Vec<f64>>,
}

impl ClassificationHead {
    pub fn new(n_classes: usize) -> Self {
        Self {
            n_classes,
            intercepts: vec![0.0; n_classes],
            weights: Vec::new(),
        }
    }

    pub fn predict(&self, function_values: &[bool]) -> usize {
        argmax(&self.logits(function_values))
    }

    pub fn logits(&self, function_values: &[bool]) -> Vec<f64> {
        let mut logits = self.intercepts.clone();
        for (class_weights, value) in self.weights.iter().zip(function_values) {
            let centered = 2.0 * f64::from(*value) - 1.0;
            for (logit, weight) in logits.iter_mut().zip(class_weights) {
                *logit += weight * centered;
            }
        }
        logits
    }
}

impl EnsembleHead for ClassificationHead {
    type Target = usize;

    fn validate_target(&self, target: Self::Target) -> Result<(), EnsembleError> {
        if target < self.n_classes {
            Ok(())
        } else {
            Err(EnsembleError::InvalidInput)
        }
    }

    fn batch_sign(&self, target: Self::Target, function_values: &[bool]) -> bool {
        self.predict(function_values) != target
    }

    fn update(&mut self, target: Self::Target, function_values: &[bool], rate: f64, l2: f64) {
        let logits = self.logits(function_values);
        let probabilities = softmax(&logits);
        let decay = 1.0 - rate * l2;
        for class_weights in &mut self.weights {
            for weight in class_weights {
                *weight *= decay;
            }
        }
        for (class_index, intercept) in self.intercepts.iter_mut().enumerate() {
            let error = probabilities[class_index] - f64::from(class_index == target);
            *intercept -= rate * error;
        }
        for (class_weights, value) in self.weights.iter_mut().zip(function_values) {
            let centered = 2.0 * f64::from(*value) - 1.0;
            for (class_index, weight) in class_weights.iter_mut().enumerate() {
                let error = probabilities[class_index] - f64::from(class_index == target);
                *weight -= rate * error * centered;
            }
        }
    }

    fn append_function(&mut self) {
        self.weights.push(vec![0.0; self.n_classes]);
    }

    fn remove_function(&mut self, index: usize) {
        self.weights.remove(index);
    }

    fn prune_index(&self) -> usize {
        self.weights
            .iter()
            .enumerate()
            .min_by(|(left_index, left_weights), (right_index, right_weights)| {
                left_weights
                    .iter()
                    .map(|weight| weight.abs())
                    .sum::<f64>()
                    .total_cmp(&right_weights.iter().map(|weight| weight.abs()).sum::<f64>())
                    .then_with(|| left_index.cmp(right_index))
            })
            .map(|(index, _)| index)
            .expect("non-empty ensemble")
    }
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities: Vec<f64> = logits
        .iter()
        .map(|logit| (logit - max_logit).exp())
        .collect();
    let normalizer = probabilities.iter().sum::<f64>();
    if normalizer > 0.0 {
        for probability in &mut probabilities {
            *probability /= normalizer;
        }
    } else {
        let uniform = 1.0 / logits.len() as f64;
        probabilities.fill(uniform);
    }
    probabilities
}
