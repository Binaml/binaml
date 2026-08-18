use crate::function_compact::{compact_with_limit_and_invert, CompactError};
use crate::workspace::{EnsembleWorkspace, ModelCapacity, WorkspaceError};
use crate::{
    FunctionBuildConfig, FunctionBuildError, FunctionBuilder, FunctionGraph, SignBatch,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct EnsembleConfig {
    pub learning_rate: f64,
    pub l2: f64,
    pub sgd_steps: usize,
    pub batch_size: usize,
    pub parent_top_k: usize,
    pub max_functions: usize,
    pub max_expert_nodes: usize,
}

impl EnsembleConfig {
    pub fn validate(&self, source_feature_count: usize) -> Result<(), EnsembleError> {
        let capacity = ModelCapacity::new(
            source_feature_count,
            self.batch_size,
            self.parent_top_k,
            self.max_functions,
            self.max_expert_nodes,
            0,
        );
        if source_feature_count == 0
            || self.max_functions == 0
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.l2.is_finite()
            || self.l2 < 0.0
            || self.sgd_steps == 0
            || self.parent_top_k < 2
            || self.max_expert_nodes == 0
            || capacity.validate().is_err()
        {
            return Err(EnsembleError::InvalidConfig);
        }
        Ok(())
    }

    fn build_config(&self, source_feature_count: usize) -> FunctionBuildConfig {
        FunctionBuildConfig::new(
            self.batch_size,
            self.parent_top_k,
            source_feature_count,
            self.max_expert_nodes,
        )
    }

    fn capacity(&self, source_feature_count: usize, n_classes: usize) -> ModelCapacity {
        ModelCapacity::new(
            source_feature_count,
            self.batch_size,
            self.parent_top_k,
            self.max_functions,
            self.max_expert_nodes,
            n_classes,
        )
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
    fn from(error: CompactError) -> Self {
        match error {
            CompactError::ExpertTooLarge => Self::Compact,
            CompactError::InvalidOutput => Self::Build,
        }
    }
}

impl From<WorkspaceError> for EnsembleError {
    fn from(_: WorkspaceError) -> Self {
        Self::InvalidConfig
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
pub(crate) struct BooleanEnsemble<H: EnsembleHead> {
    pub config: EnsembleConfig,
    pub source_feature_count: usize,
    pub head: H,
    pub functions: Vec<FunctionGraph>,
    pub n_observed: usize,
    pub(crate) workspace: EnsembleWorkspace,
    pending: bool,
}

impl<H: EnsembleHead> BooleanEnsemble<H> {
    pub fn new(
        source_feature_count: usize,
        head: H,
        config: EnsembleConfig,
        n_classes: usize,
    ) -> Result<Self, EnsembleError> {
        config.validate(source_feature_count)?;
        let capacity = config.capacity(source_feature_count, n_classes);
        let workspace = EnsembleWorkspace::new(capacity)?;
        Ok(Self {
            config,
            source_feature_count,
            head,
            functions: Vec::new(),
            n_observed: 0,
            workspace,
            pending: false,
        })
    }

    pub fn validate_features(&self, features: &[bool]) -> Result<(), EnsembleError> {
        (features.len() == self.source_feature_count)
            .then_some(())
            .ok_or(EnsembleError::InvalidInput)
    }

    fn function_count(&self) -> usize {
        self.functions.len()
    }

    fn function_values(&mut self, features: &[bool]) -> &[bool] {
        let count = self.function_count();
        for index in 0..count {
            self.workspace.function_values[index] = self.functions[index]
                .evaluate_with_scratch(features, &mut self.workspace.eval_scratch);
        }
        &self.workspace.function_values[..count]
    }

    pub fn begin_predict(&mut self, features: &[bool]) -> Result<(), EnsembleError> {
        if self.pending {
            return Err(EnsembleError::PendingPrediction);
        }
        self.validate_features(features)?;
        self.workspace
            .pending_features
            .copy_from_slice(features);
        let count = self.function_count();
        for index in 0..count {
            self.workspace.function_values[index] = self.functions[index]
                .evaluate_with_scratch(features, &mut self.workspace.eval_scratch);
        }
        self.workspace.pending_function_values[..count]
            .copy_from_slice(&self.workspace.function_values[..count]);
        self.pending = true;
        Ok(())
    }

    pub fn update(&mut self, target: H::Target) -> Result<(), EnsembleError> {
        self.head.validate_target(target)?;
        if !self.pending {
            return Err(EnsembleError::NoPendingPrediction);
        }
        self.pending = false;
        self.n_observed += 1;

        let count = self.function_count();
        let sign = self
            .head
            .batch_sign(target, &self.workspace.pending_function_values[..count]);
        let batch_index = self.workspace.batch_len;
        for feature in 0..self.source_feature_count {
            self.workspace.batch_features[feature * self.config.batch_size + batch_index] =
                self.workspace.pending_features[feature];
        }
        self.workspace.batch_signs[batch_index] = sign;
        self.workspace.batch_len += 1;

        for _ in 0..self.config.sgd_steps {
            self.head.update(
                target,
                &self.workspace.pending_function_values[..count],
                self.config.learning_rate,
                self.config.l2,
            );
        }

        if self.workspace.batch_len == self.config.batch_size {
            self.finish_batch()?;
            self.workspace.batch_len = 0;
        }
        Ok(())
    }

    fn finish_batch(&mut self) -> Result<(), EnsembleError> {
        let batch_size = self.config.batch_size;
        let feature_count = self.source_feature_count;
        let build_config = self.config.build_config(self.source_feature_count);
        let model = {
            let batch_features = &self.workspace.batch_features;
            let mut column_refs = Vec::with_capacity(feature_count);
            for feature in 0..feature_count {
                let start = feature * batch_size;
                column_refs.push(&batch_features[start..start + batch_size]);
            }
            let batch = SignBatch {
                feature_columns: &column_refs,
                signs: &self.workspace.batch_signs[..batch_size],
            };
            match FunctionBuilder::build_in_workspace(batch, build_config, &mut self.workspace.build)
            {
                Ok(model) => model,
                Err(FunctionBuildError::InvalidBatch) => return Ok(()),
                Err(error) => return Err(error.into()),
            }
        };
        let compacted = compact_with_limit_and_invert(
            model.graph,
            model.output,
            self.config.max_expert_nodes,
            model.invert_output,
        )?;
        if self.functions.len() >= self.config.max_functions {
            let index = self.head.prune_index();
            self.functions.remove(index);
            self.head.remove_function(index);
        }
        self.functions.push(compacted);
        self.head.append_function();
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct RegressionHead {
    pub intercept: f64,
    pub weights: Box<[f64]>,
    pub active: usize,
}

impl RegressionHead {
    pub fn new(max_functions: usize) -> Self {
        Self {
            intercept: 0.0,
            weights: vec![0.0; max_functions].into_boxed_slice(),
            active: 0,
        }
    }

    pub fn predict(&self, function_values: &[bool]) -> f64 {
        self.intercept
            + function_values
                .iter()
                .zip(self.weights.iter().take(self.active))
                .map(|(value, weight)| weight * f64::from(*value))
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
        for weight in self.weights.iter_mut().take(self.active) {
            *weight *= decay;
        }
        self.intercept += rate * error;
        for (weight, value) in self.weights.iter_mut().take(self.active).zip(function_values) {
            *weight += rate * error * f64::from(*value);
        }
    }

    fn append_function(&mut self) {
        if self.active < self.weights.len() {
            self.weights[self.active] = 0.0;
            self.active += 1;
        }
    }

    fn remove_function(&mut self, index: usize) {
        for slot in index..self.active - 1 {
            self.weights[slot] = self.weights[slot + 1];
        }
        self.weights[self.active - 1] = 0.0;
        self.active -= 1;
    }

    fn prune_index(&self) -> usize {
        self.weights
            .iter()
            .take(self.active)
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
    pub intercepts: Box<[f64]>,
    pub weights: Box<[f64]>,
    pub active: usize,
    pub max_functions: usize,
    pub logits_scratch: Box<[f64]>,
    pub probabilities_scratch: Box<[f64]>,
}

impl ClassificationHead {
    pub fn new(n_classes: usize, max_functions: usize) -> Self {
        Self {
            n_classes,
            intercepts: vec![0.0; n_classes].into_boxed_slice(),
            weights: vec![0.0; max_functions * n_classes].into_boxed_slice(),
            active: 0,
            max_functions,
            logits_scratch: vec![0.0; n_classes].into_boxed_slice(),
            probabilities_scratch: vec![0.0; n_classes].into_boxed_slice(),
        }
    }

    pub(crate) fn expert_weights(&self, function_index: usize) -> &[f64] {
        let start = function_index * self.n_classes;
        &self.weights[start..start + self.n_classes]
    }

    fn expert_weights_mut(&mut self, function_index: usize) -> &mut [f64] {
        let start = function_index * self.n_classes;
        let end = start + self.n_classes;
        &mut self.weights[start..end]
    }

    pub fn logits_into(&self, function_values: &[bool], logits: &mut [f64]) {
        logits.copy_from_slice(&self.intercepts);
        for (function_index, value) in function_values.iter().enumerate() {
            let activation = f64::from(*value);
            for (logit, weight) in logits.iter_mut().zip(self.expert_weights(function_index)) {
                *logit += weight * activation;
            }
        }
    }

    pub fn predict_with_scratch(
        &self,
        function_values: &[bool],
        logits: &mut [f64],
    ) -> usize {
        self.logits_into(function_values, logits);
        argmax(logits)
    }

    fn batch_sign_with_scratch(
        &self,
        target: usize,
        function_values: &[bool],
        logits: &mut [f64],
    ) -> bool {
        self.predict_with_scratch(function_values, logits) != target
    }

    pub fn logits(&self, function_values: &[bool]) -> Vec<f64> {
        let mut logits = vec![0.0; self.n_classes];
        self.logits_into(function_values, &mut logits);
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
        const MAX_CLASSES: usize = 64;
        let mut logits = [0.0_f64; MAX_CLASSES];
        assert!(self.n_classes <= MAX_CLASSES);
        self.batch_sign_with_scratch(target, function_values, &mut logits[..self.n_classes])
    }

    fn update(&mut self, target: Self::Target, function_values: &[bool], rate: f64, l2: f64) {
        self.logits_scratch.copy_from_slice(&self.intercepts);
        for (function_index, value) in function_values.iter().enumerate() {
            let activation = f64::from(*value);
            let start = function_index * self.n_classes;
            for class_index in 0..self.n_classes {
                self.logits_scratch[class_index] +=
                    self.weights[start + class_index] * activation;
            }
        }
        softmax_into(&self.logits_scratch, &mut self.probabilities_scratch);
        let decay = 1.0 - rate * l2;
        const MAX_CLASSES: usize = 64;
        let mut errors = [0.0_f64; MAX_CLASSES];
        assert!(self.n_classes <= MAX_CLASSES);
        for class_index in 0..self.n_classes {
            errors[class_index] = self.probabilities_scratch[class_index]
                - f64::from(class_index == target);
        }
        for function_index in 0..self.active {
            for weight in self.expert_weights_mut(function_index) {
                *weight *= decay;
            }
        }
        for (class_index, intercept) in self.intercepts.iter_mut().enumerate() {
            *intercept -= rate * errors[class_index];
        }
        for (function_index, value) in function_values.iter().enumerate() {
            let activation = f64::from(*value);
            for (class_index, weight) in self
                .expert_weights_mut(function_index)
                .iter_mut()
                .enumerate()
            {
                *weight -= rate * errors[class_index] * activation;
            }
        }
    }

    fn append_function(&mut self) {
        if self.active < self.max_functions {
            self.expert_weights_mut(self.active).fill(0.0);
            self.active += 1;
        }
    }

    fn remove_function(&mut self, index: usize) {
        for slot in index..self.active - 1 {
            let next = self.expert_weights(slot + 1).to_vec();
            self.expert_weights_mut(slot).copy_from_slice(&next);
        }
        self.expert_weights_mut(self.active - 1).fill(0.0);
        self.active -= 1;
    }

    fn prune_index(&self) -> usize {
        (0..self.active)
            .min_by(|&left_index, &right_index| {
                self.expert_weights(left_index)
                    .iter()
                    .map(|weight| weight.abs())
                    .sum::<f64>()
                    .total_cmp(
                        &self
                            .expert_weights(right_index)
                            .iter()
                            .map(|weight| weight.abs())
                            .sum::<f64>(),
                    )
                    .then_with(|| left_index.cmp(&right_index))
            })
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

fn softmax_into(logits: &[f64], probabilities: &mut [f64]) {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for (probability, logit) in probabilities.iter_mut().zip(logits) {
        *probability = (logit - max_logit).exp();
    }
    let normalizer = probabilities.iter().sum::<f64>();
    if normalizer > 0.0 {
        for probability in probabilities {
            *probability /= normalizer;
        }
    } else {
        let uniform = 1.0 / logits.len() as f64;
        probabilities.fill(uniform);
    }
}
