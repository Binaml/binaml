use crate::conjunction_build_common::{
    derive_conjunction_capacity, BeamEntry, ConjunctionBuildConfig, ExtensionCandidate,
    MAX_BATCH_SIZE,
};
use crate::SignBatch;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ModelCapacity {
    pub source_feature_count: usize,
    pub batch_size: usize,
    pub max_conjunctions: usize,
    pub max_conjunction_length: usize,
    pub max_experts: usize,
    pub max_functions: usize,
    pub n_classes: usize,
    pub max_extensions: usize,
    pub key_words: usize,
}

impl ModelCapacity {
    pub fn new(
        source_feature_count: usize,
        batch_size: usize,
        max_conjunctions: usize,
        max_conjunction_length: usize,
        max_functions: usize,
        max_experts: usize,
        n_classes: usize,
    ) -> Self {
        let derived = derive_conjunction_capacity(
            source_feature_count,
            ConjunctionBuildConfig {
                batch_size,
                max_conjunctions,
                max_conjunction_length,
                max_experts,
                stale_layers: 1,
            },
        );
        Self {
            source_feature_count,
            batch_size,
            max_conjunctions,
            max_conjunction_length,
            max_experts,
            max_functions,
            n_classes,
            max_extensions: derived.max_extensions,
            key_words: derived.key_words,
        }
    }

    pub fn validate(&self) -> Result<(), WorkspaceError> {
        if self.source_feature_count == 0
            || self.batch_size == 0
            || self.max_conjunctions == 0
            || self.max_conjunction_length == 0
            || self.max_functions == 0
            || self.max_experts == 0
            || self.batch_size > MAX_BATCH_SIZE
        {
            return Err(WorkspaceError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceError {
    InvalidConfig,
}

/// One depth of the beam: conjunctions with exactly `depth + 1` literals.
#[derive(Debug)]
pub(crate) struct DepthLayer {
    pub entries: Box<[BeamEntry]>,
    pub columns: Box<[bool]>,
    pub len: usize,
}

impl DepthLayer {
    fn new(max_conjunctions: usize, batch_size: usize) -> Self {
        Self {
            entries: vec![
                BeamEntry {
                    key: crate::conjunction_build_common::ConjunctionKey::EMPTY,
                    abs_assoc: 0,
                    accuracy: 0,
                    column_slot: 0,
                };
                max_conjunctions
            ]
            .into_boxed_slice(),
            columns: vec![false; max_conjunctions * batch_size].into_boxed_slice(),
            len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

#[derive(Debug)]
pub(crate) struct ConjunctionBuildWorkspace {
    pub literal_columns: Box<[bool]>,
    pub layers: Box<[DepthLayer]>,
    pub extension_buf: Box<[ExtensionCandidate]>,
    pub dedup_buf: Box<[ExtensionCandidate]>,
    pub z_scratch: Box<[bool]>,
    pub sort_scratch: Box<[usize]>,
    pub batch_size: usize,
    pub feature_count: usize,
    pub key_words: usize,
    pub max_conjunction_length: usize,
}

impl ConjunctionBuildWorkspace {
    pub fn new(capacity: ModelCapacity) -> Self {
        let d = capacity.source_feature_count;
        let b = capacity.batch_size;
        let k_c = capacity.max_conjunctions;
        let l_max = capacity.max_conjunction_length;
        let max_extensions = capacity.max_extensions;
        Self {
            literal_columns: vec![false; 2 * d * b].into_boxed_slice(),
            layers: (0..l_max)
                .map(|_| DepthLayer::new(k_c, b))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            extension_buf: vec![
                ExtensionCandidate {
                    key: crate::conjunction_build_common::ConjunctionKey::EMPTY,
                    abs_assoc: 0,
                    accuracy: 0,
                    parent_slot: 0,
                    literal_index: 0,
                };
                max_extensions
            ]
            .into_boxed_slice(),
            dedup_buf: vec![
                ExtensionCandidate {
                    key: crate::conjunction_build_common::ConjunctionKey::EMPTY,
                    abs_assoc: 0,
                    accuracy: 0,
                    parent_slot: 0,
                    literal_index: 0,
                };
                max_extensions
            ]
            .into_boxed_slice(),
            z_scratch: vec![false; b].into_boxed_slice(),
            sort_scratch: (0..max_extensions).collect::<Vec<_>>().into_boxed_slice(),
            batch_size: b,
            feature_count: d,
            key_words: capacity.key_words,
            max_conjunction_length: l_max,
        }
    }

    pub fn reset(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset();
        }
    }

    pub fn precompute_literal_columns(&mut self, batch: SignBatch<'_>) {
        let batch_size = self.batch_size;
        for feature_index in 0..self.feature_count {
            let column = batch.column(feature_index).expect("validated batch");
            let positive_start = feature_index * 2 * batch_size;
            let negative_start = positive_start + batch_size;
            self.literal_columns[positive_start..positive_start + batch_size]
                .copy_from_slice(column);
            for row in 0..batch_size {
                self.literal_columns[negative_start + row] = !column[row];
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct EnsembleWorkspace {
    pub function_values: Box<[bool]>,
    pub pending_features: Box<[bool]>,
    pub pending_function_values: Box<[bool]>,
    pub logits: Box<[f32]>,
    pub batch_features: Box<[bool]>,
    pub batch_signs: Box<[bool]>,
    pub batch_signs_inverted: Box<[bool]>,
    pub batch_len: usize,
    pub build: ConjunctionBuildWorkspace,
}

impl EnsembleWorkspace {
    pub fn new(capacity: ModelCapacity) -> Result<Self, WorkspaceError> {
        capacity.validate()?;
        let d = capacity.source_feature_count;
        let b = capacity.batch_size;
        let k_max = capacity.max_functions;
        let c = capacity.n_classes.max(1);
        Ok(Self {
            function_values: vec![false; k_max].into_boxed_slice(),
            pending_features: vec![false; d].into_boxed_slice(),
            pending_function_values: vec![false; k_max].into_boxed_slice(),
            logits: vec![0.0; c].into_boxed_slice(),
            batch_features: vec![false; b * d].into_boxed_slice(),
            batch_signs: vec![false; b].into_boxed_slice(),
            batch_signs_inverted: vec![false; b].into_boxed_slice(),
            batch_len: 0,
            build: ConjunctionBuildWorkspace::new(capacity),
        })
    }
}
