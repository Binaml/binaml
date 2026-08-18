use crate::function_build_common::{
    derive_build_capacity, BuildNodeId, EphemeralNode, FunctionBuildConfig, PairCandidate,
};
use crate::function_graph::CompactNode;
use crate::FeatureCounter;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ModelCapacity {
    pub source_feature_count: usize,
    pub batch_size: usize,
    pub parent_top_k: usize,
    pub max_functions: usize,
    pub max_expert_nodes: usize,
    pub n_classes: usize,
    pub l_build: usize,
    pub pair_count: usize,
    pub graph_nodes: usize,
}

impl ModelCapacity {
    pub fn new(
        source_feature_count: usize,
        batch_size: usize,
        parent_top_k: usize,
        max_functions: usize,
        max_expert_nodes: usize,
        n_classes: usize,
    ) -> Self {
        let (l_build, pair_count, graph_nodes) =
            derive_build_capacity(source_feature_count, parent_top_k, max_expert_nodes);
        Self {
            source_feature_count,
            batch_size,
            parent_top_k,
            max_functions,
            max_expert_nodes,
            n_classes,
            l_build,
            pair_count,
            graph_nodes,
        }
    }

    pub fn validate(&self) -> Result<(), WorkspaceError> {
        if self.source_feature_count == 0
            || self.batch_size == 0
            || self.parent_top_k < 2
            || self.max_functions == 0
            || self.max_expert_nodes == 0
            || self.max_expert_nodes >= self.graph_nodes
            || self.batch_size > FeatureCounter::MAX_BATCH_SIZE
        {
            return Err(WorkspaceError::InvalidConfig);
        }
        Ok(())
    }

    pub fn build_config(&self) -> FunctionBuildConfig {
        FunctionBuildConfig {
            batch_size: self.batch_size,
            parent_top_k: self.parent_top_k,
            max_composed_layers: self.l_build,
            max_graph_nodes: self.graph_nodes,
            max_expert_nodes: self.max_expert_nodes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceError {
    InvalidConfig,
}

const NO_SLOT: u16 = u16::MAX;

#[derive(Debug)]
pub(crate) struct FlatColumnCache {
    columns: Box<[bool]>,
    slot_for_id: Box<[u16]>,
    batch_size: usize,
    parent_top_k: usize,
}

impl FlatColumnCache {
    pub(crate) fn new(batch_size: usize, parent_top_k: usize, graph_nodes: usize) -> Self {
        let column_slots = parent_top_k * 2;
        Self {
            columns: vec![false; column_slots * batch_size].into_boxed_slice(),
            slot_for_id: vec![NO_SLOT; graph_nodes].into_boxed_slice(),
            batch_size,
            parent_top_k: column_slots,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.slot_for_id.fill(NO_SLOT);
    }

    pub(crate) fn ensure(
        &mut self,
        nodes: &[EphemeralNode],
        feature_columns: &[&[bool]],
        id: BuildNodeId,
    ) -> Result<(), ColumnCacheError> {
        if id.0 >= self.slot_for_id.len() {
            return Err(ColumnCacheError::GraphCapacity);
        }
        if self.slot_for_id[id.0] != NO_SLOT {
            return Ok(());
        }
        let slot = (0..self.parent_top_k)
            .find(|candidate| {
                !self
                    .slot_for_id
                    .iter()
                    .any(|&mapped| mapped != NO_SLOT && mapped as usize == *candidate)
            })
            .ok_or(ColumnCacheError::ColumnCapacity)?;
        self.compute_into_slot(nodes, feature_columns, id, slot)?;
        self.slot_for_id[id.0] = u16::try_from(slot).expect("slot fits in u16");
        Ok(())
    }

    pub(crate) fn column(&self, id: BuildNodeId) -> &[bool] {
        let slot = self.slot_for_id[id.0] as usize;
        let start = slot * self.batch_size;
        &self.columns[start..start + self.batch_size]
    }

    pub(crate) fn retain_only(&mut self, keep: &[BuildNodeId]) {
        let mut next_slot = 0_usize;
        for &id in keep {
            let old_slot = self.slot_for_id[id.0];
            if old_slot == NO_SLOT {
                continue;
            }
            let old_slot = old_slot as usize;
            let dst_start = next_slot * self.batch_size;
            let src_start = old_slot * self.batch_size;
            if src_start != dst_start {
                self.columns
                    .copy_within(src_start..src_start + self.batch_size, dst_start);
            }
            next_slot += 1;
        }
        self.slot_for_id.fill(NO_SLOT);
        for (slot, &id) in keep.iter().enumerate() {
            self.slot_for_id[id.0] = u16::try_from(slot).expect("slot fits in u16");
        }
    }

    fn compute_into_slot(
        &mut self,
        nodes: &[EphemeralNode],
        feature_columns: &[&[bool]],
        id: BuildNodeId,
        slot: usize,
    ) -> Result<(), ColumnCacheError> {
        let start = slot * self.batch_size;
        match nodes[id.0] {
            EphemeralNode::Source { input_index } => {
                let column = feature_columns
                    .get(input_index)
                    .ok_or(ColumnCacheError::InvalidBatch)?;
                self.columns[start..start + self.batch_size].copy_from_slice(column);
            }
            EphemeralNode::Composed {
                first,
                second,
                truth_table,
            } => {
                self.ensure(nodes, feature_columns, first)?;
                self.ensure(nodes, feature_columns, second)?;
                let first_start = self.slot_for_id[first.0] as usize * self.batch_size;
                let second_start = self.slot_for_id[second.0] as usize * self.batch_size;
                for index in 0..self.batch_size {
                    let first_value = self.columns[first_start + index];
                    let second_value = self.columns[second_start + index];
                    self.columns[start + index] = crate::boolean_circuit::evaluate_truth_table(
                        truth_table,
                        first_value,
                        second_value,
                    );
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnCacheError {
    InvalidBatch,
    GraphCapacity,
    ColumnCapacity,
}

#[derive(Debug)]
pub(crate) struct BuildWorkspace {
    pub nodes: Box<[EphemeralNode]>,
    pub node_len: usize,
    pub association_scores: Box<[i64]>,
    pub accuracy_scores: Box<[u8]>,
    pub layers: Vec<Vec<BuildNodeId>>,
    pub column_cache: FlatColumnCache,
    pub pair_scratch: Box<[FeatureCounter]>,
    pub pair_candidates: Box<[PairCandidate]>,
    pub compact_nodes: Box<[CompactNode]>,
    pub compact_sources: Box<[usize]>,
    pub capacity: ModelCapacity,
}

impl BuildWorkspace {
    pub fn new(capacity: ModelCapacity) -> Self {
        let v = capacity.graph_nodes;
        let p = capacity.pair_count;
        Self {
            nodes: vec![EphemeralNode::Source { input_index: 0 }; v].into_boxed_slice(),
            node_len: 0,
            association_scores: vec![0_i64; v].into_boxed_slice(),
            accuracy_scores: vec![0_u8; v].into_boxed_slice(),
            layers: Vec::with_capacity(capacity.l_build + 1),
            column_cache: FlatColumnCache::new(
                capacity.batch_size,
                capacity.parent_top_k,
                v,
            ),
            pair_scratch: vec![FeatureCounter::default(); p].into_boxed_slice(),
            pair_candidates: vec![
                PairCandidate {
                    first: BuildNodeId(0),
                    second: BuildNodeId(0),
                    truth_table: 0,
                    abs_assoc: 0,
                    matches: 0,
                };
                p
            ]
            .into_boxed_slice(),
            compact_nodes: vec![CompactNode::Constant(false); capacity.max_expert_nodes]
                .into_boxed_slice(),
            compact_sources: vec![0_usize; capacity.source_feature_count].into_boxed_slice(),
            capacity,
        }
    }

    pub fn reset(&mut self) {
        self.node_len = 0;
        self.layers.clear();
        self.layers.push(Vec::with_capacity(self.capacity.parent_top_k));
        self.column_cache.reset();
    }
}

#[derive(Debug)]
pub(crate) struct EnsembleWorkspace {
    pub function_values: Box<[bool]>,
    pub pending_features: Box<[bool]>,
    pub pending_function_values: Box<[bool]>,
    pub eval_scratch: Box<[bool]>,
    pub logits: Box<[f64]>,
    pub probabilities: Box<[f64]>,
    pub batch_features: Box<[bool]>,
    pub batch_signs: Box<[bool]>,
    pub batch_len: usize,
    pub build: BuildWorkspace,
    pub capacity: ModelCapacity,
}

impl EnsembleWorkspace {
    pub fn new(capacity: ModelCapacity) -> Result<Self, WorkspaceError> {
        capacity.validate()?;
        let d = capacity.source_feature_count;
        let b = capacity.batch_size;
        let k_max = capacity.max_functions;
        let n_max = capacity.max_expert_nodes;
        let c = capacity.n_classes.max(1);
        Ok(Self {
            function_values: vec![false; k_max].into_boxed_slice(),
            pending_features: vec![false; d].into_boxed_slice(),
            pending_function_values: vec![false; k_max].into_boxed_slice(),
            eval_scratch: vec![false; n_max].into_boxed_slice(),
            logits: vec![0.0; c].into_boxed_slice(),
            probabilities: vec![0.0; c].into_boxed_slice(),
            batch_features: vec![false; b * d].into_boxed_slice(),
            batch_signs: vec![false; b].into_boxed_slice(),
            batch_len: 0,
            build: BuildWorkspace::new(capacity),
            capacity,
        })
    }
}
