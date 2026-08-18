use crate::{boolean_circuit::evaluate_truth_table, binary_truth_table::FeatureCounter, SignBatch};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct FunctionBuildConfig {
    pub batch_size: usize,
    pub parent_top_k: usize,
    pub max_composed_layers: usize,
    pub max_graph_nodes: usize,
    pub max_expert_nodes: usize,
}

pub const DEFAULT_MAX_EXPERT_NODES: usize = 64;

pub fn derive_build_capacity(d: usize, k_p: usize, n_max: usize) -> (usize, usize, usize) {
    let pairs = k_p.saturating_sub(1) * k_p / 2;
    let source_cap = k_p.min(d.max(1));
    let l_build = n_max.saturating_sub(source_cap).max(1);
    let v = source_cap + l_build * k_p + 1;
    (l_build, pairs, v)
}

impl FunctionBuildConfig {
    pub fn new(
        batch_size: usize,
        parent_top_k: usize,
        source_count: usize,
        max_expert_nodes: usize,
    ) -> Self {
        let (max_composed_layers, _, max_graph_nodes) =
            derive_build_capacity(source_count, parent_top_k, max_expert_nodes);
        Self {
            batch_size,
            parent_top_k,
            max_composed_layers,
            max_graph_nodes,
            max_expert_nodes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionBuildError {
    InvalidConfig,
    InvalidBatch,
    BatchTooLarge,
    GraphCapacity,
}

impl From<crate::FeatureCounterError> for FunctionBuildError {
    fn from(error: crate::FeatureCounterError) -> Self {
        match error {
            crate::FeatureCounterError::BatchTooLarge => Self::BatchTooLarge,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BuildNodeId(pub usize);

#[derive(Debug, Clone)]
pub enum EphemeralNode {
    Source {
        input_index: usize,
    },
    Composed {
        first: BuildNodeId,
        second: BuildNodeId,
        truth_table: u8,
    },
}

#[derive(Debug, Clone)]
pub struct EphemeralGraph {
    pub(crate) nodes: Vec<EphemeralNode>,
    pub(crate) layers: Vec<Vec<BuildNodeId>>,
}

#[derive(Debug, Clone)]
pub struct FunctionModel {
    pub(crate) graph: EphemeralGraph,
    pub(crate) output: BuildNodeId,
    pub(crate) invert_output: bool,
}

pub(crate) fn validate_build_config(config: FunctionBuildConfig) -> Result<(), FunctionBuildError> {
    if config.batch_size == 0
        || config.parent_top_k < 2
        || config.max_composed_layers == 0
        || config.max_expert_nodes == 0
        || config.max_expert_nodes >= config.max_graph_nodes
        || config.batch_size > FeatureCounter::MAX_BATCH_SIZE
    {
        return Err(FunctionBuildError::InvalidConfig);
    }
    Ok(())
}

pub(crate) fn is_constant_column(column: &[bool]) -> bool {
    column.len() > 1
        && column
            .first()
            .is_some_and(|first| column.iter().all(|value| *value == *first))
}

pub(crate) fn is_constant_truth_table(truth_table: u8) -> bool {
    truth_table == 0b0000 || truth_table == 0b1111
}

pub(crate) fn cross_layer_pairs(
    frontier: &[BuildNodeId],
    global: &[BuildNodeId],
) -> Vec<(BuildNodeId, BuildNodeId)> {
    use std::collections::BTreeSet;

    let mut pairs = BTreeSet::new();
    for &left in frontier {
        for &right in global {
            if left == right {
                continue;
            }
            let (first, second) = if left.0 <= right.0 {
                (left, right)
            } else {
                (right, left)
            };
            pairs.insert((first, second));
        }
    }
    pairs.into_iter().collect()
}

pub(crate) fn nodes_at_depth(node_depth: &HashMap<usize, usize>, depth: usize) -> Vec<BuildNodeId> {
    let mut nodes: Vec<_> = node_depth
        .iter()
        .filter(|(_, node_depth)| **node_depth == depth)
        .map(|(id, _)| BuildNodeId(*id))
        .collect();
    nodes.sort_by_key(|id| id.0);
    nodes
}

const PAIR_GATE_TABLES: [u8; 4] = [0b1000, 0b0100, 0b0010, 0b0001];

pub(crate) fn score_all_pair_gates(
    first: &[bool],
    second: &[bool],
    signs: &[bool],
    ny: i64,
) -> [(i64, u8); 4] {
    let n = signs.len() as i64;
    let mut nz = [0_i64; 4];
    let mut nzy = [0_i64; 4];
    let mut matches = [0_u8; 4];
    for ((&first, &second), &sign) in first.iter().zip(second).zip(signs) {
        for (index, &truth_table) in PAIR_GATE_TABLES.iter().enumerate() {
            let active = evaluate_truth_table(truth_table, first, second);
            if active {
                nz[index] += 1;
                if sign {
                    nzy[index] += 1;
                }
            }
            if active == sign {
                matches[index] += 1;
            }
        }
    }
    std::array::from_fn(|index| ((n * nzy[index] - nz[index] * ny).abs(), matches[index]))
}

pub(crate) struct ColumnCache {
    columns: Vec<Option<Vec<bool>>>,
}

impl ColumnCache {
    pub(crate) fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    pub(crate) fn ensure(
        &mut self,
        graph: &EphemeralGraph,
        batch: SignBatch<'_>,
        id: BuildNodeId,
    ) -> Result<(), FunctionBuildError> {
        if id.0 >= self.columns.len() {
            self.columns.resize(id.0 + 1, None);
        }
        if self.columns[id.0].is_none() {
            let values = Self::compute(graph, batch, id, self)?;
            self.columns[id.0] = Some(values);
        }
        Ok(())
    }

    pub(crate) fn column(&self, id: BuildNodeId) -> &[bool] {
        self.columns[id.0]
            .as_deref()
            .expect("column must be populated before access")
    }

    pub(crate) fn retain_only(&mut self, keep: &[BuildNodeId]) {
        let keep_ids: std::collections::HashSet<_> = keep.iter().map(|id| id.0).collect();
        for (index, slot) in self.columns.iter_mut().enumerate() {
            if !keep_ids.contains(&index) {
                *slot = None;
            }
        }
    }

    fn compute(
        graph: &EphemeralGraph,
        batch: SignBatch<'_>,
        id: BuildNodeId,
        cache: &mut Self,
    ) -> Result<Vec<bool>, FunctionBuildError> {
        match graph.nodes[id.0] {
            EphemeralNode::Source { input_index } => batch
                .feature_columns
                .get(input_index)
                .ok_or(FunctionBuildError::InvalidBatch)
                .map(|column| column.to_vec()),
            EphemeralNode::Composed {
                first,
                second,
                truth_table,
            } => {
                cache.ensure(graph, batch, first)?;
                cache.ensure(graph, batch, second)?;
                Ok(cache
                    .columns[first.0]
                    .as_ref()
                    .expect("parent column populated")
                    .iter()
                    .zip(
                        cache.columns[second.0]
                            .as_ref()
                            .expect("parent column populated"),
                    )
                    .map(|(&first, &second)| evaluate_truth_table(truth_table, first, second))
                    .collect())
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PairCandidate {
    pub(crate) first: BuildNodeId,
    pub(crate) second: BuildNodeId,
    pub(crate) truth_table: u8,
    pub(crate) abs_assoc: i64,
    pub(crate) matches: u8,
}

pub(crate) struct PairCounterScratch {
    counters: Vec<FeatureCounter>,
}

impl PairCounterScratch {
    pub(crate) fn with_top_k(top_k: usize) -> Self {
        let pair_count = top_k.saturating_sub(1) * top_k / 2;
        Self {
            counters: vec![FeatureCounter::default(); pair_count],
        }
    }

    pub(crate) fn score_pairs(
        &mut self,
        parents: &[BuildNodeId],
        column_cache: &ColumnCache,
        batch: SignBatch<'_>,
        batch_size_i64: i64,
        ny: i64,
    ) -> Result<Vec<PairCandidate>, FunctionBuildError> {
        debug_assert!(self.counters.len() >= parents.len().saturating_sub(1) * parents.len() / 2);
        let mut candidates = Vec::new();
        let mut pair_index = 0;
        for (left_index, left) in parents.iter().enumerate() {
            for right in parents.iter().skip(left_index + 1) {
                let (first, second) = if left.0 <= right.0 {
                    (*left, *right)
                } else {
                    (*right, *left)
                };
                let first_column = column_cache.column(first);
                let second_column = column_cache.column(second);
                if is_constant_column(first_column) || is_constant_column(second_column) {
                    continue;
                }
                let counter = FeatureCounter::from_columns(first_column, second_column, batch.signs)?;
                self.counters[pair_index] = counter;
                pair_index += 1;
                let (truth_table, abs_assoc, matches) =
                    self.counters[pair_index - 1].truth_table_and_scores(batch_size_i64, ny);
                if is_constant_truth_table(truth_table) {
                    continue;
                }
                candidates.push(PairCandidate {
                    first,
                    second,
                    truth_table,
                    abs_assoc,
                    matches,
                });
            }
        }
        Ok(candidates)
    }
}

pub(crate) fn top_k_pair_candidates(
    candidates: &mut [PairCandidate],
    top_k: usize,
) -> &[PairCandidate] {
    candidates.sort_by(|left, right| {
        right
            .abs_assoc
            .cmp(&left.abs_assoc)
            .then_with(|| left.first.0.cmp(&right.first.0))
            .then_with(|| left.second.0.cmp(&right.second.0))
    });
    let keep = candidates.len().min(top_k);
    &candidates[..keep]
}

pub(crate) fn parent_top_k_slice(
    references: &[BuildNodeId],
    association_scores: &[i64],
    k: usize,
) -> Vec<BuildNodeId> {
    let mut parents: Vec<_> = references.to_vec();
    parents.sort_by(|left, right| {
        association_score_at(association_scores, right.0)
            .cmp(&association_score_at(association_scores, left.0))
            .then_with(|| left.0.cmp(&right.0))
    });
    parents.truncate(k);
    parents
}

fn association_score_at(association_scores: &[i64], id: usize) -> i64 {
    association_scores.get(id).copied().unwrap_or(0)
}

pub(crate) fn best_effective_matches_slice(accuracy_scores: &[u8], batch_size: u8) -> u8 {
    accuracy_scores
        .iter()
        .copied()
        .map(|matches| effective_matches(matches, batch_size))
        .max()
        .unwrap_or(0)
}

pub(crate) fn select_output_and_variant_slices(
    association_scores: &[i64],
    accuracy_scores: &[u8],
    batch_size: u8,
) -> (BuildNodeId, bool) {
    let output = (0..accuracy_scores.len())
        .map(BuildNodeId)
        .max_by(|left, right| {
            let left_matches = accuracy_scores.get(left.0).copied().unwrap_or(0);
            let right_matches = accuracy_scores.get(right.0).copied().unwrap_or(0);
            effective_matches(left_matches, batch_size)
                .cmp(&effective_matches(right_matches, batch_size))
                .then_with(|| {
                    association_score_at(association_scores, left.0)
                        .cmp(&association_score_at(association_scores, right.0))
                })
                .then_with(|| right.0.cmp(&left.0))
        })
        .expect("graph always has nodes");
    let matches = accuracy_scores[output.0];
    let invert_output = matches < batch_size - matches;
    (output, invert_output)
}

pub(crate) fn effective_matches(matches: u8, batch_size: u8) -> u8 {
    matches.max(batch_size - matches)
}

pub(crate) fn correct_count(values: &[bool], signs: &[bool]) -> u8 {
    u8::try_from(
        values
            .iter()
            .zip(signs)
            .filter(|(value, sign)| value == sign)
            .count(),
    )
    .expect("batch size fits in u8")
}

pub(crate) fn validate_batch(batch: SignBatch<'_>, batch_size: usize) -> Result<(), FunctionBuildError> {
    validate_feature_batch_with_size(batch, batch_size)?;
    if batch.signs.len() != batch_size {
        return Err(FunctionBuildError::InvalidBatch);
    }
    Ok(())
}

pub(crate) fn validate_feature_batch(batch: SignBatch<'_>) -> Result<(), FunctionBuildError> {
    let batch_size = batch
        .feature_columns
        .first()
        .map(|column| column.len())
        .unwrap_or(0);
    validate_feature_batch_with_size(batch, batch_size)
}

fn validate_feature_batch_with_size(
    batch: SignBatch<'_>,
    batch_size: usize,
) -> Result<(), FunctionBuildError> {
    if batch
        .feature_columns
        .iter()
        .any(|column| column.len() != batch_size)
    {
        return Err(FunctionBuildError::InvalidBatch);
    }
    Ok(())
}

pub(crate) fn node_column(
    graph: &EphemeralGraph,
    batch: SignBatch<'_>,
    id: BuildNodeId,
    cache: &mut HashMap<BuildNodeId, Vec<bool>>,
) -> Result<Vec<bool>, FunctionBuildError> {
    if let Some(values) = cache.get(&id) {
        return Ok(values.clone());
    }
    let values = match graph.nodes[id.0] {
        EphemeralNode::Source { input_index } => batch
            .feature_columns
            .get(input_index)
            .ok_or(FunctionBuildError::InvalidBatch)?
            .to_vec(),
        EphemeralNode::Composed {
            first,
            second,
            truth_table,
        } => {
            let first_col = node_column(graph, batch, first, cache)?;
            let second_col = node_column(graph, batch, second, cache)?;
            first_col
                .into_iter()
                .zip(second_col)
                .map(|(first, second)| evaluate_truth_table(truth_table, first, second))
                .collect()
        }
    };
    cache.insert(id, values.clone());
    Ok(values)
}

pub(crate) fn predict_model(
    model: &FunctionModel,
    batch: SignBatch<'_>,
) -> Result<Vec<bool>, FunctionBuildError> {
    validate_feature_batch(batch)?;
    let mut predictions = node_column(&model.graph, batch, model.output, &mut HashMap::new())?;
    if model.invert_output {
        for value in &mut predictions {
            *value = !*value;
        }
    }
    Ok(predictions)
}

pub(crate) fn select_output_top_k_by_association(
    candidates: &[BuildNodeId],
    association_scores: &[i64],
    accuracy_scores: &[u8],
    batch_size: u8,
    top_k: usize,
) -> (BuildNodeId, bool) {
    let mut ranked: Vec<BuildNodeId> = candidates.to_vec();
    ranked.sort_by(|left, right| {
        association_score_at(association_scores, right.0)
            .cmp(&association_score_at(association_scores, left.0))
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(top_k);

    let output = ranked
        .into_iter()
        .max_by(|left, right| {
            let left_matches = accuracy_scores.get(left.0).copied().unwrap_or(0);
            let right_matches = accuracy_scores.get(right.0).copied().unwrap_or(0);
            effective_matches(left_matches, batch_size)
                .cmp(&effective_matches(right_matches, batch_size))
                .then_with(|| right.0.cmp(&left.0))
        })
        .expect("graph always has nodes");
    let matches = accuracy_scores[output.0];
    let invert_output = matches < batch_size - matches;
    (output, invert_output)
}

pub(crate) fn surviving_nodes(graph: &EphemeralGraph) -> Vec<BuildNodeId> {
    graph.layers.iter().flat_map(|layer| layer.iter().copied()).collect()
}

pub(crate) fn node_layer(graph: &EphemeralGraph, id: BuildNodeId) -> usize {
    for (layer, nodes) in graph.layers.iter().enumerate() {
        if nodes.contains(&id) {
            return layer;
        }
    }
    0
}

pub(crate) fn select_output_truth_table_slices(
    graph: &EphemeralGraph,
    accuracy_scores: &[u8],
    batch_size: u8,
) -> (BuildNodeId, bool) {
    let output = (0..accuracy_scores.len())
        .map(BuildNodeId)
        .max_by(|left, right| {
            accuracy_scores
                .get(left.0)
                .copied()
                .unwrap_or(0)
                .cmp(&accuracy_scores.get(right.0).copied().unwrap_or(0))
                .then_with(|| node_layer(graph, *left).cmp(&node_layer(graph, *right)))
                .then_with(|| right.0.cmp(&left.0))
        })
        .expect("graph always has source nodes");
    let matches = accuracy_scores[output.0];
    let invert_output = matches < batch_size - matches;
    (output, invert_output)
}

#[cfg(test)]
mod tests {
    use super::{is_constant_column, is_constant_truth_table, PairCounterScratch, top_k_pair_candidates};

    #[test]
    fn detects_constant_columns() {
        assert!(is_constant_column(&[false, false, false]));
        assert!(is_constant_column(&[true, true]));
        assert!(!is_constant_column(&[false, true]));
        assert!(!is_constant_column(&[false]));
    }

    #[test]
    fn detects_constant_truth_tables() {
        assert!(is_constant_truth_table(0b0000));
        assert!(is_constant_truth_table(0b1111));
        assert!(!is_constant_truth_table(0b1010));
    }

    #[test]
    fn skips_constant_truth_tables_when_scoring_pairs() {
        let first = [false, false, true, true];
        let second = [false, true, false, true];
        let columns = [&first[..], &second[..]];
        let signs = [false, true, true, false];
        let batch = crate::SignBatch {
            feature_columns: &columns,
            signs: &signs,
        };
        let graph = super::EphemeralGraph {
            nodes: vec![
                super::EphemeralNode::Source { input_index: 0 },
                super::EphemeralNode::Source { input_index: 1 },
            ],
            layers: vec![vec![super::BuildNodeId(0), super::BuildNodeId(1)]],
        };
        let mut cache = super::ColumnCache::new();
        cache.ensure(&graph, batch, super::BuildNodeId(0)).unwrap();
        cache.ensure(&graph, batch, super::BuildNodeId(1)).unwrap();
        let mut scratch = PairCounterScratch::with_top_k(2);
        let candidates = scratch
            .score_pairs(
                &graph.layers[0],
                &cache,
                batch,
                4,
                2,
            )
            .unwrap();
        assert!(candidates.iter().all(|candidate| !is_constant_truth_table(candidate.truth_table)));
    }

    #[test]
    fn top_k_respects_constant_filtered_candidates() {
        let mut candidates = [
            super::PairCandidate {
                first: super::BuildNodeId(0),
                second: super::BuildNodeId(1),
                truth_table: 0b1010,
                abs_assoc: 1,
                matches: 2,
            },
            super::PairCandidate {
                first: super::BuildNodeId(0),
                second: super::BuildNodeId(2),
                truth_table: 0b1100,
                abs_assoc: 3,
                matches: 3,
            },
        ];
        let kept = top_k_pair_candidates(&mut candidates, 1);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].abs_assoc, 3);
    }
}
