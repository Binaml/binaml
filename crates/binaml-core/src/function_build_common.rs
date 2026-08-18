use crate::{binary_truth_table::FeatureCounter, SignBatch};

#[derive(Debug, Clone, Copy)]
pub struct FunctionBuildConfig {
    pub batch_size: usize,
    pub parent_top_k: usize,
    pub max_composed_layers: usize,
    pub max_graph_nodes: usize,
    pub max_expert_nodes: usize,
    pub l_pat: usize,
}

pub const DEFAULT_MAX_EXPERT_NODES: usize = 64;
pub const DEFAULT_L_PAT: usize = 2;

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
        l_pat: usize,
    ) -> Self {
        let (max_composed_layers, _, max_graph_nodes) =
            derive_build_capacity(source_count, parent_top_k, max_expert_nodes);
        Self {
            batch_size,
            parent_top_k,
            max_composed_layers,
            max_graph_nodes,
            max_expert_nodes,
            l_pat,
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
pub struct FunctionModel {
    pub(crate) graph: crate::function_graph::FunctionGraph,
}

pub(crate) fn validate_build_config(config: FunctionBuildConfig) -> Result<(), FunctionBuildError> {
    if config.batch_size == 0
        || config.parent_top_k < 2
        || config.max_composed_layers == 0
        || config.l_pat == 0
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct PairCandidate {
    pub(crate) first: BuildNodeId,
    pub(crate) second: BuildNodeId,
    pub(crate) truth_table: u8,
    pub(crate) abs_assoc: i64,
    pub(crate) matches: u8,
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

pub(crate) fn parent_top_k_end(
    layer: &mut [BuildNodeId],
    association_scores: &[i64],
    k: usize,
) -> usize {
    layer.sort_by(|left, right| {
        association_score_at(association_scores, right.0)
            .cmp(&association_score_at(association_scores, left.0))
            .then_with(|| left.0.cmp(&right.0))
    });
    layer.len().min(k)
}

fn association_score_at(association_scores: &[i64], id: usize) -> i64 {
    association_scores.get(id).copied().unwrap_or(0)
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

pub(crate) fn validate_batch(
    batch: SignBatch<'_>,
    batch_size: usize,
) -> Result<(), FunctionBuildError> {
    if batch.signs.len() != batch_size || batch.feature_count() == 0 {
        return Err(FunctionBuildError::InvalidBatch);
    }
    for index in 0..batch.feature_count() {
        if batch
            .column(index)
            .is_none_or(|column| column.len() != batch_size)
        {
            return Err(FunctionBuildError::InvalidBatch);
        }
    }
    Ok(())
}

pub(crate) fn validate_feature_batch(batch: SignBatch<'_>) -> Result<(), FunctionBuildError> {
    let batch_size = batch.column(0).map(|column| column.len()).unwrap_or(0);
    validate_feature_batch_with_size(batch, batch_size)
}

fn validate_feature_batch_with_size(
    batch: SignBatch<'_>,
    batch_size: usize,
) -> Result<(), FunctionBuildError> {
    if batch_size == 0 {
        return Err(FunctionBuildError::InvalidBatch);
    }
    for index in 0..batch.feature_count() {
        if batch
            .column(index)
            .is_none_or(|column| column.len() != batch_size)
        {
            return Err(FunctionBuildError::InvalidBatch);
        }
    }
    Ok(())
}

pub(crate) fn select_output_top_k_by_association_in_place(
    candidates: &mut [BuildNodeId],
    candidate_len: usize,
    association_scores: &[i64],
    accuracy_scores: &[u8],
    batch_size: u8,
    top_k: usize,
) -> (BuildNodeId, bool) {
    candidates[..candidate_len].sort_by(|left, right| {
        association_score_at(association_scores, right.0)
            .cmp(&association_score_at(association_scores, left.0))
            .then_with(|| left.0.cmp(&right.0))
    });
    let keep = candidate_len.min(top_k);

    let output = candidates[..keep]
        .iter()
        .max_by(|left, right| {
            let left_matches = accuracy_scores.get(left.0).copied().unwrap_or(0);
            let right_matches = accuracy_scores.get(right.0).copied().unwrap_or(0);
            effective_matches(left_matches, batch_size)
                .cmp(&effective_matches(right_matches, batch_size))
                .then_with(|| right.0.cmp(&left.0))
        })
        .copied()
        .expect("graph always has nodes");
    let matches = accuracy_scores[output.0];
    let invert_output = matches < batch_size - matches;
    (output, invert_output)
}

#[cfg(test)]
mod tests {
    use super::{is_constant_column, is_constant_truth_table, top_k_pair_candidates};

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
