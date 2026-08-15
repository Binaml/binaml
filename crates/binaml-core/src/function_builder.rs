use crate::{
    boolean_circuit::evaluate_truth_table, FeatureCounter, FeatureCounterError, SignBatch,
};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct FunctionBuildConfig {
    pub batch_size: usize,
    pub parent_top_k: usize,
    pub max_layers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionBuildError {
    InvalidConfig,
    InvalidBatch,
    BatchTooLarge,
}

impl From<FeatureCounterError> for FunctionBuildError {
    fn from(error: FeatureCounterError) -> Self {
        match error {
            FeatureCounterError::BatchTooLarge => Self::BatchTooLarge,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

pub struct FunctionBuilder;

impl FunctionBuilder {
    pub fn build(
        batch: SignBatch<'_>,
        config: FunctionBuildConfig,
    ) -> Result<(EphemeralGraph, BuildNodeId), FunctionBuildError> {
        if config.batch_size == 0
            || config.parent_top_k == 0
            || config.max_layers == 0
            || config.batch_size > FeatureCounter::MAX_BATCH_SIZE
        {
            return Err(FunctionBuildError::InvalidConfig);
        }
        validate_batch(batch, config.batch_size)?;

        let source_count = batch.feature_columns.len();
        let mut graph = EphemeralGraph {
            nodes: Vec::new(),
            layers: vec![Vec::new()],
        };

        for input_index in 0..source_count {
            let id = BuildNodeId(graph.nodes.len());
            graph.nodes.push(EphemeralNode::Source { input_index });
            graph.layers[0].push(id);
        }

        let mut scores = HashMap::new();
        for &id in &graph.layers[0] {
            let column = node_column(&graph, batch, id, &mut HashMap::new())?;
            scores.insert(id.0, correct_count(&column, batch.signs));
        }

        for _ in 0..config.max_layers {
            let parent_layer = graph.layers.len() - 1;
            let parents = parent_top_k(&graph.layers[parent_layer], &scores, config.parent_top_k);
            if parents.len() < 2 {
                break;
            }
            graph.layers.push(Vec::new());
            let new_layer = graph.layers.len() - 1;
            for (left_index, left) in parents.iter().enumerate() {
                for right in parents.iter().skip(left_index + 1) {
                    let (first, second) = if left.0 <= right.0 {
                        (*left, *right)
                    } else {
                        (*right, *left)
                    };
                    let mut cache = HashMap::new();
                    let first_col = node_column(&graph, batch, first, &mut cache)?;
                    let second_col = node_column(&graph, batch, second, &mut cache)?;
                    let truth_table = learn_truth_table(&first_col, &second_col, batch.signs)?;
                    let id = BuildNodeId(graph.nodes.len());
                    graph.nodes.push(EphemeralNode::Composed {
                        first,
                        second,
                        truth_table,
                    });
                    graph.layers[new_layer].push(id);
                    let column = node_column(&graph, batch, id, &mut HashMap::new())?;
                    scores.insert(id.0, correct_count(&column, batch.signs));
                }
            }
        }

        let output = select_output(&graph, &scores);
        Ok((graph, output))
    }
}

fn validate_batch(batch: SignBatch<'_>, batch_size: usize) -> Result<(), FunctionBuildError> {
    if batch.signs.len() != batch_size
        || batch
            .feature_columns
            .iter()
            .any(|column| column.len() != batch_size)
    {
        return Err(FunctionBuildError::InvalidBatch);
    }
    Ok(())
}

fn learn_truth_table(
    first: &[bool],
    second: &[bool],
    signs: &[bool],
) -> Result<u8, FunctionBuildError> {
    let examples: Vec<_> = first
        .iter()
        .zip(second)
        .zip(signs)
        .map(|((&first, &second), &sign)| (first, second, sign))
        .collect();
    Ok(FeatureCounter::from_batch(&examples)?.truth_table())
}

fn parent_top_k(
    references: &[BuildNodeId],
    scores: &HashMap<usize, u8>,
    k: usize,
) -> Vec<BuildNodeId> {
    let mut parents: Vec<_> = references.to_vec();
    parents.sort_by(|left, right| {
        scores
            .get(&right.0)
            .copied()
            .unwrap_or(0)
            .cmp(&scores.get(&left.0).copied().unwrap_or(0))
            .then_with(|| left.0.cmp(&right.0))
    });
    parents.truncate(k);
    parents
}

fn correct_count(values: &[bool], signs: &[bool]) -> u8 {
    u8::try_from(
        values
            .iter()
            .zip(signs)
            .filter(|(value, sign)| value == sign)
            .count(),
    )
    .expect("batch size fits in u8")
}

fn node_column(
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

fn node_layer(graph: &EphemeralGraph, id: BuildNodeId) -> usize {
    for (layer, nodes) in graph.layers.iter().enumerate() {
        if nodes.contains(&id) {
            return layer;
        }
    }
    0
}

fn select_output(graph: &EphemeralGraph, scores: &HashMap<usize, u8>) -> BuildNodeId {
    (0..graph.nodes.len())
        .map(BuildNodeId)
        .max_by(|left, right| {
            scores
                .get(&left.0)
                .copied()
                .unwrap_or(0)
                .cmp(&scores.get(&right.0).copied().unwrap_or(0))
                .then_with(|| node_layer(graph, *left).cmp(&node_layer(graph, *right)))
                .then_with(|| right.0.cmp(&left.0))
        })
        .expect("graph always has source nodes")
}

#[cfg(test)]
mod tests {
    use super::{FunctionBuildConfig, FunctionBuilder, SignBatch};

    #[test]
    fn builds_function_from_synthetic_batch() {
        let first = [false, true, false, true];
        let second = [false, false, true, true];
        let columns = [&first[..], &second[..]];
        let signs = [false, true, true, true];
        let config = FunctionBuildConfig {
            batch_size: 4,
            parent_top_k: 2,
            max_layers: 1,
        };
        let (graph, output) = FunctionBuilder::build(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config,
        )
        .unwrap();
        assert_eq!(graph.layers[1].len(), 1);
        assert!(matches!(
            graph.nodes[output.0],
            super::EphemeralNode::Composed { .. }
        ));
    }

    #[test]
    fn grows_all_pair_nodes_per_layer() {
        let a = [false; 4];
        let b = [true; 4];
        let c = [false, true, false, true];
        let columns = [&a[..], &b[..], &c[..]];
        let signs = [false, true, false, true];
        let config = FunctionBuildConfig {
            batch_size: 4,
            parent_top_k: 3,
            max_layers: 1,
        };
        let (graph, _) = FunctionBuilder::build(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config,
        )
        .unwrap();
        assert_eq!(graph.layers[1].len(), 3);
    }
}
