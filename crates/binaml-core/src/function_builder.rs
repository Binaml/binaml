use crate::association::association_score;
use crate::{
    boolean_circuit::evaluate_truth_table, FeatureCounter, FeatureCounterError, SignBatch,
};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct FunctionBuildConfig {
    pub batch_size: usize,
    pub parent_top_k: usize,
    pub max_layers_without_improvement: usize,
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

#[derive(Debug, Clone)]
pub struct FunctionModel {
    pub(crate) graph: EphemeralGraph,
    pub(crate) output: BuildNodeId,
    pub(crate) invert_output: bool,
}

pub struct FunctionBuilder;

impl FunctionBuilder {
    pub fn build(
        batch: SignBatch<'_>,
        config: FunctionBuildConfig,
    ) -> Result<FunctionModel, FunctionBuildError> {
        if config.batch_size == 0
            || config.parent_top_k == 0
            || config.max_layers_without_improvement == 0
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
        let mut association_scores: HashMap<usize, i64> = HashMap::new();
        let mut accuracy_scores: HashMap<usize, u8> = HashMap::new();

        for input_index in 0..source_count {
            let id = BuildNodeId(graph.nodes.len());
            graph.nodes.push(EphemeralNode::Source { input_index });
            graph.layers[0].push(id);
            register_node_scores(
                &graph,
                batch,
                id,
                &mut association_scores,
                &mut accuracy_scores,
            )?;
        }

        let batch_size_u8 = u8::try_from(config.batch_size)
            .expect("batch size validated to fit in u8");
        let mut best_accuracy = accuracy_scores.values().copied().max().unwrap_or(0);
        let mut layers_without_improvement = 0;

        while layers_without_improvement < config.max_layers_without_improvement {
            let parent_layer = graph.layers.len() - 1;
            let parents = parent_top_k(
                &graph.layers[parent_layer],
                &association_scores,
                config.parent_top_k,
            );
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
                    register_node_scores(
                        &graph,
                        batch,
                        id,
                        &mut association_scores,
                        &mut accuracy_scores,
                    )?;
                }
            }
            let new_best = accuracy_scores.values().copied().max().unwrap_or(0);
            if new_best > best_accuracy {
                best_accuracy = new_best;
                layers_without_improvement = 0;
            } else {
                layers_without_improvement += 1;
            }
            if best_accuracy == batch_size_u8 {
                break;
            }
        }

        let (output, invert_output) = select_output(&graph, &accuracy_scores, batch);

        Ok(FunctionModel {
            graph,
            output,
            invert_output,
        })
    }

    pub fn predict(
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

    pub fn fit(
        batch: SignBatch<'_>,
        config: FunctionBuildConfig,
    ) -> Result<(FunctionModel, u8), FunctionBuildError> {
        let model = Self::build(batch, config)?;
        let predictions = Self::predict(&model, batch)?;
        let score = correct_count(&predictions, batch.signs);
        Ok((model, score))
    }

    pub fn fit_predict(
        batch: SignBatch<'_>,
        config: FunctionBuildConfig,
    ) -> Result<(Vec<bool>, u8), FunctionBuildError> {
        let (model, score) = Self::fit(batch, config)?;
        let predictions = Self::predict(&model, batch)?;
        Ok((predictions, score))
    }
}

fn register_node_scores(
    graph: &EphemeralGraph,
    batch: SignBatch<'_>,
    id: BuildNodeId,
    association_scores: &mut HashMap<usize, i64>,
    accuracy_scores: &mut HashMap<usize, u8>,
) -> Result<(), FunctionBuildError> {
    let column = node_column(graph, batch, id, &mut HashMap::new())?;
    association_scores.insert(id.0, association_score(&column, batch.signs).abs());
    accuracy_scores.insert(id.0, correct_count(&column, batch.signs));
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
    association_scores: &HashMap<usize, i64>,
    k: usize,
) -> Vec<BuildNodeId> {
    let mut parents: Vec<_> = references.to_vec();
    parents.sort_by(|left, right| {
        association_scores
            .get(&right.0)
            .copied()
            .unwrap_or(0)
            .cmp(&association_scores.get(&left.0).copied().unwrap_or(0))
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

fn validate_batch(batch: SignBatch<'_>, batch_size: usize) -> Result<(), FunctionBuildError> {
    validate_feature_batch_with_size(batch, batch_size)?;
    if batch.signs.len() != batch_size {
        return Err(FunctionBuildError::InvalidBatch);
    }
    Ok(())
}

fn validate_feature_batch(batch: SignBatch<'_>) -> Result<(), FunctionBuildError> {
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

fn select_output(
    graph: &EphemeralGraph,
    accuracy_scores: &HashMap<usize, u8>,
    batch: SignBatch<'_>,
) -> (BuildNodeId, bool) {
    let batch_size = batch.signs.len();
    let output = (0..graph.nodes.len())
        .map(BuildNodeId)
        .max_by(|left, right| {
            accuracy_scores
                .get(&left.0)
                .copied()
                .unwrap_or(0)
                .cmp(&accuracy_scores.get(&right.0).copied().unwrap_or(0))
                .then_with(|| node_layer(graph, *left).cmp(&node_layer(graph, *right)))
                .then_with(|| right.0.cmp(&left.0))
        })
        .expect("graph always has source nodes");
    let matches = accuracy_scores[&output.0];
    let invert_output = matches < batch_size as u8 - matches;
    (output, invert_output)
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
            max_layers_without_improvement: 1,
        };
        let model = FunctionBuilder::build(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config,
        )
        .unwrap();
        assert_eq!(model.graph.layers[1].len(), 1);
        assert!(!model.invert_output);
        assert!(matches!(
            model.graph.nodes[model.output.0],
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
            max_layers_without_improvement: 1,
        };
        let model = FunctionBuilder::build(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config,
        )
        .unwrap();
        assert_eq!(model.graph.layers[1].len(), 3);
    }

    #[test]
    fn learns_negated_literal_target() {
        let feature = [false, true, false, true, false, true, false, true];
        let columns = [&feature[..]];
        let signs = [true, false, true, false, true, false, true, false];
        let config = FunctionBuildConfig {
            batch_size: 8,
            parent_top_k: 2,
            max_layers_without_improvement: 1,
        };
        let (predictions, score) = FunctionBuilder::fit_predict(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config,
        )
        .unwrap();
        assert_eq!(predictions, signs);
        assert_eq!(score, 8);
    }

    #[test]
    fn learns_xor_with_truth_table_composition() {
        let first = [false, false, true, true];
        let second = [false, true, false, true];
        let columns = [&first[..], &second[..]];
        let signs = [false, true, true, false];
        let config = FunctionBuildConfig {
            batch_size: 4,
            parent_top_k: 2,
            max_layers_without_improvement: 1,
        };
        let (predictions, score) = FunctionBuilder::fit_predict(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config,
        )
        .unwrap();
        assert_eq!(predictions, signs);
        assert_eq!(score, 4);
    }

    #[test]
    fn predicts_on_a_holdout_batch() {
        let first = [false, true, false, true];
        let second = [false, false, true, true];
        let columns = [&first[..], &second[..]];
        let signs = [false, true, true, true];
        let config = FunctionBuildConfig {
            batch_size: 4,
            parent_top_k: 2,
            max_layers_without_improvement: 1,
        };
        let model = FunctionBuilder::build(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config,
        )
        .unwrap();
        let holdout_first = [false, true];
        let holdout_second = [true, true];
        let holdout_columns = [&holdout_first[..], &holdout_second[..]];
        let holdout_signs = [true, true];
        let predictions = FunctionBuilder::predict(
            &model,
            SignBatch {
                feature_columns: &holdout_columns,
                signs: &holdout_signs,
            },
        )
        .unwrap();
        assert_eq!(predictions, [true, true]);
    }
}
