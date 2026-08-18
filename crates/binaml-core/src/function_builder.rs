pub use crate::function_build_common::{
    BuildNodeId, EphemeralGraph, EphemeralNode, FunctionBuildConfig, FunctionBuildError,
    FunctionModel,
};

use crate::function_build_common::{
    correct_count, is_constant_column, parent_top_k_slice, predict_model,
    select_output_top_k_by_association, surviving_nodes, top_k_pair_candidates, validate_batch,
    validate_build_config,
};
use crate::workspace::{BuildWorkspace, ColumnCacheError};
use crate::{association::association_score, SignBatch};

pub struct FunctionBuilder;

impl FunctionBuilder {
    pub fn build(
        batch: SignBatch<'_>,
        config: FunctionBuildConfig,
    ) -> Result<FunctionModel, FunctionBuildError> {
        let capacity = crate::workspace::ModelCapacity::new(
            batch.feature_columns.len(),
            config.batch_size,
            config.parent_top_k,
            1,
            config.max_expert_nodes,
            0,
        );
        let mut workspace = BuildWorkspace::new(capacity);
        Self::build_in_workspace(batch, config, &mut workspace)
    }

    pub fn build_in_workspace(
        batch: SignBatch<'_>,
        config: FunctionBuildConfig,
        workspace: &mut BuildWorkspace,
    ) -> Result<FunctionModel, FunctionBuildError> {
        validate_build_config(config)?;
        validate_batch(batch, config.batch_size)?;

        workspace.reset();
        let source_count = batch.feature_columns.len();
        let batch_size_i64 = i64::try_from(config.batch_size)
            .expect("batch size validated to fit in i64");
        let ny = batch
            .signs
            .iter()
            .filter(|sign| **sign)
            .count() as i64;

        for input_index in 0..source_count {
            let column = batch
                .feature_columns
                .get(input_index)
                .ok_or(FunctionBuildError::InvalidBatch)?;
            if is_constant_column(column) {
                continue;
            }
            if workspace.node_len >= workspace.nodes.len() {
                return Err(FunctionBuildError::GraphCapacity);
            }
            let id = BuildNodeId(workspace.node_len);
            workspace.nodes[workspace.node_len] = EphemeralNode::Source { input_index };
            workspace.node_len += 1;
            workspace.layers[0].push(id);
            push_source_scores(workspace, batch, id, input_index)?;
        }

        if workspace.layers[0].is_empty() {
            return Err(FunctionBuildError::InvalidBatch);
        }

        if workspace.layers[0].len() > config.parent_top_k {
            workspace.layers[0] = parent_top_k_slice(
                &workspace.layers[0],
                &workspace.association_scores[..workspace.node_len],
                config.parent_top_k,
            );
        }
        for parent in workspace.layers[0].clone() {
            ensure_column(workspace, batch, parent)?;
        }
        workspace
            .column_cache
            .retain_only(&workspace.layers[0]);

        let batch_size_u8 = u8::try_from(config.batch_size)
            .expect("batch size validated to fit in u8");
        let mut best_accuracy = best_layer_accuracy(
            &workspace.layers[0],
            &workspace.accuracy_scores[..workspace.node_len],
        );

        while workspace.layers.len() - 1 < config.max_composed_layers {
            let parent_layer = workspace.layers.len() - 1;
            let parents = workspace.layers[parent_layer].clone();
            for parent in &parents {
                ensure_column(workspace, batch, *parent)?;
            }
            let parents: Vec<_> = parents
                .into_iter()
                .filter(|id| {
                    !is_constant_column(match workspace.nodes[id.0] {
                        EphemeralNode::Source { input_index } => batch
                            .feature_columns
                            .get(input_index)
                            .expect("source index validated during build"),
                        EphemeralNode::Composed { .. } => workspace.column_cache.column(*id),
                    })
                })
                .collect();
            if parents.len() < 2 {
                break;
            }

            let candidate_count = score_pairs_into_workspace(
                workspace,
                &parents,
                batch,
                batch_size_i64,
                ny,
            )?;
            if candidate_count == 0 {
                break;
            }
            let keep_len = top_k_pair_candidates(
                &mut workspace.pair_candidates[..candidate_count],
                config.parent_top_k,
            )
            .len();

            workspace.layers.push(Vec::with_capacity(config.parent_top_k));
            let new_layer = workspace.layers.len() - 1;
            let mut graph_full = false;
            for index in 0..keep_len {
                if workspace.node_len >= config.max_graph_nodes {
                    graph_full = true;
                    break;
                }
                let candidate = workspace.pair_candidates[index];
                ensure_column(workspace, batch, candidate.first)?;
                ensure_column(workspace, batch, candidate.second)?;
                let id = BuildNodeId(workspace.node_len);
                workspace.nodes[workspace.node_len] = EphemeralNode::Composed {
                    first: candidate.first,
                    second: candidate.second,
                    truth_table: candidate.truth_table,
                };
                workspace.node_len += 1;
                workspace.layers[new_layer].push(id);
                workspace.association_scores[id.0] = candidate.abs_assoc;
                workspace.accuracy_scores[id.0] = candidate.matches;
                ensure_column(workspace, batch, id)?;
                debug_assert!(!is_constant_column(workspace.column_cache.column(id)));
                best_accuracy = best_accuracy.max(candidate.matches);
            }

            if graph_full {
                if workspace.layers[new_layer].is_empty() {
                    workspace.layers.pop();
                }
                break;
            }

            if workspace.layers[new_layer].is_empty() {
                workspace.layers.pop();
                break;
            }

            workspace
                .column_cache
                .retain_only(&workspace.layers[new_layer]);

            if best_accuracy == batch_size_u8 {
                break;
            }
        }

        let graph = EphemeralGraph {
            nodes: workspace.nodes[..workspace.node_len].to_vec(),
            layers: workspace.layers.clone(),
        };
        let (output, invert_output) = select_output_top_k_by_association(
            &surviving_nodes(&graph),
            &workspace.association_scores[..workspace.node_len],
            &workspace.accuracy_scores[..workspace.node_len],
            batch_size_u8,
            config.parent_top_k,
        );

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
        predict_model(model, batch)
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

fn ensure_column(
    workspace: &mut BuildWorkspace,
    batch: SignBatch<'_>,
    id: BuildNodeId,
) -> Result<(), FunctionBuildError> {
    workspace
        .column_cache
        .ensure(
            &workspace.nodes[..workspace.node_len],
            batch.feature_columns,
            id,
        )
        .map_err(column_cache_error)
}

fn column_cache_error(error: ColumnCacheError) -> FunctionBuildError {
    match error {
        ColumnCacheError::InvalidBatch => FunctionBuildError::InvalidBatch,
        ColumnCacheError::GraphCapacity | ColumnCacheError::ColumnCapacity => {
            FunctionBuildError::GraphCapacity
        }
    }
}

fn score_pairs_into_workspace(
    workspace: &mut BuildWorkspace,
    parents: &[BuildNodeId],
    batch: SignBatch<'_>,
    batch_size_i64: i64,
    ny: i64,
) -> Result<usize, FunctionBuildError> {
    use crate::function_build_common::{is_constant_truth_table, PairCandidate};

    let mut candidate_count = 0;
    let mut pair_index = 0;
    for (left_index, left) in parents.iter().enumerate() {
        for right in parents.iter().skip(left_index + 1) {
            let (first, second) = if left.0 <= right.0 {
                (*left, *right)
            } else {
                (*right, *left)
            };
            let first_column = parent_column(workspace, batch, first);
            let second_column = parent_column(workspace, batch, second);
            if is_constant_column(first_column) || is_constant_column(second_column) {
                continue;
            }
            let counter = crate::FeatureCounter::from_columns(
                first_column,
                second_column,
                batch.signs,
            )?;
            workspace.pair_scratch[pair_index] = counter;
            pair_index += 1;
            let (truth_table, abs_assoc, matches) =
                workspace.pair_scratch[pair_index - 1].truth_table_and_scores(batch_size_i64, ny);
            if is_constant_truth_table(truth_table) {
                continue;
            }
            workspace.pair_candidates[candidate_count] = PairCandidate {
                first,
                second,
                truth_table,
                abs_assoc,
                matches,
            };
            candidate_count += 1;
        }
    }
    Ok(candidate_count)
}

fn best_layer_accuracy(layer: &[BuildNodeId], accuracy_scores: &[u8]) -> u8 {
    layer
        .iter()
        .map(|id| accuracy_scores[id.0])
        .max()
        .unwrap_or(0)
}

fn parent_column<'a>(
    workspace: &'a BuildWorkspace,
    batch: SignBatch<'a>,
    id: BuildNodeId,
) -> &'a [bool] {
    match workspace.nodes[id.0] {
        EphemeralNode::Source { input_index } => batch
            .feature_columns
            .get(input_index)
            .expect("source index validated during build"),
        EphemeralNode::Composed { .. } => workspace.column_cache.column(id),
    }
}

fn push_source_scores(
    workspace: &mut BuildWorkspace,
    batch: SignBatch<'_>,
    id: BuildNodeId,
    input_index: usize,
) -> Result<(), FunctionBuildError> {
    let column = batch
        .feature_columns
        .get(input_index)
        .ok_or(FunctionBuildError::InvalidBatch)?;
    workspace.association_scores[id.0] = association_score(column, batch.signs).abs();
    workspace.accuracy_scores[id.0] = correct_count(column, batch.signs);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{FunctionBuildConfig, FunctionBuilder, SignBatch};
    use crate::function_build_common::EphemeralNode;
    use crate::function_build_common::DEFAULT_MAX_EXPERT_NODES;

    fn config(batch_size: usize, parent_top_k: usize, source_count: usize) -> FunctionBuildConfig {
        FunctionBuildConfig::new(
            batch_size,
            parent_top_k,
            source_count,
            DEFAULT_MAX_EXPERT_NODES,
        )
    }

    #[test]
    fn builds_function_from_synthetic_batch() {
        let first = [false, true, false, true];
        let second = [false, false, true, true];
        let columns = [&first[..], &second[..]];
        let signs = [false, true, true, true];
        let model = FunctionBuilder::build(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config(4, 2, 2),
        )
        .unwrap();
        assert_eq!(model.graph.layers[1].len(), 1);
        assert!(!model.invert_output);
        assert!(matches!(
            model.graph.nodes[model.output.0],
            EphemeralNode::Composed { .. }
        ));
    }

    #[test]
    fn skips_constant_source_columns() {
        let constant = [false; 4];
        let varying = [false, true, false, true];
        let columns = [&constant[..], &varying[..]];
        let signs = [false, true, false, true];
        let model = FunctionBuilder::build(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config(4, 2, 2),
        )
        .unwrap();
        assert_eq!(model.graph.layers[0].len(), 1);
        assert_eq!(model.graph.layers.len(), 1);
    }

    #[test]
    fn keeps_at_most_top_k_nodes_per_layer() {
        let a = [false; 4];
        let b = [true; 4];
        let c = [false, true, false, true];
        let columns = [&a[..], &b[..], &c[..]];
        let signs = [false, true, false, true];
        let model = FunctionBuilder::build(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config(4, 2, 3),
        )
        .unwrap();
        assert_eq!(model.graph.layers[0].len(), 1);
        assert_eq!(model.graph.layers.len(), 1);
    }

    #[test]
    fn learns_negated_literal_target() {
        let feature = [false, true, false, true, false, true, false, true];
        let columns = [&feature[..]];
        let signs = [true, false, true, false, true, false, true, false];
        let (predictions, score) = FunctionBuilder::fit_predict(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config(8, 2, 1),
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
        let (predictions, score) = FunctionBuilder::fit_predict(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config(4, 2, 2),
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
        let model = FunctionBuilder::build(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config(4, 2, 2),
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
