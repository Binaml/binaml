pub use crate::function_build_common::{
    BuildNodeId, EphemeralNode, FunctionBuildConfig, FunctionBuildError, FunctionModel,
};

use crate::function_build_common::{
    correct_count, is_constant_column, parent_top_k_end, top_k_pair_candidates, validate_batch,
    validate_build_config, validate_feature_batch,
};
use crate::function_compact::{compact_build_workspace_into, CompactError};
use crate::function_graph::FunctionGraph;
use crate::workspace::{BuildWorkspace, ColumnCacheError, ModelCapacity};
use crate::{association::association_score, SignBatch};

pub struct FunctionBuilder;

/// Preallocated builder workspace; zero heap growth on [`Self::build`] after construction.
pub struct FunctionBuildSession {
    workspace: BuildWorkspace,
    config: FunctionBuildConfig,
    graph: FunctionGraph,
}

impl FunctionBuildSession {
    pub fn new(
        config: FunctionBuildConfig,
        source_feature_count: usize,
    ) -> Result<Self, FunctionBuildError> {
        validate_build_config(config)?;
        let capacity = ModelCapacity::new(
            source_feature_count,
            config.batch_size,
            config.parent_top_k,
            1,
            config.max_expert_nodes,
            0,
        );
        if capacity.validate().is_err() {
            return Err(FunctionBuildError::InvalidConfig);
        }
        Ok(Self {
            workspace: BuildWorkspace::new(capacity),
            config,
            graph: FunctionGraph::empty(source_feature_count, config.max_expert_nodes),
        })
    }

    pub fn build(
        &mut self,
        batch: SignBatch<'_>,
    ) -> Result<(BuildNodeId, bool), FunctionBuildError> {
        FunctionBuilder::build_in_workspace(batch, self.config, &mut self.workspace)
    }

    pub fn build_model(
        &mut self,
        batch: SignBatch<'_>,
    ) -> Result<FunctionModel, FunctionBuildError> {
        let (output, invert_output) = self.build(batch)?;
        compact_build_workspace_into(
            &mut self.workspace,
            output,
            invert_output,
            self.config.max_expert_nodes,
            &mut self.graph,
        )?;
        Ok(FunctionModel {
            graph: self.graph.clone(),
        })
    }
}

impl From<CompactError> for FunctionBuildError {
    fn from(error: CompactError) -> Self {
        match error {
            CompactError::ExpertTooLarge | CompactError::InvalidOutput => Self::GraphCapacity,
        }
    }
}

impl FunctionBuilder {
    pub fn build(
        batch: SignBatch<'_>,
        config: FunctionBuildConfig,
    ) -> Result<FunctionModel, FunctionBuildError> {
        FunctionBuildSession::new(config, batch.feature_count())?.build_model(batch)
    }

    pub(crate) fn build_in_workspace(
        batch: SignBatch<'_>,
        config: FunctionBuildConfig,
        workspace: &mut BuildWorkspace,
    ) -> Result<(BuildNodeId, bool), FunctionBuildError> {
        validate_build_config(config)?;
        validate_batch(batch, config.batch_size)?;

        workspace.reset();
        let source_count = batch.feature_count();
        let batch_size_i64 =
            i64::try_from(config.batch_size).expect("batch size validated to fit in i64");
        let ny = batch.signs.iter().filter(|sign| **sign).count() as i64;

        for input_index in 0..source_count {
            let column = batch
                .column(input_index)
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
            workspace.push_to_current_layer(id);
            push_source_scores(workspace, batch, id, input_index)?;
        }

        if workspace.current_layer().is_empty() {
            return Err(FunctionBuildError::InvalidBatch);
        }

        if workspace.current_layer().len() > config.parent_top_k {
            let start = workspace.layer_ends[0] as usize;
            let end = workspace.layer_ends[1] as usize;
            let keep = parent_top_k_end(
                &mut workspace.layer_ids[start..end],
                &workspace.association_scores[..workspace.node_len],
                config.parent_top_k,
            );
            workspace.layer_ends[1] = u16::try_from(start + keep).expect("layer fits in u16");
        }
        let layer_zero_start = workspace.layer_ends[0] as usize;
        let layer_zero_end = workspace.layer_ends[1] as usize;
        let layer_zero_len = layer_zero_end - layer_zero_start;
        workspace.parent_buf[..layer_zero_len]
            .copy_from_slice(&workspace.layer_ids[layer_zero_start..layer_zero_end]);
        for index in 0..layer_zero_len {
            ensure_column(workspace, batch, workspace.parent_buf[index])?;
        }
        workspace
            .column_cache
            .retain_only(&workspace.parent_buf[..layer_zero_len]);

        let batch_size_u8 =
            u8::try_from(config.batch_size).expect("batch size validated to fit in u8");
        let mut best_accuracy = workspace.accuracy_scores[..workspace.node_len]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let mut layers_without_improvement = 0usize;

        while layers_without_improvement < config.l_pat
            && workspace.layer_count as usize - 1 < config.max_composed_layers
        {
            let parent_layer = workspace.layer_count as usize - 1;
            let layer_start = workspace.layer_ends[parent_layer] as usize;
            let layer_end = workspace.layer_ends[parent_layer + 1] as usize;
            let mut parent_len = layer_end - layer_start;
            workspace.parent_buf[..parent_len]
                .copy_from_slice(&workspace.layer_ids[layer_start..layer_end]);
            for index in 0..parent_len {
                ensure_column(workspace, batch, workspace.parent_buf[index])?;
            }
            let mut filtered_len = 0usize;
            for index in 0..parent_len {
                let id = workspace.parent_buf[index];
                if !is_constant_column(workspace.column_cache.column(id)) {
                    workspace.parent_buf[filtered_len] = id;
                    filtered_len += 1;
                }
            }
            parent_len = filtered_len;
            if parent_len < 2 {
                break;
            }

            let candidate_count =
                score_pairs_into_workspace(workspace, parent_len, batch, batch_size_i64, ny)?;
            if candidate_count == 0 {
                break;
            }
            let keep_len = top_k_pair_candidates(
                &mut workspace.pair_candidates[..candidate_count],
                config.parent_top_k,
            )
            .len();

            workspace.start_new_layer();
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
                workspace.push_to_current_layer(id);
                workspace.association_scores[id.0] = candidate.abs_assoc;
                workspace.accuracy_scores[id.0] = candidate.matches;
                ensure_column(workspace, batch, id)?;
                debug_assert!(!is_constant_column(workspace.column_cache.column(id)));
            }

            if graph_full {
                if workspace.current_layer().is_empty() {
                    workspace.pop_layer();
                }
                break;
            }

            if workspace.current_layer().is_empty() {
                workspace.pop_layer();
                break;
            }

            let new_layer_start = workspace.layer_ends[workspace.layer_count as usize - 1] as usize;
            let new_layer_end = workspace.current_layer_end();
            workspace
                .column_cache
                .retain_only(&workspace.layer_ids[new_layer_start..new_layer_end]);

            let new_best = workspace.accuracy_scores[..workspace.node_len]
                .iter()
                .copied()
                .max()
                .unwrap_or(0);
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

        let surviving_len = workspace.current_layer_end();
        workspace.rank_scratch[..surviving_len]
            .copy_from_slice(&workspace.layer_ids[..surviving_len]);
        let (output, invert_output) =
            crate::function_build_common::select_output_top_k_by_association_in_place(
                &mut workspace.rank_scratch,
                surviving_len,
                &workspace.association_scores[..workspace.node_len],
                &workspace.accuracy_scores[..workspace.node_len],
                batch_size_u8,
                config.parent_top_k,
            );

        Ok((output, invert_output))
    }

    pub fn predict(
        model: &FunctionModel,
        batch: SignBatch<'_>,
    ) -> Result<Vec<bool>, FunctionBuildError> {
        validate_feature_batch(batch)?;
        let batch_size = batch
            .column(0)
            .map(|column| column.len())
            .ok_or(FunctionBuildError::InvalidBatch)?;
        let feature_count = batch.feature_count();
        let mut predictions = Vec::with_capacity(batch_size);
        let mut scratch = vec![false; model.graph.eval_scratch_len()];
        for row in 0..batch_size {
            let mut features = Vec::with_capacity(feature_count);
            for index in 0..feature_count {
                features.push(
                    batch
                        .column(index)
                        .ok_or(FunctionBuildError::InvalidBatch)?[row],
                );
            }
            predictions.push(model.graph.evaluate_with_scratch(&features, &mut scratch));
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

fn ensure_column(
    workspace: &mut BuildWorkspace,
    batch: SignBatch<'_>,
    id: BuildNodeId,
) -> Result<(), FunctionBuildError> {
    workspace
        .column_cache
        .ensure(&workspace.nodes[..workspace.node_len], batch, id)
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
    parent_len: usize,
    batch: SignBatch<'_>,
    batch_size_i64: i64,
    ny: i64,
) -> Result<usize, FunctionBuildError> {
    use crate::function_build_common::{is_constant_truth_table, PairCandidate};

    let mut candidate_count = 0;
    let mut pair_index = 0;
    for left_index in 0..parent_len {
        for right_index in (left_index + 1)..parent_len {
            let left = workspace.parent_buf[left_index];
            let right = workspace.parent_buf[right_index];
            let (first, second) = if left.0 <= right.0 {
                (left, right)
            } else {
                (right, left)
            };
            let first_column = workspace.column_cache.column(first);
            let second_column = workspace.column_cache.column(second);
            if is_constant_column(first_column) || is_constant_column(second_column) {
                continue;
            }
            let counter =
                crate::FeatureCounter::from_columns(first_column, second_column, batch.signs)?;
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

fn push_source_scores(
    workspace: &mut BuildWorkspace,
    batch: SignBatch<'_>,
    id: BuildNodeId,
    input_index: usize,
) -> Result<(), FunctionBuildError> {
    let column = batch
        .column(input_index)
        .ok_or(FunctionBuildError::InvalidBatch)?;
    workspace.association_scores[id.0] = association_score(column, batch.signs).abs();
    workspace.accuracy_scores[id.0] = correct_count(column, batch.signs);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{FunctionBuildConfig, FunctionBuilder, SignBatch};
    use crate::function_build_common::{DEFAULT_L_PAT, DEFAULT_MAX_EXPERT_NODES};
    use crate::function_graph::CompactNode;

    fn config(batch_size: usize, parent_top_k: usize, source_count: usize) -> FunctionBuildConfig {
        FunctionBuildConfig::new(
            batch_size,
            parent_top_k,
            source_count,
            DEFAULT_MAX_EXPERT_NODES,
            DEFAULT_L_PAT,
        )
    }

    #[test]
    fn builds_function_from_synthetic_batch() {
        let first = [false, true, false, true];
        let second = [false, false, true, true];
        let columns = [&first[..], &second[..]];
        let signs = [false, true, true, true];
        let model =
            FunctionBuilder::build(SignBatch::from_columns(&columns, &signs), config(4, 2, 2))
                .unwrap();
        assert!(model.graph.node_count() >= 2);
        assert!(!model.graph.invert_output());
        assert!(model
            .graph
            .nodes()
            .iter()
            .any(|node| matches!(node, CompactNode::Composed { .. })));
    }

    #[test]
    fn skips_constant_source_columns() {
        let constant = [false; 4];
        let varying = [false, true, false, true];
        let columns = [&constant[..], &varying[..]];
        let signs = [false, true, false, true];
        let model =
            FunctionBuilder::build(SignBatch::from_columns(&columns, &signs), config(4, 2, 2))
                .unwrap();
        assert_eq!(model.graph.source_count(), 1);
        assert_eq!(model.graph.node_count(), 1);
    }

    #[test]
    fn keeps_at_most_top_k_nodes_per_layer() {
        let a = [false; 4];
        let b = [true; 4];
        let c = [false, true, false, true];
        let columns = [&a[..], &b[..], &c[..]];
        let signs = [false, true, false, true];
        let model =
            FunctionBuilder::build(SignBatch::from_columns(&columns, &signs), config(4, 2, 3))
                .unwrap();
        assert_eq!(model.graph.source_count(), 1);
        assert_eq!(model.graph.node_count(), 1);
    }

    #[test]
    fn learns_negated_literal_target() {
        let feature = [false, true, false, true, false, true, false, true];
        let columns = [&feature[..]];
        let signs = [true, false, true, false, true, false, true, false];
        let (predictions, score) = FunctionBuilder::fit_predict(
            SignBatch::from_columns(&columns, &signs),
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
            SignBatch::from_columns(&columns, &signs),
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
        let model =
            FunctionBuilder::build(SignBatch::from_columns(&columns, &signs), config(4, 2, 2))
                .unwrap();
        let holdout_first = [false, true];
        let holdout_second = [true, true];
        let holdout_columns = [&holdout_first[..], &holdout_second[..]];
        let holdout_signs = [true, true];
        let predictions = FunctionBuilder::predict(
            &model,
            SignBatch::from_columns(&holdout_columns, &holdout_signs),
        )
        .unwrap();
        assert_eq!(predictions, [true, true]);
    }
}
