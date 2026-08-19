pub use crate::conjunction_build_common::{
    BeamEntry, ConjunctionBuildConfig, ConjunctionBuildError,
};
pub use crate::conjunction_expert::ConjunctionExpert;

use crate::conjunction_build_common::{
    correct_count, dedup_extensions, is_constant_column, pick_winner, prune_extensions,
    score_column, sort_extension_indices, validate_batch, validate_build_config,
    validate_feature_batch, ConjunctionKey as Key, ExtensionCandidate,
};
use crate::workspace::{ConjunctionBuildWorkspace, DepthLayer, ModelCapacity};
use crate::SignBatch;

pub struct ConjunctionBuilder;

/// Preallocated builder workspace; zero heap growth on [`Self::build`] after construction.
pub struct ConjunctionBuildSession {
    workspace: ConjunctionBuildWorkspace,
    config: ConjunctionBuildConfig,
    feature_count: usize,
}

impl ConjunctionBuildSession {
    pub fn new(
        config: ConjunctionBuildConfig,
        source_feature_count: usize,
    ) -> Result<Self, ConjunctionBuildError> {
        validate_build_config(config)?;
        let capacity = ModelCapacity::new(
            source_feature_count,
            config.batch_size,
            config.max_conjunctions,
            config.max_conjunction_length,
            1,
            config.max_experts,
            0,
        );
        if capacity.validate().is_err() {
            return Err(ConjunctionBuildError::InvalidConfig);
        }
        Ok(Self {
            workspace: ConjunctionBuildWorkspace::new(capacity),
            config,
            feature_count: source_feature_count,
        })
    }

    pub fn build(&mut self, batch: SignBatch<'_>) -> Result<BeamEntry, ConjunctionBuildError> {
        ConjunctionBuilder::build_in_workspace(batch, self.config, &mut self.workspace)
    }

    pub fn build_expert(
        &mut self,
        batch: SignBatch<'_>,
    ) -> Result<ConjunctionExpert, ConjunctionBuildError> {
        let winner = self.build(batch)?;
        ConjunctionExpert::from_key(
            &winner.key,
            self.workspace.key_words,
            self.feature_count,
            self.config.max_conjunction_length,
        )
    }
}

impl ConjunctionBuilder {
    pub fn build(
        batch: SignBatch<'_>,
        config: ConjunctionBuildConfig,
    ) -> Result<ConjunctionExpert, ConjunctionBuildError> {
        ConjunctionBuildSession::new(config, batch.feature_count())?.build_expert(batch)
    }

    pub(crate) fn build_in_workspace(
        batch: SignBatch<'_>,
        config: ConjunctionBuildConfig,
        workspace: &mut ConjunctionBuildWorkspace,
    ) -> Result<BeamEntry, ConjunctionBuildError> {
        validate_build_config(config)?;
        validate_batch(batch, config.batch_size)?;

        workspace.reset();
        workspace.precompute_literal_columns(batch);

        let feature_count = batch.feature_count();
        let batch_size = config.batch_size;
        let batch_size_u8 =
            u8::try_from(batch_size).expect("batch size validated to fit in u8");
        let ny = batch.signs.iter().filter(|sign| **sign).count() as i64;
        let word_count = workspace.key_words;

        populate_depth_zero(
            batch,
            workspace,
            feature_count,
            ny,
            word_count,
            config.max_conjunctions,
        )?;
        if workspace.layers[0].len == 0 {
            return Err(ConjunctionBuildError::InvalidBatch);
        }

        let mut best_accuracy = best_across_depths(&workspace.layers).accuracy;
        let mut stale_depths = 0_usize;

        for depth in 0..workspace.max_conjunction_length - 1 {
            if stale_depths >= config.stale_layers {
                break;
            }
            if workspace.layers[depth].len == 0 {
                break;
            }

            let child_depth = depth + 1;
            let new_len = extend_into_depth(
                workspace,
                depth,
                child_depth,
                feature_count,
                batch_size,
                batch.signs,
                ny,
                word_count,
                config.max_conjunctions,
            )?;
            if new_len == 0 {
                stale_depths += 1;
                continue;
            }

            let depth_best = pick_winner(&workspace.layers[child_depth].entries, new_len).accuracy;
            if depth_best > best_accuracy {
                best_accuracy = depth_best;
                stale_depths = 0;
            } else {
                stale_depths += 1;
            }
            if best_accuracy == batch_size_u8 {
                break;
            }
        }

        Ok(best_across_depths(&workspace.layers))
    }

    pub fn predict(
        expert: &ConjunctionExpert,
        batch: SignBatch<'_>,
    ) -> Result<Vec<bool>, ConjunctionBuildError> {
        validate_feature_batch(batch)?;
        let batch_size = batch
            .column(0)
            .map(|column| column.len())
            .ok_or(ConjunctionBuildError::InvalidBatch)?;
        let feature_count = batch.feature_count();
        let mut predictions = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            let mut features = Vec::with_capacity(feature_count);
            for index in 0..feature_count {
                features.push(
                    batch
                        .column(index)
                        .ok_or(ConjunctionBuildError::InvalidBatch)?[row],
                );
            }
            predictions.push(expert.evaluate(&features));
        }
        Ok(predictions)
    }

    pub fn fit(
        batch: SignBatch<'_>,
        config: ConjunctionBuildConfig,
    ) -> Result<(ConjunctionExpert, u8), ConjunctionBuildError> {
        let expert = Self::build(batch, config)?;
        let predictions = Self::predict(&expert, batch)?;
        let score = correct_count(&predictions, batch.signs);
        Ok((expert, score))
    }

    pub fn fit_predict(
        batch: SignBatch<'_>,
        config: ConjunctionBuildConfig,
    ) -> Result<(Vec<bool>, u8), ConjunctionBuildError> {
        let (expert, score) = Self::fit(batch, config)?;
        let predictions = Self::predict(&expert, batch)?;
        Ok((predictions, score))
    }
}

fn best_across_depths(layers: &[DepthLayer]) -> BeamEntry {
    let mut best: Option<BeamEntry> = None;
    for layer in layers {
        if layer.len == 0 {
            continue;
        }
        let candidate = pick_winner(&layer.entries, layer.len);
        best = Some(match best {
            None => candidate,
            Some(current) => pick_winner(&[current, candidate], 2),
        });
    }
    best.expect("at least depth zero is populated")
}

fn populate_depth_zero(
    batch: SignBatch<'_>,
    workspace: &mut ConjunctionBuildWorkspace,
    feature_count: usize,
    ny: i64,
    word_count: usize,
    max_conjunctions: usize,
) -> Result<(), ConjunctionBuildError> {
    let mut count = 0_usize;
    for feature_index in 0..feature_count {
        let column = batch.column(feature_index).ok_or(ConjunctionBuildError::InvalidBatch)?;
        if is_constant_column(column) {
            continue;
        }
        for negated in [false, true] {
            let literal_index = feature_index * 2 + usize::from(negated);
            let literal_start = literal_index * workspace.batch_size;
            let literal_column =
                &workspace.literal_columns[literal_start..literal_start + workspace.batch_size];
            if is_constant_column(literal_column) {
                continue;
            }
            let (abs_assoc, accuracy) = score_column(literal_column, batch.signs, ny);
            let key = Key::EMPTY
                .with_literal(feature_index, negated, word_count)
                .expect("fresh literal");
            workspace.extension_buf[count] = ExtensionCandidate {
                key,
                abs_assoc,
                accuracy,
                parent_slot: 0,
                literal_index: u16::try_from(literal_index).expect("literal index fits in u16"),
            };
            count += 1;
        }
    }

    finalize_layer_from_extensions(workspace, 0, count, max_conjunctions, word_count, true);
    Ok(())
}

fn extend_into_depth(
    workspace: &mut ConjunctionBuildWorkspace,
    parent_depth: usize,
    child_depth: usize,
    feature_count: usize,
    batch_size: usize,
    signs: &[bool],
    ny: i64,
    word_count: usize,
    max_conjunctions: usize,
) -> Result<usize, ConjunctionBuildError> {
    let parent_len = workspace.layers[parent_depth].len;
    let ConjunctionBuildWorkspace {
        layers,
        literal_columns,
        extension_buf,
        z_scratch,
        ..
    } = workspace;

    let parent_columns = &layers[parent_depth].columns;
    let parent_entries = &layers[parent_depth].entries[..parent_len];

    let mut count = 0_usize;
    for parent in parent_entries {
        let parent_start = parent.column_slot * batch_size;
        for feature_index in 0..feature_count {
            if parent.key.contains(feature_index, word_count) {
                continue;
            }
            let feature_literal_start = feature_index * 2 * batch_size;
            if is_constant_column(
                &literal_columns[feature_literal_start..feature_literal_start + batch_size],
            ) {
                continue;
            }
            for negated in [false, true] {
                if count >= extension_buf.len() {
                    return Err(ConjunctionBuildError::ExpertTooLarge);
                }
                let literal_index = feature_index * 2 + usize::from(negated);
                let literal_start = literal_index * batch_size;
                z_scratch[..batch_size]
                    .copy_from_slice(&parent_columns[parent_start..parent_start + batch_size]);
                for row in 0..batch_size {
                    z_scratch[row] &= literal_columns[literal_start + row];
                }
                if is_constant_column(&z_scratch[..batch_size]) {
                    continue;
                }
                let (abs_assoc, accuracy) = score_column(&z_scratch[..batch_size], signs, ny);
                let key = parent
                    .key
                    .with_literal(feature_index, negated, word_count)
                    .expect("unused feature");
                extension_buf[count] = ExtensionCandidate {
                    key,
                    abs_assoc,
                    accuracy,
                    parent_slot: u16::try_from(parent.column_slot).expect("slot fits in u16"),
                    literal_index: u16::try_from(literal_index).expect("literal index fits in u16"),
                };
                count += 1;
            }
        }
    }

    finalize_layer_from_extensions(
        workspace,
        child_depth,
        count,
        max_conjunctions,
        word_count,
        false,
    );
    Ok(workspace.layers[child_depth].len)
}

fn finalize_layer_from_extensions(
    workspace: &mut ConjunctionBuildWorkspace,
    depth: usize,
    extension_count: usize,
    max_conjunctions: usize,
    word_count: usize,
    from_literals: bool,
) {
    if extension_count == 0 {
        workspace.layers[depth].len = 0;
        return;
    }

    sort_extension_indices(
        &workspace.extension_buf,
        &mut workspace.sort_scratch,
        extension_count,
        word_count,
    );
    let dedup_count = dedup_extensions(
        &workspace.extension_buf,
        &workspace.sort_scratch[..extension_count],
        extension_count,
        &mut workspace.dedup_buf,
        word_count,
    );
    let keep = prune_extensions(
        &mut workspace.dedup_buf,
        dedup_count,
        max_conjunctions,
        word_count,
    );

    let batch_size = workspace.batch_size;
    let literal_columns = &workspace.literal_columns;

    if from_literals {
        let layer = &mut workspace.layers[depth];
        for slot in 0..keep {
            let survivor = workspace.dedup_buf[slot];
            let dst_start = slot * batch_size;
            let literal_start = survivor.literal_index as usize * batch_size;
            layer.columns[dst_start..dst_start + batch_size]
                .copy_from_slice(&literal_columns[literal_start..literal_start + batch_size]);
            layer.entries[slot] = BeamEntry {
                key: survivor.key,
                abs_assoc: survivor.abs_assoc,
                accuracy: survivor.accuracy,
                column_slot: slot,
            };
        }
        layer.len = keep;
        return;
    }

    let (prefix, suffix) = workspace.layers.split_at_mut(depth);
    let parent_columns = &prefix[depth - 1].columns;
    let layer = &mut suffix[0];
    for slot in 0..keep {
        let survivor = workspace.dedup_buf[slot];
        let dst_start = slot * batch_size;
        let parent_start = survivor.parent_slot as usize * batch_size;
        let literal_start = survivor.literal_index as usize * batch_size;
        for row in 0..batch_size {
            layer.columns[dst_start + row] =
                parent_columns[parent_start + row] & literal_columns[literal_start + row];
        }
        layer.entries[slot] = BeamEntry {
            key: survivor.key,
            abs_assoc: survivor.abs_assoc,
            accuracy: survivor.accuracy,
            column_slot: slot,
        };
    }
    layer.len = keep;
}

#[cfg(test)]
mod tests {
    use super::{ConjunctionBuildConfig, ConjunctionBuilder, SignBatch};
    use crate::conjunction_build_common::{DEFAULT_MAX_EXPERTS, DEFAULT_STALE_LAYERS};

    fn config(batch_size: usize, max_conjunctions: usize) -> ConjunctionBuildConfig {
        ConjunctionBuildConfig::new(
            batch_size,
            max_conjunctions,
            7,
            DEFAULT_MAX_EXPERTS,
            DEFAULT_STALE_LAYERS,
        )
    }

    #[test]
    fn builds_conjunction_from_synthetic_batch() {
        let first = [false, true, false, true];
        let second = [false, false, true, true];
        let columns = [&first[..], &second[..]];
        let signs = [false, true, true, true];
        let expert = ConjunctionBuilder::build(
            SignBatch::from_columns(&columns, &signs),
            config(4, 2),
        )
        .unwrap();
        assert!(expert.literal_count() >= 1);
    }

    #[test]
    fn skips_constant_source_columns() {
        let constant = [false; 4];
        let varying = [false, true, false, true];
        let columns = [&constant[..], &varying[..]];
        let signs = [false, true, false, true];
        let expert = ConjunctionBuilder::build(
            SignBatch::from_columns(&columns, &signs),
            config(4, 2),
        )
        .unwrap();
        assert_eq!(expert.literal_count(), 1);
    }

    #[test]
    fn learns_negated_literal_target() {
        let feature = [false, true, false, true, false, true, false, true];
        let columns = [&feature[..]];
        let signs = [true, false, true, false, true, false, true, false];
        let (predictions, score) = ConjunctionBuilder::fit_predict(
            SignBatch::from_columns(&columns, &signs),
            config(8, 2),
        )
        .unwrap();
        assert_eq!(predictions, signs);
        assert_eq!(score, 8);
    }

    #[test]
    fn learns_two_literal_conjunction() {
        let first = [false, true, false, true];
        let second = [false, false, true, true];
        let columns = [&first[..], &second[..]];
        let signs = [false, false, false, true];
        let (predictions, score) = ConjunctionBuilder::fit_predict(
            SignBatch::from_columns(&columns, &signs),
            config(4, 4),
        )
        .unwrap();
        assert_eq!(predictions, signs);
        assert_eq!(score, 4);
    }

    #[test]
    fn learns_single_literal_best_match() {
        let first = [false, true, false, true];
        let second = [false, false, true, true];
        let columns = [&first[..], &second[..]];
        let signs = [false, true, true, true];
        let (predictions, score) = ConjunctionBuilder::fit_predict(
            SignBatch::from_columns(&columns, &signs),
            config(4, 4),
        )
        .unwrap();
        assert_eq!(score, 3);
        assert_eq!(predictions, [false, true, false, true]);
    }

    #[test]
    fn predicts_on_a_holdout_batch() {
        let first = [false, true, false, true];
        let second = [false, false, true, true];
        let columns = [&first[..], &second[..]];
        let signs = [false, false, false, true];
        let expert = ConjunctionBuilder::build(
            SignBatch::from_columns(&columns, &signs),
            config(4, 4),
        )
        .unwrap();
        let holdout_first = [false, true];
        let holdout_second = [true, true];
        let holdout_columns = [&holdout_first[..], &holdout_second[..]];
        let holdout_signs = [true, true];
        let predictions = ConjunctionBuilder::predict(
            &expert,
            SignBatch::from_columns(&holdout_columns, &holdout_signs),
        )
        .unwrap();
        assert_eq!(predictions, [false, true]);
    }
}
