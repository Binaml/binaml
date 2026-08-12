use crate::{
    Feature, FeatureCounter, FeatureCounterError, FeatureId, FeatureStore, InsertFeatureError,
};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct SignBatch<'a> {
    pub feature_columns: &'a [&'a [bool]],
    pub signs: &'a [bool],
}

#[derive(Debug, Clone, Copy)]
pub struct FeatureLearningConfig {
    pub batch_size: usize,
    pub parent_top_k: usize,
    pub features_per_layer: usize,
    pub candidate_capacity: usize,
    pub max_layers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureLearningError {
    InvalidConfig,
    InvalidBatch,
    BatchTooLarge,
    Store(InsertFeatureError),
}

impl From<InsertFeatureError> for FeatureLearningError {
    fn from(error: InsertFeatureError) -> Self {
        Self::Store(error)
    }
}

impl From<FeatureCounterError> for FeatureLearningError {
    fn from(error: FeatureCounterError) -> Self {
        match error {
            FeatureCounterError::BatchTooLarge => Self::BatchTooLarge,
        }
    }
}

/// Learns composed features over sequential fixed-size residual-sign batches.
#[derive(Debug)]
pub struct FeatureLearner {
    config: FeatureLearningConfig,
    store: FeatureStore,
    has_previous_batch: bool,
}

impl FeatureLearner {
    pub fn new(
        source_feature_count: usize,
        config: FeatureLearningConfig,
    ) -> Result<Self, FeatureLearningError> {
        if config.batch_size == 0
            || config.parent_top_k == 0
            || config.features_per_layer == 0
            || config.candidate_capacity == 0
            || config.max_layers == 0
            || config.batch_size > FeatureCounter::MAX_BATCH_SIZE
        {
            return Err(FeatureLearningError::InvalidConfig);
        }
        Ok(Self {
            config,
            store: FeatureStore::new(source_feature_count),
            has_previous_batch: false,
        })
    }

    #[must_use]
    pub fn store(&self) -> &FeatureStore {
        &self.store
    }

    /// Scores the preceding batch's candidates, prunes, then trains new candidates.
    pub fn observe_batch(&mut self, batch: SignBatch<'_>) -> Result<(), FeatureLearningError> {
        self.validate_batch(batch)?;
        if self.has_previous_batch {
            self.refresh_scores(batch)?;
            self.promote_and_prune();
        }
        self.learn_candidates(batch)?;
        self.has_previous_batch = true;
        Ok(())
    }

    fn validate_batch(&self, batch: SignBatch<'_>) -> Result<(), FeatureLearningError> {
        if batch.feature_columns.len() != self.store.source_feature_count()
            || batch.signs.len() != self.config.batch_size
            || batch
                .feature_columns
                .iter()
                .any(|column| column.len() != self.config.batch_size)
        {
            return Err(FeatureLearningError::InvalidBatch);
        }
        Ok(())
    }

    fn refresh_scores(&mut self, batch: SignBatch<'_>) -> Result<(), FeatureLearningError> {
        for index in 0..self.store.source_feature_count() {
            let score = correct_count(batch.feature_columns[index], batch.signs);
            self.store.set_score(FeatureId { layer: 0, index }, score)?;
        }

        let mut cache = HashMap::new();
        for layer in 1..=self.store.learned_layer_count() {
            for reference in self.store.live_refs_in_layer(layer) {
                let values = evaluate_reference(&self.store, batch, reference, &mut cache)?;
                self.store
                    .set_score(reference, correct_count(&values, batch.signs))?;
            }
            let scores: Vec<u8> = self
                .store
                .candidates(layer)
                .unwrap_or_default()
                .iter()
                .map(|feature| {
                    evaluate_feature(&self.store, batch, feature, &mut cache)
                        .map(|values| correct_count(&values, batch.signs))
                })
                .collect::<Result<_, _>>()?;
            for (feature, score) in self
                .store
                .candidates_mut(layer)
                .unwrap_or_default()
                .iter_mut()
                .zip(scores)
            {
                feature.score = score;
            }
        }
        Ok(())
    }

    fn promote_and_prune(&mut self) {
        for layer in 1..=self.store.learned_layer_count() {
            self.store.promote_candidates(layer);
        }
        for layer in 1..=self.store.learned_layer_count() {
            let mut references = self.store.live_refs_in_layer(layer);
            if references.len() <= self.config.features_per_layer {
                continue;
            }
            references.sort_by(|left, right| {
                propagated_score(&self.store, *left)
                    .cmp(&propagated_score(&self.store, *right))
                    .then_with(|| left.cmp(right))
            });
            let remove_count = references.len() - self.config.features_per_layer;
            for reference in references.into_iter().take(remove_count) {
                tombstone_with_dependents(&mut self.store, reference);
            }
        }
    }

    fn learn_candidates(&mut self, batch: SignBatch<'_>) -> Result<(), FeatureLearningError> {
        for layer in 1..=self.config.max_layers {
            if self.store.candidate_len(layer).unwrap_or(0) >= self.config.candidate_capacity {
                continue;
            }
            let parents = parent_top_k(
                &self.store,
                self.store.live_refs_in_layer(layer - 1),
                self.config.parent_top_k,
            );
            if parents.len() < 2 {
                continue;
            }
            for (left_index, left) in parents.iter().enumerate() {
                for right in parents.iter().skip(left_index + 1) {
                    if self.store.candidate_len(layer).unwrap_or(0)
                        >= self.config.candidate_capacity
                    {
                        break;
                    }
                    let inputs = if left < right {
                        [*left, *right]
                    } else {
                        [*right, *left]
                    };
                    let mut cache = HashMap::new();
                    let first = evaluate_reference(&self.store, batch, inputs[0], &mut cache)?;
                    let second = evaluate_reference(&self.store, batch, inputs[1], &mut cache)?;
                    let truth_table = learn_truth_table(&first, &second, batch.signs)?;
                    self.store.insert_candidate(layer, inputs, truth_table)?;
                }
            }
        }
        Ok(())
    }
}

fn learn_truth_table(
    first: &[bool],
    second: &[bool],
    signs: &[bool],
) -> Result<u8, FeatureLearningError> {
    let examples: Vec<_> = first
        .iter()
        .zip(second)
        .zip(signs)
        .map(|((&first, &second), &sign)| (first, second, sign))
        .collect();
    Ok(FeatureCounter::from_batch(&examples)?.truth_table())
}

fn parent_top_k(store: &FeatureStore, mut references: Vec<FeatureId>, k: usize) -> Vec<FeatureId> {
    references.sort_by(|left, right| {
        store
            .score(*right)
            .unwrap_or(0)
            .cmp(&store.score(*left).unwrap_or(0))
            .then_with(|| left.cmp(right))
    });
    references.truncate(k);
    references
}

fn correct_count(values: &[bool], signs: &[bool]) -> u8 {
    u8::try_from(
        values
            .iter()
            .zip(signs)
            .filter(|(value, sign)| value == sign)
            .count(),
    )
    .expect("batch size is validated to fit in u8")
}

fn evaluate_reference(
    store: &FeatureStore,
    batch: SignBatch<'_>,
    reference: FeatureId,
    cache: &mut HashMap<FeatureId, Vec<bool>>,
) -> Result<Vec<bool>, FeatureLearningError> {
    if let Some(values) = cache.get(&reference) {
        return Ok(values.clone());
    }
    let values = if reference.layer == 0 {
        batch
            .feature_columns
            .get(reference.index)
            .ok_or(FeatureLearningError::InvalidBatch)?
            .to_vec()
    } else {
        let feature = store
            .get(reference)
            .ok_or(FeatureLearningError::InvalidBatch)?;
        evaluate_feature(store, batch, feature, cache)?
    };
    cache.insert(reference, values.clone());
    Ok(values)
}

fn evaluate_feature(
    store: &FeatureStore,
    batch: SignBatch<'_>,
    feature: &Feature,
    cache: &mut HashMap<FeatureId, Vec<bool>>,
) -> Result<Vec<bool>, FeatureLearningError> {
    let first = evaluate_reference(store, batch, feature.inputs[0], cache)?;
    let second = evaluate_reference(store, batch, feature.inputs[1], cache)?;
    Ok(first
        .into_iter()
        .zip(second)
        .map(|(first, second)| {
            feature.truth_table & (1 << ((u8::from(first) << 1) | u8::from(second))) != 0
        })
        .collect())
}

fn propagated_score(store: &FeatureStore, reference: FeatureId) -> u8 {
    let mut score = store.score(reference).unwrap_or(0);
    for layer in reference.layer + 1..=store.learned_layer_count() {
        for dependent in store.live_refs_in_layer(layer) {
            if store
                .get(dependent)
                .is_some_and(|feature| feature.inputs.contains(&reference))
            {
                score = score.max(propagated_score(store, dependent));
            }
        }
    }
    score
}

fn tombstone_with_dependents(store: &mut FeatureStore, reference: FeatureId) {
    let dependents: Vec<_> = (reference.layer + 1..=store.learned_layer_count())
        .flat_map(|layer| store.live_refs_in_layer(layer))
        .filter(|dependent| {
            store
                .get(*dependent)
                .is_some_and(|feature| feature.inputs.contains(&reference))
        })
        .collect();
    store.tombstone(reference);
    for dependent in dependents {
        tombstone_with_dependents(store, dependent);
    }
}

#[cfg(test)]
mod tests {
    use super::{FeatureLearner, FeatureLearningConfig, SignBatch};
    use crate::FeatureId;

    fn config() -> FeatureLearningConfig {
        FeatureLearningConfig {
            batch_size: 2,
            parent_top_k: 2,
            features_per_layer: 1,
            candidate_capacity: 4,
            max_layers: 2,
        }
    }

    #[test]
    fn trains_on_one_batch_and_scores_on_the_next() {
        let mut learner = FeatureLearner::new(2, config()).unwrap();
        let first = [false, true];
        let second = [false, false];
        let columns = [&first[..], &second[..]];
        let signs = [false, true];

        learner
            .observe_batch(SignBatch {
                feature_columns: &columns,
                signs: &signs,
            })
            .unwrap();
        assert_eq!(learner.store().candidate_len(1), Some(1));
        assert_eq!(learner.store().layer_len(1), Some(0));

        learner
            .observe_batch(SignBatch {
                feature_columns: &columns,
                signs: &signs,
            })
            .unwrap();

        let layer_one = learner.store().live_refs_in_layer(1);
        assert_eq!(layer_one.len(), 1);
        assert_eq!(learner.store().score(layer_one[0]), Some(2));
        assert_eq!(
            learner.store().score(FeatureId { layer: 0, index: 0 }),
            Some(2)
        );
        assert_eq!(learner.store().candidate_len(2), None);
    }

    #[test]
    fn candidate_score_is_taken_from_the_following_batch() {
        let mut learner = FeatureLearner::new(2, config()).unwrap();
        let first = [false, true];
        let second = [false, false];
        let columns = [&first[..], &second[..]];

        learner
            .observe_batch(SignBatch {
                feature_columns: &columns,
                signs: &[false, true],
            })
            .unwrap();
        learner
            .observe_batch(SignBatch {
                feature_columns: &columns,
                signs: &[true, false],
            })
            .unwrap();

        let feature = learner.store().live_refs_in_layer(1)[0];
        assert_eq!(learner.store().score(feature), Some(0));
    }

    #[test]
    fn pruning_cascades_to_dependents() {
        let mut learner = FeatureLearner::new(2, config()).unwrap();
        let first = learner
            .store
            .insert(
                [
                    FeatureId { layer: 0, index: 0 },
                    FeatureId { layer: 0, index: 1 },
                ],
                0,
                0,
            )
            .unwrap();
        let retained = learner
            .store
            .insert(
                [
                    FeatureId { layer: 0, index: 1 },
                    FeatureId { layer: 0, index: 0 },
                ],
                0,
                5,
            )
            .unwrap();
        let dependent = learner
            .store
            .insert([first, FeatureId { layer: 0, index: 0 }], 0, 3)
            .unwrap();

        learner.promote_and_prune();

        assert!(learner.store().get(first).is_none());
        assert!(learner.store().get(dependent).is_none());
        assert!(learner.store().get(retained).is_some());
    }
}
