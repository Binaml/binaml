/// A reference to either a source feature (layer zero) or a composed feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FeatureId {
    pub layer: usize,
    pub index: usize,
}

/// A composed two-input boolean feature and its pruning score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Feature {
    pub inputs: [FeatureId; 2],
    /// A four-bit truth table: bit `(x0 << 1) | x1` is the output.
    pub truth_table: u8,
    pub score: u8,
}

/// Why a composed feature could not be added to a [`FeatureStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertFeatureError {
    InvalidInputRef(FeatureId),
}

/// Stores composed features grouped by their dependency depth.
///
/// Layer zero is the input-feature layer. Composed features start
/// at layer one; `layers[0]` consequently stores features at layer one.
#[derive(Debug, Clone, Default)]
pub struct FeatureStore {
    source_scores: Vec<u8>,
    layers: Vec<FeatureLayer>,
}

#[derive(Debug, Clone, Default)]
struct FeatureLayer {
    live: Vec<Option<Feature>>,
    candidates: Vec<Feature>,
}

impl FeatureStore {
    #[must_use]
    pub fn new(source_feature_count: usize) -> Self {
        Self {
            source_scores: vec![0; source_feature_count],
            layers: Vec::new(),
        }
    }

    #[must_use]
    pub fn source_feature_count(&self) -> usize {
        self.source_scores.len()
    }

    #[must_use]
    pub fn learned_layer_count(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    #[allow(dead_code)]
    pub fn layer_len(&self, layer: usize) -> Option<usize> {
        if layer == 0 {
            Some(self.source_feature_count())
        } else {
            self.layers
                .get(layer - 1)
                .map(|features| features.live.iter().flatten().count())
        }
    }

    #[must_use]
    pub fn get(&self, reference: FeatureId) -> Option<&Feature> {
        reference
            .layer
            .checked_sub(1)
            .and_then(|layer| self.layers.get(layer))
            .and_then(|features| features.live.get(reference.index))
            .and_then(Option::as_ref)
    }

    #[must_use]
    pub fn score(&self, reference: FeatureId) -> Option<u8> {
        if reference.layer == 0 {
            self.source_scores.get(reference.index).copied()
        } else {
            self.get(reference).map(|feature| feature.score)
        }
    }

    pub fn set_score(&mut self, reference: FeatureId, score: u8) -> Result<(), InsertFeatureError> {
        if reference.layer == 0 {
            let Some(source_score) = self.source_scores.get_mut(reference.index) else {
                return Err(InsertFeatureError::InvalidInputRef(reference));
            };
            *source_score = score;
            return Ok(());
        }
        let Some(feature) = self
            .layers
            .get_mut(reference.layer - 1)
            .and_then(|layer| layer.live.get_mut(reference.index))
            .and_then(Option::as_mut)
        else {
            return Err(InsertFeatureError::InvalidInputRef(reference));
        };
        feature.score = score;
        Ok(())
    }

    #[must_use]
    pub fn live_refs_in_layer(&self, layer: usize) -> Vec<FeatureId> {
        if layer == 0 {
            return (0..self.source_feature_count())
                .map(|index| FeatureId { layer, index })
                .collect();
        }
        self.layers
            .get(layer - 1)
            .map(|features| {
                features
                    .live
                    .iter()
                    .enumerate()
                    .filter_map(|(index, feature)| {
                        feature.as_ref().map(|_| FeatureId { layer, index })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    #[must_use]
    pub fn candidate_len(&self, layer: usize) -> Option<usize> {
        self.layers
            .get(layer.checked_sub(1)?)
            .map(|entry| entry.candidates.len())
    }

    pub(crate) fn candidates(&self, layer: usize) -> Option<&[Feature]> {
        Some(&self.layers.get(layer.checked_sub(1)?)?.candidates)
    }

    pub(crate) fn candidates_mut(&mut self, layer: usize) -> Option<&mut [Feature]> {
        Some(&mut self.layers.get_mut(layer.checked_sub(1)?)?.candidates)
    }

    /// Adds a pre-scored live feature. Batch learning uses [`Self::insert_candidate`].
    #[allow(dead_code)]
    pub fn insert(
        &mut self,
        inputs: [FeatureId; 2],
        truth_table: u8,
        score: u8,
    ) -> Result<FeatureId, InsertFeatureError> {
        for input in inputs {
            if !self.is_valid_input_ref(input) {
                return Err(InsertFeatureError::InvalidInputRef(input));
            }
        }

        let layer = inputs[0].layer.max(inputs[1].layer) + 1;
        debug_assert!(inputs.iter().any(|input| input.layer == layer - 1));

        self.ensure_layer(layer);
        let features = &mut self.layers[layer - 1].live;
        let reference = FeatureId {
            layer,
            index: features.len(),
        };
        features.push(Some(Feature {
            inputs,
            truth_table,
            score,
        }));
        Ok(reference)
    }

    pub(crate) fn insert_candidate(
        &mut self,
        layer: usize,
        inputs: [FeatureId; 2],
        truth_table: u8,
    ) -> Result<(), InsertFeatureError> {
        if inputs[0] >= inputs[1] {
            return Err(InsertFeatureError::InvalidInputRef(inputs[0]));
        }
        if layer != inputs[0].layer.max(inputs[1].layer) + 1 {
            return Err(InsertFeatureError::InvalidInputRef(inputs[1]));
        }
        for input in inputs {
            if !self.is_valid_input_ref(input) {
                return Err(InsertFeatureError::InvalidInputRef(input));
            }
        }
        self.ensure_layer(layer);
        self.layers[layer - 1].candidates.push(Feature {
            inputs,
            truth_table,
            score: 0,
        });
        Ok(())
    }

    pub(crate) fn promote_candidates(&mut self, layer: usize) -> Vec<FeatureId> {
        let Some(features) = self.layers.get_mut(layer.saturating_sub(1)) else {
            return Vec::new();
        };
        let start = features.live.len();
        let pending = std::mem::take(&mut features.candidates);
        let count = pending.len();
        features.live.extend(pending.into_iter().map(Some));
        (start..start + count)
            .map(|index| FeatureId { layer, index })
            .collect()
    }

    pub(crate) fn tombstone(&mut self, reference: FeatureId) {
        if let Some(feature) = self
            .layers
            .get_mut(reference.layer.saturating_sub(1))
            .and_then(|layer| layer.live.get_mut(reference.index))
        {
            *feature = None;
        }
    }

    fn ensure_layer(&mut self, layer: usize) {
        while self.layers.len() < layer {
            self.layers.push(FeatureLayer::default());
        }
    }

    fn is_valid_input_ref(&self, reference: FeatureId) -> bool {
        if reference.layer == 0 {
            reference.index < self.source_feature_count()
        } else {
            self.get(reference).is_some()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FeatureId, FeatureStore, InsertFeatureError};

    #[test]
    fn stores_source_feature_function_at_first_learned_layer() {
        let mut store = FeatureStore::new(2);
        let feature = store
            .insert(
                [
                    FeatureId { layer: 0, index: 0 },
                    FeatureId { layer: 0, index: 1 },
                ],
                0b0110,
                3,
            )
            .unwrap();

        assert_eq!(feature, FeatureId { layer: 1, index: 0 });
        assert_eq!(store.source_feature_count(), 2);
        assert_eq!(store.learned_layer_count(), 1);
        assert_eq!(store.layer_len(0), Some(2));
        assert_eq!(store.layer_len(1), Some(1));
        assert_eq!(store.get(feature).unwrap().truth_table, 0b0110);
        assert_eq!(store.get(feature).unwrap().score, 3);
    }

    #[test]
    fn derives_depth_and_preserves_stable_indexes() {
        let mut store = FeatureStore::new(2);
        let layer_one = store
            .insert(
                [
                    FeatureId { layer: 0, index: 0 },
                    FeatureId { layer: 0, index: 1 },
                ],
                0,
                4,
            )
            .unwrap();
        let same_layer = store
            .insert(
                [
                    FeatureId { layer: 0, index: 1 },
                    FeatureId { layer: 0, index: 0 },
                ],
                1,
                2,
            )
            .unwrap();
        let layer_two = store
            .insert([layer_one, FeatureId { layer: 0, index: 0 }], 0b1001, 1)
            .unwrap();

        assert_eq!(same_layer, FeatureId { layer: 1, index: 1 });
        assert_eq!(layer_two, FeatureId { layer: 2, index: 0 });
        assert_eq!(
            store.get(layer_two).unwrap().inputs,
            [layer_one, FeatureId { layer: 0, index: 0 }]
        );
        assert_eq!(store.get(layer_one).unwrap().score, 4);
    }

    #[test]
    fn rejects_invalid_references_without_mutation() {
        let mut store = FeatureStore::new(1);
        let invalid_source = FeatureId { layer: 0, index: 1 };
        let missing_layer = FeatureId { layer: 1, index: 0 };

        assert_eq!(
            store.insert([invalid_source, invalid_source], 0, 1),
            Err(InsertFeatureError::InvalidInputRef(invalid_source))
        );
        assert_eq!(
            store.insert([missing_layer, missing_layer], 0, 1),
            Err(InsertFeatureError::InvalidInputRef(missing_layer))
        );
        assert_eq!(store.learned_layer_count(), 0);
    }
}
