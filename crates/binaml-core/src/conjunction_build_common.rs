use crate::SignBatch;

pub const MAX_KEY_WORDS: usize = 4;
pub const MAX_LITERALS: usize = 64;
pub const DEFAULT_MAX_CONJUNCTION_LENGTH: usize = 7;
pub const DEFAULT_MAX_EXPERTS: usize = 64;
pub const DEFAULT_STALE_LAYERS: usize = 2;
pub const MAX_BATCH_SIZE: usize = u8::MAX as usize;

#[derive(Debug, Clone, Copy)]
pub struct ConjunctionBuildConfig {
    pub batch_size: usize,
    pub max_conjunctions: usize,
    pub max_conjunction_length: usize,
    pub max_experts: usize,
    pub stale_layers: usize,
}

impl ConjunctionBuildConfig {
    pub fn new(
        batch_size: usize,
        max_conjunctions: usize,
        max_conjunction_length: usize,
        max_experts: usize,
        stale_layers: usize,
    ) -> Self {
        Self {
            batch_size,
            max_conjunctions,
            max_conjunction_length,
            max_experts,
            stale_layers,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConjunctionCapacity {
    pub feature_count: usize,
    pub batch_size: usize,
    pub max_conjunctions: usize,
    pub max_conjunction_length: usize,
    pub max_extensions: usize,
    pub key_words: usize,
}

pub fn derive_conjunction_capacity(
    feature_count: usize,
    config: ConjunctionBuildConfig,
) -> ConjunctionCapacity {
    // Each beam parent can emit up to 2 * feature_count extensions; keep headroom.
    let max_extensions = (config.max_conjunctions + 1) * 2 * feature_count.max(1);
    ConjunctionCapacity {
        feature_count,
        batch_size: config.batch_size,
        max_conjunctions: config.max_conjunctions,
        max_conjunction_length: config.max_conjunction_length,
        max_extensions,
        key_words: feature_count.div_ceil(64).min(MAX_KEY_WORDS),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConjunctionBuildError {
    InvalidConfig,
    InvalidBatch,
    BatchTooLarge,
    ExpertTooLarge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConjunctionKey {
    pub positive: [u64; MAX_KEY_WORDS],
    pub negative: [u64; MAX_KEY_WORDS],
}

impl ConjunctionKey {
    pub const EMPTY: Self = Self {
        positive: [0; MAX_KEY_WORDS],
        negative: [0; MAX_KEY_WORDS],
    };

    pub fn cmp_keys(left: Self, right: Self, word_count: usize) -> std::cmp::Ordering {
        for index in 0..word_count {
            match left.positive[index].cmp(&right.positive[index]) {
                std::cmp::Ordering::Equal => {}
                ordering => return ordering,
            }
        }
        for index in 0..word_count {
            match left.negative[index].cmp(&right.negative[index]) {
                std::cmp::Ordering::Equal => {}
                ordering => return ordering,
            }
        }
        std::cmp::Ordering::Equal
    }

    pub fn contains(&self, feature_index: usize, word_count: usize) -> bool {
        let word = feature_index / 64;
        if word >= word_count {
            return false;
        }
        let mask = 1_u64 << (feature_index % 64);
        (self.positive[word] | self.negative[word]) & mask != 0
    }

    pub fn len(&self, word_count: usize) -> usize {
        (0..word_count)
            .map(|word| (self.positive[word] | self.negative[word]).count_ones() as usize)
            .sum()
    }

    pub fn with_literal(
        &self,
        feature_index: usize,
        negated: bool,
        word_count: usize,
    ) -> Option<Self> {
        if self.contains(feature_index, word_count) {
            return None;
        }
        let word = feature_index / 64;
        if word >= word_count {
            return None;
        }
        let mask = 1_u64 << (feature_index % 64);
        let mut key = *self;
        if negated {
            key.negative[word] |= mask;
        } else {
            key.positive[word] |= mask;
        }
        Some(key)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BeamEntry {
    pub key: ConjunctionKey,
    pub abs_assoc: i64,
    pub accuracy: u8,
    pub column_slot: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ExtensionCandidate {
    pub key: ConjunctionKey,
    pub abs_assoc: i64,
    pub accuracy: u8,
    pub parent_slot: u16,
    pub literal_index: u16,
}

pub(crate) fn validate_build_config(config: ConjunctionBuildConfig) -> Result<(), ConjunctionBuildError> {
    if config.batch_size == 0
        || config.max_conjunctions == 0
        || config.max_conjunction_length == 0
        || config.max_conjunction_length > MAX_LITERALS
        || config.stale_layers == 0
        || config.max_experts == 0
        || config.batch_size > MAX_BATCH_SIZE
    {
        return Err(ConjunctionBuildError::InvalidConfig);
    }
    Ok(())
}

pub(crate) fn is_constant_column(column: &[bool]) -> bool {
    column.len() > 1
        && column
            .first()
            .is_some_and(|first| column.iter().all(|value| *value == *first))
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
) -> Result<(), ConjunctionBuildError> {
    if batch.signs.len() != batch_size || batch.feature_count() == 0 {
        return Err(ConjunctionBuildError::InvalidBatch);
    }
    for index in 0..batch.feature_count() {
        if batch
            .column(index)
            .is_none_or(|column| column.len() != batch_size)
        {
            return Err(ConjunctionBuildError::InvalidBatch);
        }
    }
    Ok(())
}

pub(crate) fn validate_feature_batch(batch: SignBatch<'_>) -> Result<(), ConjunctionBuildError> {
    let batch_size = batch.column(0).map(|column| column.len()).unwrap_or(0);
    if batch_size == 0 {
        return Err(ConjunctionBuildError::InvalidBatch);
    }
    for index in 0..batch.feature_count() {
        if batch
            .column(index)
            .is_none_or(|column| column.len() != batch_size)
        {
            return Err(ConjunctionBuildError::InvalidBatch);
        }
    }
    Ok(())
}

pub(crate) fn score_column(values: &[bool], signs: &[bool], ny: i64) -> (i64, u8) {
    let batch_size = i64::try_from(signs.len()).expect("batch size fits in i64");
    let mut nz = 0_i64;
    let mut nyz = 0_i64;
    for (&value, &sign) in values.iter().zip(signs) {
        if value {
            nz += 1;
            if sign {
                nyz += 1;
            }
        }
    }
    let abs_assoc = (batch_size * nyz - nz * ny).abs();
    let accuracy = u8::try_from(batch_size - nz - ny + 2 * nyz).expect("accuracy fits in u8");
    (abs_assoc, accuracy)
}

pub(crate) fn dedup_extensions(
    extensions: &[ExtensionCandidate],
    order: &[usize],
    count: usize,
    dedup: &mut [ExtensionCandidate],
    word_count: usize,
) -> usize {
    if count == 0 {
        return 0;
    }
    let mut out = 0_usize;
    let mut index = 0_usize;
    while index < count {
        let first = order[index];
        let mut best = extensions[first];
        let mut next = index + 1;
        while next < count
            && ConjunctionKey::cmp_keys(extensions[order[next]].key, best.key, word_count)
                == std::cmp::Ordering::Equal
        {
            let candidate = extensions[order[next]];
            if candidate.abs_assoc > best.abs_assoc
                || (candidate.abs_assoc == best.abs_assoc && candidate.accuracy > best.accuracy)
            {
                best = candidate;
            }
            next += 1;
        }
        dedup[out] = best;
        out += 1;
        index = next;
    }
    out
}

pub(crate) fn sort_extension_indices(
    extensions: &[ExtensionCandidate],
    order: &mut [usize],
    count: usize,
    word_count: usize,
) {
    order[..count].sort_unstable_by(|&left, &right| {
        ConjunctionKey::cmp_keys(extensions[left].key, extensions[right].key, word_count)
    });
}

pub(crate) fn prune_extensions(
    extensions: &mut [ExtensionCandidate],
    count: usize,
    max_conjunctions: usize,
    word_count: usize,
) -> usize {
    extensions[..count].sort_unstable_by(|left, right| {
        right
            .abs_assoc
            .cmp(&left.abs_assoc)
            .then_with(|| right.accuracy.cmp(&left.accuracy))
            .then_with(|| {
                ConjunctionKey::cmp_keys(left.key, right.key, word_count).reverse()
            })
    });
    count.min(max_conjunctions)
}

pub(crate) fn pick_winner(beam: &[BeamEntry], beam_len: usize) -> BeamEntry {
    beam[..beam_len]
        .iter()
        .copied()
        .max_by(|left, right| {
            left.accuracy
                .cmp(&right.accuracy)
                .then_with(|| left.abs_assoc.cmp(&right.abs_assoc))
                .then_with(|| {
                    ConjunctionKey::cmp_keys(left.key, right.key, MAX_KEY_WORDS).reverse()
                })
        })
        .expect("beam has at least one entry")
}

#[cfg(test)]
mod tests {
    use super::{
        dedup_extensions, pick_winner, score_column, BeamEntry, ConjunctionBuildConfig,
        ConjunctionKey, ExtensionCandidate,
    };
    use crate::association::association_score;

    #[test]
    fn conjunction_key_tracks_literals() {
        let word_count = 1;
        let key = ConjunctionKey::EMPTY
            .with_literal(1, false, word_count)
            .unwrap()
            .with_literal(3, true, word_count)
            .unwrap();
        assert_eq!(key.len(word_count), 2);
        assert!(key.contains(1, word_count));
        assert!(key.contains(3, word_count));
        assert!(!key.contains(2, word_count));
        assert!(key.with_literal(1, false, word_count).is_none());
    }

    #[test]
    fn dedup_keeps_better_association_then_accuracy() {
        let key = ConjunctionKey::EMPTY.with_literal(0, false, 1).unwrap();
        let extensions = [
            ExtensionCandidate {
                key,
                abs_assoc: 1,
                accuracy: 2,
                parent_slot: 0,
                literal_index: 0,
            },
            ExtensionCandidate {
                key,
                abs_assoc: 3,
                accuracy: 1,
                parent_slot: 0,
                literal_index: 1,
            },
            ExtensionCandidate {
                key,
                abs_assoc: 3,
                accuracy: 4,
                parent_slot: 0,
                literal_index: 2,
            },
        ];
        let order = [0, 1, 2];
        let mut dedup = extensions;
        let kept = dedup_extensions(&extensions, &order, 3, &mut dedup, 1);
        assert_eq!(kept, 1);
        assert_eq!(dedup[0].abs_assoc, 3);
        assert_eq!(dedup[0].accuracy, 4);
    }

    #[test]
    fn score_column_matches_association_and_accuracy() {
        let values = [true, true, false, false];
        let signs = [true, false, true, false];
        let ny = signs.iter().filter(|sign| **sign).count() as i64;
        let (abs_assoc, accuracy) = score_column(&values, &signs, ny);
        assert_eq!(abs_assoc, association_score(&values, &signs).abs());
        assert_eq!(
            accuracy,
            super::correct_count(&values, &signs),
        );
    }

    #[test]
    fn derive_capacity_scales_with_beam_and_features() {
        let config = ConjunctionBuildConfig::new(8, 4, 7, 64, 2);
        let capacity = super::derive_conjunction_capacity(32, config);
        assert_eq!(capacity.max_extensions, 4 * 2 * 32);
        assert_eq!(capacity.key_words, 1);
    }

    #[test]
    fn pick_winner_uses_accuracy_then_association() {
        let high_accuracy = BeamEntry {
            key: ConjunctionKey::EMPTY.with_literal(0, false, 1).unwrap(),
            abs_assoc: 1,
            accuracy: 4,
            column_slot: 0,
        };
        let high_assoc = BeamEntry {
            key: ConjunctionKey::EMPTY.with_literal(1, false, 1).unwrap(),
            abs_assoc: 5,
            accuracy: 2,
            column_slot: 1,
        };
        let beam = [high_accuracy, high_assoc];
        assert_eq!(pick_winner(&beam, 2).column_slot, 0);
        assert_eq!(pick_winner(&beam, 2).accuracy, 4);
    }
}
