use crate::conjunction_build_common::{
    ConjunctionBuildError, ConjunctionKey, MAX_LITERALS,
};

/// Stored conjunction expert: AND of signed literals, sorted by feature index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConjunctionExpert {
    pub(crate) len: u8,
    pub(crate) literals: [(u8, bool); MAX_LITERALS],
}

impl ConjunctionExpert {
    pub(crate) fn empty() -> Self {
        Self {
            len: 0,
            literals: [(0, false); MAX_LITERALS],
        }
    }

    pub fn from_key(
        key: &ConjunctionKey,
        word_count: usize,
        feature_count: usize,
        max_conjunction_length: usize,
    ) -> Result<Self, ConjunctionBuildError> {
        let len = key.len(word_count);
        if len == 0 || len > max_conjunction_length {
            return Err(ConjunctionBuildError::ExpertTooLarge);
        }
        let mut expert = Self::empty();
        let mut slot = 0_usize;
        for feature_index in 0..feature_count {
            let word = feature_index / 64;
            if word >= word_count {
                break;
            }
            let mask = 1_u64 << (feature_index % 64);
            if key.positive[word] & mask != 0 {
                expert.literals[slot] = (feature_index as u8, false);
                slot += 1;
            } else if key.negative[word] & mask != 0 {
                expert.literals[slot] = (feature_index as u8, true);
                slot += 1;
            }
        }
        expert.len = u8::try_from(slot).expect("literal count fits in u8");
        Ok(expert)
    }

    pub(crate) fn copy_from(&mut self, other: &Self) {
        self.len = other.len;
        self.literals.copy_from_slice(&other.literals);
    }

    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    #[must_use]
    pub fn evaluate(&self, features: &[bool]) -> bool {
        for index in 0..self.len as usize {
            let (feature_index, negated) = self.literals[index];
            let value = features
                .get(feature_index as usize)
                .copied()
                .unwrap_or(false);
            let literal = if negated { !value } else { value };
            if !literal {
                return false;
            }
        }
        true
    }

    #[cfg(test)]
    #[must_use]
    pub fn literal_count(&self) -> usize {
        self.len as usize
    }

    #[cfg(test)]
    pub fn literals(&self) -> &[(u8, bool)] {
        &self.literals[..self.len as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::ConjunctionExpert;
    use crate::conjunction_build_common::ConjunctionKey;

    #[test]
    fn from_key_decodes_negated_literals_sorted() {
        let key = ConjunctionKey::EMPTY
            .with_literal(2, true, 1)
            .unwrap()
            .with_literal(0, false, 1)
            .unwrap();
        let expert = ConjunctionExpert::from_key(&key, 1, 4, 7).unwrap();
        assert_eq!(expert.literals(), &[(0, false), (2, true)]);
    }

    #[test]
    fn evaluate_ands_literals() {
        let key = ConjunctionKey::EMPTY
            .with_literal(0, false, 1)
            .unwrap()
            .with_literal(1, true, 1)
            .unwrap();
        let expert = ConjunctionExpert::from_key(&key, 1, 2, 7).unwrap();
        assert!(expert.evaluate(&[true, false]));
        assert!(!expert.evaluate(&[false, false]));
        assert!(!expert.evaluate(&[true, true]));
    }
}
