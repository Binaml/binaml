/// Eight-slot majority counter for two-binary-feature truth-table learning.
///
/// Each of the eight `(x0, x1, output)` combinations has its own `u8` count.
/// The batch-size limit guarantees no slot can overflow, so the observation
/// loop only performs additions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FeatureCounter {
    counts: [u8; 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureCounterError {
    BatchTooLarge,
}

impl FeatureCounter {
    pub const MAX_BATCH_SIZE: usize = u8::MAX as usize;

    /// Learns the error-minimizing table for a bounded batch.
    ///
    /// Returns `BatchTooLarge` when the batch exceeds the capacity of one
    /// counter slot. With a valid batch, no slot can exceed `u8::MAX`, even if
    /// every observation has the same value.
    #[inline]
    pub fn from_batch(batch: &[(bool, bool, bool)]) -> Result<Self, FeatureCounterError> {
        if batch.len() > Self::MAX_BATCH_SIZE {
            return Err(FeatureCounterError::BatchTooLarge);
        }

        let mut counts = [0_u8; 8];
        for &(x0, x1, output) in batch {
            let slot = (u8::from(x0) << 2) | (u8::from(x1) << 1) | u8::from(output);
            counts[slot as usize] += 1;
        }
        Ok(Self { counts })
    }

    /// Returns the packed four-entry, error-minimizing truth table.
    ///
    /// Bit `p` is the output for partition
    /// `p = (x0 << 1) | x1`. Ties resolve to zero.
    #[inline]
    pub fn truth_table(&self) -> u8 {
        let mut table = 0_u8;
        for partition in 0_u8..4 {
            let base = (partition << 1) as usize;
            if self.counts[base | 1] > self.counts[base] {
                table |= 1 << partition;
            }
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use super::{FeatureCounter, FeatureCounterError};

    #[test]
    fn learner_selects_majorities() {
        let learner = FeatureCounter::from_batch(&[
            (false, false, false),
            (false, false, true),
            (false, true, true),
            (true, false, false),
            (true, true, true),
            (true, true, true),
        ])
        .unwrap();

        assert_eq!(learner.truth_table(), 0b1010);
    }

    #[test]
    fn slots_are_independent() {
        let learner = FeatureCounter::from_batch(&[
            (false, false, true),
            (false, true, false),
            (true, false, true),
            (true, true, false),
        ])
        .unwrap();

        assert_eq!(learner.counts, [0, 1, 1, 0, 0, 1, 1, 0]);
    }

    #[test]
    fn accepts_its_maximum_batch() {
        let batch = vec![(false, false, true); FeatureCounter::MAX_BATCH_SIZE];
        assert_eq!(FeatureCounter::from_batch(&batch).unwrap().counts[1], 255);
    }

    #[test]
    fn rejects_an_oversized_batch_without_learning() {
        assert_eq!(
            FeatureCounter::from_batch(&[(false, false, false); 256]),
            Err(FeatureCounterError::BatchTooLarge)
        );
    }

    #[test]
    fn all_truth_tables_are_reconstructed() {
        for table in 0..16 {
            let batch: Vec<_> = (0..4)
                .map(|partition| {
                    let x0 = partition & 0b10 != 0;
                    let x1 = partition & 0b01 != 0;
                    let output = table & (1 << partition) != 0;
                    (x0, x1, output)
                })
                .collect();

            assert_eq!(
                FeatureCounter::from_batch(&batch).unwrap().truth_table(),
                table
            );
        }
    }

    #[test]
    fn empty_batch_has_zero_truth_table() {
        assert_eq!(FeatureCounter::from_batch(&[]).unwrap().truth_table(), 0);
    }
}
