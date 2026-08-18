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

    #[inline]
    pub fn from_columns(
        first: &[bool],
        second: &[bool],
        signs: &[bool],
    ) -> Result<Self, FeatureCounterError> {
        if first.len() > Self::MAX_BATCH_SIZE {
            return Err(FeatureCounterError::BatchTooLarge);
        }
        let mut counts = [0_u8; 8];
        for ((&x0, &x1), &output) in first.iter().zip(second).zip(signs) {
            let slot = (u8::from(x0) << 2) | (u8::from(x1) << 1) | u8::from(output);
            counts[slot as usize] += 1;
        }
        Ok(Self { counts })
    }

    /// Returns the learned table plus batch scores without materializing the column.
    #[inline]
    pub fn truth_table_and_scores(&self, batch_size: i64, ny: i64) -> (u8, i64, u8) {
        let mut table = 0_u8;
        let mut nz = 0_i64;
        let mut nzy = 0_i64;
        let mut matches = 0_u8;
        for partition in 0_u8..4 {
            let base = (partition << 1) as usize;
            let output_bit = self.counts[base | 1] > self.counts[base];
            if output_bit {
                table |= 1 << partition;
                nz += i64::from(self.counts[base]) + i64::from(self.counts[base | 1]);
                nzy += i64::from(self.counts[base | 1]);
                matches += self.counts[base | 1];
            } else {
                matches += self.counts[base];
            }
        }
        let abs_assoc = (batch_size * nzy - nz * ny).abs();
        (table, abs_assoc, matches)
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
    fn from_columns_matches_from_batch() {
        let first = [false, false, true, true];
        let second = [false, true, false, true];
        let signs = [false, true, true, false];
        let batch: Vec<_> = first
            .iter()
            .zip(second.iter())
            .zip(signs.iter())
            .map(|((&x0, &x1), &output)| (x0, x1, output))
            .collect();
        let from_batch = FeatureCounter::from_batch(&batch).unwrap();
        let from_columns = FeatureCounter::from_columns(&first, &second, &signs).unwrap();
        assert_eq!(from_batch.counts, from_columns.counts);
        assert_eq!(
            from_batch.truth_table_and_scores(4, 2),
            from_columns.truth_table_and_scores(4, 2)
        );
    }

    #[test]
    fn truth_table_and_scores_match_column_metrics() {
        use crate::association::association_score;
        use crate::boolean_circuit::evaluate_truth_table;

        let first = [false, false, true, true, false, true];
        let second = [false, true, false, true, true, false];
        let signs = [false, true, true, false, true, false];
        let counter = FeatureCounter::from_columns(&first, &second, &signs).unwrap();
        let ny = signs.iter().filter(|&&sign| sign).count() as i64;
        let (table, abs_assoc, matches) = counter.truth_table_and_scores(6, ny);
        let column: Vec<bool> = first
            .iter()
            .zip(second.iter())
            .map(|(&x0, &x1)| evaluate_truth_table(table, x0, x1))
            .collect();
        assert_eq!(table, counter.truth_table());
        assert_eq!(abs_assoc, association_score(&column, &signs).abs());
        assert_eq!(
            matches,
            u8::try_from(
                column
                    .iter()
                    .zip(signs.iter())
                    .filter(|(value, sign)| **value == **sign)
                    .count()
            )
            .unwrap()
        );
    }

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
