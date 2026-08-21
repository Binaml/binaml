//! Symmetric latest-label memory: one packed byte per input u8.
//!
//! Store `target` when the bit is 1, `!target` when 0. At predict time read
//! `mem` or `!mem` per bit — score is matching-bit count via `popcount`.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::bits::{count_errors, sample, score_sym_u8};
use super::dgp::BITS_PER_U8;
use super::learner_mem::label_u8;

#[derive(Debug, Clone)]
pub struct U8SymLearner {
    pub n_inputs: usize,
    /// One packed byte per input u8 (bit i = last label polarity for bit i).
    pub mem: Vec<u8>,
    midpoint: i32,
}

impl U8SymLearner {
    pub fn new(n_inputs: usize) -> Self {
        let nb = n_inputs * BITS_PER_U8;
        Self {
            n_inputs,
            mem: vec![0u8; n_inputs],
            midpoint: (nb / 2) as i32,
        }
    }

    pub fn predict(&self, x: &[u8]) -> bool {
        assert_eq!(x.len(), self.n_inputs);
        score_sym_u8(x, &self.mem, self.midpoint) >= 0
    }

    pub fn fit(&mut self, x_flat: &[u8], y: &[bool], epochs: usize) -> usize {
        assert_eq!(y.len() * self.n_inputs, x_flat.len());
        let n = y.len();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut indices: Vec<usize> = (0..n).collect();
        let mut epochs_run = 0;

        for _ in 0..epochs {
            epochs_run += 1;
            indices.shuffle(&mut rng);
            for &i in &indices {
                self.update_one(sample(x_flat, self.n_inputs, i), y[i]);
            }
            if count_errors(x_flat, y, self.n_inputs, |xs| self.predict(xs)) == 0 {
                break;
            }
        }
        epochs_run
    }

    fn update_one(&mut self, x: &[u8], y: bool) {
        let t = label_u8(y);
        for (j, &byte) in x.iter().enumerate() {
            self.mem[j] = if t != 0 { byte } else { !byte };
        }
    }

    pub fn accuracy(&self, x_flat: &[u8], y: &[bool]) -> f64 {
        let n = y.len();
        if n == 0 {
            return 1.0;
        }
        let errors = count_errors(x_flat, y, self.n_inputs, |xs| self.predict(xs));
        1.0 - errors as f64 / n as f64
    }

    pub fn error_count(&self, x_flat: &[u8], y: &[bool]) -> usize {
        count_errors(x_flat, y, self.n_inputs, |xs| self.predict(xs))
    }

    pub fn weights(&self) -> &[u8] {
        &self.mem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::u8_linear::dgp::LinearU8Dgp;
    use crate::u8_linear::dataset::U8Dataset;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn stores_target_on_one_not_on_zero() {
        let mut learner = U8SymLearner::new(1);
        learner.update_one(&[0b0000_0000], false);
        assert_eq!(learner.mem[0], 0b1111_1111);
        learner.update_one(&[0b0000_0001], true);
        assert_eq!(learner.mem[0], 0b0000_0001);
    }

    #[test]
    fn overwrite_replaces_byte() {
        let mut learner = U8SymLearner::new(1);
        learner.update_one(&[0], false);
        assert_eq!(learner.mem[0], 0xFF);
        learner.update_one(&[0], true);
        assert_eq!(learner.mem[0], 0);
    }

    #[test]
    fn predict_uses_symmetric_readout() {
        let mut learner = U8SymLearner::new(1);
        learner.update_one(&[0b0000_0001], true);
        assert!(learner.predict(&[0b0000_0001]));
    }

    #[test]
    fn runs_on_random_dgp() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let dgp = LinearU8Dgp::random(&mut rng, 2);
        let data = U8Dataset::split(11, &dgp, 0.8);
        let mut learner = U8SymLearner::new(2);
        learner.fit(&data.train_x, &data.train_y, 10);
        let acc = learner.accuracy(&data.test_x, &data.test_y);
        assert!(acc.is_finite());
    }
}
