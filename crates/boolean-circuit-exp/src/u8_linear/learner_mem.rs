//! Latest-label memory learner over N × u8 inputs.
//!
//! Weights are packed: one byte per input u8 for `w0` (labels when bit=0) and `w1`
//! (labels when bit=1). Inference uses `popcount((w0 & !x) | (w1 & x))`.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::bits::{count_errors, sample, score_mem_u8, update_byte_mem};
use super::dgp::{BITS_PER_U8, n_bits};

#[inline]
pub fn label_u8(y: bool) -> u8 {
    u8::from(y)
}

pub fn score_mem(xs: &[u8], w0: &[u8], w1: &[u8]) -> i32 {
    score_mem_u8(xs, w0, w1, (n_bits(xs.len()) / 2) as i32)
}

#[derive(Debug, Clone)]
pub struct U8MemLearner {
    pub n_inputs: usize,
    pub w0: Vec<u8>,
    pub w1: Vec<u8>,
    midpoint: i32,
}

impl U8MemLearner {
    pub fn new(n_inputs: usize) -> Self {
        let nb = n_inputs * BITS_PER_U8;
        Self {
            n_inputs,
            w0: vec![0u8; n_inputs],
            w1: vec![0u8; n_inputs],
            midpoint: (nb / 2) as i32,
        }
    }

    pub fn predict(&self, x: &[u8]) -> bool {
        assert_eq!(x.len(), self.n_inputs);
        score_mem_u8(x, &self.w0, &self.w1, self.midpoint) >= 0
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
            update_byte_mem(&mut self.w0[j], &mut self.w1[j], byte, t);
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

    pub fn weights(&self) -> (&[u8], &[u8]) {
        (&self.w0, &self.w1)
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
    fn stores_latest_per_bit_value() {
        let mut learner = U8MemLearner::new(1);
        learner.update_one(&[0b0000_0000], false);
        learner.update_one(&[0b0000_0001], true);
        assert_eq!(learner.w0[0] & 1, 0);
        assert_eq!(learner.w1[0] & 1, 1);
    }

    #[test]
    fn overwrite_replaces_cell() {
        let mut learner = U8MemLearner::new(1);
        learner.update_one(&[0], false);
        assert_eq!(learner.w0[0], 0);
        learner.update_one(&[0], true);
        assert_eq!(learner.w0[0], 0xFF);
    }

    #[test]
    fn runs_on_random_dgp() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let dgp = LinearU8Dgp::random(&mut rng, 2);
        let data = U8Dataset::split(11, &dgp, 0.8);
        let mut learner = U8MemLearner::new(2);
        learner.fit(&data.train_x, &data.train_y, 10);
        let acc = learner.accuracy(&data.test_x, &data.test_y);
        assert!(acc.is_finite());
    }
}
