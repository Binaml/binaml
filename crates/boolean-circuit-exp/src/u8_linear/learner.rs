//! Batch perceptron learner for N × u8 linear threshold functions.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::bits::{count_errors, sample, score_byte, score_i8};
use super::dgp::BITS_PER_U8;

#[derive(Debug, Clone)]
pub struct U8LinearLearner {
    pub n_inputs: usize,
    pub w0: Vec<i8>,
    pub w1: Vec<i8>,
}

impl U8LinearLearner {
    pub fn new(n_inputs: usize) -> Self {
        let nb = n_inputs * BITS_PER_U8;
        Self {
            n_inputs,
            w0: vec![0i8; nb],
            w1: vec![0i8; nb],
        }
    }

    pub fn predict(&self, x: &[u8]) -> bool {
        assert_eq!(x.len(), self.n_inputs);
        score_i8(x, &self.w0, &self.w1) >= 0
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
        let mut s = 0i32;
        for (j, &byte) in x.iter().enumerate() {
            let base = j * BITS_PER_U8;
            s += score_byte(byte, &self.w0[base..base + BITS_PER_U8], &self.w1[base..base + BITS_PER_U8]);
        }
        if (s >= 0) == y {
            return;
        }
        let delta: i8 = if y { 1 } else { -1 };
        for (j, &byte) in x.iter().enumerate() {
            let base = j * BITS_PER_U8;
            let mut b = byte;
            for i in 0..BITS_PER_U8 {
                if b & 1 != 0 {
                    self.w1[base + i] = self.w1[base + i].saturating_add(delta);
                } else {
                    self.w0[base + i] = self.w0[base + i].saturating_add(delta);
                }
                b >>= 1;
            }
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

    pub fn weights(&self) -> (&[i8], &[i8]) {
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
    fn recovers_full_train() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dgp = LinearU8Dgp::random(&mut rng, 1);
        let xs: Vec<u8> = (0..=255).collect();
        let ys: Vec<bool> = xs.iter().map(|&x| dgp.eval(&[x])).collect();
        let mut learner = U8LinearLearner::new(1);
        learner.fit(&xs, &ys, 50);
        assert!(learner.accuracy(&xs, &ys) >= 0.99);
    }

    #[test]
    fn generalizes_on_holdout() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let dgp = LinearU8Dgp::random(&mut rng, 1);
        let data = U8Dataset::split(99, &dgp, 0.8);
        let mut learner = U8LinearLearner::new(1);
        learner.fit(&data.train_x, &data.train_y, 200);
        assert!(learner.accuracy(&data.test_x, &data.test_y) > 0.85);
    }

    #[test]
    fn no_update_on_correct() {
        let mut learner = U8LinearLearner::new(1);
        let w0_before = learner.w0.clone();
        let w1_before = learner.w1.clone();
        learner.update_one(&[0], true);
        assert_eq!(learner.w0, w0_before);
        assert_eq!(learner.w1, w1_before);
    }
}
