//! Train / test split over N-u8 input samples (flat storage).

use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::bits::sample;
use super::dgp::{domain_size, sample_pool_size, LinearU8Dgp};

#[derive(Debug, Clone, Copy)]
pub enum TrainSplit {
    Frac(f64),
    Count(usize),
}

impl TrainSplit {
    pub fn kind(self) -> &'static str {
        match self {
            TrainSplit::Frac(_) => "frac",
            TrainSplit::Count(_) => "count",
        }
    }

    pub fn frac_value(self) -> Option<f64> {
        match self {
            TrainSplit::Frac(f) => Some(f),
            TrainSplit::Count(_) => None,
        }
    }

    pub fn count_value(self) -> Option<usize> {
        match self {
            TrainSplit::Frac(_) => None,
            TrainSplit::Count(n) => Some(n),
        }
    }

    fn train_n(self, n_samples: usize) -> usize {
        let max_train = n_samples.saturating_sub(1).max(1);
        match self {
            TrainSplit::Frac(f) => {
                ((n_samples as f64 * f).round() as usize).clamp(1, max_train)
            }
            TrainSplit::Count(n) => n.clamp(1, max_train),
        }
    }
}

#[derive(Debug, Clone)]
pub struct U8Dataset {
    pub n_inputs: usize,
    pub train_n: usize,
    pub train_x: Vec<u8>,
    pub train_y: Vec<bool>,
    pub test_x: Vec<u8>,
    pub test_y: Vec<bool>,
}

impl U8Dataset {
    pub fn train_len(&self) -> usize {
        self.train_y.len()
    }

    pub fn test_len(&self) -> usize {
        self.test_y.len()
    }

    pub fn train_sample(&self, idx: usize) -> &[u8] {
        sample(&self.train_x, self.n_inputs, idx)
    }

    pub fn test_sample(&self, idx: usize) -> &[u8] {
        sample(&self.test_x, self.n_inputs, idx)
    }

    pub fn split(seed: u64, dgp: &LinearU8Dgp, train_frac: f64) -> Self {
        Self::split_with(seed, dgp, TrainSplit::Frac(train_frac))
    }

    pub fn split_with(seed: u64, dgp: &LinearU8Dgp, split: TrainSplit) -> Self {
        let n_inputs = dgp.n_inputs;
        let x_flat = generate_samples_flat(n_inputs, seed);
        let n_samples = x_flat.len() / n_inputs;
        let train_n = split.train_n(n_samples);
        let test_n = n_samples - train_n;

        let mut perm: Vec<u32> = (0..n_samples as u32).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        perm.shuffle(&mut rng);

        let mut train_x = Vec::with_capacity(train_n * n_inputs);
        let mut train_y = Vec::with_capacity(train_n);
        let mut test_x = Vec::with_capacity(test_n * n_inputs);
        let mut test_y = Vec::with_capacity(test_n);

        for (slot, &pi) in perm.iter().enumerate() {
            let i = pi as usize;
            let off = i * n_inputs;
            let xs = &x_flat[off..off + n_inputs];
            let y = dgp.eval(xs);
            if slot < train_n {
                train_x.extend_from_slice(xs);
                train_y.push(y);
            } else {
                test_x.extend_from_slice(xs);
                test_y.push(y);
            }
        }

        Self {
            n_inputs,
            train_n,
            train_x,
            train_y,
            test_x,
            test_y,
        }
    }
}

fn generate_samples_flat(n_inputs: usize, seed: u64) -> Vec<u8> {
    if domain_size(n_inputs) <= sample_pool_size(n_inputs) as u128 {
        enumerate_all_flat(n_inputs)
    } else {
        random_samples_flat(seed.wrapping_add(999), n_inputs, sample_pool_size(n_inputs))
    }
}

fn enumerate_all_flat(n_inputs: usize) -> Vec<u8> {
    let n = sample_pool_size(n_inputs);
    let mut out = vec![0u8; n * n_inputs];
    let mut v = vec![0u8; n_inputs];
    for i in 0..n {
        let off = i * n_inputs;
        out[off..off + n_inputs].copy_from_slice(&v);
        increment_mixed_radix(&mut v);
    }
    out
}

fn increment_mixed_radix(v: &mut [u8]) {
    for byte in v.iter_mut() {
        if *byte < 255 {
            *byte += 1;
            return;
        }
        *byte = 0;
    }
}

fn random_samples_flat(seed: u64, n_inputs: usize, count: usize) -> Vec<u8> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = vec![0u8; count * n_inputs];
    for i in 0..count {
        let off = i * n_inputs;
        for j in 0..n_inputs {
            out[off + j] = rng.gen();
        }
    }
    out
}
