//! Random linear threshold DGP over N × u8 bits (LSB = bit 0 per byte).

use rand::Rng;

use super::bits::score_i8;

pub const BITS_PER_U8: usize = 8;
pub const MAX_SAMPLE_POOL: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightMode {
    I8,
    Binary,
}

impl WeightMode {
    pub fn as_str(self) -> &'static str {
        match self {
            WeightMode::I8 => "i8",
            WeightMode::Binary => "binary",
        }
    }
}

pub fn n_bits(n_inputs: usize) -> usize {
    n_inputs * BITS_PER_U8
}

pub fn domain_size(n_inputs: usize) -> u128 {
    256u128.pow(n_inputs as u32)
}

pub fn sample_pool_size(n_inputs: usize) -> usize {
    let domain = domain_size(n_inputs);
    if domain <= MAX_SAMPLE_POOL as u128 {
        domain as usize
    } else {
        MAX_SAMPLE_POOL
    }
}

pub fn score(xs: &[u8], w0: &[i8], w1: &[i8]) -> i32 {
    score_i8(xs, w0, w1)
}

#[derive(Debug, Clone)]
pub struct LinearU8Dgp {
    pub n_inputs: usize,
    pub w0: Vec<i8>,
    pub w1: Vec<i8>,
}

impl LinearU8Dgp {
    pub fn random<R: Rng>(rng: &mut R, n_inputs: usize) -> Self {
        Self::random_with_mode(rng, n_inputs, WeightMode::I8)
    }

    pub fn random_with_mode<R: Rng>(rng: &mut R, n_inputs: usize, mode: WeightMode) -> Self {
        let nb = n_bits(n_inputs);
        let mut w0 = vec![0i8; nb];
        let mut w1 = vec![0i8; nb];
        for i in 0..nb {
            w0[i] = sample_weight(rng, mode);
            w1[i] = sample_weight(rng, mode);
        }
        Self { n_inputs, w0, w1 }
    }

    pub fn eval(&self, xs: &[u8]) -> bool {
        assert_eq!(xs.len(), self.n_inputs);
        score(xs, &self.w0, &self.w1) >= 0
    }
}

fn sample_weight<R: Rng>(rng: &mut R, mode: WeightMode) -> i8 {
    match mode {
        WeightMode::I8 => rng.gen(),
        WeightMode::Binary => i8::from(rng.gen_bool(0.5)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn eval_matches_score_sign() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let dgp = LinearU8Dgp::random(&mut rng, 1);
        for x in 0..=255u8 {
            assert_eq!(dgp.eval(&[x]), score(&[x], &dgp.w0, &dgp.w1) >= 0);
        }
    }

    #[test]
    fn two_input_score() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let dgp = LinearU8Dgp::random(&mut rng, 2);
        let xs = [10u8, 20];
        assert_eq!(dgp.eval(&xs), score(&xs, &dgp.w0, &dgp.w1) >= 0);
    }

    #[test]
    fn trivial_lsb() {
        let mut w0 = vec![0i8; 8];
        let mut w1 = vec![0i8; 8];
        w0[0] = -1;
        w1[0] = 127;
        let dgp = LinearU8Dgp {
            n_inputs: 1,
            w0,
            w1,
        };
        for x in 0..=255u8 {
            assert_eq!(dgp.eval(&[x]), x & 1 != 0);
        }
    }

    #[test]
    fn binary_weights_are_zero_or_one() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let dgp = LinearU8Dgp::random_with_mode(&mut rng, 2, WeightMode::Binary);
        for &w in dgp.w0.iter().chain(dgp.w1.iter()) {
            assert!(w == 0 || w == 1);
        }
    }
}
