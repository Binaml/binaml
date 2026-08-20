//! Random DGP construction and i.i.d. Bernoulli stream generation.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::circuit::{build_random_dag, matched_learner, FixedCircuit, TopologyMode};

/// One streaming step: input bits and DGP sink label.
#[derive(Debug, Clone)]
pub struct StreamSample {
    pub x: Vec<bool>,
    pub y: bool,
}

pub fn build_dgp(rng: &mut ChaCha8Rng, d: usize, depth: usize, width: usize) -> FixedCircuit {
    let (dgp, _) = build_random_dag(rng, d, depth, width);
    dgp
}

pub fn build_learner_topology(
    rng: &mut ChaCha8Rng,
    dgp: &FixedCircuit,
    mode: TopologyMode,
) -> FixedCircuit {
    match mode {
        TopologyMode::Matched => matched_learner(dgp),
        TopologyMode::Independent => {
            let depth = infer_depth(dgp);
            let width = infer_width(dgp);
            let (_, learner) = build_random_dag(rng, dgp.d, depth, width);
            learner
        }
    }
}

fn infer_depth(dgp: &FixedCircuit) -> usize {
    // topo_order is layer-major; recover depth from gate count / width heuristic.
    let width = infer_width(dgp);
    dgp.num_gates / width.max(1)
}

fn infer_width(dgp: &FixedCircuit) -> usize {
    // Default: assume uniform layers; sqrt-ish fallback.
    if dgp.num_gates == 0 {
        return 1;
    }
    // Try divisors near sqrt(G).
    let g = dgp.num_gates;
    for w in (1..=g).rev() {
        if g % w == 0 {
            return w;
        }
    }
    g
}

pub fn stream_sample<R: Rng>(rng: &mut R, d: usize) -> StreamSample {
    let x: Vec<bool> = (0..d).map(|_| rng.gen_bool(0.5)).collect();
    StreamSample { x, y: false }
}

pub fn labeled_sample(dgp: &FixedCircuit, x: &[bool]) -> StreamSample {
    StreamSample {
        x: x.to_vec(),
        y: dgp.eval_sink(x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn stream_labels_consistent() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let dgp = build_dgp(&mut rng, 4, 2, 2);
        let x = vec![true, false, true, false];
        let sample = labeled_sample(&dgp, &x);
        assert_eq!(sample.y, dgp.eval_sink(&x));
    }
}
