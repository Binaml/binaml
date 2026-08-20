//! Fixed DAG topology for DGP (truth-table gates) and the learner (i8 row weights).

use rand::Rng;

/// How the learner wiring relates to the DGP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyMode {
    /// Random DAG independent of the DGP.
    Independent,
    /// Same `(left, right)` parents as the DGP.
    Matched,
}

/// Immutable circuit graph: sources `0..d`, gates `d..d+G`, fixed sink.
#[derive(Debug, Clone)]
pub struct FixedCircuit {
    pub d: usize,
    pub num_gates: usize,
    pub left: Box<[u8]>,
    pub right: Box<[u8]>,
    /// DGP truth table per gate (4 bits, index `2*a+b`).
    pub truth_table: Box<[u8]>,
    pub sink: u8,
    /// Gate indices in topological order.
    pub topo_order: Box<[usize]>,
}

impl FixedCircuit {
    pub fn num_nodes(&self) -> usize {
        self.d + self.num_gates
    }

    /// Evaluate the DGP boolean value at every node for input `x`.
    pub fn eval_dgp(&self, x: &[bool]) -> Vec<bool> {
        assert_eq!(x.len(), self.d);
        let n = self.num_nodes();
        let mut values = vec![false; n];
        for (i, &xi) in x.iter().enumerate() {
            values[i] = xi;
        }
        for &g in self.topo_order.iter() {
            let a = values[self.left[g] as usize];
            let b = values[self.right[g] as usize];
            let idx = 2 * a as u8 + b as u8;
            values[self.d + g] = (self.truth_table[g] >> idx) & 1 == 1;
        }
        values
    }

    pub fn eval_sink(&self, x: &[bool]) -> bool {
        self.eval_dgp(x)[self.sink as usize]
    }
}

/// Build a random layered DAG with `depth` layers and `width` gates per layer.
pub fn build_random_dag<R: Rng>(
    rng: &mut R,
    d: usize,
    depth: usize,
    width: usize,
) -> (FixedCircuit, FixedCircuit) {
    let num_gates = depth * width;
    let n = d + num_gates;
    assert!(n <= 256, "N = d + G must be <= 256");

    let mut left = Vec::with_capacity(num_gates);
    let mut right = Vec::with_capacity(num_gates);
    let mut truth_table = Vec::with_capacity(num_gates);
    let mut topo_order = Vec::with_capacity(num_gates);

    for layer in 0..depth {
        for w in 0..width {
            let g = layer * width + w;
            let max_parent = d + g; // nodes 0..max_parent are available
            let a = rng.gen_range(0..max_parent) as u8;
            let mut b = rng.gen_range(0..max_parent) as u8;
            if a == b && max_parent > 1 {
                b = ((b as usize + 1) % max_parent) as u8;
            }
            left.push(a);
            right.push(b);
            truth_table.push(rng.gen_range(0..16u8));
            topo_order.push(g);
        }
    }

    let sink = (d + num_gates - 1) as u8;

    let dgp = FixedCircuit {
        d,
        num_gates,
        left: left.clone().into_boxed_slice(),
        right: right.clone().into_boxed_slice(),
        truth_table: truth_table.into_boxed_slice(),
        sink,
        topo_order: topo_order.clone().into_boxed_slice(),
    };

    // Learner with independent wiring (same shape, different parents).
    let mut learner_left = left.clone();
    let mut learner_right = right.clone();
    for g in 0..num_gates {
        let max_parent = d + g;
        learner_left[g] = rng.gen_range(0..max_parent) as u8;
        let mut b = rng.gen_range(0..max_parent) as u8;
        if learner_left[g] == b && max_parent > 1 {
            b = ((b as usize + 1) % max_parent) as u8;
        }
        learner_right[g] = b;
    }

    let learner = FixedCircuit {
        d,
        num_gates,
        left: learner_left.into_boxed_slice(),
        right: learner_right.into_boxed_slice(),
        truth_table: vec![0; num_gates].into_boxed_slice(), // unused for learner
        sink,
        topo_order: topo_order.into_boxed_slice(),
    };

    (dgp, learner)
}

/// Copy DGP parent wiring into a learner topology (matched mode).
pub fn matched_learner(dgp: &FixedCircuit) -> FixedCircuit {
    FixedCircuit {
        d: dgp.d,
        num_gates: dgp.num_gates,
        left: dgp.left.clone(),
        right: dgp.right.clone(),
        truth_table: vec![0; dgp.num_gates].into_boxed_slice(),
        sink: dgp.sink,
        topo_order: dgp.topo_order.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    #[test]
    fn dgp_eval_matches_sink() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (dgp, _) = build_random_dag(&mut rng, 4, 2, 2);
        let x = [true, false, true, false];
        let vals = dgp.eval_dgp(&x);
        assert_eq!(dgp.eval_sink(&x), vals[dgp.sink as usize]);
    }

    #[test]
    fn matched_learner_same_parents() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (dgp, _) = build_random_dag(&mut rng, 8, 2, 4);
        let matched = matched_learner(&dgp);
        assert_eq!(matched.left, dgp.left);
        assert_eq!(matched.right, dgp.right);
    }
}
