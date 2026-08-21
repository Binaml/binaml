//! `StreamLearner`: predict / observe with FixGate backward propagation.

use std::collections::VecDeque;

use crate::circuit::FixedCircuit;
use crate::gate::{
    best_row_at_most_one_flip, bool_target, get_weight, lane, nudge_weight, pole, row_weights,
    set_weight, sign, sign_mismatches,
};

pub struct StreamLearner {
    d: usize,
    sink: u8,
    left: Box<[u8]>,
    right: Box<[u8]>,
    topo_order: Box<[usize]>,
    gate_of_node: Box<[u8]>,
    weights: Box<[u32]>,
    activations: Box<[i8]>,
    targets: Box<[i8]>,
    active_row: Box<[u8]>,
    t: u64,
}

impl StreamLearner {
    pub fn new(topology: FixedCircuit) -> Self {
        let d = topology.d;
        let g = topology.num_gates;
        let n = d + g;
        let mut gate_of_node = vec![255u8; n];
        for &gidx in topology.topo_order.iter() {
            gate_of_node[d + gidx] = gidx as u8;
        }
        Self {
            d,
            sink: topology.sink,
            left: topology.left,
            right: topology.right,
            topo_order: topology.topo_order,
            gate_of_node: gate_of_node.into_boxed_slice(),
            weights: vec![0u32; g].into_boxed_slice(),
            activations: vec![0i8; n].into_boxed_slice(),
            targets: vec![0i8; n].into_boxed_slice(),
            active_row: vec![0u8; g].into_boxed_slice(),
            t: 0,
        }
    }

    pub fn step_count(&self) -> u64 {
        self.t
    }

    pub fn predict(&mut self, x: &[bool]) -> bool {
        assert_eq!(x.len(), self.d);
        for (i, &xi) in x.iter().enumerate() {
            self.activations[i] = if xi { 1 } else { -1 };
        }
        for &g in self.topo_order.iter() {
            let a = self.left[g] as usize;
            let b = self.right[g] as usize;
            let ln = lane(self.activations[a], self.activations[b]);
            self.active_row[g] = ln;
            self.activations[self.d + g] = get_weight(self.weights[g], ln);
        }
        self.activations[self.sink as usize] >= 0
    }

    pub fn observe(&mut self, y: bool) {
        self.t += 1;
        let sink = self.sink as usize;
        self.targets[sink] = bool_target(y);

        let mut pending = VecDeque::new();
        if self.activations[sink] != self.targets[sink] {
            let sg = self.gate_of_node[sink];
            if sg != 255 {
                pending.push_back(sg as usize);
            }
        }

        while let Some(g) = pending.pop_front() {
            self.fix_gate(g, &mut pending);
        }
    }

    /// Nudge `w[lane]` only when `lane` is the best ≤1-flip row; backprop via worklist.
    fn fix_gate(&mut self, g: usize, pending: &mut VecDeque<usize>) {
        let node = self.d + g;
        let target = self.targets[node];
        if self.activations[node] == target {
            return;
        }

        let lane_idx = self.active_row[g];
        let a = self.left[g] as usize;
        let b = self.right[g] as usize;
        let act_a = self.activations[a];
        let act_b = self.activations[b];
        let weights = row_weights(self.weights[g]);

        let s_star = best_row_at_most_one_flip(act_a, act_b, weights, target);

        if lane_idx == s_star {
            let w_lane = weights[lane_idx as usize];
            if w_lane != target {
                set_weight(
                    &mut self.weights[g],
                    lane_idx,
                    nudge_weight(w_lane, target),
                );
            }
        }

        if sign_mismatches(s_star, act_a, act_b) != 1 {
            return;
        }

        let s_a = s_star >> 1;
        let s_b = s_star & 1;

        if sign(act_a) != s_a && a >= self.d {
            self.targets[a] = pole(s_a);
            self.propagate_producer(a, pending);
        } else if sign(act_b) != s_b && b >= self.d {
            self.targets[b] = pole(s_b);
            self.propagate_producer(b, pending);
        }
    }

    /// Enqueue upstream gate; duplicates allowed (multiple children may write same parent).
    fn propagate_producer(&mut self, parent: usize, pending: &mut VecDeque<usize>) {
        if parent < self.d {
            return;
        }
        if self.activations[parent] != self.targets[parent] {
            let g = self.gate_of_node[parent] as usize;
            if g != 255 {
                pending.push_back(g);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{build_random_dag, matched_learner};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn predict_observe_runs() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (dgp, _) = build_random_dag(&mut rng, 4, 2, 2);
        let topo = matched_learner(&dgp);
        let mut learner = StreamLearner::new(topo);
        let x = [true, false, true, false];
        let y = dgp.eval_sink(&x);
        learner.predict(&x);
        learner.observe(y);
        assert_eq!(learner.step_count(), 1);
    }
}
