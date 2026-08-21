//! `StreamLearnerWire`: per-wire weights, summed gate output, pole targets.

use std::collections::VecDeque;

use crate::circuit::FixedCircuit;
use crate::gate::sign;
use crate::gate_wire::{
    activation_matches_target, backprop_outcome, bool_target_pole, clamp_i8, forward_sum, lane_a,
    lane_b, nudge_weight, parent_weights, unpack_sa, unpack_sb, BackpropOutcome,
};
use crate::gate::{get_weight, pole, set_weight};

pub struct StreamLearnerWire {
    d: usize,
    sink: u8,
    left: Box<[u8]>,
    right: Box<[u8]>,
    topo_order: Box<[usize]>,
    gate_of_node: Box<[u8]>,
    weights: Box<[u32]>,
    activations: Box<[i8]>,
    targets: Box<[i8]>,
    active_signs: Box<[u8]>,
    t: u64,
}

impl StreamLearnerWire {
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
            active_signs: vec![0u8; g].into_boxed_slice(),
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
            let (sum, signs) = forward_sum(self.weights[g], self.activations[a], self.activations[b]);
            self.active_signs[g] = signs;
            self.activations[self.d + g] = clamp_i8(sum);
        }
        self.activations[self.sink as usize] >= 0
    }

    pub fn observe(&mut self, y: bool) {
        self.observe_with_diag(y, None);
    }

    pub fn observe_with_diag(
        &mut self,
        y: bool,
        mut diag: Option<&mut crate::diag_wire::WireLearnerDiag>,
    ) {
        if let Some(d) = diag.as_deref_mut() {
            d.begin_observe();
        }
        self.t += 1;
        let sink = self.sink as usize;
        self.targets[sink] = bool_target_pole(y);

        let mut pending = VecDeque::new();
        let sg = self.gate_of_node[sink];
        if sg != 255 {
            pending.push_back(sg as usize);
        }

        while let Some(g) = pending.pop_front() {
            self.fix_gate(g, &mut pending, diag.as_deref_mut());
        }
    }

    pub fn activations_snapshot(&self) -> &[i8] {
        &self.activations
    }

    fn fix_gate(
        &mut self,
        g: usize,
        pending: &mut VecDeque<usize>,
        mut diag: Option<&mut crate::diag_wire::WireLearnerDiag>,
    ) {
        if let Some(d) = diag.as_deref_mut() {
            d.record_fix_gate(g);
        }
        let node = self.d + g;
        let target = self.targets[node];
        let sa = unpack_sa(self.active_signs[g]);
        let sb = unpack_sb(self.active_signs[g]);

        let lane_a_idx = lane_a(sa);
        let lane_b_idx = lane_b(sb);
        let w_a = get_weight(self.weights[g], lane_a_idx);
        let w_b = get_weight(self.weights[g], lane_b_idx);
        set_weight(
            &mut self.weights[g],
            lane_a_idx,
            nudge_weight(w_a, target),
        );
        set_weight(
            &mut self.weights[g],
            lane_b_idx,
            nudge_weight(w_b, target),
        );

        if activation_matches_target(self.activations[node], target) {
            return;
        }

        let a = self.left[g] as usize;
        let b = self.right[g] as usize;
        self.backprop_parent(g, a, false, target, pending, diag.as_deref_mut());
        self.backprop_parent(g, b, true, target, pending, diag);
    }

    fn backprop_parent(
        &mut self,
        g: usize,
        parent: usize,
        is_parent_b: bool,
        target: i8,
        pending: &mut VecDeque<usize>,
        mut diag: Option<&mut crate::diag_wire::WireLearnerDiag>,
    ) {
        if parent < self.d {
            return;
        }
        let (w0, w1) = parent_weights(self.weights[g], is_parent_b);
        let act_p = self.activations[parent];
        let outcome = backprop_outcome(target, w0, w1, act_p);
        if let Some(d) = diag.as_deref_mut() {
            d.record_backprop(g, parent, outcome);
        }
        let Some(ws) = (match outcome {
            BackpropOutcome::Fired { want } => Some(want),
            _ => None,
        }) else {
            return;
        };
        if sign(act_p) == ws {
            return;
        }
        self.targets[parent] = pole(ws);
        let pg = self.gate_of_node[parent] as usize;
        if pg != 255 {
            pending.push_back(pg);
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
        let mut learner = StreamLearnerWire::new(topo);
        let x = [true, false, true, false];
        let y = dgp.eval_sink(&x);
        learner.predict(&x);
        learner.observe(y);
        assert_eq!(learner.step_count(), 1);
    }

    #[test]
    fn single_gate_and_learns() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (mut dgp, _) = build_random_dag(&mut rng, 2, 1, 1);
        // AND: out = a AND b
        dgp.truth_table[0] = 0b0001;
        dgp.left[0] = 0;
        dgp.right[0] = 1;
        dgp.sink = 2;
        let topo = matched_learner(&dgp);
        let mut learner = StreamLearnerWire::new(topo);

        let mut rng = ChaCha8Rng::seed_from_u64(99);
        for _ in 0..8192 {
            let x: [bool; 2] = [
                rand::Rng::gen_bool(&mut rng, 0.5),
                rand::Rng::gen_bool(&mut rng, 0.5),
            ];
            let y = dgp.eval_sink(&x);
            learner.predict(&x);
            learner.observe(y);
        }
        for a in [false, true] {
            for b in [false, true] {
                let x = [a, b];
                let y = dgp.eval_sink(&x);
                assert_eq!(learner.predict(&x), y);
            }
        }
    }
}
