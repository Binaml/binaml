//! `StreamLearner`: predict / observe with FixGate event queue.

use std::collections::VecDeque;

use crate::circuit::FixedCircuit;
use crate::gate::{
    argmin_row, bool_target, get_weight, lane, min_total, nudge_weight, pole, row_weights,
    set_weight,
};

enum Event {
    FixGate(usize),
}

#[derive(Clone, Copy)]
enum Move {
    Weight,
    ParentA { pole_val: i8 },
    ParentB { pole_val: i8 },
}

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
    event_queue: VecDeque<Event>,
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
            event_queue: VecDeque::new(),
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

        self.event_queue.clear();
        if self.activations[sink] != self.targets[sink] {
            let sg = self.gate_of_node[sink];
            if sg != 255 {
                self.event_queue.push_back(Event::FixGate(sg as usize));
            }
        }

        while let Some(Event::FixGate(g)) = self.event_queue.pop_front() {
            self.fix_gate(g);
        }
    }

    fn fix_gate(&mut self, g: usize) {
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
        let w_lane = weights[lane_idx as usize];

        let mut candidates = Vec::new();
        if w_lane != target {
            candidates.push(Move::Weight);
        }
        for sign_bit in [0u8, 1] {
            let p = pole(sign_bit);
            if a >= self.d && self.activations[a] != p {
                candidates.push(Move::ParentA { pole_val: p });
            }
            if b >= self.d && self.activations[b] != p {
                candidates.push(Move::ParentB { pole_val: p });
            }
        }

        if candidates.is_empty() {
            return;
        }

        let mut best: Option<(Move, u8, u8)> = None;
        for &mv in &candidates {
            let (cost, s_star) =
                self.simulate_move(mv, target, lane_idx, act_a, act_b, weights, w_lane);
            let replace = match best {
                None => true,
                Some((best_mv, best_cost, _)) => {
                    cost < best_cost || (cost == best_cost && move_tie_break(mv, best_mv))
                }
            };
            if replace {
                best = Some((mv, cost, s_star));
            }
        }

        let Some((mv, _, s_star)) = best else {
            return;
        };

        match mv {
            Move::Weight => {
                set_weight(
                    &mut self.weights[g],
                    lane_idx,
                    nudge_weight(w_lane, target),
                );
            }
            Move::ParentA { .. } => {
                let p = pole(s_star >> 1);
                if self.activations[a] != p {
                    self.targets[a] = p;
                    self.maybe_enqueue_parent(a);
                }
            }
            Move::ParentB { .. } => {
                let p = pole(s_star & 1);
                if self.activations[b] != p {
                    self.targets[b] = p;
                    self.maybe_enqueue_parent(b);
                }
            }
        }
    }

    fn simulate_move(
        &self,
        mv: Move,
        target: i8,
        lane_idx: u8,
        act_a: i8,
        act_b: i8,
        mut weights: [i8; 4],
        w_lane: i8,
    ) -> (u8, u8) {
        let (sim_a, sim_b) = match mv {
            Move::Weight => {
                weights[lane_idx as usize] = nudge_weight(w_lane, target);
                (act_a, act_b)
            }
            Move::ParentA { pole_val } => (pole_val, act_b),
            Move::ParentB { pole_val } => (act_a, pole_val),
        };

        let cost = min_total(sim_a, sim_b, weights, target);
        let s_star = argmin_row(sim_a, sim_b, weights, target);
        (cost, s_star)
    }

    fn maybe_enqueue_parent(&mut self, parent: usize) {
        if parent < self.d {
            return;
        }
        if self.activations[parent] != self.targets[parent] {
            let g = self.gate_of_node[parent] as usize;
            if g != 255 {
                self.event_queue.push_back(Event::FixGate(g));
            }
        }
    }
}

/// Tie-break: weight > parent a > parent b; parent +127 > -128.
fn move_tie_break(a: Move, b: Move) -> bool {
    fn rank(m: Move) -> u8 {
        match m {
            Move::Weight => 0,
            Move::ParentA { pole_val } => {
                if pole_val > 0 { 1 } else { 2 }
            }
            Move::ParentB { pole_val } => {
                if pole_val > 0 { 3 } else { 4 }
            }
        }
    }
    rank(a) < rank(b)
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
