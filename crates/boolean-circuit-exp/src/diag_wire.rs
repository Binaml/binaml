//! Diagnostics for the per-wire learner.

use crate::gate_wire::BackpropOutcome;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::circuit::{FixedCircuit, TopologyMode};
use crate::dgp::{build_dgp, build_learner_topology, labeled_sample};
use crate::learner_wire::StreamLearnerWire;

#[derive(Debug, Clone, Default)]
pub struct GateDiag {
    pub fix_gate_visits: u64,
    pub backprop_checks: u64,
    pub backprop_fired: u64,
    pub skip_cur_agrees: u64,
    pub skip_other_not_better: u64,
    pub target_conflicts: u64,
    /// Fraction of eval samples where gate boolean output mismatches DGP.
    pub node_error_rate: f64,
}

#[derive(Debug, Clone)]
pub struct WireDiagReport {
    pub seed: u64,
    pub stream_len: u64,
    pub warmup_len: u64,
    pub accuracy_after_warmup: f64,
    pub accuracy_last_n: f64,
    pub gate_diags: Vec<GateDiag>,
    pub total_backprop_fired: u64,
    pub total_backprop_checks: u64,
    pub total_target_conflicts: u64,
}

pub struct WireLearnerDiag {
    pub gates: Vec<GateDiag>,
    /// Per-node pole target written this observe wave (for fan-out conflict detection).
    wave_targets: Vec<i8>,
}

impl WireLearnerDiag {
    pub fn new(num_gates: usize, num_nodes: usize) -> Self {
        Self {
            gates: vec![GateDiag::default(); num_gates],
            wave_targets: vec![0; num_nodes],
        }
    }

    pub fn begin_observe(&mut self) {
        self.wave_targets.fill(0);
    }

    pub fn record_fix_gate(&mut self, g: usize) {
        self.gates[g].fix_gate_visits += 1;
    }

    pub fn record_backprop(&mut self, g: usize, parent: usize, outcome: BackpropOutcome) {
        self.gates[g].backprop_checks += 1;
        match outcome {
            BackpropOutcome::Fired { want } => {
                self.gates[g].backprop_fired += 1;
                let pole = crate::gate::pole(want);
                if self.wave_targets[parent] != 0 && self.wave_targets[parent] != pole {
                    self.gates[g].target_conflicts += 1;
                }
                self.wave_targets[parent] = pole;
            }
            BackpropOutcome::AlreadyAligned => self.gates[g].skip_cur_agrees += 1,
            BackpropOutcome::WeightTie => self.gates[g].skip_other_not_better += 1,
        }
    }
}

pub fn run_wire_diagnostic(
    seed: u64,
    d: usize,
    depth: usize,
    width: usize,
    stream_len: u64,
    warmup_len: u64,
    topology: TopologyMode,
    eval_samples: u64,
) -> WireDiagReport {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dgp = build_dgp(&mut rng, d, depth, width);
    let learner_topo = build_learner_topology(&mut rng, &dgp, topology);
    let num_gates = dgp.num_gates;
    let num_nodes = dgp.num_nodes();

    let mut learner = StreamLearnerWire::new(learner_topo);
    let mut diag = WireLearnerDiag::new(num_gates, num_nodes);

    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(1));
    let mut post_correct = 0u64;
    let mut post_steps = 0u64;
    let mut last_n: std::collections::VecDeque<bool> = std::collections::VecDeque::with_capacity(100);

    for _ in 0..stream_len {
        let x: Vec<bool> = (0..dgp.d).map(|_| rand::Rng::gen_bool(&mut rng, 0.5)).collect();
        let sample = labeled_sample(&dgp, &x);
        let step = learner.step_count() + 1;

        let pred = learner.predict(&x);
        learner.observe_with_diag(sample.y, Some(&mut diag));

        let correct = pred == sample.y;
        if step > warmup_len {
            post_steps += 1;
            if correct {
                post_correct += 1;
            }
        }
        if last_n.len() == 100 {
            last_n.pop_front();
        }
        last_n.push_back(correct);
    }

    let node_errors = eval_per_gate_error(&mut learner, &dgp, eval_samples, seed.wrapping_add(99));
    for (g, err) in node_errors.into_iter().enumerate() {
        diag.gates[g].node_error_rate = err;
    }

    let accuracy_after_warmup = if post_steps > 0 {
        post_correct as f64 / post_steps as f64
    } else {
        0.0
    };
    let accuracy_last_n = if last_n.is_empty() {
        0.0
    } else {
        last_n.iter().filter(|&&c| c).count() as f64 / last_n.len() as f64
    };

    let total_backprop_fired = diag.gates.iter().map(|g| g.backprop_fired).sum();
    let total_backprop_checks = diag.gates.iter().map(|g| g.backprop_checks).sum();
    let total_target_conflicts = diag.gates.iter().map(|g| g.target_conflicts).sum();

    WireDiagReport {
        seed,
        stream_len,
        warmup_len,
        accuracy_after_warmup,
        accuracy_last_n,
        gate_diags: diag.gates,
        total_backprop_fired,
        total_backprop_checks,
        total_target_conflicts,
    }
}

fn eval_per_gate_error(
    learner: &mut StreamLearnerWire,
    dgp: &FixedCircuit,
    samples: u64,
    stream_seed: u64,
) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(stream_seed);
    let g = dgp.num_gates;
    let mut wrong = vec![0u64; g];
    let mut seen = vec![0u64; g];

    for _ in 0..samples {
        let x: Vec<bool> = (0..dgp.d).map(|_| rand::Rng::gen_bool(&mut rng, 0.5)).collect();
        let dgp_vals = dgp.eval_dgp(&x);
        learner.predict(&x);
        let acts = learner.activations_snapshot();
        for gi in 0..g {
            let node = dgp.d + gi;
            let pred = acts[node] >= 0;
            let truth = dgp_vals[node];
            seen[gi] += 1;
            if pred != truth {
                wrong[gi] += 1;
            }
        }
    }

    wrong
        .into_iter()
        .zip(seen)
        .map(|(w, s)| if s > 0 { w as f64 / s as f64 } else { 0.0 })
        .collect()
}

pub fn print_wire_diag_summary(report: &WireDiagReport) {
    println!("=== Wire learner diagnostic (seed {}) ===", report.seed);
    println!(
        "T={} warmup={} acc_after_warmup={:.4} acc_last100={:.4}",
        report.stream_len, report.warmup_len, report.accuracy_after_warmup, report.accuracy_last_n
    );
    println!(
        "backprop: {}/{} fired ({:.1}%)  target_conflicts={}",
        report.total_backprop_fired,
        report.total_backprop_checks,
        if report.total_backprop_checks > 0 {
            100.0 * report.total_backprop_fired as f64 / report.total_backprop_checks as f64
        } else {
            0.0
        },
        report.total_target_conflicts,
    );

    let mut gates: Vec<(usize, &GateDiag)> = report.gate_diags.iter().enumerate().collect();
    gates.sort_by(|a, b| b.1.node_error_rate.partial_cmp(&a.1.node_error_rate).unwrap());

    println!("\nTop gates by node error rate:");
    println!(
        "{:>5} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "gate", "err_rate", "fix_vis", "bp_fire", "bp_check", "skip_aln", "skip_tie", "tgt_conf"
    );
    for (g, d) in gates.iter().take(12) {
        println!(
            "{:>5} {:>8.3} {:>8} {:>8} {:>10} {:>10} {:>10} {:>8}",
            g,
            d.node_error_rate,
            d.fix_gate_visits,
            d.backprop_fired,
            d.backprop_checks,
            d.skip_cur_agrees,
            d.skip_other_not_better,
            d.target_conflicts,
        );
    }

    let high_err: Vec<_> = gates.iter().filter(|(_, d)| d.node_error_rate > 0.1).collect();
    let low_bp: Vec<_> = gates
        .iter()
        .filter(|(_, d)| {
            d.backprop_checks > 0 && (d.backprop_fired as f64 / d.backprop_checks as f64) < 0.05
        })
        .collect();

    println!(
        "\nGates with err_rate > 10%: {} / {}",
        high_err.len(),
        report.gate_diags.len()
    );
    println!(
        "Gates with backprop fire rate < 5%: {} / {}",
        low_bp.len(),
        report.gate_diags.len()
    );

    // Correlation: mean backprop fire rate on high vs low error gates
    if !high_err.is_empty() && gates.len() > high_err.len() {
        let high_bp: f64 = high_err
            .iter()
            .map(|(_, d)| {
                if d.backprop_checks > 0 {
                    d.backprop_fired as f64 / d.backprop_checks as f64
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / high_err.len() as f64;
        let low_err: Vec<_> = gates.iter().filter(|(_, d)| d.node_error_rate <= 0.1).collect();
        let low_bp_rate: f64 = if low_err.is_empty() {
            0.0
        } else {
            low_err
                .iter()
                .map(|(_, d)| {
                    if d.backprop_checks > 0 {
                        d.backprop_fired as f64 / d.backprop_checks as f64
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / low_err.len() as f64
        };
        println!(
            "Mean backprop fire rate: high-err gates={:.3}  low-err gates={:.3}",
            high_bp, low_bp_rate
        );
    }

    let skip_ob_err: f64 = gates
        .iter()
        .map(|(_, d)| d.skip_other_not_better as f64)
        .sum::<f64>()
        / report.total_backprop_checks.max(1) as f64;
    let skip_agr: f64 = gates
        .iter()
        .map(|(_, d)| d.skip_cur_agrees as f64)
        .sum::<f64>()
        / report.total_backprop_checks.max(1) as f64;
    println!(
        "\nSkip reasons (fraction of backprop checks): already_aligned={:.3}  weight_tie={:.3}",
        skip_agr, skip_ob_err
    );
}
