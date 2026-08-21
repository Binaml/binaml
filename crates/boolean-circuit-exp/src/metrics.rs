//! Streaming accuracy (after warmup, last-N), steps-to-threshold, and timing.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::circuit::TopologyMode;
use crate::dgp::{build_dgp, build_learner_topology, labeled_sample};
use crate::learner::StreamLearner;
use crate::learner_wire::StreamLearnerWire;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearnerKind {
    Pair,
    Wire,
}

impl LearnerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            LearnerKind::Pair => "pair",
            LearnerKind::Wire => "wire",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct StepTimings {
    pub predict_ns: u64,
    pub observe_ns: u64,
}

const LAST_N: usize = 100;

#[derive(Debug, Clone)]
pub struct StreamMetrics {
    /// Fraction correct on steps `warmup_len + 1 .. stream_len`.
    pub accuracy_after_warmup: f64,
    /// Fraction correct on the last `LAST_N` steps (or all steps if `T < LAST_N`).
    pub accuracy_last_n: f64,
    pub steps_to_95pct: Option<u64>,
    pub total_steps: u64,
    pub warmup_len: u64,
    pub timings: StepTimings,
}

#[derive(Debug, Clone)]
pub struct RunMetrics {
    pub learner: LearnerKind,
    pub topology: TopologyMode,
    pub d: usize,
    pub depth: usize,
    pub width: usize,
    pub num_gates: usize,
    pub stream_len: u64,
    pub warmup_len: u64,
    pub seed: u64,
    pub accuracy_after_warmup: f64,
    pub accuracy_last_n: f64,
    pub steps_to_95pct: Option<u64>,
    pub mean_predict_ns: f64,
    pub mean_observe_ns: f64,
}

pub struct MetricsTracker {
    t: u64,
    warmup_len: u64,
    post_warmup_correct: u64,
    post_warmup_steps: u64,
    /// Prequential cumulative (from step 1) for steps-to-95% only.
    prequential_correct: u64,
    steps_to_95: Option<u64>,
    last_n: VecDeque<bool>,
    predict_ns: u128,
    observe_ns: u128,
}

impl MetricsTracker {
    pub fn new(warmup_len: u64) -> Self {
        Self {
            t: 0,
            warmup_len,
            post_warmup_correct: 0,
            post_warmup_steps: 0,
            prequential_correct: 0,
            steps_to_95: None,
            last_n: VecDeque::with_capacity(LAST_N),
            predict_ns: 0,
            observe_ns: 0,
        }
    }

    pub fn record(&mut self, pred: bool, y: bool, predict: Duration, observe: Duration) {
        self.t += 1;
        let correct = pred == y;
        if correct {
            self.prequential_correct += 1;
        }
        if self.t > self.warmup_len {
            self.post_warmup_steps += 1;
            if correct {
                self.post_warmup_correct += 1;
            }
        }
        if self.last_n.len() == LAST_N {
            self.last_n.pop_front();
        }
        self.last_n.push_back(correct);
        let prequential_acc = self.prequential_correct as f64 / self.t as f64;
        if self.steps_to_95.is_none() && prequential_acc >= 0.95 {
            self.steps_to_95 = Some(self.t);
        }
        self.predict_ns += predict.as_nanos();
        self.observe_ns += observe.as_nanos();
    }

    pub fn finish(self) -> StreamMetrics {
        let t = self.t.max(1);
        let accuracy_after_warmup = if self.post_warmup_steps > 0 {
            self.post_warmup_correct as f64 / self.post_warmup_steps as f64
        } else {
            0.0
        };
        let last_n_correct = self.last_n.iter().filter(|&&c| c).count();
        let accuracy_last_n = if self.last_n.is_empty() {
            0.0
        } else {
            last_n_correct as f64 / self.last_n.len() as f64
        };
        StreamMetrics {
            accuracy_after_warmup,
            accuracy_last_n,
            steps_to_95pct: self.steps_to_95,
            total_steps: self.t,
            warmup_len: self.warmup_len,
            timings: StepTimings {
                predict_ns: (self.predict_ns / t as u128) as u64,
                observe_ns: (self.observe_ns / t as u128) as u64,
            },
        }
    }
}

pub fn run_stream(
    dgp: &crate::circuit::FixedCircuit,
    learner_topo: crate::circuit::FixedCircuit,
    stream_len: u64,
    warmup_len: u64,
    stream_seed: u64,
    learner: LearnerKind,
) -> StreamMetrics {
    assert!(warmup_len < stream_len, "warmup_len must be < stream_len");
    let mut tracker = MetricsTracker::new(warmup_len);
    let mut rng = ChaCha8Rng::seed_from_u64(stream_seed);

    macro_rules! stream_loop {
        ($learner:expr) => {
            for _ in 0..stream_len {
                let x: Vec<bool> = (0..dgp.d).map(|_| rand::Rng::gen_bool(&mut rng, 0.5)).collect();
                let sample = labeled_sample(dgp, &x);

                let t0 = Instant::now();
                let pred = $learner.predict(&sample.x);
                let predict_dt = t0.elapsed();

                let t1 = Instant::now();
                $learner.observe(sample.y);
                let observe_dt = t1.elapsed();

                tracker.record(pred, sample.y, predict_dt, observe_dt);
            }
        };
    }

    match learner {
        LearnerKind::Pair => {
            let mut learner = StreamLearner::new(learner_topo);
            stream_loop!(learner);
        }
        LearnerKind::Wire => {
            let mut learner = StreamLearnerWire::new(learner_topo);
            stream_loop!(learner);
        }
    }

    tracker.finish()
}

pub fn run_single(
    seed: u64,
    d: usize,
    depth: usize,
    width: usize,
    stream_len: u64,
    warmup_len: u64,
    topology: TopologyMode,
    learner: LearnerKind,
) -> RunMetrics {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dgp = build_dgp(&mut rng, d, depth, width);
    let learner_topo = build_learner_topology(&mut rng, &dgp, topology);
    let num_gates = dgp.num_gates;
    let m = run_stream(
        &dgp,
        learner_topo,
        stream_len,
        warmup_len,
        seed.wrapping_add(1),
        learner,
    );
    RunMetrics {
        learner,
        topology,
        d,
        depth,
        width,
        num_gates,
        stream_len,
        warmup_len,
        seed,
        accuracy_after_warmup: m.accuracy_after_warmup,
        accuracy_last_n: m.accuracy_last_n,
        steps_to_95pct: m.steps_to_95pct,
        mean_predict_ns: m.timings.predict_ns as f64,
        mean_observe_ns: m.timings.observe_ns as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy_after_warmup_excludes_prefix() {
        let mut tr = MetricsTracker::new(2);
        tr.record(false, true, Duration::ZERO, Duration::ZERO);
        tr.record(false, true, Duration::ZERO, Duration::ZERO);
        tr.record(true, true, Duration::ZERO, Duration::ZERO);
        tr.record(false, true, Duration::ZERO, Duration::ZERO);
        let m = tr.finish();
        assert!((m.accuracy_after_warmup - 0.5).abs() < 1e-9);
    }

    #[test]
    fn accuracy_last_n_tracks_tail() {
        let mut tr = MetricsTracker::new(0);
        for _ in 0..99 {
            tr.record(false, true, Duration::ZERO, Duration::ZERO);
        }
        tr.record(true, true, Duration::ZERO, Duration::ZERO);
        let m = tr.finish();
        assert!((m.accuracy_last_n - 0.01).abs() < 1e-9);
    }
}
