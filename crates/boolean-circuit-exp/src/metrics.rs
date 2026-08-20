//! Prequential accuracy, steps-to-threshold, and timing.

use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::circuit::TopologyMode;
use crate::dgp::{build_dgp, build_learner_topology, labeled_sample};
use crate::learner::StreamLearner;

#[derive(Debug, Clone, Default)]
pub struct StepTimings {
    pub predict_ns: u64,
    pub observe_ns: u64,
}

#[derive(Debug, Clone)]
pub struct StreamMetrics {
    pub final_accuracy: f64,
    pub steps_to_95pct: Option<u64>,
    pub total_steps: u64,
    pub timings: StepTimings,
}

#[derive(Debug, Clone)]
pub struct RunMetrics {
    pub topology: TopologyMode,
    pub d: usize,
    pub depth: usize,
    pub width: usize,
    pub num_gates: usize,
    pub stream_len: u64,
    pub seed: u64,
    pub final_accuracy: f64,
    pub steps_to_95pct: Option<u64>,
    pub mean_predict_ns: f64,
    pub mean_observe_ns: f64,
}

pub struct MetricsTracker {
    correct: u64,
    t: u64,
    steps_to_95: Option<u64>,
    predict_ns: u128,
    observe_ns: u128,
}

impl MetricsTracker {
    pub fn new() -> Self {
        Self {
            correct: 0,
            t: 0,
            steps_to_95: None,
            predict_ns: 0,
            observe_ns: 0,
        }
    }

    pub fn record(&mut self, pred: bool, y: bool, predict: Duration, observe: Duration) {
        self.t += 1;
        if pred == y {
            self.correct += 1;
        }
        let acc = self.correct as f64 / self.t as f64;
        if self.steps_to_95.is_none() && acc >= 0.95 {
            self.steps_to_95 = Some(self.t);
        }
        self.predict_ns += predict.as_nanos();
        self.observe_ns += observe.as_nanos();
    }

    pub fn finish(self) -> StreamMetrics {
        let t = self.t.max(1);
        StreamMetrics {
            final_accuracy: self.correct as f64 / self.t as f64,
            steps_to_95pct: self.steps_to_95,
            total_steps: self.t,
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
) -> StreamMetrics {
    let mut learner = StreamLearner::new(learner_topo);
    let mut tracker = MetricsTracker::new();
    let mut rng = ChaCha8Rng::seed_from_u64(0);

    for _ in 0..stream_len {
        let x: Vec<bool> = (0..dgp.d).map(|_| rand::Rng::gen_bool(&mut rng, 0.5)).collect();
        let sample = labeled_sample(dgp, &x);

        let t0 = Instant::now();
        let pred = learner.predict(&sample.x);
        let predict_dt = t0.elapsed();

        let t1 = Instant::now();
        learner.observe(sample.y);
        let observe_dt = t1.elapsed();

        tracker.record(pred, sample.y, predict_dt, observe_dt);
    }

    tracker.finish()
}

pub fn run_single(
    seed: u64,
    d: usize,
    depth: usize,
    width: usize,
    stream_len: u64,
    topology: TopologyMode,
) -> RunMetrics {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dgp = build_dgp(&mut rng, d, depth, width);
    let learner_topo = build_learner_topology(&mut rng, &dgp, topology);
    let num_gates = dgp.num_gates;
    let m = run_stream(&dgp, learner_topo, stream_len);
    RunMetrics {
        topology,
        d,
        depth,
        width,
        num_gates,
        stream_len,
        seed,
        final_accuracy: m.final_accuracy,
        steps_to_95pct: m.steps_to_95pct,
        mean_predict_ns: m.timings.predict_ns as f64,
        mean_observe_ns: m.timings.observe_ns as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_accuracy() {
        let mut tr = MetricsTracker::new();
        tr.record(true, true, Duration::from_nanos(10), Duration::from_nanos(20));
        tr.record(false, true, Duration::from_nanos(10), Duration::from_nanos(20));
        let m = tr.finish();
        assert!((m.final_accuracy - 0.5).abs() < 1e-9);
    }
}
