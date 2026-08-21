//! Streaming boolean circuit learning experiments.

pub mod circuit;
pub mod diag_wire;
pub mod dgp;
pub mod gate;
pub mod gate_wire;
pub mod learner;
pub mod learner_wire;
pub mod metrics;
pub mod u8_linear;

pub use circuit::{FixedCircuit, TopologyMode};
pub use dgp::{build_dgp, build_learner_topology, StreamSample};
pub use gate::{bool_target, get_weight, lane, pole, set_weight, sign};
pub use gate_wire::{bool_target_pole, forward_sum, nudge_weight as nudge_weight_wire};
pub use learner::StreamLearner;
pub use learner_wire::StreamLearnerWire;
pub use metrics::{LearnerKind, RunMetrics, StepTimings, StreamMetrics};
pub use u8_linear::{
    LinearU8Dgp, U8Dataset, U8LinearLearner, U8MemLearner, U8RunMetrics, U8Summary, U8SymLearner,
    U8Timings, U8Variant, WeightMode,
};
