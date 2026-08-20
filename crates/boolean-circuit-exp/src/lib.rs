//! Streaming boolean circuit learning experiments.

pub mod circuit;
pub mod dgp;
pub mod gate;
pub mod learner;
pub mod metrics;

pub use circuit::{FixedCircuit, TopologyMode};
pub use dgp::{build_dgp, build_learner_topology, StreamSample};
pub use gate::{bool_target, get_weight, lane, pole, set_weight, sign};
pub use learner::StreamLearner;
pub use metrics::{RunMetrics, StepTimings, StreamMetrics};
