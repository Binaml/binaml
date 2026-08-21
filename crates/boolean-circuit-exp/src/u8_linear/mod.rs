//! Batch learning of N × u8 → bool linear threshold functions.

mod bits;
mod dataset;
mod dgp;
mod learner;
mod learner_mem;
mod learner_sym;
mod metrics;

pub use dataset::{TrainSplit, U8Dataset};
pub use dgp::{domain_size, sample_pool_size, score, LinearU8Dgp, WeightMode, BITS_PER_U8, MAX_SAMPLE_POOL};
pub use learner::U8LinearLearner;
pub use learner_mem::{label_u8, score_mem, U8MemLearner};
pub use learner_sym::U8SymLearner;
pub use metrics::{
    pool_size_for, run_seed, run_seed_mem, run_seed_perceptron, run_seed_sym, run_seed_with_split,
    summarize, U8RunMetrics, U8Summary, U8Timings, U8Variant,
};
