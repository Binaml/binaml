//! Online regression with residual-learned boolean features.

mod association;
mod batch;
mod classifier;
mod conjunction_build_common;
mod conjunction_builder;
mod conjunction_expert;
mod ensemble;
mod regressor;
mod workspace;

pub use association::association_score;
pub use batch::SignBatch;
pub use classifier::{BClassifier, BClassifierError};
pub use conjunction_build_common::{
    derive_conjunction_capacity, ConjunctionBuildConfig, ConjunctionBuildError, ConjunctionKey,
    DEFAULT_MAX_CONJUNCTION_LENGTH, DEFAULT_MAX_EXPERTS, DEFAULT_STALE_LAYERS, MAX_BATCH_SIZE,
};
pub use conjunction_builder::{ConjunctionBuildSession, ConjunctionBuilder};
pub use conjunction_expert::ConjunctionExpert;
pub use regressor::{BRegressor, BRegressorError};
