//! Online regression with residual-learned boolean features.

mod association;
mod batch;
mod binary_truth_table;
mod boolean_circuit;
mod classifier;
mod ensemble;
mod function_build_common;
mod function_builder;
mod function_compact;
mod function_graph;
mod regressor;
mod workspace;

pub use association::association_score;
pub use batch::SignBatch;
pub use classifier::{BClassifier, BClassifierError};
pub use function_build_common::{
    derive_build_capacity, BuildNodeId, DEFAULT_L_PAT, DEFAULT_MAX_EXPERT_NODES, EphemeralGraph,
    EphemeralNode, FunctionBuildConfig, FunctionBuildError, FunctionModel,
};
pub use function_builder::FunctionBuilder;
pub use regressor::{BRegressor, BRegressorError};

pub(crate) use binary_truth_table::{FeatureCounter, FeatureCounterError};
pub(crate) use function_compact::compact;
pub(crate) use function_graph::FunctionGraph;
