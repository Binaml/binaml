//! Online regression with residual-learned boolean features.

mod association;
mod batch;
mod binary_truth_table;
mod boolean_circuit;
mod classifier;
mod ensemble;
mod function_builder;
mod function_compact;
mod function_graph;
mod regressor;

pub use association::association_score;
pub use batch::SignBatch;
pub use classifier::{BClassifier, BClassifierError};
pub use function_builder::{
    BuildNodeId, EphemeralGraph, EphemeralNode, FunctionBuildConfig, FunctionBuildError,
    FunctionBuilder, FunctionModel,
};
pub use regressor::{BRegressor, BRegressorError};

pub(crate) use binary_truth_table::{FeatureCounter, FeatureCounterError};
pub(crate) use function_compact::compact;
pub(crate) use function_graph::FunctionGraph;
