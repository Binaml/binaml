//! Online regression with residual-learned boolean features.

mod batch;
mod binary_truth_table;
mod boolean_circuit;
mod function_builder;
mod function_compact;
mod function_graph;
mod regressor;

pub(crate) use batch::SignBatch;
pub(crate) use binary_truth_table::{FeatureCounter, FeatureCounterError};
pub(crate) use boolean_circuit::evaluate_truth_table;
pub(crate) use function_builder::{FunctionBuildConfig, FunctionBuilder, FunctionBuildError};
pub(crate) use function_compact::compact;
pub(crate) use function_graph::FunctionGraph;
pub use regressor::{BRegressor, BRegressorError};
