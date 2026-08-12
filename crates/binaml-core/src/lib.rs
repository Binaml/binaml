//! Online regression with residual-learned boolean features.

mod binary_truth_table;
mod feature_learning;
mod feature_store;
mod regressor;

pub(crate) use binary_truth_table::{FeatureCounter, FeatureCounterError};
pub(crate) use feature_learning::{
    FeatureLearner, FeatureLearningConfig, FeatureLearningError, SignBatch,
};
pub use regressor::{BRegressor, BRegressorError};

pub(crate) use feature_store::{Feature, FeatureId, FeatureStore, InsertFeatureError};
