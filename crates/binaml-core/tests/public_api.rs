use binaml_core::{BRegressor, BRegressorError};

#[test]
fn rejects_invalid_public_config() {
    let error = BRegressor::with_hyperparameters(0, 0.01, 1e-4, 32, 1, 8, 32, 32, 2)
        .expect_err("zero-width models are invalid");
    assert_eq!(error, BRegressorError::InvalidConfig);
}
