use binaml_core::BRegressor;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const DEFAULT_LEARNING_RATE: f64 = 5e-3;

fn model_error(error: impl std::fmt::Debug) -> PyErr {
    PyValueError::new_err(format!("{error:?}"))
}

#[pyclass]
struct BRegressorCore {
    model: BRegressor,
}

fn binary_features(input: PyReadonlyArray1<'_, u8>) -> PyResult<Vec<bool>> {
    let values = input
        .as_slice()
        .map_err(|_| PyValueError::new_err("features must be contiguous"))?;
    if values.iter().any(|value| *value > 1) {
        return Err(PyValueError::new_err("features must contain only 0 and 1"));
    }
    Ok(values.iter().map(|value| *value == 1).collect())
}

#[pymethods]
impl BRegressorCore {
    #[new]
    #[pyo3(signature = (source_feature_count, learning_rate=DEFAULT_LEARNING_RATE, l2=1e-4, batch_size=16, sgd_steps=5, parent_top_k=8, max_layers=3, max_functions=64))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        source_feature_count: usize,
        learning_rate: f64,
        l2: f64,
        batch_size: usize,
        sgd_steps: usize,
        parent_top_k: usize,
        max_layers: usize,
        max_functions: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            model: BRegressor::with_hyperparameters(
                source_feature_count,
                learning_rate,
                l2,
                batch_size,
                sgd_steps,
                parent_top_k,
                max_layers,
                max_functions,
            )
            .map_err(model_error)?,
        })
    }

    fn predict(&self, features: PyReadonlyArray1<'_, u8>) -> PyResult<f64> {
        self.model
            .predict(&binary_features(features)?)
            .map_err(model_error)
    }

    fn observe(&mut self, features: PyReadonlyArray1<'_, u8>, target: f64) -> PyResult<()> {
        self.model
            .observe(&binary_features(features)?, target)
            .map_err(model_error)
    }

    #[getter]
    fn intercept(&self) -> f64 {
        self.model.intercept()
    }

    #[getter]
    fn n_observed(&self) -> usize {
        self.model.n_observed()
    }

    #[getter]
    fn function_count(&self) -> usize {
        self.model.function_count()
    }

    fn weight(&self, index: usize) -> Option<f64> {
        self.model.weight(index)
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BRegressorCore>()?;
    Ok(())
}
