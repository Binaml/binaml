use binaml_core::{
    BClassifier, BRegressor, FunctionBuildConfig, FunctionBuilder, FunctionModel, SignBatch,
    DEFAULT_MAX_EXPERT_NODES,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::Instant;

const DEFAULT_LEARNING_RATE: f64 = 0.03;
const DEFAULT_L2: f64 = 5e-4;
const DEFAULT_CLASSIFIER_LEARNING_RATE: f64 = 0.12;
const DEFAULT_CLASSIFIER_BATCH_SIZE: usize = 12;
const DEFAULT_CLASSIFIER_SGD_STEPS: usize = 30;
const DEFAULT_SGD_STEPS: usize = 20;

fn model_error(error: impl std::fmt::Debug) -> PyErr {
    PyValueError::new_err(format!("{error:?}"))
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

fn binary_signs(input: PyReadonlyArray1<'_, bool>) -> PyResult<Vec<bool>> {
    input
        .as_slice()
        .map_err(|_| PyValueError::new_err("signs must be contiguous"))
        .map(|values| values.to_vec())
}

fn feature_columns(input: PyReadonlyArray2<'_, u8>) -> PyResult<(Vec<Vec<bool>>, usize)> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("features must be a 2D array"));
    }
    let n_samples = shape[0];
    let n_features = shape[1];
    let values = input
        .as_slice()
        .map_err(|_| PyValueError::new_err("features must be contiguous"))?;
    if values.iter().any(|value| *value > 1) {
        return Err(PyValueError::new_err("features must contain only 0 and 1"));
    }
    let mut columns = vec![Vec::with_capacity(n_samples); n_features];
    for row in 0..n_samples {
        for feature in 0..n_features {
            columns[feature].push(values[row * n_features + feature] == 1);
        }
    }
    Ok((columns, n_samples))
}

fn predictions_to_py<'py>(
    py: Python<'py>,
    predictions: Vec<bool>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    Ok(predictions.into_pyarray(py))
}

fn fit_result_to_py<'py>(
    py: Python<'py>,
    predictions: Vec<bool>,
    score: i64,
    elapsed_seconds: f64,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("predictions", predictions_to_py(py, predictions)?)?;
    dict.set_item("score", score)?;
    dict.set_item("elapsed_seconds", elapsed_seconds)?;
    Ok(dict.into())
}

fn feature_batch(input: PyReadonlyArray2<'_, u8>) -> PyResult<(Vec<Vec<bool>>, usize)> {
    feature_columns(input)
}

fn sign_batch_from_arrays(
    features: PyReadonlyArray2<'_, u8>,
    signs: PyReadonlyArray1<'_, bool>,
) -> PyResult<(Vec<Vec<bool>>, Vec<bool>, usize)> {
    let (columns, batch_size) = feature_columns(features)?;
    let signs = binary_signs(signs)?;
    if signs.len() != batch_size {
        return Err(PyValueError::new_err(
            "signs length must match feature rows",
        ));
    }
    Ok((columns, signs, batch_size))
}

#[pyclass]
struct FunctionLearner {
    parent_top_k: usize,
    model: Option<FunctionModel>,
}

impl FunctionLearner {
    fn build_config(&self, batch_size: usize, source_count: usize) -> FunctionBuildConfig {
        FunctionBuildConfig::new(
            batch_size,
            self.parent_top_k,
            source_count,
            DEFAULT_MAX_EXPERT_NODES,
        )
    }
}

#[pymethods]
impl FunctionLearner {
    #[new]
    #[pyo3(signature = (parent_top_k=8))]
    fn new(parent_top_k: usize) -> Self {
        Self {
            parent_top_k,
            model: None,
        }
    }

    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        features: PyReadonlyArray2<'_, u8>,
        signs: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<PyObject> {
        let (columns, signs, batch_size) = sign_batch_from_arrays(features, signs)?;
        let column_refs: Vec<&[bool]> = columns.iter().map(Vec::as_slice).collect();
        let batch = SignBatch {
            feature_columns: &column_refs,
            signs: &signs,
        };
        let started = Instant::now();
        let (model, score) =
            FunctionBuilder::fit(batch, self.build_config(batch_size, column_refs.len())).map_err(model_error)?;
        self.model = Some(model);
        let dict = PyDict::new(py);
        dict.set_item("score", i64::from(score))?;
        dict.set_item("elapsed_seconds", started.elapsed().as_secs_f64())?;
        Ok(dict.into())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'_, u8>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("learner is not fit"))?;
        let (columns, batch_size) = feature_batch(features)?;
        let dummy_signs = vec![false; batch_size];
        let column_refs: Vec<&[bool]> = columns.iter().map(Vec::as_slice).collect();
        let batch = SignBatch {
            feature_columns: &column_refs,
            signs: &dummy_signs,
        };
        let predictions = FunctionBuilder::predict(model, batch).map_err(model_error)?;
        predictions_to_py(py, predictions)
    }

    fn fit_predict<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'_, u8>,
        signs: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<PyObject> {
        let (columns, signs, batch_size) = sign_batch_from_arrays(features, signs)?;
        let column_refs: Vec<&[bool]> = columns.iter().map(Vec::as_slice).collect();
        let batch = SignBatch {
            feature_columns: &column_refs,
            signs: &signs,
        };
        let started = Instant::now();
        let (predictions, score) =
            FunctionBuilder::fit_predict(batch, self.build_config(batch_size, column_refs.len()))
                .map_err(model_error)?;
        fit_result_to_py(
            py,
            predictions,
            i64::from(score),
            started.elapsed().as_secs_f64(),
        )
    }
}

#[pyclass]
struct BRegressorCore {
    model: BRegressor,
}

#[pymethods]
impl BRegressorCore {
    #[new]
    #[pyo3(signature = (source_feature_count, learning_rate=DEFAULT_LEARNING_RATE, l2=DEFAULT_L2, batch_size=16, sgd_steps=DEFAULT_SGD_STEPS, parent_top_k=8, max_functions=64, max_expert_nodes=DEFAULT_MAX_EXPERT_NODES))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        source_feature_count: usize,
        learning_rate: f64,
        l2: f64,
        batch_size: usize,
        sgd_steps: usize,
        parent_top_k: usize,
        max_functions: usize,
        max_expert_nodes: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            model: BRegressor::with_hyperparameters(
                source_feature_count,
                learning_rate,
                l2,
                batch_size,
                sgd_steps,
                parent_top_k,
                max_functions,
                max_expert_nodes,
            )
            .map_err(model_error)?,
        })
    }

    fn predict(&mut self, features: PyReadonlyArray1<'_, u8>) -> PyResult<f64> {
        self.model
            .predict(&binary_features(features)?)
            .map_err(model_error)
    }

    fn update(&mut self, target: f64) -> PyResult<()> {
        self.model.update(target).map_err(model_error)
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

#[pyclass]
struct BClassifierCore {
    model: BClassifier,
}

#[pymethods]
impl BClassifierCore {
    #[new]
    #[pyo3(signature = (source_feature_count, n_classes, learning_rate=DEFAULT_CLASSIFIER_LEARNING_RATE, l2=DEFAULT_L2, batch_size=DEFAULT_CLASSIFIER_BATCH_SIZE, sgd_steps=DEFAULT_CLASSIFIER_SGD_STEPS, parent_top_k=8, max_functions=96, max_expert_nodes=DEFAULT_MAX_EXPERT_NODES))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        source_feature_count: usize,
        n_classes: usize,
        learning_rate: f64,
        l2: f64,
        batch_size: usize,
        sgd_steps: usize,
        parent_top_k: usize,
        max_functions: usize,
        max_expert_nodes: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            model: BClassifier::with_hyperparameters(
                source_feature_count,
                n_classes,
                learning_rate,
                l2,
                batch_size,
                sgd_steps,
                parent_top_k,
                max_functions,
                max_expert_nodes,
            )
            .map_err(model_error)?,
        })
    }

    fn predict(&mut self, features: PyReadonlyArray1<'_, u8>) -> PyResult<usize> {
        self.model
            .predict(&binary_features(features)?)
            .map_err(model_error)
    }

    fn update(&mut self, target: usize) -> PyResult<()> {
        self.model.update(target).map_err(model_error)
    }

    #[getter]
    fn n_observed(&self) -> usize {
        self.model.n_observed()
    }

    #[getter]
    fn function_count(&self) -> usize {
        self.model.function_count()
    }

    fn intercept(&self, class_index: usize) -> Option<f64> {
        self.model.intercept(class_index)
    }

    fn weight(&self, function_index: usize, class_index: usize) -> Option<f64> {
        self.model.weight(function_index, class_index)
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BRegressorCore>()?;
    module.add_class::<BClassifierCore>()?;
    module.add_class::<FunctionLearner>()?;
    Ok(())
}
