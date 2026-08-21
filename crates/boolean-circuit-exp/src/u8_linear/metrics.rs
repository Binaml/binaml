//! Batch train / test metrics for u8 linear experiments.

use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::bits::sample;
use super::dataset::{TrainSplit, U8Dataset};
use super::dgp::{sample_pool_size, LinearU8Dgp, WeightMode};
use super::learner::U8LinearLearner;
use super::learner_mem::U8MemLearner;
use super::learner_sym::U8SymLearner;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum U8Variant {
    Perceptron,
    Mem,
    Sym,
}

impl U8Variant {
    pub fn as_str(self) -> &'static str {
        match self {
            U8Variant::Perceptron => "perceptron",
            U8Variant::Mem => "mem",
            U8Variant::Sym => "sym",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct U8Timings {
    pub fit_ns: u64,
    pub predict_ns: u64,
}

#[derive(Debug, Clone)]
pub struct U8RunMetrics {
    pub seed: u64,
    pub n_inputs: usize,
    pub weight_mode: WeightMode,
    pub variant: U8Variant,
    pub n_samples: usize,
    pub split_kind: &'static str,
    pub train_frac: Option<f64>,
    pub train_n: usize,
    pub train_accuracy: f64,
    pub test_accuracy: f64,
    pub epochs_run: usize,
    pub train_errors: usize,
    pub test_errors: usize,
    pub timings: U8Timings,
}

#[derive(Debug, Clone)]
pub struct U8Summary {
    pub mean_train_acc: f64,
    pub std_train_acc: f64,
    pub mean_test_acc: f64,
    pub std_test_acc: f64,
    pub mean_epochs: f64,
    pub mean_fit_ns: f64,
    pub mean_predict_ns: f64,
}

pub fn run_seed(
    seed: u64,
    n_inputs: usize,
    train_frac: f64,
    epochs: usize,
    weight_mode: WeightMode,
    variant: U8Variant,
) -> U8RunMetrics {
    run_seed_with_split(
        seed,
        n_inputs,
        TrainSplit::Frac(train_frac),
        epochs,
        weight_mode,
        variant,
    )
}

pub fn run_seed_with_split(
    seed: u64,
    n_inputs: usize,
    split: TrainSplit,
    epochs: usize,
    weight_mode: WeightMode,
    variant: U8Variant,
) -> U8RunMetrics {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dgp = LinearU8Dgp::random_with_mode(&mut rng, n_inputs, weight_mode);
    let data = U8Dataset::split_with(seed, &dgp, split);
    let n_samples = data.train_len() + data.test_len();

    let (
        epochs_run,
        train_accuracy,
        test_accuracy,
        train_errors,
        test_errors,
        timings,
    ) = match variant {
        U8Variant::Perceptron => {
            let mut learner = U8LinearLearner::new(n_inputs);
            let t0 = Instant::now();
            let epochs_run = learner.fit(&data.train_x, &data.train_y, epochs);
            let fit_ns = t0.elapsed().as_nanos() as u64;
            let predict_ns = mean_predict_ns(&data, true, |x| learner.predict(x));
            (
                epochs_run,
                learner.accuracy(&data.train_x, &data.train_y),
                learner.accuracy(&data.test_x, &data.test_y),
                learner.error_count(&data.train_x, &data.train_y),
                learner.error_count(&data.test_x, &data.test_y),
                U8Timings { fit_ns, predict_ns },
            )
        }
        U8Variant::Mem => {
            let mut learner = U8MemLearner::new(n_inputs);
            let t0 = Instant::now();
            let epochs_run = learner.fit(&data.train_x, &data.train_y, epochs);
            let fit_ns = t0.elapsed().as_nanos() as u64;
            let predict_ns = mean_predict_ns(&data, true, |x| learner.predict(x));
            (
                epochs_run,
                learner.accuracy(&data.train_x, &data.train_y),
                learner.accuracy(&data.test_x, &data.test_y),
                learner.error_count(&data.train_x, &data.train_y),
                learner.error_count(&data.test_x, &data.test_y),
                U8Timings { fit_ns, predict_ns },
            )
        }
        U8Variant::Sym => {
            let mut learner = U8SymLearner::new(n_inputs);
            let t0 = Instant::now();
            let epochs_run = learner.fit(&data.train_x, &data.train_y, epochs);
            let fit_ns = t0.elapsed().as_nanos() as u64;
            let predict_ns = mean_predict_ns(&data, true, |x| learner.predict(x));
            (
                epochs_run,
                learner.accuracy(&data.train_x, &data.train_y),
                learner.accuracy(&data.test_x, &data.test_y),
                learner.error_count(&data.train_x, &data.train_y),
                learner.error_count(&data.test_x, &data.test_y),
                U8Timings { fit_ns, predict_ns },
            )
        }
    };

    U8RunMetrics {
        seed,
        n_inputs,
        weight_mode,
        variant,
        n_samples,
        split_kind: split.kind(),
        train_frac: split.frac_value(),
        train_n: data.train_n,
        train_accuracy,
        test_accuracy,
        epochs_run,
        train_errors,
        test_errors,
        timings,
    }
}

fn mean_predict_ns<F>(data: &U8Dataset, test: bool, mut predict: F) -> u64
where
    F: FnMut(&[u8]) -> bool,
{
    let (x_flat, n) = if test {
        (&data.test_x[..], data.test_len())
    } else {
        (&data.train_x[..], data.train_len())
    };
    if n == 0 {
        return 0;
    }
    let mut total = 0u128;
    for i in 0..n {
        let x = sample(x_flat, data.n_inputs, i);
        let t0 = Instant::now();
        let _ = predict(x);
        total += t0.elapsed().as_nanos();
    }
    (total / n as u128) as u64
}

pub fn run_seed_perceptron(
    seed: u64,
    n_inputs: usize,
    train_frac: f64,
    epochs: usize,
) -> U8RunMetrics {
    run_seed(
        seed,
        n_inputs,
        train_frac,
        epochs,
        WeightMode::I8,
        U8Variant::Perceptron,
    )
}

pub fn run_seed_mem(seed: u64, n_inputs: usize, train_frac: f64, epochs: usize) -> U8RunMetrics {
    run_seed(
        seed,
        n_inputs,
        train_frac,
        epochs,
        WeightMode::I8,
        U8Variant::Mem,
    )
}

pub fn run_seed_sym(seed: u64, n_inputs: usize, train_frac: f64, epochs: usize) -> U8RunMetrics {
    run_seed(
        seed,
        n_inputs,
        train_frac,
        epochs,
        WeightMode::I8,
        U8Variant::Sym,
    )
}

pub fn summarize(results: &[U8RunMetrics]) -> U8Summary {
    let n = results.len() as f64;
    let mean_train_acc = results.iter().map(|r| r.train_accuracy).sum::<f64>() / n;
    let var_train = results
        .iter()
        .map(|r| (r.train_accuracy - mean_train_acc).powi(2))
        .sum::<f64>()
        / n;
    let mean_test_acc = results.iter().map(|r| r.test_accuracy).sum::<f64>() / n;
    let var_test = results
        .iter()
        .map(|r| (r.test_accuracy - mean_test_acc).powi(2))
        .sum::<f64>()
        / n;
    let mean_epochs = results.iter().map(|r| r.epochs_run as f64).sum::<f64>() / n;
    let mean_fit_ns = results.iter().map(|r| r.timings.fit_ns as f64).sum::<f64>() / n;
    let mean_predict_ns = results
        .iter()
        .map(|r| r.timings.predict_ns as f64)
        .sum::<f64>()
        / n;

    U8Summary {
        mean_train_acc,
        std_train_acc: var_train.sqrt(),
        mean_test_acc,
        std_test_acc: var_test.sqrt(),
        mean_epochs,
        mean_fit_ns,
        mean_predict_ns,
    }
}

pub fn pool_size_for(n_inputs: usize) -> usize {
    sample_pool_size(n_inputs)
}
