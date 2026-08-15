![Binaml](https://raw.githubusercontent.com/Binaml/binaml/main/assets/banner/binaml-banner.svg)

Binaml focuses on continual learning models over streams of binary features
subject to data drift.
Binary features provide a task-agnostic representation and enable models
optimized for memory, latency, and energy efficiency.

## Core concepts

Binaml starts with input bits and learns boolean functions by composing pairs of
Boolean features. Each composition is one of the 16 Boolean functions of arity
two, represented as a four-bit truth table. Repeated composition yields richer,
inspectable boolean functions.

`BRegressor` is the current Binaml model: an online ensemble of batch-learned
boolean functions, each with a scalar weight, updated by residual-sign SGD.

See the [Binaml paper](papers/binaml/) for the model specification.

The supported Python API is `binaml.BRegressor`. The supported Rust API is
`binaml_core::BRegressor`. `binaml._core` is a private implementation detail.

## Quick start: Python

Install the alpha package with uv:

```bash
uv add "binaml==0.1.0a2"
```

To run from a checkout, install Python 3.13+, a Rust toolchain, and the project
dependencies:

```bash
uv sync
```

```python
import numpy as np
from binaml import BRegressor

model = BRegressor(n_features=2)

for features, target in [
    (np.array([0, 1], dtype=np.uint8), 1.0),
    (np.array([1, 0], dtype=np.uint8), 0.0),
]:
    prediction = model.predict(features)
    model.observe(features, target)
```

Each `predict(features)` call must be followed by `observe(features, target)`.

## Quick start: Rust

Add `binaml-core` to your `Cargo.toml`:

```toml
[dependencies]
binaml-core = "0.1.0-alpha.2"
```

```rust
use binaml_core::{BRegressor, BRegressorError};

fn main() -> Result<(), BRegressorError> {
    let mut model = BRegressor::with_hyperparameters(
        2, 5e-3, 1e-4, 16, 5, 8, 3, 64,
    )?;

    for (features, target) in [([false, true], 1.0), ([true, false], 0.0)] {
        let prediction = model.predict(&features)?;
        model.observe(&features, target)?;
        println!("{prediction}");
    }

    Ok(())
}
```

## Benchmarks

The included synthetic streaming environments and prequential evaluation
protocol compare `BRegressor` and `BClassifier` with linear and MLP baselines.
See the
[synthetic drifting streams paper](papers/synthetic-drifting-streams/)
for the benchmark specification. Plotting support is optional:

```bash
uv run --extra benchmarks python -m binaml.benchmarks.synthetic_streaming_regression.cli \
  --scenario python/binaml/benchmarks/synthetic_streaming_regression/scenarios/default.json

uv run --extra benchmarks python -m binaml.benchmarks.synthetic_streaming_classification.cli \
  --scenario python/binaml/benchmarks/synthetic_streaming_classification/scenarios/default.json
```

## Citation

If you use Binaml, cite the software metadata in
[CITATION.cff](CITATION.cff). Cite [our papers](papers/) separately when their model or
benchmark specification informs your work.

## License

Source code is licensed under [Apache-2.0](LICENSE).
