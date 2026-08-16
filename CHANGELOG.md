# Changelog

All notable changes to Binaml are documented here.

## [Unreleased]

## [0.1.0-alpha.3] - 2026-08-16

### Added
- Internal function-learning pipeline: `FunctionBuilder`, ephemeral-to-compact
  lowering, and immutable `FunctionGraph` storage for learned boolean functions.
- `BRegressor.function_count` and `BRegressor.weight()` Python accessors.
- Unit tests for `FunctionGraph` evaluation (sources, constants, composed gates,
  missing features, output selection).

### Changed
- **Breaking:** `BRegressor` now uses a function-ensemble architecture instead
  of a shared growing feature store. Removed `features_per_layer` and
  `candidate_capacity`. Added `max_functions`. Default hyperparameters updated.
- Binaml paper, README, and benchmark model config updated for the
  function-ensemble model.

### Removed
- Shared-graph `FeatureStore` / `FeatureLearner` implementation.
- `FunctionEnsembleRegressor` (merged into `BRegressor`).

## [0.1.0-alpha.2] - 2026-08-12

### Changed
- Unified the package description with the paper title.
- Fixed the PyPI banner by using an absolute image URL.

## [0.1.0-alpha.1] - 2026-08-12

### Added
- `binaml-core` Rust crate with the online `BRegressor` model.
- `binaml` Python package and native bindings for `BRegressor`.
- Synthetic drifting binary-regression benchmark and prequential evaluation protocol.
- Model and benchmark specification papers with citation metadata.
