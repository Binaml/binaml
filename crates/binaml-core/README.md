# Binaml Core

The target-neutral Rust implementation of online boolean-function ensemble
regression. Its supported public interface is `BRegressor` and
`BRegressorError`; graph building, compaction, and truth-table machinery are
internal implementation details.

`binaml-core` is designed to support future server, embedded, and C-ABI crates
without depending on those targets or on Python bindings.

## Build

```bash
cargo build --release
```

## Tests

```bash
cargo test
```
