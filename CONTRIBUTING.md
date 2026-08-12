# Contributing to Binaml

Thank you for contributing. Binaml is early research software; please discuss substantial API or algorithm changes in an issue before opening a pull request.

## Development setup

Install Python and Rust dependencies:

```bash
uv sync
```

Run the checks before opening a pull request:

```bash
uv run ruff check .
uv run pytest
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

After editing Rust code under `crates/` or `bindings/`, rebuild the extension:

```bash
uv sync --reinstall-package binaml
```

## Pull requests

- Keep changes focused and include tests for behavior changes.
- Update public documentation, benchmark scenarios, and papers when public behavior changes.
- Do not commit generated binaries, benchmark outputs, LaTeX auxiliary files,
  local editor configuration, or credentials.
- By submitting a contribution, you agree to license it under Apache-2.0.

Create short-lived branches from `main` using `feature/`, `fix/`, `docs/`, or
`release/` prefixes. Merge through squash pull requests and delete the branch
afterward. `main` must remain releasable.

Prepare each release in a `release/<version>` pull request. Update the package
version, lockfiles, changelog, and citation metadata there. After the pull
request passes the release checks and is merged, tag the resulting `main`
commit with the matching `v<version>` tag.
