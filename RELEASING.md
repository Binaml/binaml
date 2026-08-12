# Releasing Binaml

The Cargo workspace version is authoritative. Maturin converts Rust prerelease
versions to Python versions, for example:

- Cargo and crates.io: `0.1.0-alpha.1`
- PyPI: `0.1.0a1`
- Git tag: `v0.1.0-alpha.1`

`CHANGELOG.md` is the source for GitHub release notes.

## Prepare a release

1. Create `release/<version>` from `main`.
2. Update the workspace version in `Cargo.toml`.
3. Update `Cargo.lock`.
4. Add the matching version and release date to `CHANGELOG.md` and
   `CITATION.cff`.
5. Run:

   ```bash
   uv sync --reinstall-package binaml
   uv run pytest
   uv run ruff check .
   cargo fmt --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   cargo publish -p binaml-core --dry-run --locked
   ```

6. Open a pull request and wait for CI and the release checks.
7. Squash-merge the pull request and delete the release branch.

## First binaml-core release

Crates.io requires the crate to exist before trusted publishing can be
configured.

1. Create a short-lived crates.io token.
2. From the reviewed `main` commit, publish `binaml-core`:

   ```bash
   cargo publish -p binaml-core --locked
   ```

3. In crates.io, register the trusted publisher for:
   - owner: `Binaml`
   - repository: `binaml`
   - workflow: `release.yml`
   - environment: `release`
4. Revoke the bootstrap token.

The release workflow detects that this first crate version already exists and
does not try to publish it again.

## Publish

Create and push an annotated tag on the reviewed `main` commit:

```bash
git tag -a v0.1.0-alpha.1 -m "Binaml 0.1.0-alpha.1"
git push origin v0.1.0-alpha.1
```

Approve the protected `release` environment. The workflow then:

1. verifies the tag and package metadata;
2. rebuilds and tests the Python distributions;
3. publishes `binaml-core` when the version is not already on crates.io;
4. publishes `binaml` to PyPI through trusted publishing;
5. creates a GitHub release from the matching changelog section.

Alpha, beta, and release-candidate tags become GitHub prereleases.

## Verify

Confirm that both registry versions and the GitHub release exist. Install the
published Python package in a clean directory:

```bash
uv run --isolated --no-project --with "binaml==0.1.0a1" \
  python -c "from binaml import BRegressor; print(BRegressor)"
```

Published versions and tags are immutable. Fix release problems with a new
prerelease version rather than replacing an existing one.
