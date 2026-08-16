# Agent notes

## Python / tests

Always use **uv** (not bare `python`, `pip`, or `pytest`):

```bash
uv sync
uv run pytest
uv run --extra benchmarks python -m binaml.benchmarks.synthetic_streaming_regression.cli \
 --scenario python/binaml/benchmarks/synthetic_streaming_regression/scenarios/default.json
uv run --extra benchmarks python -m binaml.benchmarks.synthetic_streaming_regression.cli \
 --scenario python/binaml/benchmarks/synthetic_streaming_regression/scenarios/default.json \
 --plots
```

JAX baselines are Python-only; they do not require a Rust rebuild.
After editing Rust under `crates/` or `bindings/`, rebuild the extension:

```bash
uv sync --reinstall-package binaml
```

## Papers / LaTeX

White papers live under `papers/<paper-name>/` (see `papers/README.md`).
`main.tex` is the source; `main.pdf` is the reviewed artifact to keep beside it.
When editing or adding papers, keep `papers/README.md` current with their
directory names, titles, and citation links.

**Never compile from the repo root.** LaTeX writes outputs into the current
working directory, which is how stray `main.pdf` / `main.aux` / etc. appear at
the project root.

```bash
cd papers/<paper-name>
latexmk -pdf main.tex
# or: pdflatex main.tex
```

Do not commit auxiliary files (`*.aux`, `*.log`, `*.fls`, `*.fdb_latexmk`,
`*.out`, …). Only `main.tex`, reviewed `main.pdf`, and paper assets belong in
git.
