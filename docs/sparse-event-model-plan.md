# Sparse event-based Binaml: implementation plan

This document specifies a **parallel implementation** of Binaml for large binary
feature spaces. The goal is to ingest **batches of change events** instead of
dense feature vectors, while keeping the same learning algorithm (conjunction
experts + linear head, batch beam search, online SGD). The dense model
(`BRegressor` / `BClassifier` in `crates/binaml-core`) stays unchanged so we
can compare accuracy, latency, and memory on identical streams.

## Motivation

The current hot path scales with input width `d`:

| Step | Current cost | Dominates when |
|------|--------------|----------------|
| `begin_predict` | copy `d` bools + evaluate `F` experts | large `d` |
| `update` | write `d` bools into batch buffer | large `d` |
| batch finalize | `precompute_literal_columns` over all `d` features | large `d`, beam search |

Expert evaluation is already sparse (each expert reads ≤ `max_conjunction_length`
literals). The mismatch is **feature ingest and batch materialization**.

Target workloads: high `d` (10⁴–10⁶+), low active count per observation (`k ≪ d`),
default-off bits, upstream producers that naturally emit deltas.

## Goals

- Same semantics as the dense model on equivalent inputs (bit-identical
  predictions/updates when the event stream materializes the same dense rows).
- Public API: apply a sparse event patch, then `predict` / `update` as today.
- Fixed workspace bounds driven by **sparsity caps**, not `d`.
- Side-by-side Rust types and Python wrappers for A/B benchmarks.
- Reuse unchanged pieces: `ConjunctionExpert`, `ConjunctionBuilder` beam logic,
  ensemble head SGD, pruning.

## Non-goals (v1)

- Replacing or removing the dense API.
- Dynamic heap growth on the predict/update hot path (preallocate to configured
  caps; reject or degrade gracefully when exceeded).
- Out-of-order or cross-session event merging (assume in-order patches per model
  instance unless a snapshot reset is explicit).
- Sparse internal representation of expert weights or the linear head.

## Comparison strategy

Implement as a **sibling crate module**, not a refactor of `BooleanEnsemble`.

```
crates/binaml-core/src/
  ensemble.rs              # dense (unchanged)
  sparse/
    mod.rs
    feature_state.rs       # sparse current features
    sparse_batch.rs        # sparse batch buffer + active index set
    sparse_ensemble.rs     # event ingest + incremental eval
    sparse_workspace.rs    # bounded buffers
  sparse_regressor.rs      # BRegressorSparse public wrapper
  sparse_classifier.rs     # BClassifierSparse public wrapper
```

Python mirrors under `python/binaml/models/`:

- `binaml_regressor_sparse.py` → `BRegressorSparseCore`
- `binaml_classifier_sparse.py` → `BClassifierSparseCore`

Benchmarks add a **sparse stream adapter** that emits events from the same DGP
as the dense benchmarks, plus micro-benchmarks at varying `(d, k)`.

Correctness checks:

1. **Dense equivalence:** for each step, materialize dense `x_t` from the event
   log; assert `predict_sparse(events) == predict_dense(x_t)` and same post-
   update weights (within float tolerance).
2. **Regression suite:** existing pytest scenarios run both models.
3. **Stress:** random sparse patches with `k` near cap; verify no panic and
   deterministic errors when caps are exceeded.

## Architecture overview

```mermaid
flowchart LR
  subgraph ingest
    E[Event patch\n(index, value)*]
    FS[FeatureState\nsparse set of 1s]
  end
  subgraph predict
    DIR[Dirty expert indices]
    EV[Expert evaluate\nliteral lookup]
    HEAD[Linear head]
  end
  subgraph update
    SB[SparseBatchRow]
    SGD[Head SGD]
    BF[Batch finalize\nactive features only]
  end
  E --> FS
  FS --> DIR --> EV --> HEAD
  FS --> SB
  EV --> SGD
  SB --> BF
  BF --> NEW[New ConjunctionExpert]
```

**Observation lifecycle (unchanged contract):**

1. `apply_events(patch)` — merge changes into internal state.
2. `predict()` — uses current state; sets pending flag.
3. `update(target)` — head SGD + append sparse row to batch; maybe finalize.

Unlike the dense API, `predict` does not take features: state is model-owned.
Callers that still have full vectors can use a helper
`apply_dense_snapshot(x: &[bool])` implemented as “set all indices to match `x`”
(for tests and dense equivalence only; not for production hot path).

## Core data structures

### `FeatureEvent`

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureEvent {
    pub index: u32,   // global feature index, 0 .. d-1
    pub value: bool,  // almost always true in v1 workloads; false = explicit unset
}
```

Patches are `&[FeatureEvent]`. Duplicate indices in one patch: **last wins**
(document and test this).

### `FeatureState`

Represents `x ∈ {0,1}^d` without storing `d` bits.

**v1 representation:** sorted unique `Vec<u32>` (or fixed buffer) of indices
where `x[i] == true`. Default is empty (= all zeros).

```rust
pub struct FeatureState {
    active: Vec<u32>,           // sorted, unique, len ≤ max_active_state
    scratch: Vec<u32>,          // merge buffer for apply_events
}
```

**Operations:**

- `apply_events(patch)` — for each event, insert or remove `index` in `active`
  (maintain sorted order via scratch merge or binary search + shift within cap).
- `contains(index) -> bool` — binary search on `active`.
- `evaluate_literal(index, negated) -> bool` — O(log k) or O(1) with small k.

**Alternative (v2):** roaring bitmap or word-block bitset when `k` is large but
still ≪ `d`. Start with sorted index list for simplicity and predictable bounds.

**Config:** `max_active_state` — max number of simultaneous 1-bits in state.
Exceeding → `SparseError::ActiveStateFull`.

### `ExpertIndex` (incremental evaluation)

Inverted index from feature index → expert slots that reference that literal.

```rust
pub struct ExpertIndex {
    /// For each feature index that appears in any stored expert literal,
    /// list of expert slot indices. Stored in a flat buffer + offsets,
    /// or `Vec<Vec<u16>>` built only at batch finalize / expert append (cold path).
    ///
    /// v1: rebuild on expert append/prune (F ≤ max_functions, cheap).
    /// v2: incremental maintenance on append/prune.
    by_feature: Vec<Vec<u16>>,
}
```

On `apply_events`, collect **dirty features** = indices touched in the patch.
**Candidate experts** = union of `by_feature[f]` for dirty `f`, plus always
re-evaluate if patch is empty (predict-only tick with no changes).

### `SparseBatchRow`

One observation in the expert-construction batch.

```rust
pub struct SparseBatchRow {
    /// Indices where x_t == true at this step (snapshot of FeatureState.active
    /// at update time, copied into batch storage).
    ones: Vec<u32>,   // len ≤ max_active_row
}
```

### `SparseBatchBuffer`

Replaces dense `batch_features: [bool; B * d]`.

```rust
pub struct SparseBatchBuffer {
    rows: Vec<SparseBatchRow>,       // len ≤ batch_size, fixed capacity at new
    signs: Vec<bool>,                // supervision bits, same as today
    len: usize,                      // current batch fill

    /// Union of all indices appearing in any row (including “stable 1” bits).
    active_indices: Vec<u32>,        // sorted unique, rebuilt on each append
    /// Column materialization cache for beam search: index → [bool; B]
    column_cache: ColumnCache,
}
```

**`active_indices` rebuild (on each `append`):**

```
active = sort_unique( ⋃ row.ones for row in rows[0..len] )
```

Only these indices get literal columns during batch finalize. A feature that is
0 in every row of the batch is correctly skipped (constant false column).

**Important:** include indices that are **stable 1** across rows (present in
`row.ones` even if they did not change on that tick). Snapshot `ones` from full
state at update time, not from the event patch alone.

### `ColumnCache`

Bounded workspace for sparse beam search. Replaces
`literal_columns: [bool; 2 * d * B]`.

```rust
pub struct ColumnCache {
    /// Maps feature index → slot 0..A-1 where A = active_indices.len()
    index_to_slot: HashMap<u32, u16>,  // cold path OK at batch finalize only

    /// Packed columns: positive literal then negative literal, each length B.
    /// Size: 2 * max_active_batch * batch_size bools (preallocated).
    literals: Vec<bool>,
    batch_size: usize,
}
```

For each `f` in `active_indices`, materialize column `b` as
`row[b].contains(f)` and negated column as `!that`. Feed columns into existing
`ConjunctionBuilder` via `SignBatch::from_columns` (already supports column
pointers in `batch.rs`).

### `SparseModelCapacity`

Replaces `ModelCapacity` fields that scale with `d`.

```rust
pub struct SparseModelCapacity {
    pub feature_count: u32,          // logical d (index space size)
    pub batch_size: usize,
    pub max_active_state: usize,     // cap k for FeatureState
    pub max_active_row: usize,       // cap per observation (usually = max_active_state)
    pub max_active_batch: usize,     // cap |active_indices| per batch window
    // ... same max_conjunctions, max_conjunction_length, max_functions, etc.
}
```

Memory is **O(max_active_batch * B + max_functions * ℓ)**, not **O(d * B)**.

### `SparseEnsemble<H>`

Parallel to `BooleanEnsemble<H>` in `ensemble.rs`:

```rust
pub struct SparseEnsemble<H: EnsembleHead> {
    config: EnsembleConfig,
    feature_count: u32,
    state: FeatureState,
    head: H,
    functions: Box<[ConjunctionExpert]>,
    expert_index: ExpertIndex,
    n_observed: usize,
    workspace: SparseEnsembleWorkspace,
    pending: bool,
    dirty_experts: Vec<u16>,   // scratch for incremental predict
}
```

Reuse `EnsembleHead` trait and head implementations from `ensemble.rs` unchanged.

## Hot-path logic

### `apply_events`

```
for (index, value) in patch:
    if value: state.insert(index)
    else:     state.remove(index)
mark dirty feature indices from patch
```

No expert evaluation here.

### `begin_predict` (incremental)

```
if pending: error

if patch was empty since last predict:
    candidate_experts = all active experts [0..F)
else:
    candidate_experts = union(expert_index.by_feature[f] for f in dirty_features)

for slot in candidate_experts:
    function_values[slot] = functions[slot].evaluate_sparse(&state)

for slot not in candidate_experts:
    reuse cached function_values[slot] from previous predict if state unchanged
    on those literals — OR simply re-evaluate all F if F is small (v1 fallback)

copy function_values → pending_function_values
pending = true
```

**v1 simplification:** if `F * ℓ_max < re_eval_threshold`, re-evaluate all
experts every predict (still avoids O(d) copy). Incremental index is opt-in when
profiling shows expert eval dominates.

Add `ConjunctionExpert::evaluate_sparse(&self, state: &FeatureState) -> bool`:
same short-circuit AND as `evaluate`, but `state.contains(index)` instead of
`features[index]`.

### `update`

Same as dense ensemble:

1. Validate target; require pending.
2. Compute batch sign from head + pending function values.
3. **Append** `SparseBatchRow { ones: state.active.clone() }` and sign.
4. Rebuild `active_indices` for the buffer.
5. Run `sgd_steps` head updates on pending function values.
6. If batch full → `finish_batch_sparse()`.

### `finish_batch_sparse`

```
build column_cache from sparse batch rows + active_indices
construct SignBatch::from_columns(column_ptrs, signs)
run ConjunctionBuilder::build_in_workspace (unchanged)
append/prune expert (unchanged)
rebuild expert_index
```

Beam search loops **only `active_indices`** instead of `0..d`. Constant-column
skips remain in `conjunction_builder.rs`.

## Python API (sketch)

```python
class BRegressorSparse(PredictUpdateState):
    def __init__(
        self,
        n_features: int,
        *,
        max_active_state: int = 256,
        max_active_batch: int = 512,
        **same_hyperparameters_as_BRegressor,
    ): ...

    def apply_events(self, events: np.ndarray) -> None:
        """events: shape (n, 2), dtype int32 — columns [index, value]."""

    def predict(self) -> float:
        """Uses internal state; no feature argument."""

    def update(self, target: float) -> None: ...
```

Classifier mirror with `predict() -> int`.

For benchmark compatibility:

```python
def dense_row_to_events(x: np.ndarray) -> np.ndarray:
    """Return events that set all 1-bits (for equivalence tests)."""
```

## Benchmarks and comparison plan

### 1. Correctness harness

New test module `tests/test_sparse_dense_equivalence.py`:

- Small `d` (32, same as default scenarios).
- Random dense rows → convert to events → compare trajectories over 10⁴ steps.
- Assert matching `function_count`, weights, predictions.

### 2. Sparse synthetic environment

Extend or parallel `synthetic_drifting_*` with:

- `n_features = d` large (e.g. 2¹⁶ logical index space).
- `active_per_step = k` sampled indices set to 1.
- Optional drift on which index subsets matter (mirrors paper DGP spirit at scale).

### 3. Micro-benchmarks

Report separately:

| Metric | Dense | Sparse |
|--------|-------|--------|
| predict latency | O(d + F·ℓ) | O(k log k + F'·ℓ) |
| update latency | O(d + S·F) | O(k + S·F) |
| batch finalize | O(d·B) | O(\|active\|·B) |
| RSS at rest | O(d·B) workspace | O(max_active·B) |

Sweep `(d, k)` with fixed hyperparameters.

### 4. CLI flag

```
--model dense|sparse --max-active-state K --max-active-batch Kb
```

Same scenario JSON; sparse path uses event generator.

## Phased rollout

### Phase 0 — scaffolding

- Add `docs/sparse-event-model-plan.md` (this file).
- Empty `sparse` module + `BRegressorSparse` stub returning `todo!()` behind
  feature flag `sparse` (default off).

### Phase 1 — state + equivalence

- `FeatureState`, `FeatureEvent`, `apply_events`.
- `evaluate_sparse` on `ConjunctionExpert`.
- `SparseEnsemble` with dense-equivalent path: re-evaluate all experts, sparse
  batch rows but **materialize all `d` columns** via state lookup (slow but
  correct). Validates learning parity before optimizations.

### Phase 2 — sparse batch finalize

- `SparseBatchBuffer`, `active_indices`, `ColumnCache`.
- Beam search over active set only.
- Dense equivalence tests at `d=32` and `d=4096`, small `k`.

### Phase 3 — incremental predict + Python

- `ExpertIndex`, dirty expert tracking.
- PyO3 bindings + Python wrappers.
- Benchmark CLI integration.

### Phase 4 — tuning

- Roaring bitmap option for large `k`.
- Incremental `ExpertIndex` maintenance.
- Document caps and error behavior for production.

## Configuration defaults (starting point)

| Parameter | Suggested default | Notes |
|-----------|-------------------|-------|
| `max_active_state` | 256 | max 1-bits in current state |
| `max_active_row` | 256 | same as state cap in v1 |
| `max_active_batch` | 512 | union size across batch window |
| `feature_count` | user-defined | logical index space `d` |

Reject observation if `state.active.len() > max_active_row` at update time.
Reject batch finalize if `active_indices.len() > max_active_batch` (or truncate
with logged warning — prefer hard error for determinism in v1).

## Edge cases

| Case | Behavior |
|------|----------|
| Empty patch, `predict` | Valid; uses current state |
| Event sets unknown index ≥ `d` | `InvalidInput` |
| Duplicate indices in patch | Last wins |
| All-zero state | Valid; conjunctions use negative literals only |
| `max_active_*` exceeded | `SparseError` variant, no allocation |
| Expert prune / append | Rebuild `ExpertIndex` (v1) |

## Open questions

1. **Explicit unset events** — required if state persists across ticks and upstream
   only sends deltas. v1: yes, support `(i, false)`.
2. **Session reset** — `reset_state()` vs full model reset; needed for bounded
   `FeatureState` in long runs.
3. **Feature flag vs separate types** — prefer separate public types
   (`BRegressorSparse`) over changing existing constructors.
4. **Hash collisions** — document that index semantics are opaque; collision
   handling stays upstream of Binaml.

## Success criteria

- Bit-identical learning to dense model on small-`d` equivalence tests.
- At `d ≥ 10⁴`, `k ≤ 256`: predict+update p99 latency lower than dense by ≥10×
  (target, to validate with benchmarks).
- Fixed memory independent of `d` given configured sparsity caps.
- Side-by-side benchmark reports in repo without removing dense baselines.
