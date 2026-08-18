# Runtime memory and preallocation plan

Plan for making Binaml **fully memory-bounded at construction**: zero heap
growth on every path (`predict`, `update`, and expert batch build). Targets
`BooleanEnsemble`, the regression/classification heads, `FunctionGraph`,
and `FunctionBuilder` in `crates/binaml-core`.

Notation matches `papers/binaml/main.tex` § Notation and the Rust/Python APIs.
In this document, paper symbols are written `K_max`, `N_max`, etc. (LaTeX:
`K_{\max}`, `N_{\max}`, …).

## Current implementation (narrow builder, still allocating)

`FunctionBuilder` lives in `function_builder.rs` with shared helpers in
`function_build_common.rs`. The builder is **narrow** (`K_p` nodes per layer),
not wide (`P` pairs kept per layer). Several optimizations are in place; full
preallocation at `BooleanEnsemble::new` is still outstanding.

| Piece | Location | Status |
|-------|----------|--------|
| **Top-`K_p` per layer** | `function_builder.rs` | Layer 0 trimmed to top-`K_p` sources by \|association\|. Each composed layer evaluates all `P` parent pairs but **keeps only top-`K_p`** candidates by \|association\|. |
| **`ColumnCache`** | `workspace.rs` (`FlatColumnCache`) | Fixed `2·K_p` column slots × `B` in `BuildWorkspace`; **`retain_only`** after each layer. Legacy heap `ColumnCache` remains in `function_build_common.rs` for tests/helpers. |
| **`PairCounterScratch`** | `function_build_common.rs` | Pre-sized once to **`P = K_p(K_p−1)/2`** `FeatureCounter` slots; reused each layer during pair scoring. |
| **Pair scoring** | `binary_truth_table.rs` | `from_columns` + `truth_table_and_scores` on cached parent columns. Constant truth tables (`0b0000`, `0b1111`) and constant parent columns are **skipped**. Composed column materialized only for kept nodes. |
| **Output selection** | `select_output_top_k_by_association` | Among **surviving** nodes, take top-`K_p` by \|association\|, then pick best **accuracy** (with invert). Not global over all ephemeral nodes. |
| **`L_build` cap** | `FunctionBuildConfig::max_composed_layers` | Derived from `DEFAULT_MAX_EXPERT_NODES` (64), `d`, and `K_p` via `derive_build_capacity`. Hard memory ceiling on composed layers. |
| **`l_pat` stop** | `FunctionBuildConfig::l_pat` | Public patience: stop after `l_pat` consecutive composed layers with no global in-sample accuracy improvement (`DEFAULT_L_PAT = 2`). |
| **Graph / scores** | `function_builder.rs` | Preallocated workspace buffers sized by **`V`**; builder loop bounded by **`l_pat`** and **`L_build`**. |
| **Predict** | `FunctionGraph::eval` | Preallocated `eval_scratch: [bool; N_max]` on ensemble hot path; standalone `predict_model` still uses heap `HashMap` for tests. |

**Target mapping:** replace heap pools with fixed workspace sized from `K_p`,
`L_build`, and `N_max` (see layouts below). Algorithm semantics above should
be preserved; only storage moves into `EnsembleWorkspace`.

## Notation (paper ↔ code)

| Symbol | Meaning | Rust (proposed / existing) | Python (proposed / existing) |
|--------|---------|---------------------------|------------------------------|
| `d` | Input width | `source_feature_count` | `n_features` |
| `B` | Expert batch size | `batch_size` | `batch_size` |
| `S` | SGD steps per observation | `sgd_steps` | `sgd_steps` |
| η | Learning rate | `learning_rate` | `learning_rate` |
| λ | L2 decay | `l2` | `l2` |
| `K_p` | Per-layer width cap | `parent_top_k` | `parent_top_k` |
| `L_build` | Max composed layers (derived, memory) | `max_composed_layers` in `FunctionBuildConfig` | — |
| `l_pat` | Layer patience (runtime early stop) | `l_pat` in `FunctionBuildConfig` / `EnsembleConfig` | `l_pat` |
| `K_max` | Ensemble capacity | `max_functions` | `max_functions` |
| `N_max` | Max compact nodes per stored expert | `max_expert_nodes` | `max_expert_nodes` |
| `P` | Pair **scratch** slots per layer | `K_p*(K_p-1)/2` | — |
| `V` | Max ephemeral **graph** nodes | `K_p.min(d) + L_build*K_p + 1` | — |
| `C` | Number of classes | `n_classes` | `n_classes` |
| `F`, `F_t` | Current expert count | `function_count()` | `function_count()` |
| `N_k` | Compact nodes in expert k (`N_k` ≤ `N_max`) | `FunctionGraph::node_count()` | — |

Do **not** use bare `K` for multiple meanings: class count is `C`; layer width
is `K_p`; experts are `F` capped by `K_max`.

## Config simplification

Build graphs and build workspace are **narrow** (`K_p` wide); stored experts are
**narrower still** (`N_k` ≤ `N_max` ≪ `V` after compaction). **`max_expert_nodes`**
(`N_max`) is the new memory cap on stored experts. **`l_pat`** is the restored
public patience for batch graph growth. Build pool size **`V`**, pair scratch
**`P`**, and layer cap **`L_build`** are **derived** from `N_max`, `K_p`, and
`d` at `BooleanEnsemble::new`.

**Policy:**

| | User-facing? | Role |
|--|--------------|------|
| `max_expert_nodes` (`N_max`) | **Yes** | Caps stored `CompactNode` count, inference scratch, and (via derivation) build workspace |
| `l_pat` | **Yes** | Stop after `l_pat` consecutive composed layers without global accuracy improvement |
| `parent_top_k` (`K_p`) | **Yes** (existing) | Caps nodes **kept** per layer; sizes parent pool, pair scratch, and column cache |
| `P`, `V`, `L_build` | **No** — derived | Preallocated builder buffers |

### What `N_max` counts (compacted graph)

After `compact()`, a stored expert is a `FunctionGraph`:

- **`nodes`**: topological list of **`CompactNode`** records — this is what
  `node_count()` returns and what **`N_max` caps**.
- **`source_indices`**: raw input coordinates used by `CompactNode::Source`
  entries (`source_count()` ≤ `d`; not a separate cap).

Each entry in `nodes` is exactly one of:

| `CompactNode` | Counted in `N_max`? |
|---------------|---------------------|
| `Source(local_idx)` | Yes — one node per referenced source in the backward closure |
| `Constant(bool)` | Yes — rare after simplify (builder avoids constant gates at source) |
| `Composed { first, second, truth_table }` | Yes — one per gate on paths to the output |

Compaction keeps only **backward reachability** from the selected output on the
ephemeral graph, then simplifies (alias, constant fold). Sibling candidates from
the same builder layer that are **not** ancestors of the output are dropped — so
`N_k` ≪ `V` in typical runs.

**Not** counted toward `N_max`: ephemeral builder nodes that were never on the
output path, `source_indices` metadata beyond the `Source` nodes already in
`nodes`, ensemble weights, batch buffers.

### Derivation (from `N_max` and `K_p`)

Per composed layer the builder:

1. Takes the previous layer (already ≤ `K_p` nodes) as parents.
2. Evaluates all **`P = K_p(K_p−1)/2`** distinct parent pairs (pair scratch).
3. Filters constant parents / constant truth tables.
4. Keeps top-**`K_p`** candidates by \|association\| as the new layer.

So **`P`** sizes pair-evaluation scratch; **`K_p`** sizes layer width and
active column cache. Ephemeral graph node count is bounded by:

```text
V = K_p.min(d) + L_build * K_p + 1
```

(path worst case: `min(d, K_p)` non-constant sources + one composed chain of
length `L_build` with up to `K_p` siblings per layer retained in the ephemeral
graph).

#### `L_build` from `N_max`

**Recommended (valid worst case over compact shapes):**

```text
L_build = max(1, N_max.saturating_sub(K_p.min(d)))
```

A compact expert with `N_max` nodes on a single path needs at least that many
composed layers once source literals are capped at `K_p`. The older
`N_max − d` formula remains safe when `d > K_p` but over-estimates
`L_build`.

Then at construction:

```text
validate: N_max >= 1, K_p >= 2, N_max <= V - 1
```

Example: `N_max = 64`, `K_p = 8`, `d = 32` → `P = 28`, `L_build = 56`,
`V = 8 + 56×8 + 1 = 457`. (Tighter than the pre–top-k formula
`d + L_build·P + 1 = 929`.)

**Roles split:**

| Parameter | Meaning |
|-----------|---------|
| `L_build` | Hard max composed layers (memory derivation + safety ceiling) |
| `l_pat` | Runtime patience: max consecutive composed layers without global accuracy gain |

Builder stops when **any** of: `layers_without_improvement` ≥
`l_pat`, composed layers ≥ `L_build`, `build_node_len` ≥ `V`, batch accuracy
perfect, fewer than two parents, or no valid pair candidates.

After each composed layer, compare the global max of `accuracy_scores` to the
running best; reset the patience counter on improvement, else increment it.

```rust
fn derive_build_capacity(d: usize, k_p: usize, n_max: usize) -> (usize, usize, usize) {
    let pairs = k_p.saturating_sub(1) * k_p / 2;
    let source_cap = k_p.min(d);
    let l_build = n_max.saturating_sub(source_cap).max(1);
    let v = source_cap + l_build * k_p + 1;
    (l_build, pairs, v)
}
```

## Build vs compact bounds (why one public cap suffices)

```text
FunctionBuilder  →  EphemeralGraph (≤ V nodes)  →  compact()  →  FunctionGraph (≤ N_max)
                         │
                         ├─ ColumnCache (active): ≤ K_p columns × B  [retain_only per layer]
                         └─ PairCounterScratch: P counters (fixed at init)
```

Always `N_k` ≤ |reachable| ≤ `V`. Inference memory scales with `K_max` ×
`N_max`. Build **hot** workspace scales with **`K_p` × B** (columns) + **`P`**
(pair counters), not `V × B`. Graph metadata scales with **`V`**.

| Event | Cap |
|-------|-----|
| Layer push | `\|layer\| ≤ K_p`; `build_node_len < V` |
| Pair eval | scratch index `< P` |
| Compaction result | `n_nodes` ≤ `N_max` or `ExpertTooLarge` |
| Expert eval | scratch length `N_max` |

## Goals

- **Fixed resident memory** at model construction from
  (`d`, `B`, `K_max`, `N_max`, `K_p`, `l_pat`, `C`) — builder size derived from
  `N_max` + `K_p`; layer depth bounded at runtime by `l_pat` and at construction
  by `L_build`.
- **Zero heap allocation** after `new` on all paths.
- **Public build controls:** `max_expert_nodes` (memory) and `l_pat` (patience).
- **Dense weights**, fixed expert slots, preallocated builder workspace of
  derived size (`V`, `P`, `K_p`).

Non-goals: SIMD/GPU; zero-copy Python input (separate plan).

## Existing configuration

| Parameter | Symbol | Notes |
|-----------|--------|-------|
| `max_functions` | `K_max` | unchanged |
| `batch_size` | `B` | unchanged |
| `n_features` / `source_feature_count` | `d` | unchanged |
| `parent_top_k` | `K_p` | caps layer width, pair scratch, column cache, and output candidate pool |
| `l_pat` | `l_pat` | consecutive composed layers without accuracy improvement before early stop (default 2) |
| `n_classes` | `C` | classifier only |
| **`max_expert_nodes`** | `N_max` | **new** |

## Suggested default for `max_expert_nodes`

Measure compacted node counts on versioned benchmark scenarios; pick a default
with margin (e.g. 64 for `d=32`, `K_p=8`). Check derived capacity:

Example: `N_max = 64`, `K_p = 8`, `d = 32` → `P = 28`, `L_build = 56`,
`V = 457`.

## Proposed layouts

### `EnsembleWorkspace` (allocated once; derived caps)

```text
// Hot path
function_values:     [bool; K_max]
pending_features:    [bool; d]
pending_activations: [bool; K_max]
eval_scratch:        [bool; N_max]
logits:              [f64; C]               // classifier
probabilities:       [f64; C]
batch_features:      [bool; B * d]
batch_signs:         [bool; B]
batch_len:           usize

// Batch build
build_nodes:         [EphemeralNode; V]
build_assoc:         [i64; V]
build_accuracy:      [u8; V]
build_columns:       [bool; 2 * K_p * B]   // parent + current layer peak; or K_p*B with in-place reuse
build_column_ready:  [u8; ceil(2*K_p/8)]
build_layer_ends:    [u16; L_build + 1]    // layer i spans build_nodes[ends[i]..ends[i+1]), each width ≤ K_p
build_parent_buf:    [BuildNodeId; K_p]
build_pair_scratch:  [FeatureCounter; P]   // matches PairCounterScratch today
build_pair_candidates: [PairCandidate; P]  // optional; or stack buffer during sort
compact_nodes:       [CompactNode; N_max]
compact_sources:     [usize; d]
```

`build_columns` + `build_column_ready` replace the heap `ColumnCache`.
`build_pair_scratch` replaces `PairCounterScratch`'s `Vec<FeatureCounter>`.
During pair eval, read two parent rows from `build_columns`, write into
`build_pair_scratch[i]`, score with `truth_table_and_scores` (already implemented).

(See `derive_build_capacity` above.)

### Stored expert slot

```rust
struct FunctionGraph {
    source_indices: Box<[usize]>,  // cap d
    nodes: Box<[CompactNode]>,     // cap N_max
    n_nodes: u16,
    output: u16,
}
```

`K_max` slots preallocated at model construction.

### Heads

Dense `weights`: length `K_max` (regression) or `K_max · C` (classification,
expert-major).

## Builder changes (with simplified config)

1. **Memory layer budget:** never exceed derived `L_build`.
2. **Patience layer budget (done):** stop after `l_pat` consecutive layers with
   no global accuracy improvement.
3. **Width budget:** at most **`K_p`** nodes per layer (already implemented).
4. **Pair scratch:** at most **`P`** counters per layer (already implemented).
5. **Node budget:** stop before `build_node_len == V`.
6. **Compact:** error if `n_nodes > N_max`.
7. **Constant gate (done):** skip constant source columns, constant parent
   columns, and truth tables `0b0000` / `0b1111`. Unary literals (varying
   single source) remain valid.
8. **Output (done):** top-`K_p` by association, then best accuracy among
   survivors.
9. **Layer cap (done):** derive `L_build` via
   `FunctionBuildConfig::new` / `derive_build_capacity`.
10. **Column cache (done):** flat `build_columns[2·K_p·B]` + ready flags in
    workspace (replaces heap `ColumnCache`).
11. **Predict / eval:** replace `node_column` + `HashMap` with
    `eval_scratch: [bool; N_max]`.
12. Replace remaining allocating `Vec` paths at end of `build_in_workspace`
    (`nodes.to_vec()`, `layers.clone()`) where they affect post-`new` growth.

## Implementation phases

### Phase 0 — Config

- Add **`max_expert_nodes`** and **`l_pat`** (Rust, Python, PyO3, paper §
  Notation, benchmarks).
- Add `derive_build_capacity(d, k_p, n_max)`; validate at `new` (`l_pat > 0`,
  `N_max < V`, etc.).
- Enforce `L_build` and `V` in builder; wire `l_pat` patience in the composed-layer loop.

### Phase 1 — Dense weights + fixed expert slots (`N_max`)

### Phase 2 — Hot-path workspace

### Phase 3 — Flat batch buffer

### Phase 4 — Preallocated builder

- [x] **Top-`K_p` per layer** — evaluate `P` pairs, keep `K_p` nodes.
- [x] **`PairCounterScratch`** — fixed `P`-sized counter pool per build.
- [x] **Column-cache semantics** — lazy columns, parent reuse, single-layer
  `retain_only`, stack scoring via `truth_table_and_scores`.
- [x] **Constant gate** — no constant sources, constant gates, or constant
  parents in pair eval.
- [x] **Association-constrained output** — `select_output_top_k_by_association`.
- [x] **Layer cap** — derive `L_build`
  (`FunctionBuildConfig::new`, `derive_build_capacity`, `DEFAULT_MAX_EXPERT_NODES`).
- [x] **`l_pat` patience** — early stop after consecutive non-improving layers
  (`DEFAULT_L_PAT = 2`; paper benchmarks use `l_pat = 5`).
- [x] Flat `build_columns[2·K_p·B]` + ready flags (drop `ColumnCache` heap).
- [x] Fixed `build_nodes`, `build_assoc`, `build_accuracy`, layer index buffers.
- [x] Wire `FunctionBuilder::build` to workspace instead of local `Vec`s.

### Phase 5 — Tests, benchmarks, paper memory §

## Memory budget

| Component | Bytes |
|-----------|-------|
| Regression weights | 8·K_max |
| Classification weights + intercepts | 8·C·(1 + K_max) |
| Expert slots | K_max·(8d + 24·N_max) |
| Hot workspace | K_max + 2d + N_max + 2C + Bd + B |
| Builder graph pool | V·(24 + 8) + O(L_build) |
| Builder column cache | 2·K_p·B + ceil(2·K_p/8) |
| Builder pair scratch | 8·P |

`P = K_p(K_p−1)/2`, `L_build = max(1, N_max − K_p.min(d))`,
`V = K_p.min(d) + L_build·K_p + 1`.

Transient allocations from end-of-build `EphemeralGraph` materialization
(`nodes.to_vec()`, `layers.clone()`) are what Phase 4 still removes.

## Paper updates

Add to § Notation in `main.tex` (LaTeX):

```tex
\item[$N_{\max}$] Max compact nodes per stored expert
  (\texttt{max\_expert\_nodes}): length of \texttt{FunctionGraph.nodes}.
  Per-layer width $K_p$ (\texttt{parent\_top\_k}), pair scratch
  $P=\binom{K_p}{2}$, layer cap
  $L_{\mathrm{build}}=\max(1,N_{\max}-K_p\wedge d)$, graph pool
  $V=(K_p\wedge d)+L_{\mathrm{build}}K_p+1$ derived at construction.
\item[$l_{\mathrm{pat}}$] Layer patience (\texttt{l\_pat}): stop after
  $l_{\mathrm{pat}}$ consecutive composed layers with no in-sample accuracy
  improvement (default $2$; paper benchmarks use $5$).
```

## Success criteria

- [x] Public caps: `max_expert_nodes` and `l_pat`.
- [x] `V`, `P`, `L_build` derived; `l_pat` is the runtime early-stop control.
- [x] `N_max < V` and `l_pat > 0` validated at construction.
- [x] Column cache fixed at `O(K_p·B)`; pair scratch fixed at `P` (no heap growth).
- [x] Predict / expert eval use preallocated scratch (no per-call `HashMap` / `Vec`).
- [ ] Zero alloc after `new`; tests green.
