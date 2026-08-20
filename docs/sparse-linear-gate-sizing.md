# Sparse linear gate: representation & ops cost by fan-in

Reference for the **fixed-topology, bias-free sparse linear gate** used in
[boolean-circuit-exp-plan.md](./boolean-circuit-exp-plan.md).

Each gate has **`k` binary parents**. This doc covers two **`k = 2`** layouts:

| Layout | Weights | Forward | Used in |
|--------|---------|---------|---------|
| **Per wire** | **`w_a0, w_a1, w_b0, w_b1`** — two per parent | **`sum = w_a[a] + w_b[b]`**, threshold | general fan-in tables below |
| **Per pair (v1)** | **`w_00 … w_11`** — one per **`(sign(a),sign(b))`** | **`out = w[lane]`** (`i8`) | [boolean-circuit-exp-plan.md](./boolean-circuit-exp-plan.md) |

Both pack **4 × `i8`** in one **`u32`** (zero padding). The experiment plan uses
**per pair**; tables below keep **per wire** for **`k > 2`** unless noted.

**Per wire:** each parent stores **`w0, w1` as `i8`**. Forward: sum active weights,
threshold `sum > 0` → gate output. Update: nudge the **`k` active weights** on mistake.

**Per pair (`k = 2`, exp plan):** one weight per sign-row; parent **`i8`** activations
carry confidence; **`sign(x) = (x >= 0)`** (**128** values per side); row =
**`2·sign(a)+sign(b)`**; output **is** the selected weight.
On mismatch, score weight **`±1`** and parent **pole (`+127`/`−128`)** moves via
**`input_cost + |w[s]−T|`**; parent target uses **`pole(sign)`** from the selected row
(see exp plan).

Assumptions unless noted:

- **`G`** gates, **`N = d + G`** total nodes (sources + gates).
- Parent indices **`u8`** when **`N ≤ 256`**, else **`u16`** (tables below use `u8`).
- Node **`i8` activations** and **`i8` targets** (exp plan); per-wire tables below
  still assume bitpacked bool reads for generic **`k`**.
- Weights clamped to **`±127`** (full `i8`).
- Sink-supervised: stream **`y: bool` → targets[sink] = ±1`**; gates compare full
  **`i8` activation** to **`i8` target** on update (see exp plan).

**Weight bytes (per wire)** = **`2k`**. **Weight bytes (per pair, `k = 2`)** = **`2^k = 4`**.
**Parent index bytes** = **`k`** (with `u8` indices).

---

## Summary table (per gate, per-wire layout)

For **`k = 2` per-pair (experiment plan):** same **4 B / `u32`**, **1 sel** on predict
(gate output = selected **`i8`**), **`active_row: u8`** scratch. Observe: score **2**
weight **`±1`** + **4** parent-pole moves via **`total(s)`** — see
[Per-pair `k = 2`](#per-pair-k--2-boolean-circuit-exp) below.

| Fan-in `k` | Weight bytes | Weight pack (recommended) | Parent bytes (`u8`) | Scratch bytes | Output range | Predict ops | Observe ops (on error) |
|------------|--------------|----------------------------|---------------------|---------------|--------------|-------------|------------------------|
| 2 (per wire) | 4 | `u32` | 2 | 3 | bool | 2 sel + 1 add + 1 cmp + 1 bit write | 1 cmp + 2 × `i8` sat add |
| 2 (per pair, exp) | 4 | `u32` | 2 | 4 | `i8` | 2 sign + 1 sel + 1 store | 1 cmp + move eval + 1 update |
| 3 | 6 | `u64` (6 used) | 3 | 3 | bool | 3 sel + 2 add + 1 cmp + 1 bit write | 1 cmp + 3 × `i8` sat add |
| 4 | 8 | `u64` | 4 | 3 | bool | 4 sel + 3 add + 1 cmp + 1 bit write | 1 cmp + 4 × `i8` sat add |
| 5 | 10 | `u128` (10 used) | 5 | 3 | bool | 5 sel + 4 add + 1 cmp + 1 bit write | 1 cmp + 5 × `i8` sat add |
| 6 | 12 | `u128` (12 used) | 6 | 3 | bool | 6 sel + 5 add + 1 cmp + 1 bit write | 1 cmp + 6 × `i8` sat add |
| 7 | 14 | `u128` (14 used) | 7 | 3 | bool | 7 sel + 6 add + 1 cmp + 1 bit write | 1 cmp + 7 × `i8` sat add |
| 8 | 16 | `u128` | 8 | 3 | bool | 8 sel + 7 add + 1 cmp + 1 bit write | 1 cmp + 8 × `i8` sat add |
| 9 | 18 | `u128` + `u16` (2 loads) | 9 | 4 | bool | 9 sel + 8 add + 1 cmp + 1 bit write | 1 cmp + 9 × `i8` sat add |
| 10 | 20 | `u128` + `u32` (2 loads) | 10 | 4 | bool | 10 sel + 9 add + 1 cmp + 1 bit write | 1 cmp + 10 × `i8` sat add |
| 11 | 22 | `u128` + `u64` (22 used, 2 loads) | 11 | 4 | bool | 11 sel + 10 add + 1 cmp + 1 bit write | 1 cmp + 11 × `i8` sat add |
| 12 | 24 | `u128` × 2 (2 loads) | 12 | 4 | bool | 12 sel + 11 add + 1 cmp + 1 bit write | 1 cmp + 12 × `i8` sat add |
| 13 | 26 | `u128` × 2 (26 used, 2 loads) | 13 | 4 | bool | 13 sel + 12 add + 1 cmp + 1 bit write | 1 cmp + 13 × `i8` sat add |
| 14 | 28 | `u128` × 2 (28 used, 2 loads) | 14 | 4 | bool | 14 sel + 13 add + 1 cmp + 1 bit write | 1 cmp + 14 × `i8` sat add |
| 15 | 30 | `u128` × 2 (30 used, 2 loads) | 15 | 4 | bool | 15 sel + 14 add + 1 cmp + 1 bit write | 1 cmp + 15 × `i8` sat add |
| 16 | 32 | `u128` × 2 (2 loads) | 16 | 4 | bool | 16 sel + 15 add + 1 cmp + 1 bit write | 1 cmp + 16 × `i8` sat add |

**Scratch (observe):** `sum: i16` (2 B) + **`active` bit pattern** (1 B if `k ≤ 8`,
else 2 B as `u16`) storing which **`k` weights** were selected. Optional: omit
`active` and re-read parent bits from `values` during update.

**Predict ops** include **`k` parent bit reads** from packed node values (not
counted again in the sel column above).

**Observe ops (per wire)** are **on mistake only** (gate output ≠ target): one compare plus
`k` saturating `i8` adds. If the gate already matches, skip weight updates.

**Scratch (per pair exp):** `active_row: u8` (1 B/gate) + parent activations read
from **`Box<[i8]>`** (not gate-local).

**Observe ops (per pair, exp)** on mismatch: compare **`i8`** activation vs target;
evaluate **`total(s)`** for four rows and up to six moves (2 weight + 4 pole); one update.

**Full step (`predict` + `observe`, worst case — every gate wrong, per-wire layout):**

```text
O(G) × [ k bit reads + k weight selects + (k−1) adds + 1 cmp + 1 bit write   // predict
         + 1 cmp + k saturating i8 adds ]                                     // observe
```

Serial **depth** of the DAG still dominates wall time when `G` is large and the
graph is deep; fan-in raises **per-gate constant factor** linearly in `k`.

---

## Memory waste vs operational sweet spots

### Weight packing waste (per gate)

Actual payload is always **`2k` bytes**. Padding appears when the pack chunk is
rounded up to **`u32` / `u64` / `u128`** boundaries.

| `k` | Payload | Pack size | Padding | Waste % | Loads |
|-----|---------|-----------|---------|---------|-------|
| 2 | 4 B | `u32` (4) | 0 | **0%** | 1 |
| 3 | 6 B | `u64` (8) | 2 B | **25%** | 1 |
| 4 | 8 B | `u64` (8) | 0 | **0%** | 1 |
| 5 | 10 B | `u128` (16) | 6 B | **37.5%** | 1 |
| 6 | 12 B | `u128` (16) | 4 B | **25%** | 1 |
| 7 | 14 B | `u128` (16) | 2 B | **12.5%** | 1 |
| 8 | 16 B | `u128` (16) | 0 | **0%** | 1 |
| 9 | 18 B | 16+2 | 0* | 0%* | **2** |
| 10 | 20 B | 16+4 | 0* | 0%* | **2** |
| 11 | 22 B | 16+8 | 2 B† | ~8%† | **2** |
| 12 | 24 B | 32 | 8 B | **25%** | **2** |
| 13 | 26 B | 32 | 6 B | **18.8%** | **2** |
| 14 | 28 B | 32 | 4 B | **12.5%** | **2** |
| 15 | 30 B | 32 | 2 B | **6.3%** | **2** |
| 16 | 32 B | 32 | 0 | **0%** | **2** |

\*Tight tail struct; in an array, **`{ u128, u16 }` may align to 24–32 B** per
element — profile before assuming 18 B stride.

†`tail: u64` uses 6 of 8 bytes.

**Worst padding (single-load tier):** **`k = 5`** (37.5%). **Avoid `k = 3, 5, 6`
if memory matters**; prefer **`k = 4`** over **`k = 3`**, and **`k = 8`** over
**`k = 5..7`**.

**Zero-waste fan-ins (exact fill):** **`k ∈ {2, 4, 8, 16}`** — natural word boundaries.

At **`G = 256`**, worst case **`k = 5`**: **~1.5 KiB** padding in weights alone
(`256 × 6` B).

**No padding alternative:** `Box<[i8; 2*k]>` per gate — **0% weight waste**, but
**`k` loads** (or one vector load only if contiguous and compiler vectorizes).
Usually worse for **`k ≥ 4`** on predict.

### Node-value bitpack waste (depends on `N = d + G`, not `k`)

| `N` range | Storage | Unused bits (worst case) |
|-----------|---------|--------------------------|
| 1–64 | 1 × `u64` | `64 − N` (e.g. **`N = 17` → 47 bits idle**) |
| 65–128 | 2 × `u64` | `128 − N` |
| 129–256 | 4 × `u64` | `256 − N` |

Pick **`d + G`** near **64, 128, or 256** to minimize idle bits. **`N ≤ 64`**
also keeps all parent reads in **one `u64`** (simplest bit extract).

### Scratch waste

| `k` | Scratch | Note |
|-----|---------|------|
| 2–8 | 3 B/gate | `active` fits in **`u8`** (only **`k` bits** used) |
| 9–16 | 4 B/gate | needs **`u16`** for parent bit pattern |

Jump at **`k = 9`** costs **`+1 B × G`** (e.g. **+256 B** at `G = 256`) — minor.

### Where operations are simplest / cheapest

**Cheapest overall (per wire):** **`k = 2`**

- Fewest selects/adds/updates among per-wire layouts (see summary table).
- One **`u32`** load; four weights in one GPR.
- Can fuse select+add with **`cmov`** / **`(1-a)*w0 + a*w1`** (no loop).
- **`active`**: 2 bits in scratch.

**Cheapest predict (per pair, experiment):** **`k = 2`**

- Same **`u32`** footprint as per-wire **`k = 2`**.
- Two sign tests + one lane select; output written as **`i8`** (no threshold).
- Node state **`2 × N`** bytes (**`activations + targets`**) vs bitpacked bool — simpler
  observe, slightly more RAM at large **`N`**.

**Best single-load balance:** **`k = 4`** (per wire)

- **`u64`** fills exactly — **no pad waste** (unlike **`k = 3`**).
- Still small op count (4 sel + 3 add).
- Natural on 64-bit hosts (one load = one register).

**Best wide single-load:** **`k = 8`**

- **`u128`** fills exactly — **no pad waste** (unlike **`k = 5..7`**).
- Last fan-in with **one load** and **3 B scratch**.

**First fan-in with extra load + higher scratch:** **`k = 9`**

- **2 memory loads** per gate on predict.
- Scratch **3 → 4 B**; **`k`** adds/selects step up linearly.

**Power-of-two rule:** **`k = 2, 4, 8, 16`** align payload, pack size, and (for
8/16) unroll-friendly loop counts. Prefer these over **`k ± 1`**.

### Quick decision guide

| Goal | Prefer | Avoid |
|------|--------|-------|
| Min predict/observe ops (per wire) | **`k = 2`** | **`k ≥ 9`** (2 loads + large scratch) |
| Min predict ops (exp plan) | **`k = 2` per pair** | per-wire dual select + add |
| Min padding, 1 load | **`k = 4` or `k = 8`** | **`k = 3, 5, 6`** |
| Min padding, any `k` | **`k ∈ {2,4,8,16}`** | **`k = 5`** (worst %) |
| Min node-value waste | **`N ≈ 64, 128, 256`** | **`N = 65`** (second `u64` mostly empty) |
| Simplest codegen | **`k = 2`** fixed, or **`[i8; 2k]`** uniform | Mixed pack per `k` |

**Practical default unchanged:** **`k = 2`** for speed, **`k = 4`** if you need
wider gates without crossing into the **2-load** regime.

---

## Weight packing detail

Store weights contiguously per gate, little-endian `i8` lanes in integer chunks
so predict does **1–2 loads** per gate.

### Per wire (general `k`)

| `k` | Layout | Rust sketch |
|-----|--------|-------------|
| 2 | 4 B | `weights: Box<[u32]>` |
| 3–4 | ≤ 8 B | `weights: Box<[u64]>` |
| 5–8 | ≤ 16 B | `weights: Box<[u128]>` |
| 9 | 18 B | `{ lo: u128, tail: u16 }` (2 loads) |
| 10 | 20 B | `{ lo: u128, tail: u32 }` |
| 11 | 22 B | `{ lo: u128, tail: u64 }` (6 B used) |
| 12–16 | ≤ 32 B | `{ lo: u128, hi: u128 }` or `Box<[u8; 2*k]>` |

Extract weight for parent `j` with value `b ∈ {0,1}` (per wire):

```text
lane = 2*j + b
w    = (((packed >> (8*lane)) & 0xFF) as i8)   // mask before cast → correct sign
```

### Per-pair `k = 2` (boolean-circuit-exp)

| Layout | Rust sketch |
|--------|-------------|
| 4 B (rows `00,01,10,11`) | `weights: Box<[u32]>` |

Extract weight for parent activations (per pair, exp):

```text
lane = 2*(act_a >= 0) + (act_b >= 0)
w    = (((packed >> (8*lane)) & 0xFF) as i8)
out  = w                              // gate i8 activation
```

Forward: **`out = w`**. Observe: minimal-cost weight **`±1`** and/or parent **`pole(sign)`**
**`∈ {+127, −128}`** from **`argmin total(s)`** (see exp plan).

On targets without cheap `u128` (some embedded), use `Box<[i8; 2*k]>` for all
`k` — same ops, more loads.

For `k ≤ 8`, unrolling all selects in one register is usually faster than a loop.

**Alternative (uniform, any `k`):** `weights: Box<[i8]>` with stride **`2k`**, no
packing — simpler codegen, extra bandwidth for large `k`. Prefer packing for
`k ≤ 8`.

---

## Parent index layout

SoA (gate-major loop):

```rust
// k = 4 example
parent: [[u8; 4]; G]   // or parent0..parent3: Box<[u8]>
```

| `k` | Bytes/gate (`u8`) | Bytes/gate (`u16`) |
|-----|-------------------|---------------------|
| 2 | 2 | 4 |
| 4 | 4 | 8 |
| 8 | 8 | 16 |
| 16 | 16 | 32 |

Gates must be stored in **topological order**; parents always refer to lower
indices.

---

## Node values (exp plan vs generic)

| Layout | Representation |
|--------|----------------|
| **Exp plan** | **`activations: Box<[i8]>`**, **`targets: Box<[i8]>`** — **`2N`** bytes |
| Generic / per-wire tables | bitpacked **`u64`** (bool), see below |

Generic bitpacked (per-wire reference only):

| `N` | Representation |
|-----|----------------|
| ≤ 64 | single `u64` |
| ≤ 128 | `[u64; 2]` |
| ≤ 256 | `[u64; 4]` |
| > 256 | `⌈N/64⌉` × `u64` |

Parent `p` bool value: `(values[p/64] >> (p%64)) & 1`.

---

## Memory at scale (`G = 256` gates)

Weights = `256 × 2k` B; parents (`u8`) = `256 × k` B; scratch = `256 × (3 or 4)` B.

| Fan-in `k` | Weights only | Weights + parents (`u8`) | + scratch (one step) |
|------------|--------------|--------------------------|----------------------|
| 2 (per wire or per pair) | 1.0 KiB | 1.5 KiB | +768 B (3 B/gate) |
| 4 | 2.0 KiB | 3.0 KiB | +768 B |
| 8 | 4.0 KiB | 6.0 KiB | +768 B |
| 16 | 8.0 KiB | 12.0 KiB | +1.0 KiB (4 B/gate) |

Memory is **not** the limiting factor for any `k ≤ 16` at experiment scale.

---

## Accumulator width

If all active weights are **`+127`**, max sum = **`127 × k`**. With mixed signs,
`|sum| ≤ 127 × k` still holds. For `k = 16` → **2032** ≪ **`i16::MAX` (32767)**.

| Range | Accumulator |
|-------|-------------|
| `k = 2` per pair (exp) | **`i8`** (direct) |
| `k = 2` per wire (sum of two weights) | **`i16`** (max **254**) |
| `k = 3..=16` per wire | **`i16`** (recommended) |
| clamp weights to **`±63`** | optional **`i8`** sum (max `63k`; at `k=16` → 1008) |

No `i32`/`f32` needed on the hot path.

---

## Choosing fan-in

| Priority | Prefer |
|----------|--------|
| Min ops/gate, min memory | **`k = 2`** |
| Balance depth vs gate cost | **`k = 3` or `k = 4`** (`u64` single load) |
| Max expressivity per gate (still linear) | **`k = 8`** (`u128` single load) |
| Rare wide gates | **`k = 16`** (2× `u128` load; consider if depth savings justify 2× weight traffic) |

**Rule of thumb:** move from 2 → 4 if you need **shallower graphs**; go beyond 8
only if profiling shows depth dominates and sink-supervision still learns.

---

## Recommended defaults (experiments)

| Parameter | Value |
|-----------|--------|
| Fan-in `k` | **2** (baseline, per pair + **`i8`** nodes in exp plan) |
| Weight layout (`k = 2`) | **per pair** (`w_00…w_11`), sign lane select |
| Node state | **`i8` activations + `i8` targets** |
| Sources `d` | **32** (independent of `k`) |
| Gates `G` | **64–256** |
| Weights | **`i8`**, packed per table above |
| Parents | **`u8`** if `N ≤ 256` |
| Sum | **`i16`** |
| Threshold | **`sum > 0`** |
| Update step `η` | **`1`**, saturating `i8` |

---

## Related

- [boolean-circuit-exp-plan.md](./boolean-circuit-exp-plan.md) — stream protocol,
  DGP, experiments (**`k = 2` per-pair**, **`i8`** activations/targets; this doc
  generalizes fan-in).
