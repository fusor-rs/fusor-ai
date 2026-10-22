# knowledge.md

What measuring the browser runtimes taught, so that the next model's kernels start from it.
Each entry says what to do, what not to do and the number behind it. The numbers are from
PaddleOCR-VL-1.6's `web-threaded-relaxed` runtime (8 workers, Chrome 156, an M4 Pro) and a
kernel harness replaying `kernels::matmul` and `ops::attend` in a page. An entry is here only
if it was measured in the page, not just in the harness; the per-attempt record is in
paddle-ocr-wasm's optimization ledger.

## V8 compiles a function, not a loop

- V8 runs a wasm function in its baseline compiler (Liftoff) until a *later call* finds its
  tiering budget spent, and never optimizes a loop already running: no on-stack
  replacement. A hot loop inlined into a function entered once per image runs the whole
  image in baseline code, 3–4× slower on dot loops. Keep each hot loop in its own
  `#[inline(never)]` function (`quant_tiles`, the attention's `unit`), and make its first
  calls short: `quant_matmul` calls `quant_tiles` per `ROWS_PER_CALL` (64) rows, so an
  image's first product spends the budget on its first rows and runs the rest optimized.
  That made a session's first read cost what a warm one does (10.1 → 8.7 s).
- `wasm-opt -O3` undoes `#[inline(never)]`: its one-caller inlining is unbounded by default
  and folds the kernel back into its caller. `build-runtime.sh` passes `-ocimfs=20`.
- Shared memory (the threaded builds) costs the same code 12–15 % against a single-threaded
  module; nothing in the kernels changes that.

## The kernel is op-bound, not dot-bound

- Relaxed SIMD's `i32x4.relaxed_dot_i8x16_i7x16_add_s` reaches 170 GMAC/s on one core; a
  kernel reaches 115–130 only when the other operations per dot are few. Count every
  operation per block (loads, broadcasts, seeds, scales, stores), not the dots: the quad
  layout (one output column per lane, the weights of four rows interleaved) won because it
  halved the broadcasts and shuffles per dot, not because of the dots.
- A run-time trip count anywhere near the unrolled chunk loop costs 10–30 % on every shape,
  even as a separate call site: LLVM outlines the shared closure, or the larger function
  spills in V8. A partial last block (72 of 96 inputs) saved 25 % of the dots on paper and
  lost on every shape in practice. Pad instead.
- One row (decode) reads its weights once from memory: stream one quad per pass; four
  interleaved streams run at half the bandwidth.

## The store is part of the kernel

- Hand the store a whole quad (`[f32; 4]`) and let the caller copy it; a per-value closure
  (range filter, bias lookup, one store each) costs 6 % on a 1152-deep product and halves a
  96-deep one. Harness, GMAC/s on the QK shape: copy 90, scaled per value 68, scaled as a
  map over the quad 77, per-value adapter 48, adapter with a GELU through a function
  pointer 21. `Linear::project_quads` is the primitive; `project` is its per-value adapter.
- Never call through a function pointer per value or per quad in a store. Activations are
  applied four lanes at a time (`gelu_tanh_lanes`) and chosen by an enum matched inside an
  always-inlined method (`layer::Activation::apply`).
- Scale in the store as a map over the quad (one vector multiply), not per value.

## Plain Rust vectorizes only with constant lengths

- A loop over four rows with a run-time length (`out[i] += w_k · row_k[i]`, the head size
  read from the shape) ran scalar in the page, 2× slower than one row at a time, while the
  harness with a constant length showed it 24 % faster. Write such loops as explicit SIMD
  in `arch` (`axpy_f32_many`, `dot_f32_many`), and give the harness the page's run-time
  sizes before trusting it.

## Bit-exact or nothing

- A kernel change leaves every output bit-identical unless it is meant to move the numbers.
  Folding the attention's `1/√d` into the keys' quantization moved a few levels by one and
  dropped a line of a reading; `exp` with fused multiply-adds, a shorter polynomial or a
  different summation order are the same kind of change.
- Vectorized transcendental functions mirror the scalar one operation for operation (the
  same clamp, range reduction and Horner order, no fused ops), and a test asserts the bits
  against the scalar for a sweep of inputs. `exp`, `gelu_tanh` and `axpy_f32_many` are done
  this way.
- A softmax's sum stays in index order (`scores.iter().sum()` after the vector exp); the
  per-lane partial sums the sweep returns are a different association.
- Fusing the softmax's maximum, exponentials and int8 quantization into one sweep gained
  nothing: the row sits in L1 and the exp polynomial is the cost. Fewer passes over L1 data
  is not a win by itself.

## Shared preparation, split by group

- Quantize each row of a layer's input once per team, not once per thread: each thread
  normalizes and quantizes a contiguous range of (row, group) units. Splitting by whole rows
  leaves the one-row (decode) case to one thread; splitting by groups balances it.
- Reading activations another core quantized costs nothing measurable; streaming
  activations and outputs through 5–64 MB buffers costs ≤ 2 %. Do not reorganize splits for
  cache locality without a measurement.

## Measure in the page, explain with the harness

- The page's kernels ran 15–26 % below the same code in a harness page; the difference was
  in the page's stores, not in the kernel. Build the harness variant of the page's exact
  closure, with the page's run-time sizes, before changing the kernel.
- Nothing else runs on the machine during a measurement or a dump, the harness included: one
  overlapping harness run doubled a phase's traced time. Load averages above ~4 move the
  page's first token by 10–20 %; a change smaller than the spread of three runs is not a
  result; compare traced self time per op and per shape when the load differs.
- The first run in a fresh browser profile is often the slowest by seconds (V8 compiles,
  the cache is cold); take medians of three, never the first run alone.
- Greedy decoding flips a token on a one-ulp change somewhere in a near-tie. A character
  error rate threshold does not see that; compare the text of a fixed image to the last
  accepted build's, character for character.
- Barrier waits summed over threads are the load indicator: they tripled with other work on
  the machine while no kernel changed.

## Tiles and shapes

- Four rows per tile is the sweet spot in V8: six- and eight-row tiles run at 68–71 GMAC/s
  against 116–124 for four (register pressure); two rows 102–108, one row 72–91.
- The attention's `Q·Kᵀ` (32 queries × thousands of keys × a head of 72–128) is the shape
  the store dominates: with 96 inputs there are 24 dot chunks per output quad, so anything
  per store shows. `P·Vᵀ` (thousands of inputs) hides its store.
- Liftoff is not uniformly slow: a register-only dot loop runs 3.4× slower under it, the
  tile kernels at full speed. What suffers is specific; trace per phase in a cold run
  (cold/warm ratio per layer) rather than assume.

## What did not matter

- Cache locality of the activations (which core quantized them), streaming outputs through
  64 MB buffers, fusing sweeps over an L1-resident row, and `wasm-opt`'s passes on the
  kernel code: each ≤ 2 % or nothing. Measure before reorganizing for any of them.

## This machine (M4 Pro, Chrome 156), for scaling expectations

- `relaxed_dot_i8x16_i7x16_add_s`: 170 GMAC/s per core, 3 per cycle. f32 multiply and add
  without fused ops: ~28 GMAC/s per core. The vector `exp`: ~14 cycles per four lanes (30
  operations, two dependent chains). Eight workers get 85–96 GMAC/s each on the real
  kernels in a shared-memory module.
- Decode is bound by the 361 MB of weights per token; the layer matmuls stream at 25–30 GB/s
  per worker, the language model head at ~3 GB/s per worker, unexplained and the next lead
  if tokens per second ever matters.

## Keep the harness in the repository

- A page that replays `kernels::matmul` and `ops::attend` with the page's exact store
  closures and run-time sizes answered every "why is the page slower" question of the
  session in minutes. It lives in `tools/kernel-harness` (from paddle-ocr-wasm's
  `dump/optimizations/harness`), with its `real_*` exports and the all-core `bench-many`
  runner.
