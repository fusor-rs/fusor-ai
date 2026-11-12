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
- A function with one caller joins it, generics included. The f32 product's dots, made
  generic over how many run together and instantiated at eight for the product alone, were
  inlined with their callers into the product itself, which a patch embedding enters once
  per image and worker: the whole embedding ran in baseline code and the receipt's first
  token went from 4.22 to 4.41 s, while the traced build, whose spans hold those functions
  apart, showed the embedding improved (0.34 → 0.22 s summed). With the block of rows out of
  line (`f32_block`, called per block): 4.14 s. After a change to how a hot loop is reached,
  measure the uninstrumented page.
- `wasm-opt -O3` undoes `#[inline(never)]`: its one-caller inlining is unbounded by default
  and folds the kernel back into its caller. `build-runtime.sh` passes `-ocimfs=20`.
- The threaded builds' toolchain sets the kernels' speed: `nightly-2025-11-15` generated the
  quad kernel at 88–94 GMAC/s per core on shared memory, where Rust 1.99 and
  `nightly-2026-10-20` give 108–118, the same as a single-threaded module (kernel harness,
  four builds back to back). What looked like a 12–15 % cost of shared memory was the
  compiler; shared memory itself costs the matmul nothing measurable. Measure the kernel
  before moving `THREADS_TOOLCHAIN`.

## The kernel is op-bound, not dot-bound

- Relaxed SIMD's `i32x4.relaxed_dot_i8x16_i7x16_add_s` reaches 241 GMAC/s on one core (15
  dots per ns) with twelve or more accumulators in flight; with eight it stops at 170, each
  waiting three cycles for its own result. A kernel gets a fraction of that: count every
  operation per block (broadcasts, seeds, scales), not the dots, and count the accumulators
  in flight. The quad layout (one output column per lane, the weights of four rows
  interleaved) won because it halved the broadcasts and shuffles per dot, not because of the
  dots.
- V8 runs wasm in the order it is written, and LLVM's wasm backend nests a value used once
  into the expression that uses it: an unrolled chunk loop reaches V8 as each accumulator's
  eight dots in one dependent chain, one accumulator after another, about six chains in
  flight of sixteen. Keep the chunk step a real loop, with a trip count the compiler cannot
  see (`quad_pass`'s `words`), so that every accumulator takes one dot per step.
- V8 allocates 28 SIMD registers on arm64 (15 on x86-64). Sixteen accumulators, sixteen
  float sums and a block's chunks and broadcasts do not fit: that pass stored 60 values and
  reloaded 78 per 128 dots, accumulators in the middle of their dots among them. Keep the
  float sums in memory behind a stride the compiler cannot see, loaded and stored once per
  group. The loop and the sums together took Qwen3.5's prefill linears from 203–220 to
  222–230 GOPS per worker on the page and the receipt's warm first token from 4.74 to 4.48 s
  traced, bit-identical.
- A `v128.load32_splat` is a SIMD operation (an add and an `ld1r`); a plain `v128.load` is
  none. A full tile's activations broadcast once into a stack scratch and loaded in the dots
  gave the same linears 231–243 GOPS and the first token 4.32 s, for rows up to 2048 wide
  (`BROADCAST_COLS`: six rows of it are 49 KB, inside the first-level cache).
- A partial last block (72 of 96 inputs, a run-time trip count on the unrolled chunk loop)
  saved 25 % of the dots on paper and lost 10–30 % on every shape while the sums lived in
  registers: LLVM outlined the shared closure, or the larger function spilled in V8. Pad
  instead.
- One row (decode) reads its weights once from memory: stream one quad per pass; four
  interleaved streams run at half the bandwidth.

- The quad kernel's non-dot operations are the format's: per group of 32 inputs and pass
  of 4 rows × 4 quads, 32 loads, 32 broadcasts, 16 seeds and 48 scaling operations for 128
  dots (1.0 per dot). One scale per 128 values (`Q7b128`, the weights and the activations
  alike) cuts it to 0.625 and ran the real kernel 15–22 % faster in the harness and the
  prefill linears at 226–243 GOPS per worker on the page (203–215 at 32): the receipt read
  −9 % traced. It is not free: int8 activations per 128 values lose the language model's
  outlier channels (parity's hidden-state errors +50 %, a next-token count down), and on
  the vision tower alone the merged tokens' error grows 27 % and a reply drifts from the
  reference's. Parked with its patch in qwen3-5's ledger. The loss is two of the same size
  (Qwen3.5's parity, each half emulated alone inside `Q7b32`): the weights' scale per 128
  alone moves the merged tokens' error by 15 % and costs a next token, the activations' alone
  by 12 %. The activations' half has a cure (the group's largest step shared only among the
  blocks within 25 % of it: +1 %); the weights' has none short of a better quantizer, a row's
  largest of 128 weights being 0.26 bit above its largest of 32. Halving the activations'
  levels multiplies the error by 1.36: the weights' rounding is about 72 % of its power.

## The store is part of the kernel

- Hand the store a whole quad (`[f32; 4]`) and let the caller copy it; a per-value closure
  (range filter, bias lookup, one store each) costs 6 % on a 1152-deep product and halves a
  96-deep one. Harness, GMAC/s on the QK shape: copy 90, scaled per value 68, scaled as a
  map over the quad 77, per-value adapter 48, adapter with a GELU through a function
  pointer 21. `Linear::project_quads` is the primitive; `project` is its per-value adapter.
- Never call through a function pointer per value or per quad in a store.
- Scale in the store as a map over the quad (one vector multiply), not per value.

- An activation does not go in a store. With its GELU in the store the vision ffn_up ran at
  197–201 GOPS per worker against 233 for the qkv projection of the same depth. Half the gap
  was the GELU's arithmetic. The other half was the store: once it holds the activation's
  match LLVM outlines it, the quad goes to it through memory (four scalar stores read back
  as one vector) and the bias is added a lane at a time. A store built per activation closed
  that half (200 → 212 GOPS on the page) for 115 KB of module: not kept. With a copying
  store and the activation swept over each block of `ROWS_PER_CALL` rows after its projection
  (`gelu_tanh_sweep`, `silu_sweep`), the product runs at 235 GOPS and the sweep costs 0.25 s
  summed over the workers (3.48 → 2.96 + 0.25 s on the receipt); the gated feed-forward's
  product went from 229 to 239.
- A sweep is its own `#[inline(never)]` function with its constants held as values. The same
  GELU as a lane function inlined into the layer's feed-forward, which is entered once a
  layer, ran in baseline code and built fifteen constants again at every quad: 0.42 s where
  the sweep takes 0.25, and a cold read 1.5 % longer.

- A store's state goes in by value. A closure that reaches its buffer, a width and a scale
  through references loads each of them again at every store, since the store before may
  have written them: the attention's score store spent 33 scalar instructions on a store of
  four values. As a `move` closure over a slice and the two scalars its state sits in the
  environment the kernel holds uniquely, and the loads leave the loop: the vision `Q·Kᵀ` went
  from 186 to 216 GOPS per worker on the page together with the next entry.
- A pass's fixed costs are a tenth of a pass over two blocks. Its sums zeroed as an array are
  a `memory.fill`, which V8 calls out of line (3.6 ns for 256 bytes); zeroed by volatile
  stores in a loop they are stores. Under 1 % on rows of 1024 and more, it was a third of
  what the two changes gained on `Q·Kᵀ`. Together they took 2.5 % off Qwen3.5's first token
  (4.25 → 4.14 s traced), bit-identical.

## Plain Rust vectorizes only with constant lengths

- A loop over four rows with a run-time length (`out[i] += w_k · row_k[i]`, the head size
  read from the shape) ran scalar in the page, 2× slower than one row at a time, while the
  harness with a constant length showed it 24 % faster. Write such loops as explicit SIMD
  in `arch` (`axpy_f32_many`, `dot_f32_many`), and give the harness the page's run-time
  sizes before trusting it.

## Read the generated code

- `f32::max` skips NaNs: reducing a vector's lanes with it is a compare and a select per
  operand. Reduce as vectors (`f32x4_max` of the shuffled halves) and extract one lane. LLVM
  turns `i16x8.narrow_i32x4` and `i8x16.narrow_i16x8` into two clamps per value before the
  same shuffles; where the values are known to fit (the activation quantizer's levels are
  within ±127) take their low bytes with the shuffles alone. Both in the quantizer: 0.33 →
  0.21 ns per value in the harness, its self time on Qwen3.5's vision tower 1.87 → 1.04 s
  summed over the workers, bit-identical.
- A reduction on one accumulator waits for itself at every step. The softmax's row maximum
  on four maxima over runs of 16 values (a maximum is exact in any order) took 18 % off the
  softmax's sweep on the page (1.88 → 1.54 s summed over the workers).
- V8 keeps a value in a register across a loop and builds a constant again where it is
  used: the fast exponential's sweep made seven vector constants per iteration (6 `dup`,
  4 `movk`, 1 `movi` of 104 instructions). The same constants as values the compiler cannot
  see (`black_box` once per sweep) stay in registers: the softmax's sweep 1.56 → 1.43 s
  summed over the workers on the page, bit-identical. Count the registers first: a value
  V8 has no register for is spilled, a constant is only rebuilt.
- `--js-flags=--perf-prof --perf-prof-path=<dir>` makes release Chrome write its optimized
  wasm code to `jit-<pid>.dump` (macOS included). The 2 MB log buffer is lost when the
  renderer is killed: compile padded copies of the module until the functions wanted are on
  disk, then disassemble with `llvm-mc`. The stores between the dots in that listing are
  what showed the spills above; the wasm text shows neither the registers nor the order V8
  keeps.

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
- Two identities are the same bits for fewer operations. `poly · 2ⁿ` is the addition of `n`
  to the float's exponent bits where the result is normal, as the exponentials' clamps keep
  it; `trunc(nearest(q))` for a level within an int8 is the low byte of `q + 1.5 · 2²³`
  (floats are a unit apart there, so the sum is the level rounded to even). On the receipt
  the softmax's sweep went 1.43 → 1.34 s summed and the activations' quantizer 1.06 → 0.92 s.
- Fusing the softmax's maximum, exponentials and int8 quantization into one sweep gained
  nothing: the row sits in L1 and the exp polynomial is the cost. Fewer passes over L1 data
  is not a win by itself.

- Where the exponentials feed an int8 quantization (the vision attention's probabilities,
  7-bit levels per block), an exp within 6e-5 (degree-4 Taylor on `|r| ≤ ln 2 / 2` with
  fused multiply-adds, no upper clamp since `score − max ≤ 0`) is a hundred times under the
  quantization step: the page's replies stayed byte-identical, the encoder's parity errors
  moved within ±10 % in both directions, and the softmax's exp self time fell 42 % (0.42 →
  0.24 s per worker on the receipt; harness 0.68 → 0.41 ns per value). It is the model's
  choice (`Exponentials::Fast` on its tower's shape), backed by that model's evidence; the
  f32 softmaxes keep the exact exp. SIMD128 has no fused form and lands up to 64 ulp away,
  within the same bound; on x86-64 the scalar backend's fused multiply-add is a library call.
- The step after that one moves outputs, and is a candidate, not merged (`kernel-candidates`).
  A block's levels are `round(127 · e^(score − m))`, `m` its largest score: a function of the
  scores alone, zero below `m − 5.6`. One degree-7 polynomial with the rounding constant in
  its last multiply-add stands for the exponential, the division and the rounding, a block's
  scale is one `exp_fast(m − M) / 127`, and the row's normalizer is its levels' own sum. On
  the page the scores' way to int8 went 2.24 → 1.47 s summed and the receipt's first token
  −2.9 % traced, −3.3 % in rounds; every reply stayed byte-identical and the tower's parity
  errors moved −17 % to +10 %, their mean −4 % (the exact exponential alone moves them −10 %
  to +3.5 %: that is the noise). The exponentials' own sum does not go with it: the clamped
  polynomial counts 0.47 of a level for every key below `m − 5.6`, and that normalizer was
  40–65 % worse.

## Shared preparation, split by group

- Quantize each row of a layer's input once per team, not once per thread: each thread
  normalizes and quantizes a contiguous range of (row, group) units. Splitting by whole rows
  leaves the one-row (decode) case to one thread; splitting by groups balances it.
- Reading activations another core quantized costs nothing measurable; streaming
  activations and outputs through 5–64 MB buffers costs ≤ 2 %. Do not reorganize splits for
  cache locality without a measurement.

## Convolutions

- Quantize a pixel once, not once per window over it (9 times in a 3 × 3 convolution, 27 in
  a causal 3 × 3 × 3 one). Where the channels are whole groups of the weights' format, a
  window's quantization groups are its pixels' and a padded tap is the group of zeros: each
  thread quantizes the rows its windows read and gathers the windows in int8 with their
  scales and corrections (`conv`'s `Band`). Realistic Vision's generation (8 steps at
  512 × 512): 13.1 → 12.5 s over seven rotated rounds, bit-identical.
- A GroupNorm takes all of a thread's groups of an image in one pass (each group's sum still
  in the order of its pixels), not one strided pass per group, and normalizes a pixel a
  group's run at a time, the mean, the deviation and the affine pair hoisted, so that the
  activation's loop is vector code: 13.6 → 13.1 s on the same page.

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
- The traced build is another build. Its spans are calls the compiler does not inline
  through, so a loop can sit in a function of its own there and in a caller entered once on
  the page (the f32 product above: its span −35 %, the page +4.5 %). A kept change is
  measured in rotated rounds on the uninstrumented site as well as by its span: the span
  says where the time went, the page whether it went.
- The machine heats within a run of reads. Eight reads back to back gave the same sites'
  first token as 4.00, 4.10, 4.20, 4.31, 4.41, 4.40, 4.44 and 4.53 s whatever the site (the
  load from 3 to 8): 12 %, more than most changes, and the site read first always wins.
  Rotate the order every round and discard a preheating read; or let the load fall before
  every read (under 2.8 on this machine, about a minute): six sites read that way repeated
  to 0.01–0.05 s over two rounds and put five commits of 0.01–0.03 s each in their traced
  order (3.73, 3.71, 3.69, 3.68, 3.65 s), which heated rounds could not tell apart.
- The traced build's barrier waits are not the page's. Its clock worker spins on a core
  beside the eight workers and the engine thread, so on ten performance cores a worker is
  displaced at every phase: 0.9 s of waits per worker on an 8.3 s traced read of Qwen3.5,
  1.9 ms per decode token. The uninstrumented page keeps a core per worker and its static
  shares finish together: claiming the shares dynamically (a static three quarters, the last
  quarter in chunks from one atomic counter) gained nothing on the page and cost 1–4 % of
  decode; guided chunks down to one unit cost 13 %. Read the dump's barrier column as the
  trace's load, and balance nothing without a page measurement.
- A chain of dependent f32 dots runs at latency, not throughput. The gated delta rule's step
  (per memory row: a dot, an update, a dot) halved with four rows in flight through
  `dot_f32_many` and a four-row update (`decay_rows`), each row's arithmetic in its own
  order (4.45 → 2.23 µs per 128 rows of 128; the rule's self time −36 % in the page), and the
  softmax's `exp_sweep` gained 8 % from two runs per iteration into the same accumulators in
  the same order. Both bit-identical: only the interleaving changed.

- The machine's other work (a VM, another session's rustc, macOS daemons on 2–3 cores)
  moved the receipt's first token by 10–20 % and its spread to 0.3–0.8 s within one session.
  Interleaved rounds and the dump's self time per op, which the load barely moves, resolved
  4 % changes that the medians could not. When a format changes, each A/B site keeps its
  own bundle.

## Tiles and shapes

- Four rows per tile is the sweet spot in V8: six- and eight-row tiles run at 68–71 GMAC/s
  against 116–124 for four (register pressure); two rows 102–108, one row 72–91.
- The attention's `Q·Kᵀ` (32 queries × thousands of keys × a head of 72–128) is the shape
  the store dominates: with 96 inputs there are 24 dot chunks per output quad, so anything
  per store shows. `P·Vᵀ` (thousands of inputs) hides its store.
- Liftoff is not uniformly slow: a register-only dot loop runs 3.4× slower under it, the
  tile kernels at full speed. What suffers is specific; trace per phase in a cold run
  (cold/warm ratio per layer) rather than assume.
- Four quads per pass beside four rows is the register optimum too: six quads run 2 %
  slower, eight at half the rate (32 accumulators spill). With the sums in memory and the
  activations broadcast, a pass is 128 dots and 48 scaling operations (11 SIMD operations
  per 8 dots), and the page runs the prefill linears at 115–121 GMAC/s per worker with every
  core busy; the harness gives the same kernel 113–134 on eight pages at once and 131–144
  alone. What is left is the format's per-block scaling.

## Sums are chains, and a block fits a cache

- A scalar sum is a chain too. `layer_norm`'s mean and variance are two sums over a row,
  1024 additions each waiting three cycles for the last: 1.4 µs a row, 0.24 s summed over the
  workers on Qwen3.5's receipt. Eight rows' sums interleaved (eight accumulators in plain
  Rust, each row's values added in their order from the `−0.0` a float sum starts at) are
  the same operations at the adder's throughput: 0.066 s, bit-identical.
- The f32 product wants its block of rows in the first-level cache and eight dots in flight,
  and neither alone. A column at a time across all the rows streamed a receipt's 21 MB of
  patches again for each of a worker's 128 columns: 23.5 GOPS per worker. Blocks of 64 rows
  (393 KB): 32. Blocks of the eight rows that fit 64 KB, eight dots against one row of
  weights: 49, the patch embedding 0.47 → 0.22 s summed. In the harness eight dots in flight
  on the 64-row block gained 5 %, the small block with four dots 27 %, both 55 %: a chain's
  bound shows only once the data is near. Rows of 4096 values lose with eight in flight
  (nine rows of 16 KB are past the cache) and keep four.
- An attention's keys and values, quantized at run time, went through the formats' scalar
  quantizer and a packing that works out a code's position per code: 3.4 ns a value where
  the activations' quantizer takes 0.15. Through the backends' `absmax` and `quantize_block`
  (the same scale, the same levels; the format's clamp never acts) and a chunk's four codes
  at a time, `attn.quantize_kv` went 0.58 → 0.36 s summed on the receipt. What is left is
  three passes over an image's rows 12 KB apart.
- A call's rotary angles are computed once, not per thread, layer and row: into a shared
  buffer before the layers (`Stack::turn`), the vision `qkv_rope` phase's own time went
  0.26 → 0.10 s summed.
- The delta rule's convolution a tap of every channel at a time (the taps stored tap-major,
  the sums started at `−0.0`, the SiLU swept after) where it went a channel of every tap:
  0.49 → 0.165 s summed, bit-identical.

## Work nobody reads

- Of a prompt's last layer one row's output is read; the rows before it matter to later
  tokens only as that layer's keys and values. Projecting just those (a fifth of the
  projection's columns) for them and running the layer on the last row alone took 4 % of the
  prefill's products (10.40 → 10.11 s summed) and a sixth of its cached attention (0.68 →
  0.57 s): the receipt's first token −1.5 % traced, bit-identical, since a row's arithmetic
  does not depend on the rows computed with it (`Hybrid::forward_to_last`).

## What did not matter

- Cache locality of the activations (which core quantized them), streaming outputs through
  64 MB buffers, fusing sweeps over an L1-resident row, and `wasm-opt`'s passes on the
  kernel code: each ≤ 2 % or nothing. Measure before reorganizing for any of them.
- The residual store's form (a zip loop over a range slice, a `[f32; 4]`, or one v128 add),
  the number of activation rows per matmul call (64 to 512, same rate), and a straight-line
  depthwise convolution over transposed taps (1.3× plain, nothing on shared memory): none
  moved the page.

- The prefill's chunking (512 + 384 rows for a 896-row prompt): the per-row cost is the same
  in both chunks; one call of 896 rows would save ≤ 10 ms of fixed costs for 50 MB of
  buffers.

## This machine (M4 Pro, Chrome 156), for scaling expectations

- `relaxed_dot_i8x16_i7x16_add_s`: 241 GMAC/s per core with twelve or more accumulators in
  flight (170 with eight), about 15 G SIMD operations per second in all. f32 multiply and
  add without fused ops: ~28 GMAC/s per core. The vector `exp`: ~14 cycles per four lanes (30
  operations, two dependent chains). Built with the pinned `nightly-2026-10-20`, the page
  runs Qwen3.5's prefill linears at 230–242 GOPS (115–121 GMAC/s) per worker, the vision
  QK/PV at 216/237 GOPS and its f32 patch embedding at 49 (24.6 GMAC/s); 85–96 GMAC/s per
  worker was `nightly-2025-11-15`'s code.
- Decode is bound by the 361 MB of weights per token; the layer matmuls stream at 25–30 GB/s
  per worker, PaddleOCR-VL's language model head at ~3 GB/s per worker, unexplained and the
  next lead if tokens per second ever matters. Qwen3.5's head streams at 29 GB/s like its
  layers, and eight native threads reading private buffers get 27 GB/s each (216 in all):
  its decode sits on the memory's bandwidth.

- Workers against cores (10 performance and 4 efficiency, 14 reported; the page's default is
  8): a pool of 10 cuts the receipt's warm first token 10–12 % (the vision and the prefill
  scale with the cores) but plain 10 workers lose 10 % of decode, which streams the weights
  at the memory's bandwidth that eight workers already reach and pays every extra thread at
  each of its 234 barriers per token. `Team::run_on` keeps the one-row calls on 8 threads
  of the 10 (the others sleep in Rayon's pool): first token −10 to −12 %, decode at the
  8-worker rate, bit-exact. 12 workers put shares on efficiency cores: +24 % first token,
  −32 % decode. The session's first image gains nothing at 10: each worker's isolate
  compiles its optimized code during that read, and ten compile at once. The default count
  is the host's: the web reports 14 cores without saying which are performance cores.
- Past the fast cores, static shares cost more than the cores add. The same page at 8, 10,
  12 and 14 workers reads the receipt's first token in 4.41, 3.95, 6.77 and 7.18 s, while
  the kernel's capacity at 14 busy cores is 28–34 % above 8 (the harness in 14 pages): every
  phase waits at its barrier for the share of a thread on an efficiency core, a third as
  fast for a kernel, and the system rotates all the threads over those cores, so no thread
  is the slow one for long. A page's default worker count is past the fast cores on any
  processor with fewer of them than workers. What to find is a count, not a set of threads:
  with as many threads as fast cores the system keeps them there.
- That count cannot be had from a race inside the sealed runtime. A race of counting loops
  before or after a run, short enough to be free (10 µs), is shorter than the scheduler's
  jitter: with more runnable threads than cores some count nothing, twelve races in a row
  kept 7 to 12 of 14 threads, and a wrong verdict costs a whole run (reads of 6.2–6.7 s at
  8 workers, where there was nothing to drop). A loop of integer steps also reads an
  efficiency core at two thirds of a fast one, a loop of float additions at half, the
  kernels at a third. The page has a clock; the runtime does not.

## Keep the harness in the repository

- A page that replays `kernels::matmul` and `ops::attend` with the page's exact store
  closures and run-time sizes answered every "why is the page slower" question of the
  session in minutes. It lives in `tools/kernel-harness` (from paddle-ocr-wasm's
  `dump/optimizations/harness`), with its `real_*` exports and the all-core `bench-many`
  runner.
