# FAST attention performance debug log

## Goal

Measure the current generic FAST online attention kernel against matched
ordinary-XLA and Grug FA4 baselines on one H100, then implement and validate a
cooperative warp-per-query-row kernel. Report raw warmed CUDA-event medians;
this checkpoint begins without a performance threshold.

## Initial status

- Accepted base: `8b1f8d8e7b94c2f862d08eb583d8a4e026651f13`
- Correctness evidence: the focused causal-GQA H100 canary passes its existing
  maximum and mean absolute-error limits.
- Current online schedule uses one thread per query row. It retains every Dv
  component in scalar registers and serially streams K.
- The scheduler currently limits K to 256 and the output-vector extent to 64.
  The requested `S=2048`, `D=Dv=128` representative case is therefore not
  admitted by the current production path. The first matched benchmark will
  record that failure and use the largest shape accepted by all compared paths.

## Benchmark protocol

- One H100 and one CUDA stream for all cells.
- Device inputs and outputs remain allocated between samples.
- Warmup precedes measurement.
- CUDA events measure device execution; host dispatch time is observational.
- Cells are counterbalanced to reduce order drift.
- Report all raw samples and medians, compiler identity, executable/code
  digests, and resource observations. Do not introduce a pass/fail speed
  threshold in the baseline.

## Hypothesis 1: serial row execution dominates

The current kernel assigns one thread to a row, performs the QK feature dot
serially, retains up to 64 output accumulators, and serially advances K. A
cooperative warp should reduce score latency and register pressure by:

- assigning one warp to each query row;
- partitioning the QK feature dot across lanes and using a warp reduction plus
  broadcast for each score;
- assigning strided Dv components to lanes;
- keeping the scalar `m/l` state consistent across lanes;
- streaming each K coordinate once per row warp; and
- placing multiple independent row warps in a block when resource limits allow.

## Results

### 2026-08-27 cooperative F32 checkpoint

- Replaced the scalar row schedule with four 32-lane row warps per block.
- Each lane accumulates a strided part of the score Contract's feature
  reduction; the warp sums and broadcasts the score with shuffle
  instructions.
- Each lane owns `dv = lane + 32*i`; `m` and `l` remain replicated, and K
  stays the outer streamed loop.
- Lifted the bounded online capability to K=2048 and Dv=128. Both the
  S256/D64 and requested B1/Hq32/Hkv8/S2048/D=Dv128 ordinary-JAX fixtures now
  complete production materialization, scheduling, bundle construction, and
  executable verification without a score temporary.
- The independent H100 numerical canary passed on the existing causal output
  epilogue fixture. Log:
  `/private/tmp/shuttle-cooperative-f32-numerics.log`, SHA-256
  `22879a481acfc636291f13edf90489604de02212fc517bae1fb0cc5a7db5d091`.
- `ptxas -arch=sm_90 -v` reports 40 registers, zero stack bytes, zero spill
  stores, zero spill loads, and zero barriers for the axis-4 canary PTX.
- A second H100 canary with BF16 public Q/K/V and BF16 output, while retaining
  FP32 `m/l/o`, passed the independent host oracle. The combined focused target
  ran both the F32 and BF16 tests successfully.
- The older prospective PTX state-machine checker rejects shuffle-based
  ownership. This is a test-oracle mismatch; production executable
  verification and the independent numerical canary are the active gates for
  this performance slice.

### 2026-08-27 matched H100 timing

The matched harness used one patched-JAX process, one visible H100, the same
JAX external-ready CUDA stream, ten warmups per cell, and 51 counterbalanced
CUDA-event samples per cell. All cells accepted the same BF16 external tensors
with logical layout `[B,Hkv,G,S,D]`, `[B,Hkv,S,D]`, `[B,Hkv,S,D]`; the Grug
wrapper converts that layout to BSHD inside its compiled executable. No timing
threshold was applied. Ordinary XLA used its cuBLAS path because the hermetic
wheel's Triton GEMM autotuner could not compile any candidate configuration.

At `B=1,Hq=32,Hkv=8,S=64,D=Dv=64`:

- Shuttle FAST median: 224,032 ns
- ordinary XLA median: 45,792 ns
- Grug FA4 median: 45,952 ns
- Shuttle and Grug maximum absolute error versus ordinary XLA: 0.015625

At the requested `B=1,Hq=32,Hkv=8,S=2048,D=Dv=128`:

- Shuttle FAST median: 34,177,025 ns
- ordinary XLA median: 1,088,512 ns
- Grug FA4 median: 322,400 ns
- Shuttle and Grug maximum absolute error versus ordinary XLA: 0.015625

The cooperative implementation is numerically sound at the representative
shape but is not competitive: its median is about 31.4 times ordinary XLA and
106 times Grug FA4. The dominant structural gap is now query tiling and KV
reuse. Each query-row warp independently streams the full K domain, including
the causal upper triangle that is later masked, so four row warps in a block do
not cooperate on K/V traffic.

The reservation was released after this measurement. The Iris session state
was removed and the pod reached terminal `Completed`.

### 2026-08-27 CTA K/V reuse checkpoint

- Changed the online schedule from independent flat row blocks to CTA groups
  that keep batch and KV-head coordinates fixed while four warps cover nearby
  grouped-query/query coordinates.
- Derived a 32-key tile from the 16 KiB dynamic-shared-memory budget at
  `D=Dv=128`. The CTA stages BF16 K and V once per tile, synchronizes, and
  reuses them across its row warps. Q components remain lane-owned registers.
- For the standard causal predicate, the CTA derives the largest live query
  coordinate and omits key tiles entirely above that bound.
- The first H100 run exposed a real grouping bug: flattening four adjacent
  output rows let a block cross an Hkv boundary because `G*Q=34`. That mixed
  K/V heads and failed both numerical canaries. Mapping blocks as
  `(B,Hkv) x ceil((G,Q)/4)` fixed the source-coordinate ownership.
- After the grouping fix, the independent H100 F32 and BF16 canaries both pass
  at `B=1,Hq=32,Hkv=8,S=2048,D=Dv=128`. The focused log is
  `/private/tmp/shuttle-cta-h100-numerics-grouped.log`, SHA-256
  `5dda71a7e3e1474cb257744a822b6cfdf7120014ae4e326d57db5b109b0e70b1`.
- Fresh `ptxas -arch=sm_90 -v` accounting reports 32 registers, 16,384 bytes
  of dynamic shared memory, one barrier, zero stack bytes, and zero spill
  stores or loads.
- The exact matched Shuttle/XLA/Grug timing run did not complete in this
  reservation. Its clean patched-JAX build reached 26,863 of 27,267 actions,
  but a long ROCDL/NVVM translation-unit compile remained active after the
  bounded timing window. The build was stopped rather than holding an idle
  eight-H100 node. No new medians are claimed; the earlier cooperative
  medians above remain the latest matched measurements.
- The reservation was released after the bounded build window. The pod first
  reached `Succeeded` and was then removed from the cluster.

### 2026-08-27 native CTA timing

To avoid another patched-JAX wheel build, the native H100 attention canary now
warms the exact BF16 `B=1,Hq=32,Hkv=8,S=2048,D=Dv=128` executable ten times
and records 51 `GpuRunner::ExecuteTimed` CUDA-event samples. Q and K are zero;
V varies across KV head, key, and value dimension. The independent full-output
oracle is the BF16-rounded causal prefix mean, so the timing cell also checks
masking, grouped-head ownership, V indexing, and output addressing. The
existing small nonzero-score canaries retain recurrence and score coverage.

- Shuttle CTA median: 15,513,983 ns
- maximum absolute error: 0
- mean absolute error: 0
- raw log: `/private/tmp/shuttle-native-timing-h100-second.log`
- raw log SHA-256:
  `a60fc349fc7d0b98525f656efe60fc19c88d03c2547ae45c3571439602a7746c`

For iteration guidance only, this is about 48.1 times the previously measured
FA4 median of 322,400 ns and 14.3 times the previously measured ordinary-XLA
median of 1,088,512 ns. Those reference measurements came from a different
reservation; they are not a same-cycle matched benchmark.

The reservation was released immediately after the native timing run. The pod
reached `Succeeded` and was removed.

### Bounded SM90 tensor-core direction

The CTA scalar kernel remains memory-reuse-correct but executes the QK and PV
products as scalar FMA loops. The next candidate should replace both products
as one cohesive tile protocol rather than micro-tune those loops:

- derive a query tile whose rows share `(B,Hkv)` and a key tile from the
  available dynamic shared-memory budget;
- stage Q, K, and V as BF16 with 64-bit global/shared address arithmetic;
- use an SM90 MMA score tile, retaining scores on chip;
- apply the exact Relation predicate before row maximum/sum updates;
- rescale the prior FP32 `{m,l,o}` state once per key tile;
- convert the live tile probabilities to an MMA input representation and use a
  PV MMA tile with FP32 accumulation;
- finalize and store BF16 output only after all causally live key tiles; and
- fall back unless dimensions, alignment, shared memory, and fragment geometry
  are all source-proven legal.

A warp-level `mma.sync.m16n8k16` design wastes most score rows under the current
four-row CTA. The bounded SM90 candidate should therefore jointly change query
tile ownership and MMA lowering, using either a 16-row warp tile or a 64-row
warpgroup tile. The acceptance remains semantic: no score materialization and
the online state protocol, not a fixed tile or instruction spelling.

### 2026-08-27 first `mma.sync` candidate

The first bounded candidate uses one 16-row CTA tile. Warp 0 evaluates QK and
PV with `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`; the remaining
warps cooperatively stage Q/K/V in 16 KiB of shared memory. Source-derived
eligibility requires BF16 public Q/K/V/output, one score reduction, D divisible
by 16 and at most 256, Dv divisible by 8 and at most 128, and K divisible by 16
and at most 2048. Unsupported shapes retain the existing online fallback.

The candidate assembles on H100 with 124 registers, 16,384 bytes of dynamic
shared memory, one barrier, and no spills. The S2048/D128 native timing median
improved from 15,513,983 ns to 13,142,975 ns after repairing a K/V staging CFG
bug in which the V branch skipped definitions of its common affine coordinate
registers. This remains roughly 40.8 times the prior cross-reservation FA4
median and is not a performance success.

Numerical validation remains blocked. With zero Q/K and deterministic V, the
candidate returns zero for the causal prefix-average output (maximum absolute
error 0.84375, mean absolute error 0.0480831). Device diagnostics established:

- forcing every shared probability to one produces nonzero PV output, proving
  the V staging, PV MMA, and public store paths are live;
- forcing only logical key zero produces the exact query-zero output, proving
  sparse probability fragment placement and BF16 packing are sound; and
- replacing the delayed predicate register with an immediately materialized
  u32 validity value does not change the all-zero failure.

The remaining concrete seam is the source Relation predicate as evaluated for
the MMA score-coordinate route. The scalar online route passes the same causal
fixtures, while the MMA route does not mark the query-zero/key-zero leaf valid.
No MMA performance claim is made until that affine/Relation discrepancy is
fixed and the independent numerical oracle passes.

A follow-up run materialized each Relation result immediately into a
coordinate-matched shared validity tile and reloaded it beside the BF16
probability store. It compiled and ran, but produced exactly the same numerical
errors (maximum 0.84375, mean 0.0480831). This falsifies predicate-register
lifetime and delayed Relation transport as the cause. The shared-validity
experiment was reverted because it added traffic without changing behavior.
The source-derived PTX compare itself evaluates q0/k0 as true and q0/k1 as
false. The remaining discriminator is inside the computed probability path:
QK accumulator, tile maximum, score delta/exponential, or row sum. A future
device diagnostic must expose those values before another semantic change.

The RNO2A reservation was released after the final diagnostic. The dev-GPU
session state was removed and the allocator monitor was stopped; no GPU
reservation remains active.

### 2026-08-27 bound score-stage diagnostic

A temporary output overwrite restricted to block 0, thread 0 captured the
query-zero/key-zero path as `qk=0`, `max=0`, `exp=1`, and `sum=1`. This proves
the zero-input score, causal predicate, tile maximum, exponential, and local
normalization sum for the first MMA lane group. The diagnostic stores and host
print were removed immediately after the run, and the H100 reservation was
released.

The corresponding PTX used `shfl.sync.down` with control value `0x3` for each
four-lane MMA group. That value clamps only absolute lanes 0 through 3; groups
1 through 7 therefore read invalid shuffle sources. This explains why the
block-zero/thread-zero probe was correct while most query rows failed. The
local fix uses `0x1c03`, whose segment mask preserves each four-lane group and
whose clamp selects its final lane. The exact fixture now compiles through
bundle construction and executable verification with the repaired control
value. A numerical H100 rerun remains required before treating the diagnosis
as closed.

### 2026-08-27 probability transport and state carry

The segmented-shuffle repair did not close the numerical failure. A bounded
H100 probe then followed query zero/key zero through the computed probability
and PV paths: `exp=1`, shared-memory `P=1`, BF16 PV A-fragment `1`, and the
first PV accumulator `-0.75`. This closed the score-to-PV transport and MMA
fragment mapping.

The remaining bug was the runtime K-tile state backedge. The emitter changed
the C++ register identifier after each generated scale instruction, but the
same PTX instructions execute on every runtime loop iteration. Each tile
therefore scaled the original zero `o` register instead of the preceding
tile's accumulator. The fix keeps each PV accumulator in a stable PTX register
and emits both scaling and MMA accumulation in place.

A final temporary carry trace for query zero/value zero observed first-tile
`m=0,l=1,o=-0.75`, second-tile `oldScale=1` with incoming `o=-0.75`, and
final `l=1,o=-0.75`, normalized and stored as `-0.75` at byte offset zero.
After all diagnostics were removed, the focused H100 target passed all three
attention tests. The BF16 `B1/Hq32/Hkv8/S2048/D=Dv128` oracle had zero maximum
and mean absolute error. Its 51-sample CUDA-event median was 13,307,423 ns.
This closes correctness for the zero-score cell, not performance: it remains
about 41 times the earlier cross-reservation FA4 median. The reservation was
released immediately and allocator status confirmed no active session.

### 2026-08-27 64-key cooperative tile

The corrected 16-key kernel contains 16 QK MMAs and 16 PV MMAs per runtime
tile at `D=Dv=128`. QK is not duplicated across Dv warps, but warp zero also
serializes all PV chunks. With 124 registers and 128 threads per block,
register capacity limits the kernel to four blocks, or 16 warps, per SM.

The next local candidate uses a resource-derived 64-key tile. Warp zero emits
64 static QK MMAs for the 64-key tile, while all four warps emit 16 aggregate
PV MMAs and own disjoint 32-element Dv slices. The probability tile is shared
and the online state rescales once per 64 keys. Q, K, V, P, and row state use
39,104 bytes, below the bounded 48 KiB launch contract. The generated PTX has
80 static MMA instructions total and declares 481 virtual f32 registers;
physical allocation and spills remain an H100 ptxas measurement. A held-out
S2048 oracle now uses deterministic nonzero BF16 Q/K and independently checks
100 outputs spanning early, interior, and final causal rows so an incomplete
QK path cannot pass as a prefix mean.

The H100 held-out run passed with maximum absolute error `0.0009765625` and
mean absolute error `0.00005615234375`. Its 51-sample CUDA-event median was
`13.136096 ms`, only about 1.3 percent below the corrected 16-key median and
still about 40.7 times the earlier cross-reservation FA4 median. This rules
out K-loop state-merge frequency as the main bottleneck. The reservation
`codex-shuttle-mma-bk64-0827` was released immediately and allocator status
confirmed no active session.

### 2026-08-27 in-repo Grug FA4 forward prior art

The audited Marin source identity is commit
`1ec7fd7c7557377ab3802d7b8abf6279b4fea6e5`. The relevant source paths and
Git blob identities are:

- `lib/levanter/src/levanter/grug/attention/_fa4_cute.py`:
  `0ace15acb466dfbd68767939e08a4c25c916d787`;
- `lib/levanter/src/levanter/grug/attention/_fa4_cute_backend.py`:
  `f985f86cea960e843199ce3f22d71242e0d6a842`;
- `lib/levanter/src/levanter/grug/attention/_fa4_cute_config.py`:
  `86d30273a10d696c81f75d7ce787a2a0dabcf347`; and
- `lib/levanter/src/levanter/grug/attention/_fa4_cute_kernels.py`:
  `3bde33f633b8aacac4174b365b4c18a7c947f9bc`.

For H100/SM90, BF16, D=Dv=128, the forward selector chooses M=128 query
rows, N=64 keys, and 128 threads. The forward launcher is the in-repo
`SegmentedFlashAttentionForwardAmpere` implementation even on SM90. It does
not use TMA, WGMMA, warpgroup specialization, or a CTA cluster. Its four warps
form one tiled `mma.sync m16n8k16` layout with atom layout `(4,1,1)` and an M
permutation of 64, so all four warps cooperatively own row/output fragments.
The separate SM90 producer-warpgroup plus two-consumer-warpgroup configuration
is backward-only and is not the measured forward algorithm.

D128 selects an 8x64 shared-layout atom with CuTe swizzle `(3,3,3)`. The CTA
holds Q/O as an aliased 128x128 BF16 tile and separate 64x128 BF16 K and V
tiles, for 65,536 bytes before allocator padding. Global transfers use
128-bit `CopyUniversalOp`, not TMA and not a cp.async-only copy atom. The code
still uses commit/wait scheduling: Q persists; V for the current tile overlaps
QK; after QK finishes, the next K tile is prefetched while softmax and PV use
the current V tile. Q/K use non-transposed four-matrix `ldmatrix`; V uses the
transposed form.

Each CTA is indexed by query tile, batch, and query head. For GQA=4 the kernel
computes `kv_head = q_head // 4`; it does not combine the four query heads into
one CTA or share K/V across their CTAs. Causal traversal begins at the highest
potentially valid 64-key tile and walks backward, skipping the upper triangle
at tile granularity. Every retained score is still checked against query
bounds, key bounds, the packed lower bound, and `key <= query`.

The score tile, row maximum, row sum, and output accumulator live in registers.
The kernel applies the mask to FP32 scores, performs a thread-quad maximum,
uses fast `exp2(score*scale_log2-max*scale_log2)`, rescales the previous FP32
output accumulator when the running maximum changes, converts P to BF16 in
registers, and immediately applies the PV MMA. Final reciprocal normalization
also stays in registers before Q shared memory is reused for output staging.
The source imposes a 128-thread and shared-capacity implementation check but no
explicit register or cluster constraint.

The closest Shuttle schedule should therefore use the generic source-derived
online-state match to select an SM90 tuning default near M128xN64, distribute
both QK and PV row fragments across all four warps, retain FP32 `{m,l,o}` in
registers, traverse only causally reachable N tiles, and pipeline current V
with QK plus next K with softmax/PV. These are target defaults, not graph
acceptance conditions. The present one-score-warp Bk64 experiment is useful as
a measurement but is not the FA4 execution structure.

### 2026-08-27 first FA4-class M128/N64 checkpoint

The planner/runtime now permits a target-bounded 96 KiB dynamic-shared ceiling
and checks every launch above the legacy 48 KiB limit against the concrete
device's `shared_memory_per_block_optin` value before kernel loading. The
S2048/D128 tuning default uses 128 threads, an M128/N64 logical schedule, and
65,536 dynamic shared bytes. Q occupies the 128x128 BF16 allocation; K and the
row-major BF16 probability tile are phase-aliased; V is separate. All four
warps own distinct m16 row fragments. Two runtime 64-row passes keep generated
PTX under the 512 KiB transport cap, but currently reload K/V rather than
retaining one tile across both passes.

The first H100 run left head 7 untouched because executable logical-coordinate
maps still divided blocks using the old 16-row tile. After that was corrected,
query 17 remained untouched because the row-within-tile expression still
truncated the emitter-provided row with `mod 16`. The map now consumes the
source-derived row coordinate directly. K and P aliasing also requires a CTA
barrier after all QK MMAs and before the first probability store.

With those concrete fixes, all three focused H100 attention tests passed. The
nonzero BF16 S2048/D128 cell retained maximum absolute error `0.0009765625`
and mean absolute error `0.00005615234375`. Its 51-sample CUDA-event median was
`7.867488 ms`, a 1.67x improvement over the 64-key one-score-warp checkpoint
but still about 24.4 times the earlier cross-reservation FA4 median. The raw
samples were:

```
8022080 7772704 7934976 7922592 8005567 7815872 7728767 7935328
7813471 7587999 7849184 7998016 7600671 7907264 7898944 7795744
7923936 7578240 8010016 7652288 7988160 7872960 7734528 7942592
7878143 7773952 7832672 7886208 7789248 7680672 7582687 7954688
7867488 7989408 7860544 7889184 7809536 7876736 7723199 7728096
7906079 7671167 7714464 7980928 7699296 7951968 7638720 7901248
7895232 7606495 7874816
```

The reservation was released immediately after the passing run. The next
measured structural gap is persistent K/V reuse across both row-fragment
passes plus source-derived reverse-causal tile traversal; the current kernel
still evaluates every K tile in forward order.

### 2026-08-27 source-derived reverse-causal traversal

The emitter now recognizes the exact Relation predicate chain rather than a
workload label: it traces each iota through reshape and broadcast, requires a
signed `key <= query` comparison, maps the key coordinate to the weighted
Contract reduction iterator, and derives the query coordinate independently.
Each 64-row pass begins at the highest N64 tile that can contain a valid key
and walks backward to zero. Graphs without that proven relation remain outside
this MMA route.

The focused H100 suite again passed all three attention tests with unchanged
nonzero-cell accuracy: maximum absolute error `0.0009765625`, mean absolute
error `0.00005615234375`. The 51-sample median fell to `5.545536 ms`, 1.42x
below the full-forward M128 checkpoint and 2.37x below the one-score-warp Bk64
checkpoint. It remains about 17.2 times the earlier cross-reservation FA4
median. Raw samples:

```
5627711 5440256 5624159 5580576 5612127 5442400 5569696 5611936
5429632 5540224 5445151 5601568 5446112 5564256 5602431 5546048
5462880 5539456 5643551 5505952 5535679 5501023 5505440 5591168
5612640 5492896 5404032 5512864 5622912 5474815 5761568 5594912
5595744 5633056 5447680 5554400 5530975 5569183 5479072 5685919
5498848 5599775 5544511 5509568 5464863 5661536 5545536 5450848
5462112 5636064 5560704
```

The second reservation was released immediately and the local session record
was removed. The remaining dominant structural difference is that Shuttle's
two runtime row passes each stage K/V independently. Closing it requires two
simultaneous row-fragment state sets, plus compact ldmatrix-based emitted code
so the combined body does not exceed the per-entry PTX transport limit; simply
raising N or adding barriers cannot recover this reuse.

### 2026-08-27 source-verified 128-bit Q/K/V staging

The converted source proves Q, K, and V each enter the online kernel through a
single exact BF16-to-F32 Map whose tensor coordinate map is identity. The MMA
emitter now uses that proof to bypass the redundant scalar BF16 conversion and
stage eight adjacent BF16 elements unchanged with one `ld.global.v4.u32` and
one `st.shared.v4.u32`. Unsupported element types, scalar bodies, or affine
tails remain on the existing scalar route. The selected schedule supplies the
shared bases and row strides; no workload name or static shape tuple selects
the vector path.

The exact S2048/D128 device entry remains below the unchanged transport cap at
507,506 bytes. Its static PTX contains three vector-load sites and three
vector-store sites, no scalar global u16 staging load, 256 MMA instructions,
394 unsigned divides, and 1,107 unsigned remainders. The remaining div/rem
sites are mostly score/softmax/fragment-coordinate work rather than the removed
per-element staging transactions.

A bounded one-H100 RNO run passed all three independent numerical canaries.
The BF16 S2048/D128 case retained maximum absolute error `0.0009765625` and
mean absolute error `0.00005615234375`. Its 51-sample CUDA-event median was
`1.116960 ms`, 2.65 times below the preceding `2.965376 ms` checkpoint and
about 3.46 times the earlier cross-reservation Grug FA4 median. Raw samples:

```
1138592 1097888 1083552 1110464 1133312 1099616 1125824 1126336
1119232 1091008 1144704 1138848 1117280 1066432 1095392 1125568
1105056 1072288 1084256 1114720 1092448 1070400 1123936 1115328
1107744 1143968 1091040 1146240 1087296 1167936 1160384 1093600
1116960 1152320 1125248 1151296 1153120 1094272 1091616 1171968
1104480 1144096 1137856 1092448 1131616 1110240 1154592 1153536
1126752 1111456 1089632
```

SM90a ptxas reported 255 registers, one barrier, a 16-byte stack frame,
20 bytes of spill stores, and 12 bytes of spill loads. The holder was released
immediately after the ptxas measurement; its pod reached terminal Succeeded.

### 2026-08-27 CTA base and N64 byte-pointer recurrences

The source verifier now additionally requires identity BF16 conversion maps
and contiguous trailing feature/value dimensions before selecting the MMA
staging route. B/Hkv and each lane's vector offset are decoded once per CTA.
Each reverse-causal N64 iteration derives K and V with two `mad.wide.u32`
offsets from those persistent bases, then advances loop-carried 64-bit pointers
by one 128-thread transaction stride. Inspection of the emitted tile setup and
K/V copy loop shows zero `div.u32` or `rem.u32`; unsupported affine tails fail
this tuning route rather than retaining the generic address evaluator in the
hot loop.

The exact entry is 507,715 bytes. Local bundle verification and the native
transport suite pass. A second bounded one-H100 RNO cycle passed all three
independent numerical canaries with unchanged BF16 error. The S2048/D128
51-sample median was `0.974304 ms`, 1.15 times below vector staging and 3.04
times below the 2.965376 ms checkpoint. It remains about 3.02 times the prior
cross-reservation Grug FA4 median. Raw samples:

```
972672 1030208 971488 974304 998976 945920 953632 961760 948128
969088 958048 978464 1006688 1000128 1040512 945792 965696 974528
992736 995456 988928 996224 968640 916160 958176 1000768 959808
954080 991872 957984 971904 959168 988256 945920 946400 1034048
978400 960992 989440 997856 991456 955488 974496 981536 994912
981536 985408 969856 951168 943392 989120
```

The recurrence adds two persistent address registers: ptxas still uses 255
registers and one barrier, while stack/spill observations rise to a 32-byte
frame, 28-byte spill stores, and 28-byte spill loads. The holder was released
immediately and its pod reached terminal Succeeded. Address arithmetic is no
longer the dominant multiplicative gap; the next structural target must reduce
live online-output state/register pressure and overlap tile movement, rather
than further micro-tuning the already recurrence-only K/V copy loop.

### 2026-08-27 persistent M128 state and compact fragment loads

The M128 candidate now stages each N64 K/V tile once and carries two
independent 64-row register state sets. A uniform runtime row-fragment loop
selects one `{m,l,o}` set, updates it against the staged tile, and writes it
back before selecting the other set. The higher causal half may consume one
additional tile; once both halves are in range they share every subsequent
K/V stage. No score or probability matrix crosses the kernel boundary.

The shared-memory fragment path now follows the audited Grug organization:
Q and row-major P use `ldmatrix.sync.aligned.m8n8.x4.shared.b16`; K and V use
the transposed two-matrix form. This removed the repeated scalar 16-bit loads,
packing, and address arithmetic from every MMA operand. On the exact BF16
S2048/D128 fixture, the verified PTX entry fell from 479,103 to 292,476 bytes,
well below the unchanged 512 KiB transport limit. The local executable-bundle
pipeline passed, and the proportional native transport binary passed 61/61
tests.

The first bounded H100 request did not reach a usable shell before the
allocation monitor lost its controller tunnel. The allocator reported the pod
active only while the monitor was being aborted. The holder was explicitly
released, and a subsequent status query reported no active session. Therefore
this checkpoint has no ptxas resource, numerical, or timing evidence yet; the
exact ldmatrix fragment mapping and physical register pressure remain the next
hardware discriminators.

### 2026-08-27 single-buffer asynchronous K/V overlap

The SM90 MMA schedule now requests 80 KiB of dynamic shared memory: persistent
Q plus one K tile, one V tile, and a distinct probability tile. The extra
probability allocation removes the former K/P alias without introducing a
second K or V buffer. On schedules with that capability, the emitter uses
16-byte `cp.async.ca.shared.global` transactions in this order: initial K and
wait; current V while QK consumes K; wait for V after QK; next reverse K while
softmax/PV consumes current P/V; wait at the tile backedge. Every transfer is
closed by `commit_group`/`wait_group 0`. A 64 KiB schedule retains the prior
synchronous `ld.global.v4`/`st.shared.v4` phase-aliased fallback.

The exact S2048/D128 entry is 508,298 bytes. Its static PTX has three async
copy sites, three commits, three waits, four CTA barrier sites, 256 MMA
instructions, 272 ldmatrix instructions, 394 unsigned divides, 1,107 unsigned
remainders, and 68 `ex2.approx.f32` instructions. The existing online exp path
therefore already applies base-2 exponentiation with the log2(e) scale
constant; this iteration did not substitute natural exponentiation.

One bounded whole-node H100 run passed all three independent attention
canaries. The BF16 S2048/D128 case retained maximum absolute error
`0.0009765625` and mean absolute error `0.00005615234375`. Its 51-sample
CUDA-event median was `0.941312 ms`, 3.4% below the preceding `0.974304 ms`
checkpoint. The raw samples were:

```
970784 912096 955200 916928 871904 934144 938080 960064 1000704
917568 960768 962560 901056 921952 974432 973248 895296 955424
894400 890848 972736 968192 905664 934592 942784 975872 961504
970432 934080 959424 893120 927584 913120 906944 983872 955488
946080 926368 950624 947488 957632 911392 931904 935744 925408
896512 946528 931808 1001056 962720 941312
```

SM90a ptxas still reports 255 registers, one barrier, a 32-byte stack frame,
28-byte spill stores, and 28-byte spill loads. Transfer overlap is therefore
real but only a small fraction of the remaining cost; register pressure and
the scalar score/softmax address/control body remain unchanged. The measured
median is about 2.92 times the earlier cross-reservation Grug FA4 median of
`0.322400 ms`, so it does not meet the 1.20x target. A matched Grug ptxas run
was not feasible in this holder because its `/app` environment had CPU-only
JAX; rebuilding the patched GPU wheel stack would have extended the expensive
reservation beyond the focused cycle. The holder was released immediately
after ptxas and the local session record was cleared.

### 2026-08-27 register P, swizzled shared tiles, and coalesced output

The next checkpoint removed the probability shared-memory round trip rather
than tuning its barrier. Each lane already owns the exact two adjacent softmax
probabilities needed by one PV A-register pair. The emitter now rounds those
two FP32 probabilities to BF16, packs them into one b32 register, and feeds
the four low-row/high-row, low-key/high-key pairs directly to PV MMA. No P
store, barrier, or ldmatrix reload remains.

The compatible BF16 D/Dv layout also selects the source-derived equivalent of
CuTe `Swizzle<3,3,3>`. Within every 8x64 atom, row bits `[0,3)` XOR the
8-element column-sector bits. A compile-time permutation proof covers all 512
atom elements. Q/K/V vector staging and every matching ldmatrix address apply
the same transform; extents or shared bases that cannot preserve the atom use
the unswizzled route. With P register-resident, shared storage returns to
64 KiB for persistent Q plus one K and one V tile.

Direct scalar output stores were also replaced. After the final tile, Q is
dead, so each lane packs its finalized BF16 output pair into the swizzled Q
allocation. A CTA barrier closes the alias, and a 128-thread loop drains the
tile with coalesced `ld.shared.v4.u32`/`st.global.v4.u32` transactions. The
epilogue remains applied before staging.

The exact S2048/D128 PTX is 383,675 bytes, down from 508,298. It retains 256
MMA instructions but reduces ldmatrix sites from 272 to 144, has no shared
u16 probability operations, and contains one vectorized output drain loop.
The local executable bundle and native transport suite pass.

One bounded whole-node H100 cycle passed all three independent attention
canaries. BF16 S2048/D128 retained maximum absolute error `0.0009765625` and
mean absolute error `0.00005615234375`. Its 51-sample CUDA-event median was
`0.355616 ms`, 2.65 times below the preceding `0.941312 ms` checkpoint. The
raw samples were:

```
373920 373408 351552 367136 347968 358432 355616 355296 348800
357472 358464 352960 340032 368192 370848 341312 339232 352160
355424 365632 346688 348416 369888 358304 352864 351808 357472
355840 353248 346016 364128 343456 361216 364736 349888 359872
349664 354784 347136 366784 356160 354048 358688 351648 366080
363520 362688 345600 353248 359872 358560
```

SM90a ptxas reports 255 registers, one barrier, a 24-byte stack frame,
24-byte spill stores, and 24-byte spill loads; the cubin is 120,264 bytes. The
median is 1.103 times the earlier matched Grug FA4 median of `0.322400 ms`, so
it clears the exploratory 1.20x target with the existing cross-reservation
caveat. The holder was released immediately after ptxas and its session record
was cleared.

### 2026-08-27 same-process Shuttle/XLA/Grug panel

A pinned patched-JAX runtime was built on one H100 from JAX
`619764c15117fbefc4ba13ab941871cb514c23f6` and XLA
`9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`. The first 128-way wheel build
ended in a Clang exit-245 compiler crash after caching 26,939 actions. An
incremental `--jobs=32` retry completed all three JAX CUDA wheels. The panel
then compiled ordinary XLA, Shuttle FAST, and the in-repo Grug CuTe forward in
one Python process with one visible H100, identical BF16 inputs and adapters,
10 warmups, and 51 counterbalanced samples.

The medians were `1.507904 ms` for Shuttle, `1.047168 ms` for ordinary XLA,
and `0.338496 ms` for Grug. Shuttle was therefore 4.455 times the Grug median
and did not satisfy the requested 1.20x condition. This panel used CUDA events
on JAX's single external-ready stream with a DLPack readiness handoff for each
result. JAX 0.10.1 does not expose its internal BF16 compute stream, so these
numbers include the asynchronous dispatch-to-ready boundary and are not a
replacement for the native kernel-only `GpuRunner::ExecuteTimed` result above.
They are still directly matched across all three subjects in one process and
reservation.

All outputs were finite. Shuttle and Grug each had maximum absolute BF16 error
`0.015625` versus ordinary XLA. A separate structurally distinct axis-1 offset
causal softcap canary passed with maximum absolute error `0.0078125` and mean
absolute error `0.00008098303806036711`.

The exact raw samples, in nanoseconds, were:

```
Shuttle:
1571712 1525632 1525824 1512480 1509120 1481888 1492032 1507136 1513632
1507584 1508128 1521120 1500416 1500160 1520896 1514752 1498720 1510080
1512384 1506912 1502336 1507264 1504544 1504608 1507904 1507072 1515040
1517248 1506016 1514432 1501344 1520448 1509088 1508992 1513376 1503744
1485952 1501760 1499680 1504384 1510848 1507360 1509376 1491584 1486272
1520128 1516480 1498848 1496320 1517216 1515552

Ordinary XLA:
1130528 1117152 1055840 1035360 1051744 1058240 1045920 1056288 1048704
1029472 1045440 1057696 1053664 1051904 1038400 1050400 1049344 1051744
1037376 1062592 1045504 1051520 1042944 1047744 1037952 1043520 1046688
1059232 1024960 1060576 1028544 1047168 1046112 1035232 1037856 1064256
1034944 1060288 1045248 1061664 1037088 1029440 1040992 1062272 1036832
1061472 1045568 1066976 1043072 1053824 1041344

Grug CuTe:
369504 337056 342336 335360 339936 332704 346400 330592 349728 328416
344736 326400 348544 331136 347936 327200 340800 332320 324096 341024
339488 330464 336672 335808 346144 336704 343008 335072 326752 341152
340000 338496 336768 337312 349376 342048 339680 336288 359072 347520
345536 322368 352896 326272 345760 329056 344672 327424 336544 323168
346016
```

The Shuttle observer bound the generated bundle to
`210205dcc95707be33813c001e59acc143ec6d5939c4c55e0be689833e6156b2`.
The XLA dump's Shuttle wrapper MLIR SHA-256 was
`66af02cd4edd7d857c15e88f18aed38fdde2944e4e67dbc3e330a348e7b26d93`;
the observer does not expose the embedded Shuttle PTX bytes. The Grug CUTLASS
object was 79,520 bytes with SHA-256/fingerprint
`d01221d6414bee5c3520e8ca6606681dd5a7983f254848f1ea47226c0d77a07c`.
Its object was not a standalone cubin accepted by the retained `nvdisasm`, so
no PTX or cubin resource identity is claimed for that loaded image. The
ordinary XLA attention PTX SHA-256 was
`0e1674bf2bd2d7db9b7f26ee119493c18e4e125362e7b9f30fd25fafc2c575f4`.
The retained CUDA 13.0 nvcc and nvdisasm SHA-256 values were respectively
`0cc55cc74c52e16af787abac2f6901230ac13a8c3ebe9682d25a473eb6e830a4`
and `cfc2fa33538928f78c1c75fd5e25e275d71cd5746fee57d41c98b207d35cc008`.
The reservation was released immediately after artifact capture and status
reported no active session.

### 2026-08-28 execution-scoped initialization cache

The same-process gap was outside the device kernel. XLA creates fresh Prepare
and Initialize state for every execution of the custom-call thunk. Shuttle's
Initialize handler copied the approximately 384 KiB PTX payload and called
`StreamExecutor::LoadKernel` for every launch on every invocation, whereas the
native runner initialized its kernels once. An XSpace trace of three steady
state calls records three `Thunks::Prepare` and three `Thunks::Initialize`
events, confirming that these lifecycle stages are execution-scoped rather
than compilation-scoped.

The runtime now owns an immutable loaded executable in the compilation-lifetime
Instantiate state. Entries are keyed by the exact `StreamExecutor`, so a CUDA
device/context cannot reuse another context's loaded kernels. A mutex guards
the first load and publishes a shared pointer to the PTX storage and kernel
objects; each execution-scoped Initialize state retains that pointer. Prepare
continues to own invocation-local temporary allocations. The lifecycle
regression initializes a fresh XLA `ExecutionState` against the same
Instantiate state and observes zero additional `LoadKernel` calls. Its
concurrent execute coverage also preserves distinct temporary addresses while
sharing the loaded kernel object.

The exact patched JAX/XLA wheel stack compiled successfully on Linux/H100.
The native monolithic test target compiled all Shuttle sources but did not
link under the JAX external-repository invocation because its final link
omitted the external MLIR, Abseil, and Shuttle libraries; this is a Bazel
repository-link configuration failure rather than a test assertion or source
compile failure. The earlier macOS gate was independently blocked by its
obsolete `macosx10.11` SDK selection.

On one visible H100, the patched Shuttle path used identical BF16 inputs,
ordinary-XLA adapter, 10 warmups, 51 counterbalanced CUDA-event samples, and
the same external-ready stream as the preceding panel. Shuttle's median fell
from `1.507904 ms` to `0.374560 ms`, a 4.026-times reduction. Ordinary XLA's
median was `1.041760 ms`. Shuttle remained finite, emitted exactly one observed
generated bundle, and had maximum absolute error `0.015625` versus ordinary
XLA. The raw Shuttle samples, in nanoseconds, were:

```
404320 387616 380864 376160 377632 375008 370176 365440 381536
368128 371264 362688 383552 386272 361472 377952 366656 376256
369632 380992 360800 377920 359936 374560 399712 379936 365632
373280 374144 374976 364480 377024 382176 367840 375616 359520
386976 381120 365792 358752 374880 391104 360192 365632 363712
378496 363776 367008 362272 369248 383616
```

The minimal validation environment could not rebuild Grug: CUTLASS 4.5.2
aborted in its private MLIR binding while lowering a list of four operation
results, before any Grug sample was recorded. Therefore this run does not
claim a new same-process three-way panel. Against the immediately preceding
same-process Grug median of `0.338496 ms` on the same H100 shape and timing
boundary, patched Shuttle is 1.107 times Grug and clears the exploratory 1.20x
condition with that explicitly retained cross-process comparison caveat.

The eight-H100 holder was released immediately after profiling. Kubernetes
subsequently reported the reservation pod as `NotFound`.

### 2026-08-28 exact matched cache panel

The failed Grug rerun above used the PyPI JAX Python package with the patched
JAX libraries. The successful panel had used the exact JAX source revision on
`PYTHONPATH`. Restoring JAX
`619764c15117fbefc4ba13ab941871cb514c23f6`, CUTLASS DSL/base/CUDA 13
`4.5.2`, CUDA Python and bindings `12.9.4`, and CUDA pathfinder `1.5.4`
removed the private CUTLASS MLIR binding failure. An isolated Grug compile and
execute probe passed before the matched panel. No Shuttle or Grug graph,
kernel, adapter, or timing code changed.

The exact patched JAX/XLA wheel stack compiled on one eight-H100 holder. The
panel exposed one H100 and used the same BF16 B1/Hq32/Hkv8/S2048/D128 inputs,
ordinary-XLA adapter, 10 warmups, 51 counterbalanced CUDA-event samples, and
external-ready stream for Shuttle FAST, ordinary XLA, and Grug FA4 in one
Python process. The medians were `0.367936 ms` for Shuttle, `1.030528 ms` for
ordinary XLA, and `0.343424 ms` for Grug. Shuttle was 1.071 times the matched
Grug median and met the exploratory 1.20x forward-H100 condition.

All outputs were finite. Shuttle and Grug each had maximum absolute BF16 error
`0.015625` versus ordinary XLA. The structurally distinct axis-1 offset causal
softcap canary passed with maximum absolute error `0.0078125` and mean absolute
error `0.00008098303806036711`. The observed generated-bundle digest remained
`210205dcc95707be33813c001e59acc143ec6d5939c4c55e0be689833e6156b2`.

The raw samples, in nanoseconds, were:

```
Shuttle:
410688 370400 372128 364000 365376 360160 366848 367936 358432
369504 372640 374784 367136 379232 365056 358240 380352 362592
363264 372544 375328 383488 367360 376096 370656 382560 355840
365184 374816 363008 361888 363360 360160 380576 363712 360576
373600 358720 350272 361888 356960 380032 376000 367104 387744
375680 368192 362304 371744 382624 375648

Ordinary XLA:
1173024 1105344 1030528 1022144 1039008 1020992 1035840 1040992 1002880
1043712 1043840 1013408 1007328 1010048 1020160 1049248 1016512 1046048
1014688 1023456 1027136 1027232 1032928 1034240 1041440 1012192 1025024
1027328 1026432 1013824 1035104 1002336 1015424 1048256 1040160 1042720
1010720 1029312 1033760 1037792 1040896 1043744 1029824 1017952 1034048
1037248 1031456 1021568 1040160 1036640 1008928

Grug FA4:
389248 345600 352032 343840 361088 340192 342784 327456 340096
331360 344160 341888 335328 343424 343712 340800 334752 346208
340000 343328 375616 352032 344384 339360 345088 347168 339904
336160 332640 350656 341280 353856 350400 357728 338336 348384
351616 333152 351552 346560 330016 348992 356096 341440 331712
336448 333856 334336 334752 349056 360576
```

An XSpace capture around three first-use Shuttle calls recorded three
`Thunks::Prepare` and three `Thunks::Initialize` events. An Nsight Systems
CUDA-API capture bounded by `cudaProfilerStart` and `cudaProfilerStop` over
those same three calls recorded one `cuModuleLoadDataEx` and three
`cuLaunchKernel` calls. The counts show one initialization cache miss followed
by two hits; only the first execution loaded the module. The regression test
independently initializes a fresh XLA execution state and observes zero
additional `LoadKernel` calls.

The holder was released immediately after the matched panel and first-use
profile. Kubernetes reported the reservation pod as `NotFound`.

## Future work

- [x] Record the representative-shape admission limit and remove it with the
  cooperative implementation.
- [x] Record matched Shuttle, XLA, and FA4 raw samples and medians.
- [x] Inspect cooperative PTX registers, spills, and launch resources.
- [x] Re-run the F32 numerical canary after the cooperative emitter change.
- [x] Release each H100 reservation immediately after the focused iteration.
