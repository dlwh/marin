# Native schema-v3 conformance capture

`jax-bf16-batched-contract-forward-sm90.json` is the request received by the
native physical-artifact provider from an actual C++ StableHLO-to-Shuttle pass
pipeline. It is not a Python-constructed request fixture.

Source program:

- JAX 0.10.1 / jaxlib 0.10.1
- `lib/shuttle/mlir/test/Inputs/jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir`
- source SHA-256: `7f6b95815dfa55e5a8bdf6617b1221f8bf791ca9beb0acec09da11637899bfd6`

Native pipeline:

1. `AnnotateSource`
2. `FormStructuralRegions` with FAST numerical policy
3. `ConvertStablehloToAlgebra`
4. `PlanRowFoldMaterialization`
5. `PlanSimt32RowFoldSchedule`
6. `PlanSourceEventLowering`
7. `RecordingFatbinProvider`

The capture was made on 2026-09-03 by the
`GpuTransportV2Test.SourceBoundOnlineRegionCanUseNativePhysicalArtifactProvider`
harness compiled with Bazelisk 1.28.1 / Bazel 9.2.0. A temporary provider-side
capture hook serialized the request before artifact attachment and was removed
after capture. The harness subsequently rejected the dummy artifact because it
does not contain physical certificates for the dense request's exact Events;
the provider had already received the complete request.

Capture identities:

- request bytes SHA-256: `6bc66bc45689936ea37685175f24fba3ae2beec34e7cd33e772058810473a940`
- native request fingerprint: `c34bcc6ea82dc94096288f4130a208798cff35254715742a374b854952335457`
- canonical TensorProgram fingerprint: `71cd4f540fd999ebbd9e5da29fcebf1f4fdd5fdd189840ef7e8585e30da0bb18`
- adapted task-plan fingerprint: `bfa757090c2bed8f3dc999e0b95bd2c86866fc81728c3b50548d1e8bd8fd853b`
- canonical Event-plan fingerprint: `0127493a6a0f5625aaf2c825657072fb272091c6eafea3383e01de01d7f34c77`
- generic full-tile task-plan fingerprint: `888e934bdd7164908127da38be9bdb9ac2287eaeeb15507752209908dbfcc034`
- generic executable physical-request fingerprint: `deae8eec60350de90854ec29bb9ce5bfc359fde652e32f223e02b8942d197952`

The conformance adapter checks two explicit quotients. BF16-to-FP32 widening is
identical whether its conversion annotation says exact or round-to-nearest-even,
because every BF16 value is exactly representable in FP32. The native plan's
unit-count, per-edge Event storage is an injective refinement of the canonical
consumer-counted Event Tensor; storage slot plus generation must remain unique
for every active relation coordinate.

The adapted native plan and generic planner intentionally have different task
fingerprints because the native SIMT schedule flattens owner tiles while the
generic comparison uses one full logical tile per axis. Their algebra identity,
operation partition, and task-dependency graph agree. The generic plan then
forms a complete executable physical request without workload-name dispatch.

## Attention capture

`jax-bf16-causal-gqa-online-axis4-sm90.json` is a second request from the same
native pipeline and provider hook, sourced from
`lib/shuttle/mlir/test/Inputs/jax-0.10.1-bf16-causal-gqa-online-axis4.mlir`.
Its source SHA-256 is
`31c410b668238c198c127892efe3cab20c47a22595f04fbe5c209b0eeb7700d2`.

Capture identities:

- request bytes SHA-256: `e189ef4bcfef68a1c27b88988fd1d8f5134ae3e1e32d45bf2ed395bd8b5bfa4c`
- native request fingerprint: `d46020f16c5d26c96bb0191ca48263b9ad158b2ba7fcee8a1c3ddc991e897ada`
- canonical TensorProgram fingerprint: `d12018ef43a750134539a775a5a64770ab3b10c2a6ec475cd04e6f474804b184`
- adapted fused task-plan fingerprint: `26731db564a37898850636b82d7b8b7c737b326f0a44ff6f75a1229271eafc7a`
- generic full-tile task-plan fingerprint: `3866ca5a3043235b36119cafd5a974dd6455bccc81210629378e6544f5d9bda9`
- generic executable physical-request fingerprint: `5a2a74f127c073eb9a91d114b46c1bf671bda27250a13f7ea3a0212997fd1fbc`

The adapter proves the native source graph and ordinary-JAX TensorProgram have
the same scalar and index algebra. It generically absorbs scalar constants and
broadcasts into scalar Maps, absorbs reducer identity constants into Folds,
maps native compare/select Relations to scalar AST nodes, verifies iota result
expressions, and checks singleton-broadcast indices by their exact bounded
index semantics. The native task is fully fused and therefore has no external
Event Tensor.

The current capture also carries the native planner's typed
`normalized_exp_contract_regions` record. Its operation/result references bind
the producer Contract, scalar score Map, both Fold states, normalized weight,
consumer Contract, and finalizer at the provider boundary; the record is part
of the request fingerprint rather than rediscovered by provider-side topology
scanning.

The remaining task-level difference is explicit: the generic composition
planner leaves index-only Relations and boundary conversions in separate task
families, while the native FAST planner internalizes them into the normalized
Contract/Fold/Contract task. Both plans cover the same typed algebra, and the
generic plan forms a valid physical request, but their task and schedule
identities do not yet share a justified coordinate refinement. The current
generic plan has 18 task families and 19 exact inter-task Event Tensors; the
native plan has one task family with an 18-block flattened launch grid and no
external Events.

The schedule records also retain two independent physical-model mismatches.
The canonical algebra exposes the producer feature contraction as a distinct
extent-8 axis and the normalized/consumer contraction as the extent-17 stream
axis. The native schema records both producer and consumer contraction
dimensions as its extent-17 stream dimension, so the adapted native task has
no separate producer contraction axis. In addition, the native record is a
source-ordered `native_structured` schedule with 98,304 bytes of scratch,
whereas the generic request is an unordered `logical_row_major` schedule with
estimated register resources. These records must not be declared identical
until the native task schema preserves the distinct contraction coordinate and
the generic planner derives a compatible physical schedule.

## Value-and-pullback capture

`jax-bf16-causal-gqa-value-and-grad-s128-d128-sm90.json` is the request
submitted by the same native provider boundary for the ordinary-JAX program
that returns the BF16 forward value and its three VJP results. Its source is
`lib/shuttle/mlir/test/Inputs/jax-0.10.1-bf16-causal-gqa-value-and-grad-s128-d128.mlir`
(SHA-256
`597cb67a79409ae2498c66ea9cd30638caec47fd99bc305e5abc0d0bea81f662`).

Capture identities:

- request bytes SHA-256: `e3d7ca179f2f644240c1eb355652ac640bd91c671894025a1a56ab6efb4f1aeb`
- native request fingerprint: `2e6ed38db23b92b8f928769871273c2b488a005b53dd37c42822bca97d4e6aca`
- materialization fingerprint: `1ce0418f895bc6594e851b6d415ec5647a2c564f32eed615170f8d85e3b5fded`
- schedule fingerprint: `88a224806bf834940771a6668cb47bea6f1618ebe93dd2e7214886cc0176b933`
- Event-lowering fingerprint: `ec101f416d36d98c3f15983930c7226159455a4c935014bbac1fb62ff2135c66`

The request contains 83 source-derived native algebra operations, three fused
Contract/Fold task families, three exact inter-task Event Tensors, and ten
imperative Event-lowering operations. The first remaining generic algebra
quotient is precise: the ordinary-JAX TensorProgram retains a dead full-shape
constant Map before its maximum Fold, while native StableHLO conversion keeps
the same reducer identity as a scalar constant consumed by the Fold. Exact SSA
grouping therefore stops at canonical operation 15 and native operation 18;
the adapter does not silently erase or manufacture that equivalence.

The exact scalar `jax.value_and_grad` objective used by the executable
reference is preserved separately as
`lib/shuttle/mlir/test/Inputs/jax-0.10.1-bf16-causal-gqa-scalar-value-and-grad-s128-d128.mlir`
(SHA-256
`30fe1a6c162528d1e91b8eed1b851bce3ac34cb25324379b6c63b9b32752972b`).

`jax-bf16-causal-gqa-scalar-value-and-grad-s128-d128-sm90.json` is the native
schema-v3 request now submitted for that scalar program. It was captured on
2026-09-03 by
`GpuTransportV2Test.ScalarObjectivePullbackPhysicalRequestCarriesExactTaskAndBufferRoles`
using Bazelisk 1.28.1 / Bazel 9.2.0. The request was serialized by the recording
provider before it deliberately declined artifact attachment.

Capture identities:

- request bytes SHA-256: `1e9a5797a1147f863ea5e426a15fbfa53ce25e56552cd5c851c8d91cf2f447e6`
- native request fingerprint: `e8bd9084c1486074f14329d14906732798e9c52364646cb71c3a6c85c9c4301e`
- materialization fingerprint: `e5318253b3d6789219feb1699e9884d02ebd3390f8ce321c3b116f991c17697f`
- schedule fingerprint: `e156d55a5450bd61be6d2c05c6dbdbb2e5ba5f172c1e567e0e893e1898de298c`
- Event-lowering fingerprint: `05faf73653fb624f8ecf94fcd9b8003db45112a562b898de5290a2a60d600533`

The request contains 91 source-derived algebra operations plus the region
yield, four materialization tasks, six exact inter-task Event Tensors, and one
schedule-local Event Tensor from the scalar Fold's partial phase to its merge
phase. The local Event has one generation, device release/acquire visibility,
and 128 storage slots, one for each private partial. It is retained as an
imperative notify/wait pair until the physical two-launch refinement proves a
strict same-stream partial-before-merge order, at which point it erases to that
exact launch-order witness. The
typed pullback record binds forward, scalar-objective, owner-pullback, and
streamed-pullback tasks; maximum, normalizer, and output-dependent state
buffers; the forward result and output cotangent; and all three terminal
cotangent buffers. The scalar objective Fold has no surviving owner axes and
retains all five reduction axes explicitly. Its hierarchical schedule uses 128
partial CTAs over 4096-element tiles, a private FP32 partial buffer of shape
`[128]`, and one 256-thread merge CTA. Both physical launches refine the same
canonical Fold task, so its incoming and outgoing Event realizations bracket
the complete partial/merge launch range.

## Natural-JAX MoE value-and-gradient capture

`jax-bf16-moe-value-and-grad-small-sm90.json.gz` is the compressed native
schema-v3 request for the ordinary-JAX `MoEDebugConfig(tokens=4, hidden=4,
intermediate=6, experts=3, top_k=2)` value-and-gradient program. It was captured
from `/private/tmp/moe-vg-small.mlir` by the recording provider after the v7
generic independent-Map co-launch planner, before artifact attachment.

Capture identities:

- source MLIR SHA-256: `8a1a5414736fc1f9ab40a438e77abed06cedb44d781ef3e2f9f6bace6747b35c`
- uncompressed request bytes SHA-256: `0229b135d018e9fcd04cdbce608f3c7a4612a8e9507d761b959b42d48de6c0e6`
- compressed fixture SHA-256: `dc7a275e61282d7b5de8825ce289c2b631f9d2f5d74732e13d5526a3bd8edbb2`
- native request fingerprint: `8c95100dcfd9f7ccf7943b4f92fdd706dea27d028eb3bafc41c9b1bf4100787a`
- materialization fingerprint: `991f7b28228cfb2b6b06dcdf3a2282b69ad49a9c23b687713127414fe1571913`
- schedule fingerprint: `51ae7c7f775c5ffd7975ee7dc6dd481f13f9bb8b2129941fa20e7bc7e6bf2cd5`
- Event-lowering fingerprint: `0807deba1ab4c8b8bab028194d8d92d822c323f628e876b87906e54f0c06caa3`

The fixture intentionally exposes the next exact normalization boundary. The
native source contains seven shared nullary literal operations before its first
Contract; the executable TensorProgram folds those literals into scalar and
index ASTs. The adapter currently requires consecutive native operation groups,
so conformance stops at canonical operation 0 rather than assigning one hoisted
literal to multiple consumer groups. This is an unavailable proof, not evidence
of equality or a semantic mismatch. Partial stateless ownership-role records
and one-thread scalar schedules are decoded before this boundary and retain
their literal native meaning.
