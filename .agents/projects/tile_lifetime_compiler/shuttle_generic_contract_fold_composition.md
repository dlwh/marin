# Generic Contract/Fold composition checkpoint

## Result

The ABI22 planner does not implement the required anonymous SSA composition:

~~~text
h = Contract0(a, b)
s = Fold(h)
z = Map(h, s)
y = Contract1(z, p)
~~~

The required physical cut has two kernels. The first evaluates `Contract0`
and Fold together and writes the private boundary `{h, s}`. The second
evaluates `Map(h, s)` and feeds `Contract1`. `h` is the full unnormalized
Contract result; `s` is the compact Fold result.

Current `deriveNormContractSubplan` cannot represent this graph. It requires
one Fold, one Contract, and one Map containing `rsqrt`, then requires an RNE
BF16 normalization boundary and result epilogue. Generic fusion handles
Map-to-Map and Fold-to-Map edges only. Its code generator has no
Contract-to-Fold or Map-to-Contract composite emitter.

This checkpoint freezes the generic behavior before production changes. It
does not change admission, emit PTX for the composition, or launch hardware.

## SSA-derived partition

The partitioner starts from the two reduction sinks and follows legal producer
edges. It does not inspect module names, function names, dimensions, source
digests, or workload identifiers.

Partition 0 has Fold execution geometry:

~~~text
sources: Contract0, Fold
outputs: full f32 Contract result h, compact f32 Fold result s
~~~

Partition 1 has Contract execution geometry:

~~~text
sources: Map, Contract1
inputs:  private h and s plus public operands
output:  public Contract1 result y
~~~

The two source sets form an exact partition of the source algebra. Every source
has one physical owner. Source coverage verifies the partition, while buffer
verification binds both SSA cut values to the producing and consuming tasks.

## Legality

The composition is admitted only when all of these properties follow from SSA
and operation attributes:

- `Contract0` and `Contract1` satisfy the existing closed projected Contract
  rules: static rank 2 through 8, one reduction iterator, FP32 accumulation,
  supported input types, default precision, and checked positive extents.
- `Contract0` has exactly the two internal users `Fold` and `Map`. Its result is
  private and crosses only from partition 0 to partition 1. An extra user or
  live-out rejects this composition.
- `Fold` is closed additive FP32 reduction with exact positive-zero initializer
  and the declared numerical order. Its result is static FP32 and has fewer
  elements than `Contract0`'s result.
- `Map` is a closed scalar map. One input is `Contract0`; another is the Fold
  result through a legal projected or broadcast indexing map. The Map result
  is consumed by `Contract1` and has no external use.
- Both boundary values are private, produced by partition 0, consumed by
  partition 1, and live only across that edge. Their tensor types, byte extents,
  producer, consumer, and lifetime are derived from the `Contract0` and Fold
  results. `h` and `s` cannot be swapped because they have distinct source
  identities and tensor types.
- Partition 1 owns every public result in function result order. Every public
  operand and result retains an external binding with its original type and
  required byte extent. Slot ordinals are not part of the contract.

Neither `h` nor `s` is a public binding. The Map result `z` remains internal to
partition 1 and cannot appear in materialization buffers or slots.

## Cost decision

The compiler constructs the ordinary materialized plan for the same source and
numerical policy before considering this composition. That baseline evaluates
each source once and materializes the producer results needed by later tasks.

The composed candidate must satisfy all of these comparisons:

- no more launches than the materialized baseline;
- no more temporary bytes or temporary traffic than the baseline;
- at least one strict improvement among launch count, temporary bytes, and
  temporary traffic;
- the only cross-partition temporaries are `h` and `s`;
- `Contract0` is evaluated exactly once and the Map result is never
  materialized.

These are relative checks against a plan derived from the same SSA graph. No
shape table or topology literal selects the candidate. The first implementation
may use one documented compiler-wide launch or temporary-traffic margin, but
that margin must be derived from the materialized baseline and mutation-tested
at its boundary.

## Red contract

Two anonymous algebra fixtures cover rank-2 and rank-3 Contract results, two
unrelated static shape sets, SOURCE_ORDERED and FAST policies, symbol renaming,
and different source ordinals. Each fixture must produce two verified semantic
fusion regions:

1. a Fold-kind region evaluating `Contract0 -> Fold` and exposing private FP32
   `h` and `s`;
2. a Contract-kind region evaluating `Map -> Contract1`, consuming both cut
   values and owning public `y`.

The test locates the two semantic groups by kind and source membership, then
locates buffers by source value, type, storage, producer, consumer, and
live-out role. It does not pin total task count, total slot count, slot
ordinals, grid dimensions, block size, PTX register allocation, fingerprints,
or exact task attribute serialization.

The red mutations cover:

- a public `h` or `s` boundary;
- a nonshrinking or non-FP32 Fold boundary;
- lost or swapped `h` and `s` outputs;
- wrong boundary storage, producer, consumer, or lifetime;
- missing, duplicated, or incorrectly owned source provenance;
- missing or rebound public inputs;
- missing, rebound, or additional public results;
- a candidate whose relative cost does not improve on the materialized plan.

The positive contract rejects materialization of the Map result `z` by allowing
only the required `h` and `s` private cut. Existing closed Contract, Fold, and
Map admission tests remain authoritative for unsupported source operations;
this checkpoint does not duplicate those source-level matrices.

The green implementation must also serialize and load through public
`GpuExecutable::Load`. A later reviewed H100 gate will run each policy twice,
compare the public result with an independent nested-reduction oracle, and
record the observed launch and slot topology as provenance. That later run will
not make its observed decomposition normative.

## Production surfaces after red review

The smallest production change spans five existing boundaries:

1. `deriveGpuPlan` gains the generic sink-rooted partition and relative cost
   comparison.
2. Materialization records bind the exact source partition and both SSA cut
   values for the two generic region kinds.
3. plan and source verifiers enforce the `{h, s}` cut, private lifetime, unique
   ownership, and public result order.
4. Fold schedule emission writes both the projected Contract result and its
   reduction; Contract schedule emission evaluates the declared Map from both
   private inputs without materializing the Map result.
5. transport and FFI retain their existing generic slot, dependency, binding,
   and lifetime validation. No transport schema change is expected unless the
   new provenance cannot remain compiler-internal.

Historical RMSNorm and LayerNorm topology records remain evidence provenance.
They do not select this composition and are not rewritten by this checkpoint.
