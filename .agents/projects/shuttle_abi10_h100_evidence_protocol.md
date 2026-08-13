# ABI 10 H100 evidence protocol

## Status

Pre-run design. No ABI 10 H100 kernel, correctness, cache, or timing result is
accepted by this protocol yet.

## Trust boundary

An evidence JSON document is not an authority. Digests embedded beside the
objects they describe are useful integrity checks, but the document author can
replace both the object and its digest. Acceptance therefore requires two
controller-separated stages and one versioned artifact:

1. A checked-in authorization manifest fixes the source revision, canonical
   inputs and reference, compiler options, generated bundle identities, exact
   collector and verifier programs, timing prohibition, artifact name, and a
   fresh challenge.
2. Iris submits a unique job whose name and argv contain that challenge and the
   authorization digest. The exact workspace ZIP is content-addressed by
   `bundle_id`. The controller records the normalized command, resources,
   authenticated submitter, timestamps, task attempt, and final state.
3. The pinned collector writes one closed artifact directory, hashes every
   file, uploads exactly one immutable-version W&B artifact, and emits the
   qualified artifact version and digest in its task log.
4. A separately submitted pinned verifier downloads that exact W&B artifact
   version, validates the raw files, joins them to the Iris submission and task
   records, and emits its own verdict receipt. The controller query rows and raw
   finelog rows, including their server-assigned sequence numbers, are exported
   before retention can remove them.

The supported claim is **controller-observed, content-addressed, independently
revalidated, and externally exported**. Iris does not currently sign individual
job or log rows, W&B is not treated as hardware attestation, and this protocol
must not be described as cryptographically unrewritable.

## Pre-run authorization

The authorization is immutable once submitted. It contains:

- the exact Marin commit/tree, JAX commit/tree, XLA revision, three JAX patch
  digests, canonical compiler options, and contract digest;
- canonical input raw/array digests and independent-reference digest/limits;
- exact generated/final observer fingerprints, transport/device descriptor
  identities, and the ordered 19-entry PTX slice manifest;
- exact `ptxas` version and executable digest;
- exact TE runner source, harness, build manifest, sealed-plan template, and
  collector/verifier program digests;
- one globally unique challenge, one job-name prefix, the one allowed W&B
  entity/project/artifact name, and a prohibition on timing fields;
- the required H100 class, GPU count, CUDA-visible device, command, and
  error-on-existing submission policy.

The workspace ZIP must contain this authorization verbatim. Its Iris
`bundle_id` is the SHA-256 of the exact ZIP bytes and is checked against the
controller's persisted launch request.

## Raw artifact

The collector artifact contains raw files, not copied claims:

- canonical BF16 input files, independent-reference output, ordinary-JAX
  output, three sequential Shuttle outputs, and two concurrent outputs;
- the serialized GPU transport and independent decoded descriptor;
- all 19 PTX slices, all 19 cubins, and one bounded per-ordinal `ptxas` record
  with input/output digests and resource data;
- the rebuilt wheel, CUDA plugin extension, registration transcript, and exact
  observer-event transcript;
- populate/reuse process records, cache directory snapshots, and the actual
  cache entry bytes;
- the sealed TE plan, execution identity, runner binary, and the four exact
  forward result artifacts validated through the existing TE harness;
- the collector's own canonical record chain and a closed file manifest.

The verifier recomputes every file size and SHA-256. It independently decodes
the transport, derives the ordered PTX slices, and compares their bytes with the
PTX files. It invokes `ptxas` identity checks and validates cubin/resource
records per ordinal. It validates the registration transcript against the
actual plugin file, the cache transition against the actual files and distinct
Iris attempts, output metrics from raw BF16 bytes, and every TE result through
the sealed plan and execution identity. No embedded `accepted` boolean can
advance a status.

## Iris receipts

The pre-execution receipt is the controller `job_submitted` row plus the
persisted `jobs`/`job_config` record. The post-execution receipt adds the final
job state, `task_attempts.attempt_uid`, and the raw postseal log row. The
verifier has a different job ID, bundle, attempt UID, and log row. Exact SQL and
raw log requests live beside the verifier so receipt collection is
reproducible.

Finished Iris rows and logs have finite retention and are not WORM. Export both
receipt sets immediately into the versioned evidence artifact and the task
logbook. A missing live controller join, missing exact W&B artifact version, or
expired/unavailable receipt leaves the observation unqualified.

## Timing gate

The untimed artifact and verifier verdict must be complete before a timing
authorization is submitted. Timing uses a separate challenge, collector, W&B
artifact, Iris jobs, and verifier receipts. It may refer to the accepted untimed
artifact by exact W&B version and digest, but cannot copy or rewrite untimed
module, executable, oracle, or qualification identities.

No H100 reservation or timing run is authorized merely by this design file.
