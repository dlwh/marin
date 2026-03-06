# Unified Constraint & Attribute System

## Status

Implemented (Stages 1–6 complete as of 2026-03-05).

Summary of what was done:
- **Stage 1**: Created `src/iris/cluster/constraints.py` with `WellKnownAttribute` enum; replaced all magic string attribute key literals across the codebase.
- **Stage 2**: Added `device_type`, `device_variant`, `device_count`, `preemptible` to `ScaleGroupResources` proto; removed `accelerator_type`/`accelerator_variant` from `ScaleGroupConfig` (reserved); config loader derives `SliceConfig` fields from `resources`; updated all YAML examples.
- **Stage 3**: `build_worker_metadata()` now reads device type/variant/preemptible from `WorkerConfig` fields rather than hardware probes; probing is diagnostic-only.
- **Stage 4**: Added `constraints_from_resources()` in `constraints.py`; `service.py` injects device constraints at job submission; `merge_constraints()` extended to treat `device-type` as a canonical key.
- **Stage 5**: Removed bespoke `device_compatible()`/`device_variant_matches()` from the scheduler; `can_fit()` handles capacity only; device/variant matching flows entirely through `matches_constraints()`.
- **Stage 6**: `DemandEntry` carries `NormalizedConstraints` instead of per-field device/preemptible/zone fields; `ScalingGroup.matches_demand()` replaces the old `matches_device_requirement()` + separate preemptible/zone checks; `vm.proto` and dashboard updated.

`SliceConfig` and `WorkerConfig` protos still carry `accelerator_type`/`accelerator_variant` as platform API parameters (derived from `ScaleGroupResources` by the config loader). These are used by platform modules (`gcp.py`, `local.py`, `coreweave.py`) to call cloud APIs and are not scheduling-relevant.

## Problem

Iris has three parallel, partially-overlapping systems for expressing what a job
needs and what a worker offers:

### 1. The DeviceConfig oneof (ResourceSpecProto.device)

`DeviceConfig` is a protobuf oneof with `CpuDevice`, `GpuDevice`, `TpuDevice`.
Jobs carry it in `ResourceSpecProto.device`, workers carry it in
`WorkerMetadata.device`. The scheduler has bespoke matching in
`WorkerCapacity.can_fit()`: `device_compatible()`, `device_variant_matches()`,
separate GPU/TPU count checks. Device type is extracted via `get_device_type()` →
magic strings `"cpu"`, `"gpu"`, `"tpu"`.

### 2. Worker attributes (map<string, AttributeValue>)

Workers self-report key-value attributes at registration: `tpu-name`,
`tpu-worker-id`, `tpu-topology`, `gpu-variant`, `gpu-count`, `preemptible`,
`zone`, `region`. Keys are magic strings scattered across `env_probe.py`,
`_build_worker_attributes()`, `controller.py`, `types.py`. Some attributes are
dynamically probed (nvidia-smi, GCP metadata server), others come from static
config (`worker.attributes` in YAML).

### 3. Job constraints (repeated Constraint)

Jobs express placement requirements via `Constraint(key, op, value)` matching
against worker attributes. There are "canonical" constraint keys with dedicated
constructors: `preemptible_constraint()`, `zone_constraint()`,
`region_constraint()`, `device_variant_constraint()`.

### What goes wrong

**A. Device matching is split across two systems.** `WorkerCapacity.can_fit()`
has hardcoded device type/variant/count checks separate from the constraint
system. The scheduler checks `device_compatible()` and
`device_variant_matches()` via bespoke code, *then also* checks
`matches_constraints()` for attribute-based constraints.

**B. Autoscaler demand routing duplicates constraint logic.** `DemandEntry`
separately carries `device_type`, `device_variants`, `preemptible`,
`required_regions`, `required_zones` — all of which are *also* expressed as
constraints. `ScalingGroup.matches_device_requirement()` reimplements matching.

**C. Magic strings everywhere.** `"tpu-name"`, `"gpu-variant"`,
`"device-variant"`, `"preemptible"` etc. are string literals scattered across
many files with no single source of truth.

**D. Config-provided values are second-class to probed values.**
`build_worker_metadata()` does prioritize explicit `accelerator_type`/`variant`
from `WorkerConfig` over probed data, but the priority chain is implicit and
spread across multiple branches. The config already declares
`accelerator_type: tpu`, `accelerator_variant: v5litepod-16`, yet the worker
still runs nvidia-smi and GCP metadata probes and uses the results for
scheduling attributes. The config values should be the *only* source for
scheduling-relevant attributes, with probes providing diagnostics only.

**E. YAML config is redundant.** Every scale group must declare
`accelerator_type`/`accelerator_variant` on the group level, *and* repeat
`accelerator_variant` in `slice_template`, *and* set `preemptible` on
`slice_template`, *and* echo `preemptible: "true"` in `worker.attributes`.
The config validator even checks that `worker.attributes.preemptible` matches
`slice_template.preemptible` — proving these are the same value stored in
three places.

## Design Principles

1. **All scheduling decisions flow through constraints.** The constraint system
   is the single mechanism for boolean worker matching (device type, variant,
   zone, preemptible, etc.).

2. **Resources are for capacity accounting.** CPU, memory, disk, and device
   *count* stay in `ResourceSpecProto` because they are deductible — capacity
   shrinks as tasks are assigned.

3. **Config is the source of truth, not probing.** The scale group config
   declares what resources a worker has. That declaration flows to workers via
   `WorkerConfig`. Workers export it as attributes. Dynamic probing becomes
   diagnostic-only.

4. **No magic strings.** All well-known attribute keys live in a single enum.

## New Config Shape

### Scale group YAML (before)

```yaml
tpu_v5e_16:
  zones: [europe-west4-b, us-west4-a]
  accelerator_type: tpu
  accelerator_variant: v5litepod-16
  num_vms: 4
  priority: 30
  resources: { cpu: 112, ram: 192GB, disk: 100GB, tpu_count: 4, gpu_count: 0 }
  min_slices: 0
  max_slices: 256
  slice_template:
    accelerator_variant: v5litepod-16
    preemptible: true
    gcp:
      runtime_version: v2-alpha-tpuv5-lite
  worker:
    attributes:
      preemptible: "true"
```

### Scale group YAML (after)

```yaml
tpu_v5e_16:
  zones: [europe-west4-b, us-west4-a]
  num_vms: 4
  priority: 30
  resources:
    cpu: 112
    ram: 192GB
    disk: 100GB
    device_type: tpu
    device_variant: v5litepod-16
    device_count: 4        # TPU chips per VM (or GPU count)
    preemptible: true
  min_slices: 0
  max_slices: 256
  slice_template:
    gcp:
      runtime_version: v2-alpha-tpuv5-lite
```

Key changes:
- `accelerator_type`/`accelerator_variant` move into `resources` as
  `device_type`/`device_variant`
- `preemptible` moves from `slice_template` into `resources`
- `worker.attributes` is eliminated for well-known keys — the system derives
  them from `resources`
- `tpu_count`/`gpu_count` collapse to `device_count`
- `resources` becomes the single declaration of what the worker provides

### CPU group (before)

```yaml
cpu_vm_ondemand:
  accelerator_type: cpu
  num_vms: 1
  resources: { cpu: 2, ram: 16GB, disk: 100GB, tpu_count: 0, gpu_count: 0 }
  slice_template:
    accelerator_type: cpu
    preemptible: false
    gcp:
      mode: GCP_SLICE_MODE_VM
      machine_type: e2-highmem-2
  worker:
    attributes:
      preemptible: "false"
```

### CPU group (after)

```yaml
cpu_vm_ondemand:
  num_vms: 1
  resources:
    cpu: 2
    ram: 16GB
    disk: 100GB
    device_type: cpu
    preemptible: false
  slice_template:
    gcp:
      mode: GCP_SLICE_MODE_VM
      machine_type: e2-highmem-2
```

## New Proto Shape

### ScaleGroupResources

```protobuf
message ScaleGroupResources {
  int32 cpu_millicores = 1;
  int64 memory_bytes = 2;
  int64 disk_bytes = 3;

  // Device configuration — replaces top-level accelerator_type/variant
  AcceleratorType device_type = 10;
  string device_variant = 11;       // e.g. "v5litepod-16", "H100"
  int32 device_count = 12;          // GPUs or TPU chips per VM

  // Scheduling attributes derived from config
  bool preemptible = 20;
}
```

### ScaleGroupConfig

Remove: `accelerator_type` (field 9), `accelerator_variant` (field 10).
These are now in `ScaleGroupResources`.

`SliceConfig`: remove `accelerator_type` (field 3), `accelerator_variant`
(field 4), `preemptible` (field 6). These are derived from `resources`.

`WorkerSettings.attributes`: remains for *custom* user-defined attributes
only. Well-known keys (`preemptible`, `device-type`, `device-variant`, etc.)
are auto-populated from `resources` — the config loader rejects duplicates.

### WorkerConfig

`WorkerConfig` already has `accelerator_type` (field 20),
`accelerator_variant` (field 21), `gpu_count` (field 22). These stay but are
populated from `resources.device_type`, `resources.device_variant`,
`resources.device_count` during config apply.

Add: `preemptible` (bool) to `WorkerConfig` so the worker knows its own
preemptibility from config.

### WorkerMetadata

The `device` field stays for capacity info (device count for resource
deduction). The TPU/GPU specific metadata fields (`tpu_name`,
`tpu_worker_hostnames`, `gpu_count`, `gpu_name`, etc.) become
diagnostic-only — populated by probing but NOT used for scheduling decisions.

The `attributes` map remains the authoritative source for constraint matching.

## Well-Known Attribute Enum

New file: `src/iris/cluster/constraints.py`

```python
from enum import StrEnum

class WellKnownAttribute(StrEnum):
    """Canonical attribute keys for constraint-based scheduling.

    These keys are auto-populated on workers from scale group config.
    Jobs can constrain on these keys using Constraint objects.
    """
    DEVICE_TYPE = "device-type"          # "cpu", "gpu", "tpu"
    DEVICE_VARIANT = "device-variant"    # "v5litepod-16", "H100"
    PREEMPTIBLE = "preemptible"          # "true", "false"
    REGION = "region"                    # "us-central1", "europe-west4"
    ZONE = "zone"                       # "us-central1-a"
    TPU_NAME = "tpu-name"               # TPU slice name (from GCP metadata)
    TPU_WORKER_ID = "tpu-worker-id"     # Worker index within slice
    TPU_TOPOLOGY = "tpu-topology"       # "v5litepod-16" (same as device-variant for TPU)
    TPU_VM_COUNT = "tpu-vm-count"       # VMs in slice (from topology table)
```

All magic string references across the codebase are replaced with this enum.

## Flow: Config → Worker → Attributes → Scheduling

### 1. Config loading (config.py)

```
YAML resources dict
  → ScaleGroupResources proto
    → device_type, device_variant, device_count, preemptible
```

Config loader auto-derives `SliceConfig` fields from `resources`:
- `slice_template.accelerator_type` ← `resources.device_type` (for platform create_slice API)
- `slice_template.accelerator_variant` ← `resources.device_variant`
- `slice_template.preemptible` ← `resources.preemptible`
- `slice_template.gpu_count` ← `resources.device_count` (if GPU)

### 2. Worker bootstrap (autoscaler → WorkerConfig)

```
ScaleGroupConfig.resources
  → WorkerConfig.accelerator_type = resources.device_type
  → WorkerConfig.accelerator_variant = resources.device_variant
  → WorkerConfig.gpu_count = resources.device_count (if GPU)
  → WorkerConfig.preemptible = resources.preemptible
  → WorkerConfig.worker_attributes = resources → standard attrs
```

The autoscaler builds `WorkerConfig` from the scale group config. Well-known
attributes are populated from `resources` — the worker doesn't need to probe.

### 3. Worker registration (env_probe.py)

```
WorkerConfig
  → worker_attributes from config (device-type, device-variant, preemptible, zone, region)
  → TPU metadata from GCP (tpu-name, tpu-worker-id — still probed, but diagnostic)
  → Hardware probe (CPU count, memory, disk, GPU info — diagnostic only)
```

`build_worker_metadata()` changes:
- Device type/variant come from `WorkerConfig`, not probing
- `preemptible` comes from `WorkerConfig`, not GCP metadata server
- `_probe_gpu_info()` / `_probe_tpu_metadata()` still run for diagnostics
  (populating `WorkerMetadata.gpu_name`, `tpu_worker_hostnames`, etc.)
- Attributes are built from config values, not probed values

**Exception: TPU multi-host metadata.** `tpu-name`, `tpu-worker-id`,
`tpu-worker-hostnames`, `tpu-chips-per-host-bounds` must still come from
GCP metadata because they identify *which specific VM in a TPU slice* this
worker is. The config says "you are a v5litepod-16 worker" but can't say
"you are worker 2 of 4 in slice my-tpu-pod-abc123". These remain probed.

### 4. Controller receives registration

```
WorkerMetadata.attributes includes:
  device-type = "tpu"
  device-variant = "v5litepod-16"
  preemptible = "true"
  zone = "europe-west4-b"
  region = "europe-west4"
  tpu-name = "my-tpu-pod-abc123"  (from probe)
  tpu-worker-id = 2                (from probe)
```

### 5. Job submission → constraint injection

When a job arrives with `resources.device = tpu_device("v5litepod-16")`, the
controller auto-generates constraints from the resource spec:

```python
def constraints_from_resources(resources: ResourceSpecProto) -> list[Constraint]:
    """Auto-generate device constraints from resource spec."""
    constraints = []
    device_type = get_device_type(resources.device)
    if device_type != "cpu":
        constraints.append(Constraint(
            key=WellKnownAttribute.DEVICE_TYPE,
            op=ConstraintOp.EQ,
            value=device_type,
        ))
    variant = get_device_variant(resources.device)
    if variant and variant != "auto":
        constraints.append(Constraint(
            key=WellKnownAttribute.DEVICE_VARIANT,
            op=ConstraintOp.EQ,
            value=variant,
        ))
    return constraints
```

Explicit user constraints merge with (and can override) auto-generated ones.
For example, `device_variant_constraint(["v5litepod-16", "v6e-16"])` overrides
the single-variant constraint from the resource spec.

### 6. Scheduler matching

`WorkerCapacity.can_fit()` is simplified to resource capacity only:
- CPU millicores check
- Memory bytes check
- Device count check (GPU/TPU count deduction)
- Building task limit

Device type and variant matching is handled entirely by
`WorkerCapacity.matches_constraints()` using the posting-list index.

Remove: `device_compatible()`, `device_variant_matches()`,
`WorkerCapacity.device_type`, `WorkerCapacity.device_variant`,
`WorkerSnapshot.device_type`, `WorkerSnapshot.device_variant`.

### 7. Autoscaler demand routing

`DemandEntry` is simplified:

```python
@dataclass
class DemandEntry:
    task_ids: list[str]
    coschedule_group_id: str | None
    normalized: NormalizedConstraints    # replaces device_type, device_variants,
                                         # preemptible, required_regions, required_zones
    resources: ResourceSpecProto
    raw_constraints: list[Constraint]    # for passing through to scheduler
    invalid_reason: str | None = None
```

`NormalizedConstraints` gains `device_type`:

```python
@dataclass(frozen=True)
class NormalizedConstraints:
    device_type: DeviceType | None
    device_variants: frozenset[str] | None
    preemptible: bool | None
    required_regions: frozenset[str] | None
    required_zones: frozenset[str] | None
```

`ScalingGroup.matches_demand(normalized: NormalizedConstraints)` replaces
both `matches_device_requirement()` and the separate preemptible/zone checks
in the autoscaler.

## Constraint Conflict Semantics

When auto-generated constraints from `ResourceSpecProto.device` coexist with
explicit user constraints, merge rules apply:

1. **Auto-generated constraints** are produced from the resource spec's device
   field (device-type, device-variant).
2. **Explicit user constraints** are provided in `LaunchJobRequest.constraints`.
3. **Merge rule**: For each well-known key (`device-type`, `device-variant`,
   `preemptible`, `region`, `zone`), explicit user constraints *replace*
   auto-generated ones entirely. This uses the existing canonical-key merge
   logic in `merge_constraints()` (types.py), extended to include
   `device-type`.
4. **Non-canonical keys** accumulate (AND semantics) — same as today.

Example: Job submits `device=tpu_device("v5litepod-16")` with explicit
`device_variant_constraint(["v5litepod-16", "v6e-16"])`. The auto-generated
`device-variant=v5litepod-16` constraint is *replaced* by the user's IN
constraint. The auto-generated `device-type=tpu` constraint is kept (user
didn't override it).

## Impacted Subsystems

Beyond the scheduler and autoscaler, the following subsystems use device/
preemptible/variant fields and must be updated:

### Reservation matching (`controller.py`)

`_worker_matches_reservation_entry()` currently uses `device_compatible()`
and `device_variant_matches()` for bespoke device matching. After this change
it should use `matches_constraints()` against the reservation entry's
constraints, with auto-generated device constraints injected the same way as
for regular jobs.

### Platform modules (`gcp.py`, `local.py`, `coreweave.py`)

Platform `create_slice()` methods read `SliceConfig.accelerator_variant`,
`SliceConfig.preemptible`, etc. These fields remain on `SliceConfig` as
*platform API parameters* — they are derived from `ScaleGroupResources` by
the config loader, not specified separately in YAML. The GCP platform passes
`--accelerator-type=<variant>` and `--preemptible` to `gcloud` commands.

### Autoscaler status API (`vm.proto`)

`DemandEntryStatus` in `vm.proto` currently has `accelerator_type`,
`accelerator_variant`, `preemptible` fields. These must be updated to
reflect the `NormalizedConstraints` structure for dashboard display.

### Service layer (`service.py`)

Job submission goes through `ControllerService.launch_job()` in `service.py`,
which calls into the controller. Constraint injection happens here, before
the job enters the state machine.

### Dashboard UI (`job-detail.js`, etc.)

Static dashboard files reference device fields for display. These need
cosmetic updates to show constraint-based attributes.

### ControllerWorker (`state.py`)

`ControllerWorker` currently exposes `device_type` and `device_variant`
properties derived from `WorkerMetadata.device`. After this change, these
properties should read from `attributes` instead (or be removed in favor
of direct attribute access).

## Implementation Stages

### Stage 1: WellKnownAttribute enum + magic string replacement

Create `src/iris/cluster/constraints.py` with the `WellKnownAttribute` enum.
Replace all magic string attribute key references across the codebase. Pure
refactor, no behavioral change.

**Files:**
- New: `src/iris/cluster/constraints.py`
- Modified: `types.py` (replace `PREEMPTIBLE_ATTRIBUTE_KEY`, `REGION_ATTRIBUTE_KEY`,
  `ZONE_ATTRIBUTE_KEY`, `DEVICE_VARIANT_ATTRIBUTE_KEY`), `env_probe.py`,
  `scheduler.py`, `controller.py`, `service.py`, `state.py`,
  `scaling_group.py`, `config.py`, `autoscaler.py`
- Tests: `test_types.py`, `test_scheduler.py`, `test_autoscaler.py`,
  `test_state.py`, `test_env_probe.py`, `test_config.py`,
  `test_scaling_group.py`, `test_client.py`

**Invariant:** All existing tests pass unchanged. No behavioral change.

### Stage 2: Proto + config: new ScaleGroupResources fields, YAML migration

Update `config.proto`: add `device_type`, `device_variant`, `device_count`,
`preemptible` to `ScaleGroupResources`. Remove `accelerator_type` (field 9),
`accelerator_variant` (field 10) from `ScaleGroupConfig`. Remove
`accelerator_type` (field 3), `accelerator_variant` (field 4), `preemptible`
(field 6) from `SliceConfig` — config loader derives these.

Update all YAML configs to new format. Update config loader to derive
`SliceConfig` fields from `resources`. Add `preemptible` to `WorkerConfig`
proto. Regenerate protos.

Clean break — no backward compatibility. All YAML files updated atomically.

**Files:**
- Modified: `config.proto`, `config.py` (loader, validators, apply_defaults),
  `scaling_group.py` (read device_type from resources)
- Modified: All `examples/*.yaml` (marin.yaml, smoke.yaml, coreweave.yaml,
  local.yaml, demo.yaml)
- Modified: Platform modules `gcp.py`, `local.py`, `coreweave.py` (read from
  derived SliceConfig fields — no change needed if derivation populates them)
- Generate: `scripts/generate_protos.py`
- Tests: `test_config.py`, `test_scaling_group.py`, `test_autoscaler.py`

**Invariant:** Config loads correctly. Platform create_slice receives correct
accelerator_variant/preemptible via derived SliceConfig fields.

### Stage 3: Worker attributes from config, probing is diagnostic-only

Change `build_worker_metadata()` so device-type/variant/preemptible attributes
come from `WorkerConfig` fields. Probes still run for diagnostic metadata
(`gpu_name`, `tpu_worker_hostnames`, etc.) but are not used for attribute
population.

`_build_worker_attributes()` reads from config:
- `WellKnownAttribute.DEVICE_TYPE` ← `WorkerConfig.accelerator_type`
- `WellKnownAttribute.DEVICE_VARIANT` ← `WorkerConfig.accelerator_variant`
- `WellKnownAttribute.PREEMPTIBLE` ← `WorkerConfig.preemptible`

TPU multi-host metadata (`tpu-name`, `tpu-worker-id`) still probed from GCP
metadata — these are per-VM identity, not config-level.

**Files:**
- Modified: `env_probe.py` (`build_worker_metadata`, `_build_worker_attributes`,
  `DefaultEnvironmentProvider`), `config.py` (WorkerConfig population path in
  autoscaler's `_build_worker_config`)
- Tests: `test_env_probe.py` — verify attributes come from config, not probes

**Invariant:** Worker registers with correct attributes. Scheduling behavior
unchanged.

### Stage 4: Controller auto-injects device constraints from ResourceSpec

Add `constraints_from_resources()` in `constraints.py`. The service layer
(`service.py:launch_job()`) merges auto-generated constraints with explicit
user constraints before storing the job. Extend `merge_constraints()` to
handle `device-type` as a canonical key.

Also update `_worker_matches_reservation_entry()` in `controller.py` to
auto-inject constraints from the reservation entry's resource spec and use
`matches_constraints()` instead of bespoke device matching.

**Files:**
- Modified: `constraints.py` (new `constraints_from_resources()`),
  `types.py` (extend `merge_constraints` canonical keys),
  `service.py` (inject constraints at job submission),
  `controller.py` (reservation matching)
- Tests: New tests in `test_types.py` for `constraints_from_resources()`,
  `test_scheduler.py` for constraint injection, `test_state.py` for
  reservation matching

**Invariant:** Jobs with `device=tpu_device("v5litepod-16")` schedule
identically to before. Reservation matching behavior unchanged.

### Stage 5: Simplify scheduler — remove bespoke device matching

Remove `device_compatible()`, `device_variant_matches()`. Remove
`device_type`/`device_variant` from `WorkerCapacity`/`WorkerSnapshot`.
Remove `device_index` from `SchedulingContext`.
`can_fit()` only checks CPU, memory, device count, building limit.
Device type/variant matching is handled entirely by `matches_constraints()`.

Also update `ControllerWorker` in `state.py` — remove `device_type`/
`device_variant` properties (consumers use `attributes` instead).

**Files:**
- Modified: `scheduler.py` (remove `device_compatible`,
  `device_variant_matches`, simplify `WorkerCapacity`, `WorkerSnapshot`,
  `SchedulingContext`), `state.py` (`ControllerWorker`),
  `controller.py` (any direct `device_type`/`device_variant` access)
- Tests: `test_scheduler.py`, `test_state.py` — same scheduling results

**Invariant:** Full scheduler test suite passes. Same matching behavior,
simpler code.

### Stage 6: Simplify DemandEntry and autoscaler routing

Replace per-field demand routing with `NormalizedConstraints`. Add
`device_type` to `NormalizedConstraints`. Simplify `DemandEntry` to carry
`NormalizedConstraints` instead of separate fields.

Unify `ScalingGroup.matches_device_requirement()` and the separate
preemptible/zone checks into `ScalingGroup.matches_demand(normalized)`.

Update `vm.proto:DemandEntryStatus` to reflect new fields. Update dashboard
status serialization in `autoscaler.py`.

**Files:**
- Modified: `autoscaler.py` (DemandEntry, demand routing, status
  serialization), `scaling_group.py` (matches_demand), `controller.py`
  (compute_demand_entries), `types.py` (NormalizedConstraints),
  `vm.proto` (DemandEntryStatus)
- Modified: `static/controller/job-detail.js` (dashboard display)
- Generate: `scripts/generate_protos.py`
- Tests: `test_autoscaler.py`, `test_scaling_group.py` — same routing

**Invariant:** Autoscaler routes demand identically. Dashboard shows
correct demand info.

### Stage 7: Cleanup and docs

Remove any remaining references to old field names. Update docs:
`AGENTS.md`, `README.md`, `OPS.md`, `docs/autoscaler-v2.md`,
`docs/controller-flow.md`, `docs/worker-flow.md`.

**Files:**
- Modified: docs, AGENTS.md

**Invariant:** Docs match code. No stale references.

## Migration Notes

- TPU-specific metadata (`tpu-name`, `tpu-worker-id`, etc.) still comes from
  GCP metadata probing. This is inherently dynamic — the config can't know
  which specific slice instance a VM belongs to.
- `worker.attributes` in YAML remains for custom user-defined attributes
  (e.g., `pool: large-jobs`). Well-known keys are auto-populated.
- The `DeviceConfig` oneof in `ResourceSpecProto` is unchanged on the wire —
  clients still submit `device=tpu_device("v5litepod-16")`. The controller
  converts this to constraints internally.
