# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Profile the ordinary-JAX dense value-and-grad composition on one H100."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from cute_launch_argument_capture import preflight_cute_launch_argument_capture
from h100_first_forward_fixture import export_first_forward_fixture

from tile_lifetime.backends.h100.cute_generic_tensor_program import (
    build_generic_component_multi_image_artifact,
    compile_generic_kernel_composition,
)
from tile_lifetime.backends.h100.cute_launch_abi import validate_cute_launch_abi
from tile_lifetime.backends.h100.cute_normalized_exp_training_component import (
    sm90_split_normalized_exp_component_compiler,
)
from tile_lifetime.canonical_kernel_composition import (
    FusedComponentEvaluation,
    coalesce_private_map_fold_components,
)
from tile_lifetime.dense_region import DenseCompositionShape, DenseCompositionWeights, dense_composition_forward
from tile_lifetime.jax_normalized_exp_contract_pullback import trace_jax_owned_pullback
from tile_lifetime.jax_normalized_exp_contract_training import (
    JaxOwnedNormalizedExpValueAndGradTrace,
    prepare_sm90_split_normalized_exp_value_and_grad,
)
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.physical_task_contract import PhysicalTarget
from tile_lifetime.streaming_attention_executable import jax_tensor_program_executable_request
from tile_lifetime.tensor_program import NumericalPolicy


def timed(call):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    result = call()
    jax.block_until_ready(result) if isinstance(result[0], jax.Array) else torch.cuda.synchronize()
    return (time.perf_counter() - begin) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4)
    parser.add_argument("--model", type=int, default=8)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--feature", type=int, default=128)
    parser.add_argument("--feed-forward", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--coalesce-private-map-fold", action="store_true")
    parser.add_argument("--describe-contract-bindings", action="store_true")
    parser.add_argument("--sm90-contract", action="store_true")
    parser.add_argument("--same-coordinate-ssa", action="store_true")
    parser.add_argument("--comparison", choices=("jax", "recursive"), default="recursive")
    parser.add_argument("--maximum-absolute-error", type=float, default=6.103515625e-05)
    parser.add_argument("--native-fixture-directory", type=Path)
    parser.add_argument("--first-forward-fixture-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.kv_heads is None:
        args.kv_heads = args.heads
    paired_baseline = args.comparison == "recursive" and (
        args.coalesce_private_map_fold or args.sm90_contract or args.same_coordinate_ssa
    )
    if args.sm90_contract or args.describe_contract_bindings:
        # The optional provider requires its recovered QuACK runtime, beyond the baseline CUDA extra.
        from tile_lifetime.backends.h100.cute_contract_component import (  # noqa: PLC0415
            sm90_contract_component_compiler,
        )
        from tile_lifetime.sm90_contract_binding import sm90_contract_component_binding  # noqa: PLC0415
    root = Path(__file__).resolve().parents[3]
    source_paths = [
        path
        for directory in ("lib/tile_lifetime/src", "lib/shuttle/src")
        for path in sorted((root / directory).rglob("*.py"))
    ]
    source_paths.append(Path(__file__))
    source_paths.append(Path(__file__).with_name("h100_first_forward_fixture.py"))
    source_paths.append(Path(__file__).with_name("cute_launch_argument_capture.py"))
    initial_digests = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
    }
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name():
        raise ValueError("This evidence runner requires exactly one visible H100")
    if args.first_forward_fixture_directory is not None:
        preflight = preflight_cute_launch_argument_capture()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "first-forward-preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        print(json.dumps({"phase": "argument_capture_preflight_passed", **preflight}), flush=True)
    planning_started = time.perf_counter()
    print(json.dumps({"phase": "planning_started"}), flush=True)
    with jax.default_device(jax.devices("cpu")[0]):
        keys = jax.random.split(jax.random.key(91), 11)

        def array(key, shape):
            return (0.05 * jax.random.normal(key, shape)).astype(jnp.bfloat16)

        source = array(keys[0], (1, args.sequence, args.model))
        weights = DenseCompositionWeights(
            input_scale=array(keys[1], (args.model,)),
            query=array(keys[2], (args.model, args.heads, args.feature)),
            key=array(keys[3], (args.model, args.kv_heads, args.feature)),
            value=array(keys[4], (args.model, args.kv_heads, args.feature)),
            output=array(keys[5], (args.heads, args.feature, args.model)),
            mlp_scale=array(keys[6], (args.model,)),
            gate=array(keys[7], (args.model, args.feed_forward)),
            up=array(keys[8], (args.model, args.feed_forward)),
            down=array(keys[9], (args.feed_forward, args.model)),
        )
        shape = DenseCompositionShape(args.heads, args.feature, args.feature)
        primals = (source, *weights)

        def forward(x, *parameter_leaves):
            return dense_composition_forward(x, DenseCompositionWeights(*parameter_leaves), shape)

        def value_and_grad(x, *parameter_leaves):
            loss, gradients = jax.value_and_grad(
                lambda primal, *parameters: jnp.mean(forward(primal, *parameters).astype(jnp.float32) ** 2),
                argnums=tuple(range(len(primals))),
            )(x, *parameter_leaves)
            return loss, *gradients

        forward_program = jax_function_to_tensor_program(forward, *primals, numerical_policy=NumericalPolicy.FAST)
        source_program = jax_function_to_tensor_program(value_and_grad, *primals, numerical_policy=NumericalPolicy.FAST)
        trace = JaxOwnedNormalizedExpValueAndGradTrace(
            forward_program,
            source_program,
            trace_jax_owned_pullback(
                forward,
                *primals,
                output_cotangent=array(keys[10], (1, args.sequence, args.model)),
                numerical_policy=NumericalPolicy.FAST,
            ),
        )
        request = jax_tensor_program_executable_request(source_program, tile_size=128, target=PhysicalTarget.SM90)
        plan = prepare_sm90_split_normalized_exp_value_and_grad(request, trace, tile_size=128)
    planning_record = {
        "phase": "planning_complete",
        "elapsed_seconds": time.perf_counter() - planning_started,
        "components": len(plan.composition.components),
        "crossing_events": len(plan.composition.concrete_events),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name(args.output.stem + "-planning.json").write_text(json.dumps(planning_record, indent=2) + "\n")
    print(json.dumps(planning_record), flush=True)
    baseline_plan = plan
    if args.coalesce_private_map_fold:
        composition = coalesce_private_map_fold_components(plan.program, plan.composition)
        plan = replace(plan, composition=composition, physical_request=composition.scheduled_request)
    if args.describe_contract_bindings:
        bindings = []
        for component in plan.composition.components:
            binding = sm90_contract_component_binding(plan.program, component)
            if binding is not None:
                bindings.append(
                    {
                        "component_id": component.id,
                        "operations": component.operation_ordinals,
                        "rows": binding.rows,
                        "reduction": binding.reduction,
                        "features": binding.features,
                    }
                )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(bindings, indent=2) + "\n")
        print(json.dumps(bindings), flush=True)
        return
    torch_inputs = tuple(torch.tensor(np.asarray(x, dtype=np.float32), device="cuda").bfloat16() for x in primals)
    gpu_inputs = tuple(jax.device_put(x, jax.devices("gpu")[0]) for x in primals)
    reference = jax.jit(value_and_grad)
    expected = reference(*gpu_inputs)
    jax.block_until_ready(expected)
    reference_samples = [timed(lambda: reference(*gpu_inputs)) for _ in range(args.iterations)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name(args.output.stem + "-jax-baseline.json").write_text(
        json.dumps(
            {
                "phase": "jax_baseline_before_generated_compilation",
                "case": {
                    key: value
                    for key, value in vars(args).items()
                    if key not in {"output", "native_fixture_directory", "first_forward_fixture_directory"}
                },
                "source_digests": initial_digests,
                "host_ms": reference_samples,
                "median_host_ms": statistics.median(reference_samples),
                "timing_note": "Diagnostic baseline before generated compilation; final timings are interleaved.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps({"phase": "jax_baseline_complete", "median_host_ms": statistics.median(reference_samples)}),
        flush=True,
    )
    physical_compilers = (sm90_split_normalized_exp_component_compiler(plan),)
    if args.sm90_contract:
        physical_compilers += (sm90_contract_component_compiler(),)
    print("Compiling generated composition", flush=True)
    executable = compile_generic_kernel_composition(
        plan.program,
        plan.composition,
        example_inputs=torch_inputs,
        physical_component_compilers=physical_compilers,
        fused_evaluation=(
            FusedComponentEvaluation.SAME_COORDINATE_SSA
            if args.same_coordinate_ssa
            else FusedComponentEvaluation.RECURSIVE
        ),
    )
    print("Generated compilation complete", flush=True)
    physical_components = []
    for component in executable.components:
        if component.physical_component is not None:
            physical_components.append(
                {"id": component.component_id, "type": type(component.physical_component).__name__}
            )
        for phase_index, phase in enumerate(component.reduction_phases):
            if phase.physical_component is not None:
                physical_components.append(
                    {"id": component.component_id, "phase": phase_index, "type": type(phase.physical_component).__name__}
                )
    diagnostic_fixture = None
    if args.first_forward_fixture_directory is not None:
        fixture = export_first_forward_fixture(
            executable, torch_inputs, args.first_forward_fixture_directory / "forward", args.maximum_absolute_error
        )
        if any(
            hashlib.sha256((root / path).read_bytes()).hexdigest() != digest for path, digest in initial_digests.items()
        ):
            raise ValueError("Source changed during diagnostic fixture export")
        record = {"source_digests": initial_digests, "artifacts": {"forward": fixture}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "first-forward-diagnostic.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            json.dumps({"phase": "first_forward_fixture_complete", "component_id": fixture["component_id"]}), flush=True
        )
        if args.native_fixture_directory is None:
            args.output.write_text(json.dumps(record, indent=2) + "\n")
            return
        diagnostic_fixture = fixture
    actual = executable(*torch_inputs)
    errors = []
    for output, target in zip(actual, expected, strict=True):
        error = np.abs(output.float().cpu().numpy() - np.asarray(target, dtype=np.float32))
        if not np.isfinite(error).all():
            raise ValueError("Nonfinite generated/reference difference; timing rejected")
        errors.append({"max_abs": float(error.max()), "mean_abs": float(error.mean())})
    numerical_acceptance = all(error["max_abs"] <= args.maximum_absolute_error for error in errors)
    calls = {"generated": lambda: executable(*torch_inputs), "jax": lambda: reference(*gpu_inputs)}
    baseline_comparison = None
    if paired_baseline:
        baseline_compilers = (sm90_split_normalized_exp_component_compiler(baseline_plan),)
        if args.sm90_contract:
            baseline_compilers += (sm90_contract_component_compiler(),)
        baseline_executable = compile_generic_kernel_composition(
            baseline_plan.program,
            baseline_plan.composition,
            example_inputs=torch_inputs,
            physical_component_compilers=baseline_compilers,
        )
        baseline_outputs = baseline_executable(*torch_inputs)
        baseline_comparison = [
            {
                "max_abs": float((a.float() - b.float()).abs().max()),
                "mean_abs": float((a.float() - b.float()).abs().mean()),
                "bitwise_equal": torch.equal(a, b),
            }
            for a, b in zip(actual, baseline_outputs, strict=True)
        ]
        calls["baseline"] = lambda: baseline_executable(*torch_inputs)
    baseline_errors = None
    if paired_baseline:
        baseline_errors = [
            float(np.abs(a.float().cpu().numpy() - np.asarray(b, dtype=np.float32)).max())
            for a, b in zip(baseline_outputs, expected, strict=True)
        ]
        numerical_acceptance = numerical_acceptance and all(
            np.isfinite(error) and error <= args.maximum_absolute_error for error in baseline_errors
        )
    if not numerical_acceptance:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "phase": "numerical_gate_failed",
                    "numerics": errors,
                    "baseline_max_abs_vs_jax": baseline_errors,
                    "maximum_absolute_error_gate": args.maximum_absolute_error,
                },
                indent=2,
            )
            + "\n"
        )
        raise ValueError("Generated or baseline numerical gate failed; timing rejected")
    for _ in range(args.warmup):
        for call in calls.values():
            timed(call)
    samples = {name: [] for name in calls}
    variants = tuple(calls)
    for iteration in range(args.iterations):
        shift = iteration % len(variants)
        order = variants[shift:] + variants[:shift]
        if (iteration // len(variants)) % 2:
            order = tuple(reversed(order))
        for name in order:
            samples[name].append(timed(calls[name]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "phase": "whole_program_complete_diagnostics_pending",
                "numerics": errors,
                "baseline_max_abs_vs_jax": baseline_errors,
                "treatment_vs_baseline": baseline_comparison,
                "numerical_acceptance": numerical_acceptance,
                "maximum_absolute_error_gate": args.maximum_absolute_error,
                "gate_provenance": (
                    "Prospective gate using retained tiny-case observed maximum; not an inherited tolerance contract."
                ),
                "host_ms": samples,
                "trace_fingerprint": trace.fingerprint,
                "program_fingerprint": plan.program.semantic_fingerprint,
                "physical_request_fingerprint": plan.physical_request.fingerprint,
                "composition_fingerprint": plan.composition.fingerprint,
                "crossing_event_count": len(plan.composition.concrete_events),
                "component_count": len(executable.components),
                "physical_components": physical_components,
                "launch_count": sum(len(c.kernel_names) for c in executable.components),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "phase": "whole_program_complete",
                "numerics": errors,
                "median_host_ms": {name: statistics.median(values) for name, values in samples.items()},
            }
        ),
        flush=True,
    )
    environment = executable.execute_environment(*torch_inputs)
    torch.cuda.synchronize()
    components = []
    for component in executable.components:
        # Execute the same runtime on a single component with its already materialized boundary.
        isolated = replace(
            executable,
            program=replace(
                plan.program,
                inputs=component.live_inputs,
                outputs=component.live_outputs,
                operations=tuple(plan.program.operations[i] for i in component.operation_ordinals),
            ),
            components=(component,),
        )
        inputs = tuple(environment[value] for value in component.live_inputs)
        for _ in range(args.warmup):
            isolated.execute_environment(*inputs)
        torch.cuda.synchronize()
        event_samples = []
        for _ in range(args.iterations):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            isolated.execute_environment(*inputs)
            end.record()
            end.synchronize()
            event_samples.append(begin.elapsed_time(end))
        components.append(
            {
                "id": component.component_id,
                "operation_ordinals": component.operation_ordinals,
                "kernel_names": component.kernel_names,
                "fusion_rejection": component.fusion_rejection,
                "boundary_input_bytes": sum(x.numel() * x.element_size() for x in inputs),
                "boundary_output_bytes": sum(
                    environment[x].numel() * environment[x].element_size() for x in component.live_outputs
                ),
                "cuda_event_ms": event_samples,
                "cuda_event_median_ms": statistics.median(event_samples),
            }
        )
    graph_record = None
    if args.cuda_graph:
        graph_executables = {"generated": executable}
        eager_outputs = {"generated": actual}
        if paired_baseline:
            graph_executables["baseline"] = baseline_executable
            eager_outputs["baseline"] = baseline_outputs
        graphs = {}
        graph_outputs = {}
        for name, compiled in graph_executables.items():
            graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[name]):
                graph_outputs[name] = compiled(*torch_inputs)
            graphs[name].replay()
        torch.cuda.synchronize()
        graph_errors = {
            name: [float((a.float() - b.float()).abs().max()) for a, b in zip(outputs, eager_outputs[name], strict=True)]
            for name, outputs in graph_outputs.items()
        }
        if not all(
            np.isfinite(error) and error <= args.maximum_absolute_error
            for errors_for_graph in graph_errors.values()
            for error in errors_for_graph
        ):
            raise ValueError("Graph replay numerical gate failed; graph timing rejected")
        graph_host = {name: [] for name in graphs}
        graph_device = {name: [] for name in graphs}
        for iteration in range(args.iterations):
            names = tuple(graphs) if iteration % 2 == 0 else tuple(reversed(graphs))
            for name in names:
                graph_host[name].append(timed(lambda captured=graphs[name]: (captured.replay(),)))
            for name in names:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                graph_device[name].append(begin.elapsed_time(end))
        graph_record = {"host_ms": graph_host, "device_ms": graph_device, "max_abs_vs_eager": graph_errors}
    root = Path(__file__).resolve().parents[3]
    digests = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("lib/tile_lifetime/src", "lib/shuttle/src")
        for path in sorted((root / directory).rglob("*.py"))
    }
    digests[str(Path(__file__).relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helper_path = Path(__file__).with_name("h100_first_forward_fixture.py")
    digests[str(helper_path.relative_to(root))] = hashlib.sha256(helper_path.read_bytes()).hexdigest()
    capture_path = Path(__file__).with_name("cute_launch_argument_capture.py")
    digests[str(capture_path.relative_to(root))] = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    if digests != initial_digests:
        raise ValueError("Source changed during benchmark; evidence rejected")
    record = {
        "case": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output", "native_fixture_directory", "first_forward_fixture_directory"}
        },
        "source_digests": digests,
        "input_digests": [
            {"shape": x.shape, "dtype": str(x.dtype), "sha256": hashlib.sha256(np.asarray(x).tobytes()).hexdigest()}
            for x in primals
        ],
        "trace_fingerprint": trace.fingerprint,
        "program_fingerprint": plan.program.semantic_fingerprint,
        "physical_request_fingerprint": plan.physical_request.fingerprint,
        "composition_fingerprint": plan.composition.fingerprint,
        "crossing_event_count": len(plan.composition.concrete_events),
        "component_count": len(executable.components),
        "physical_components": physical_components,
        "launch_count": sum(len(c.kernel_names) for c in executable.components),
        "numerics": errors,
        "baseline_max_abs_vs_jax": baseline_errors,
        "treatment_vs_baseline": baseline_comparison,
        "numerical_acceptance": numerical_acceptance,
        "maximum_absolute_error_gate": args.maximum_absolute_error,
        "gate_provenance": (
            "Prospective gate using retained tiny-case observed maximum; not an inherited tolerance contract."
        ),
        "host_ms": samples,
        "median_host_ms": {key: statistics.median(value) for key, value in samples.items()},
        "components": components,
        "cuda_graph": graph_record,
        "timing_note": (
            "Synchronized host wall time; variants rotate through every position and reverse each cycle. "
            "Isolated CUDA events include submission gaps and are not additive kernel durations. "
            "Boundary bytes exclude internal scratch and recomputation."
        ),
        "toolchain": {
            "jax": jax.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "cutlass": version("nvidia-cutlass-dsl"),
            "quack": version("quack-kernels"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    artifacts = {} if diagnostic_fixture is None else {"forward": diagnostic_fixture}
    executables = {"generated": executable}
    if paired_baseline:
        executables["baseline"] = baseline_executable
    for name, compiled in executables.items():
        artifact = build_generic_component_multi_image_artifact(compiled, function_prefix=f"shuttle_dense_{name}")
        artifacts[name] = {
            "digest": artifact.digest,
            "record": artifact.canonical_record,
            "launch_recipe": artifact.launch_recipe.canonical_record,
            "device_image_bytes": [len(image.device_image) for image in artifact.images],
            "total_device_image_bytes": sum(len(image.device_image) for image in artifact.images),
        }
        record["phase"] = "artifact_exported_validation_pending"
        record["artifacts"] = artifacts
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        if args.native_fixture_directory is not None:
            validate_cute_launch_abi(artifact)
            artifacts[name]["cuda_parameter_abi_validated"] = True
            directory = args.native_fixture_directory / name
            directory.mkdir(parents=True, exist_ok=True)
            transport = artifact.native_transport
            (directory / "bundle.bin").write_bytes(transport)
            manifest = {
                "bundle": "bundle.bin",
                "inputs": [],
                "expected_outputs": [],
                "max_absolute_error": args.maximum_absolute_error,
            }
            expected_tensors = tuple(
                torch.tensor(np.asarray(target, dtype=np.float32)).to(dtype=output.dtype)
                for target, output in zip(expected, actual, strict=True)
            )
            for key, prefix, tensors in (
                ("inputs", "input", torch_inputs),
                ("expected_outputs", "expected", expected_tensors),
            ):
                paths = []
                for index, tensor in enumerate(tensors):
                    filename = f"{prefix}-{index}.bin"
                    payload = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
                    (directory / filename).write_bytes(payload)
                    paths.append(filename)
                manifest[key] = paths
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            artifacts[name]["native_transport_bytes"] = len(transport)
            artifacts[name]["fixture_sha256"] = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.iterdir())
            }
    record["phase"] = "complete"
    record["artifacts"] = artifacts
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {key: record[key] for key in ("component_count", "launch_count", "numerics", "median_host_ms")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
