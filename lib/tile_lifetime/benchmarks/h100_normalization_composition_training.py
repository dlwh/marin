# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare explicit generic composition treatments on normalization value-and-grad."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path

import jax
import numpy as np
import torch

from tile_lifetime.backends.h100.cute_generic_tensor_program import (
    build_generic_component_multi_image_artifact,
    compile_generic_kernel_composition,
)
from tile_lifetime.canonical_kernel_composition import (
    FusedComponentEvaluation,
    coalesce_private_map_fold_components,
    lower_canonical_kernel_composition,
)
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.physical_task_contract import PhysicalTarget
from tile_lifetime.row_normalization_jax import (
    RowProjectionConfig,
    layer_projection,
    projection_value_and_grad,
    rms_projection,
)
from tile_lifetime.streaming_attention_executable import jax_tensor_program_executable_request
from tile_lifetime.tensor_program import NumericalPolicy, deserialize_tensor_program

ERROR_TOLERANCE = 2e-5


def paired_host_samples(calls, iterations):
    samples = {name: [] for name in calls}
    for iteration in range(iterations):
        names = tuple(calls)
        offset = iteration % len(names)
        names = names[offset:] + names[:offset]
        if (iteration // len(names)) % 2:
            names = tuple(reversed(names))
        for name in names:
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = calls[name]()
            if name == "jax":
                jax.block_until_ready(result)
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - started) * 1000)
    return samples


def errors(actual, expected):
    records = []
    for output, target in zip(actual, expected, strict=True):
        difference = np.abs(output - target)
        records.append(
            {
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
                "bitwise_equal": bool(np.array_equal(output, target)),
                "accepted": bool(
                    np.isfinite(difference).all()
                    and np.allclose(
                        output,
                        target,
                        atol=ERROR_TOLERANCE,
                        rtol=ERROR_TOLERANCE,
                    )
                ),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("rms", "layer"), default="rms")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--same-coordinate-ssa", action="store_true")
    parser.add_argument("--coalesce-private-map-fold", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name():
        raise ValueError("This benchmark requires exactly one visible H100")
    config = RowProjectionConfig(rows=args.rows, hidden=args.hidden, features=args.features)
    factory = {"rms": rms_projection, "layer": layer_projection}[args.kind]
    function = projection_value_and_grad(factory(config))
    rng = np.random.default_rng(113)
    arrays = tuple(
        rng.normal(size=shape).astype(np.float32)
        for shape in (
            (args.rows, args.hidden),
            (args.hidden,),
            (args.hidden, args.features),
            (args.rows, args.features),
        )
    )
    with jax.default_device(jax.devices("cpu")[0]), jax.default_matmul_precision("default"):
        cpu_inputs = tuple(jax.numpy.asarray(array) for array in arrays)
        source_jaxpr = str(jax.make_jaxpr(function)(*cpu_inputs))
        source = jax_function_to_tensor_program(function, *cpu_inputs, numerical_policy=NumericalPolicy.FAST)
        request = jax_tensor_program_executable_request(source, tile_size=128, target=PhysicalTarget.SM90)
        program = deserialize_tensor_program(json.dumps(request.algebra_record))
        baseline = lower_canonical_kernel_composition(request)
        treatment = (
            coalesce_private_map_fold_components(program, baseline) if args.coalesce_private_map_fold else baseline
        )
    compositions = {"baseline": baseline, "treatment": treatment}
    torch_inputs = tuple(torch.tensor(array, device="cuda") for array in arrays)
    executables = {
        name: compile_generic_kernel_composition(
            program,
            composition,
            example_inputs=torch_inputs,
            fused_evaluation=(
                FusedComponentEvaluation.SAME_COORDINATE_SSA
                if name == "treatment" and args.same_coordinate_ssa
                else FusedComponentEvaluation.RECURSIVE
            ),
        )
        for name, composition in compositions.items()
    }
    gpu_inputs = tuple(jax.device_put(array, jax.devices("gpu")[0]) for array in arrays)
    with jax.default_matmul_precision("default"):
        performance_lowered = jax.jit(function).lower(*gpu_inputs)
        performance_reference = performance_lowered.compile()
    with jax.default_matmul_precision("highest"):
        reference_lowered = jax.jit(function).lower(*gpu_inputs)
        accuracy_reference = reference_lowered.compile()
    expected = tuple(np.asarray(value) for value in jax.tree.leaves(accuracy_reference(*gpu_inputs)))
    default_outputs = tuple(np.asarray(value) for value in jax.tree.leaves(performance_reference(*gpu_inputs)))
    outputs = {name: executable(*torch_inputs) for name, executable in executables.items()}
    actual = {name: tuple(value.cpu().numpy() for value in result) for name, result in outputs.items()}
    numerics = {name: errors(result, expected) for name, result in actual.items()}
    numerics["treatment_vs_baseline"] = errors(actual["treatment"], actual["baseline"])
    record = {
        "case": {key: value for key, value in vars(args).items() if key != "output"},
        "precision_protocol": {
            "name": "default_performance_highest_accuracy_v1",
            "source_import_precision": "default",
            "performance_reference_precision": "default",
            "accuracy_reference_precision": "highest",
            "reason": (
                "Separate arithmetic accuracy from default JAX tensorcore performance; "
                "original default-reference rejection remains reported."
            ),
            "tolerance_changed": False,
        },
        "default_jax_vs_accuracy_reference": errors(default_outputs, expected),
        "original_default_reference_comparison": {
            name: errors(result, default_outputs) for name, result in actual.items()
        },
        "performance_reference_stablehlo": performance_lowered.as_text(),
        "source_jaxpr": source_jaxpr,
        "reference_stablehlo": reference_lowered.as_text(),
        "gate": {
            "atol": ERROR_TOLERANCE,
            "rtol": ERROR_TOLERANCE,
            "source": "test_jax_normalization_value_and_grad_lowers_to_acyclic_tasks",
        },
        "numerics": numerics,
        "numerical_acceptance": all(item["accepted"] for items in numerics.values() for item in items),
        "source_program_fingerprint": source.semantic_fingerprint,
        "program_fingerprint": program.semantic_fingerprint,
        "request_fingerprint": request.fingerprint,
        "input_digests": [
            {"shape": array.shape, "dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
            for array in arrays
        ],
        "jax_source": inspect.getsource(factory) + inspect.getsource(projection_value_and_grad),
        "boundaries": {
            name: {
                "composition_fingerprint": composition.fingerprint,
                "scheduled_request_fingerprint": composition.scheduled_request.fingerprint,
                "task_ids": [task.id for task in composition.scheduled_request.task_plan.tasks],
                "component_count": len(composition.components),
                "launch_count": sum(len(component.kernel_names) for component in executables[name].components),
                "crossing_event_count": len(composition.concrete_events),
            }
            for name, composition in compositions.items()
        },
        "toolchain": {"jax": jax.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name()},
    }
    root = Path(__file__).resolve().parents[3]
    record["source_digests"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("lib/tile_lifetime/src", "lib/shuttle/src")
        for path in sorted((root / directory).rglob("*.py"))
    }
    record["source_digests"][str(Path(__file__).relative_to(root))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    if not record["numerical_acceptance"]:
        raise ValueError("Predeclared numerical gate failed; timing rejected; see output artifact")
    calls = {name: lambda executable=executable: executable(*torch_inputs) for name, executable in executables.items()}
    calls["jax"] = lambda: performance_reference(*gpu_inputs)
    paired_host_samples(calls, args.warmup)
    record["timing_order"] = "rotate_all_positions_and_reverse_each_cycle"
    record["host_ms"] = paired_host_samples(calls, args.iterations)
    graphs = {}
    graph_outputs = {}
    for name, executable in executables.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs[name] = executable(*torch_inputs)
        graphs[name] = graph
        graph.replay()
        torch.cuda.synchronize()
    record["graph_vs_eager"] = {
        name: errors(tuple(value.cpu().numpy() for value in result), actual[name])
        for name, result in graph_outputs.items()
    }
    if not all(item["accepted"] for items in record["graph_vs_eager"].values() for item in items):
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        raise ValueError("CUDA graph replay changed numerical results")
    record["graph_host_ms"] = paired_host_samples(
        {name: graph.replay for name, graph in graphs.items()}, args.iterations
    )
    graph_samples = {name: [] for name in graphs}
    for iteration in range(args.iterations):
        for name in tuple(graphs) if iteration % 2 == 0 else tuple(reversed(graphs)):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            graph_samples[name].append(begin.elapsed_time(end))
    record["graph_device_ms"] = graph_samples
    record["medians_ms"] = {
        measurement: {name: statistics.median(samples) for name, samples in record[measurement].items()}
        for measurement in ("host_ms", "graph_host_ms", "graph_device_ms")
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    artifacts = {}
    for name, executable in executables.items():
        artifact = build_generic_component_multi_image_artifact(
            executable, function_prefix=f"shuttle_normalization_{name}"
        )
        artifacts[name] = {
            "digest": artifact.digest,
            "record": artifact.canonical_record,
            "launch_recipe": artifact.launch_recipe.canonical_record,
        }
    record["artifacts"] = artifacts
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {"numerical_acceptance": record["numerical_acceptance"], "medians_ms": record["medians_ms"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
