# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from track1_map_h100_build import JAX_COMMIT, JAX_PATCH_SHA256, JAX_TREE, XLA_COMMIT, XLA_PATCH_SHA256
from track1_map_h100_prototype import (
    BASE_IMAGE,
    BAZEL_SHA256,
    INTEGRATION_COMMIT,
    INTEGRATION_TREE,
    Authorization,
    canonical_json,
    load_authorization,
    prototype_plan,
    validate_result_pair,
    wheel_set,
)


def _document() -> dict[str, object]:
    return {
        "attempts": 1,
        "base_image": BASE_IMAGE,
        "hardware": {"gpu_count": 1, "gpu_model": "H100"},
        "integration": {"commit": INTEGRATION_COMMIT, "tree": INTEGRATION_TREE},
        "permissions": {"build": True, "network_hydration": True, "run": True},
        "schema": "track1_map_h100_prototype_authorization",
        "schema_version": 1,
        "sources": {
            "jax": {
                "commit": JAX_COMMIT,
                "patch_sha256": list(JAX_PATCH_SHA256),
                "tree": JAX_TREE,
                "url": "https://github.com/jax-ml/jax.git",
            },
            "xla": {
                "commit": XLA_COMMIT,
                "patch_sha256": list(XLA_PATCH_SHA256),
                "url": "https://github.com/openxla/xla.git",
            },
        },
        "toolchain": {
            "bazel": {
                "sha256": BAZEL_SHA256,
                "url": "https://github.com/bazelbuild/bazel/releases/download/7.7.0/bazel-7.7.0-linux-x86_64",
                "version": "7.7.0",
            },
            "cuda_root": "/usr/local/cuda-13.2",
            "python": "3.12",
        },
    }


def _authorization(tmp_path: Path) -> tuple[Path, str, Authorization]:
    path = tmp_path / "authorization.json"
    raw = canonical_json(_document())
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    return path, digest, load_authorization(path, digest)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("attempts",), 2),
        (("base_image",), "ghcr.io/marin-community/iris-task-h100-evidence:latest"),
        (("hardware", "gpu_count"), 8),
        (("hardware", "gpu_model"), "A100"),
        (("integration", "commit"), "a" * 40),
        (("integration", "tree"), "b" * 40),
        (("permissions", "network_hydration"), False),
        (("permissions", "build"), False),
        (("permissions", "run"), False),
        (("sources", "jax", "commit"), "c" * 40),
        (("sources", "jax", "tree"), "d" * 40),
        (("sources", "jax", "patch_sha256"), list(reversed(JAX_PATCH_SHA256))),
        (("sources", "xla", "commit"), "e" * 40),
        (("sources", "xla", "patch_sha256"), list(reversed(XLA_PATCH_SHA256))),
        (("toolchain", "bazel", "sha256"), "f" * 64),
        (("toolchain", "cuda_root"), "/usr/local/cuda"),
        (("toolchain", "python"), "3.13"),
    ],
)
def test_authorization_rejects_scope_or_permission_mutation(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    document = deepcopy(_document())
    target = document
    for component in path[:-1]:
        target = target[component]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]
    raw = canonical_json(document)
    authorization = tmp_path / "authorization.json"
    authorization.write_bytes(raw)

    with pytest.raises(ValueError, match="authorization"):
        load_authorization(authorization, hashlib.sha256(raw).hexdigest())


def test_authorization_requires_exact_canonical_bytes_and_cli_digest(tmp_path: Path) -> None:
    path, digest, loaded = _authorization(tmp_path)

    assert loaded.sha256 == digest
    assert loaded.document == _document()

    path.write_text(json.dumps(_document(), indent=2) + "\n")
    rewritten = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="canonical JSON"):
        load_authorization(path, rewritten)
    with pytest.raises(ValueError, match="SHA-256"):
        load_authorization(path, digest)


def test_dry_run_renders_one_network_hydration_then_offline_build_and_two_workers(tmp_path: Path) -> None:
    _, _, authorization = _authorization(tmp_path)
    workspace = tmp_path / "fresh"

    rendered = prototype_plan(
        authorization,
        workspace=workspace,
        runner_repository=Path("/app"),
        python=Path("/usr/bin/python3.12"),
        cuda_root=Path("/usr/local/cuda-13.2"),
        observed_base_image=BASE_IMAGE,
    )

    assert rendered["schema_version"] == 1
    assert rendered["workspace"] == str(workspace)
    phases = rendered["phases"]
    assert [phase["name"] for phase in phases] == [
        "clone_integration",
        "fetch_jax",
        "fetch_xla",
        "fetch_bazel",
        "hydrate_bazel_repositories",
        "offline_wheel_build",
        "create_runtime",
        "install_wheels",
        "populate",
        "reuse",
    ]
    hydration = phases[4]["argv"]
    offline = phases[5]["argv"]
    assert "--bazel_options=--repository_disable_download" not in hydration
    assert "--bazel_options=--repository_disable_download" in offline
    assert phases[8]["argv"][phases[8]["argv"].index("--role") + 1] == "populate"
    assert phases[9]["argv"][phases[9]["argv"].index("--role") + 1] == "reuse"
    assert not workspace.exists()

    workspace.mkdir()
    with pytest.raises(RuntimeError, match="fresh workspace"):
        prototype_plan(
            authorization,
            workspace=workspace,
            runner_repository=Path("/app"),
            python=Path("/usr/bin/python3.12"),
            cuda_root=Path("/usr/local/cuda-13.2"),
            observed_base_image=BASE_IMAGE,
        )


def _worker_document(worker: str, pid: int) -> dict[str, object]:
    coordinates = []
    for rank in (2, 3):
        for mode in ("disabled", "source_ordered", "fast"):
            name = f"jit_rank{rank}_{mode}-{rank}{len(mode)}-cache"
            coordinates.append(
                {
                    "cache_entry": [name, len(name), hashlib.sha256(name.encode()).hexdigest()],
                    "gradient_sha256": hashlib.sha256(f"gradient:{rank}:{mode}".encode()).hexdigest(),
                    "mode": mode,
                    "observer": (
                        None
                        if worker == "reuse" or mode == "disabled"
                        else {
                            "final_module_sha256": "1" * 64,
                            "generated_bundle_sha256": "2" * 64,
                            "invocation_id": rank,
                        }
                    ),
                    "primal_sha256": hashlib.sha256(f"primal:{rank}:{mode}".encode()).hexdigest(),
                    "rank": rank,
                }
            )
    return {"coordinates": coordinates, "pid": pid, "schema_version": 1, "worker": worker}


def test_result_pair_requires_fresh_process_reuse_of_exact_outputs_and_cache() -> None:
    populate = _worker_document("populate", 101)
    reuse = _worker_document("reuse", 202)

    validated = validate_result_pair(populate, reuse)
    assert validated == {"populate": populate, "reuse": reuse}

    for mutation in ("pid", "cache", "output", "observer"):
        changed = deepcopy(reuse)
        if mutation == "pid":
            changed["pid"] = 101
        elif mutation == "cache":
            changed["coordinates"][0]["cache_entry"][2] = "3" * 64  # type: ignore[index]
        elif mutation == "output":
            changed["coordinates"][0]["primal_sha256"] = "4" * 64  # type: ignore[index]
        else:
            changed["coordinates"][1]["observer"] = {  # type: ignore[index]
                "final_module_sha256": "1" * 64,
                "generated_bundle_sha256": "2" * 64,
                "invocation_id": 1,
            }
        with pytest.raises(ValueError, match="worker result"):
            validate_result_pair(populate, changed)


def test_wheel_set_requires_exact_regular_three_wheel_output(tmp_path: Path) -> None:
    names = (
        "jaxlib-0.10.1-cp312-cp312-manylinux_2_27_x86_64.whl",
        "jax_cuda13_plugin-0.10.1-cp312-cp312-manylinux_2_27_x86_64.whl",
        "jax_cuda13_pjrt-0.10.1-py3-none-manylinux_2_27_x86_64.whl",
    )
    for name in names:
        (tmp_path / name).write_bytes(name.encode())

    wheels = wheel_set(tmp_path)
    assert tuple(path.name for path in wheels) == names

    (tmp_path / "jaxlib-0.10.1-extra.whl").write_bytes(b"extra")
    with pytest.raises(RuntimeError, match="exactly one"):
        wheel_set(tmp_path)
    (tmp_path / "jaxlib-0.10.1-extra.whl").unlink()
    (tmp_path / names[0]).unlink()
    (tmp_path / names[0]).symlink_to(tmp_path / names[1])
    with pytest.raises(RuntimeError, match="regular"):
        wheel_set(tmp_path)
