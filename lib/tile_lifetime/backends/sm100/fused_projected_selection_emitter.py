# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a direct SM100 Contract/maximum-Fold physical instantiation."""

from __future__ import annotations

import hashlib
import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEMPLATE_SHA256 = {
    "fmha_sm100_inst.jinja": "7181b94bae8be3631824343e8ff68cbc001d38d1b929eebe9958928f968a7423",
    "fmha_sm100_variant_run.cu.jinja": "3e8af59327dcbe3bf354a17b7445c027b9aa04b664a97ff4dad4cb52e8c4a9cc",
    "fmha_sm100_params.h": "a776dc84103e87931672d903a2b5737b163182d0a1ea7d07c9d3d105e8b34782",
    "tvm_ffi_utils.h": "e77134f7fc39637bcdd490e6371ee140014dbadf9a670627b673c9737a7e9b86",
    "gmem_bounds_check.h": "c6a174cd5d392bca154c17a119da4bd2ff1116371a20609b38f4eea27fa43597",
}

GENERATED_VARIANT_NAME = "shuttle_projected_contract_maximum_fold_bf16_m256n128k128"
GENERATED_KERNEL_NAME = "shuttle_projected_contract_maximum_fold_kernel"
GENERATED_ENTRY_NAME = f"run_{GENERATED_VARIANT_NAME}"

STABLE_SELECTION_CPP_SOURCE = r"""
#include <torch/extension.h>

void shuttle_stable_topk_cuda(
    torch::Tensor score,
    torch::Tensor valid_count,
    torch::Tensor output,
    int64_t selected_count,
    int64_t invalid_index);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("stable_topk_select", &shuttle_stable_topk_cuda);
}
"""

STABLE_SELECTION_CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>

namespace {
constexpr int kMaximumSelected = 32;

__device__ __forceinline__ uint32_t total_order_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}

__global__ void stable_topk_kernel(
    const float* score,
    const int32_t* valid_count,
    int32_t* output,
    int query_count,
    int group_count,
    int score_right_stride,
    int selected_count,
    int invalid_index) {
  const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= query_count * group_count) return;
  const int query = row / group_count;
  const int group = row - query * group_count;
  const int candidates = valid_count[query];
  float selected_value[kMaximumSelected];
  int32_t selected_index[kMaximumSelected];
  int selected_size = 0;
  for (int candidate = 0; candidate < candidates; ++candidate) {
    const float candidate_value = score[(group * score_right_stride + candidate) * query_count + query];
    const uint32_t candidate_key = total_order_key(candidate_value);
    int insertion = selected_size;
    for (int slot = 0; slot < selected_size; ++slot) {
      const uint32_t selected_key = total_order_key(selected_value[slot]);
      if (candidate_key > selected_key ||
          (candidate_key == selected_key && candidate < selected_index[slot])) {
        insertion = slot;
        break;
      }
    }
    if (insertion >= selected_count) continue;
    const int next_size = selected_size < selected_count ? selected_size + 1 : selected_size;
    for (int slot = next_size - 1; slot > insertion; --slot) {
      selected_value[slot] = selected_value[slot - 1];
      selected_index[slot] = selected_index[slot - 1];
    }
    selected_value[insertion] = candidate_value;
    selected_index[insertion] = candidate;
    selected_size = next_size;
  }
  for (int left = 1; left < selected_size; ++left) {
    const int32_t value = selected_index[left];
    int right = left;
    while (right > 0 && value < selected_index[right - 1]) {
      selected_index[right] = selected_index[right - 1];
      --right;
    }
    selected_index[right] = value;
  }
  const int output_base = row * selected_count;
  for (int slot = 0; slot < selected_count; ++slot) {
    output[output_base + slot] = slot < selected_size ? selected_index[slot] : invalid_index;
  }
}
}  // namespace

void shuttle_stable_topk_cuda(
    torch::Tensor score,
    torch::Tensor valid_count,
    torch::Tensor output,
    int64_t selected_count,
    int64_t invalid_index) {
  TORCH_CHECK(score.is_cuda() && valid_count.is_cuda() && output.is_cuda(), "stable Selection requires CUDA tensors");
  TORCH_CHECK(score.scalar_type() == torch::kFloat32, "stable Selection scores must be FP32");
  TORCH_CHECK(valid_count.scalar_type() == torch::kInt32, "stable Selection valid counts must be INT32");
  TORCH_CHECK(output.scalar_type() == torch::kInt32, "stable Selection indices must be INT32");
  TORCH_CHECK(score.is_contiguous() && valid_count.is_contiguous() && output.is_contiguous(),
              "stable Selection tensors must be contiguous");
  TORCH_CHECK(score.dim() == 3 && valid_count.numel() == score.size(2),
              "stable Selection expects [group,right,query] scores and one valid count per query");
  TORCH_CHECK(output.dim() == 3 && output.size(0) == score.size(2) && output.size(1) == score.size(0) &&
                  output.size(2) == selected_count,
              "stable Selection output shape must be [query,group,selected]");
  TORCH_CHECK(selected_count > 0 && selected_count <= kMaximumSelected,
              "stable Selection supports selected counts from 1 through 32");
  TORCH_CHECK(invalid_index < 0 && invalid_index >= INT32_MIN,
              "stable Selection invalid index must be a negative INT32 value");
  const c10::cuda::CUDAGuard device_guard(score.device());
  constexpr int threads = 256;
  const int rows = static_cast<int>(score.size(0) * score.size(2));
  const int blocks = (rows + threads - 1) / threads;
  stable_topk_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream(score.get_device())>>>(
      score.data_ptr<float>(),
      valid_count.data_ptr<int32_t>(),
      output.data_ptr<int32_t>(),
      static_cast<int>(score.size(2)),
      static_cast<int>(score.size(0)),
      static_cast<int>(score.size(1)),
      static_cast<int>(selected_count),
      static_cast<int>(invalid_index));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@dataclass(frozen=True)
class DirectProjectedSelectionSources:
    """Generated direct physical instantiation and static audit."""

    instantiation_source: str
    binding_source: str
    stable_selection_source: str
    source_sha256: dict[str, str]
    retained_physical_tokens: tuple[str, ...]
    forbidden_tokens: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.forbidden_tokens


def render_direct_projected_selection_sources(msa_root: Path) -> DirectProjectedSelectionSources:
    """Render the low-level score Contract/Fold template from pinned sources."""
    csrc = msa_root / "python" / "fmha_sm100" / "csrc"
    source_hashes = _verify_templates(csrc)
    return render_direct_projected_selection_template_text(
        (csrc / "fmha_sm100_inst.jinja").read_text(),
        (csrc / "fmha_sm100_variant_run.cu.jinja").read_text(),
        source_sha256=source_hashes,
    )


def render_direct_projected_selection_template_text(
    instantiation_template: str,
    binding_template: str,
    *,
    source_sha256: dict[str, str],
) -> DirectProjectedSelectionSources:
    """Instantiate a generic score-Contract/maximum-Fold physical template."""
    jinja2 = importlib.import_module("jinja2")
    parameters = {
        "variant_name": GENERATED_VARIANT_NAME,
        "func_name": GENERATED_KERNEL_NAME,
        "dtype_in": "nv_bfloat16",
        "cutlass_dtype_out": "cutlass::bfloat16_t",
        "tile_q": "_256",
        "tile_kv": "_128",
        "thread_shape": "_2, _1, _1",
        "is_split_kv": "false",
        "single_wg": "false",
        "page_size": 128,
        "pack_factor": 1,
        "sparse_mode": "OnlyScore",
    }
    instantiation = jinja2.Template(instantiation_template).render(**parameters)
    binding = jinja2.Template(binding_template).render(**parameters)
    instantiation = instantiation.replace(
        f"Auto-generated FMHA variant: {GENERATED_VARIANT_NAME}",
        "Generated Shuttle Contract plus maximum-Fold physical instantiation",
    )
    binding = binding.replace(
        f"Auto-generated per-variant FMHA run binding: {GENERATED_VARIANT_NAME}",
        "Generated Shuttle Contract plus maximum-Fold FFI binding",
    ).replace(
        f"FMHA variant {GENERATED_VARIANT_NAME} failed",
        "Shuttle projected Contract/maximum-Fold kernel failed",
    )
    stable_selection = STABLE_SELECTION_CPP_SOURCE + STABLE_SELECTION_CUDA_SOURCE
    combined = f"{instantiation}\n{binding}\n{stable_selection}"
    forbidden_candidates = (
        "get_fmha_variant",
        "fmha_sm100(",
        "sparse_atten_func",
        "sparse_fmha(",
    )
    forbidden = tuple(token for token in forbidden_candidates if token in combined)
    required_candidates = (
        "run_fmha_fwd<",
        "SparseAttnMode::OnlyScore",
        "CausalMask",
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC",
    )
    retained = tuple(token for token in required_candidates if token in combined)
    if set(retained) != set(required_candidates):
        missing = sorted(set(required_candidates) - set(retained))
        raise ValueError(f"generated Contract/Fold instantiation lost physical tokens {missing}")
    return DirectProjectedSelectionSources(
        instantiation_source=instantiation,
        binding_source=binding,
        stable_selection_source=stable_selection,
        source_sha256={
            **source_sha256,
            "generated_stable_selection": hashlib.sha256(stable_selection.encode()).hexdigest(),
        },
        retained_physical_tokens=retained,
        forbidden_tokens=forbidden,
    )


def compile_direct_projected_selection(
    msa_root: Path,
    *,
    cache_root: Path | None = None,
) -> tuple[Any, Any, DirectProjectedSelectionSources]:
    """Compile and load the generated physical entry without a variant manager."""
    sources = render_direct_projected_selection_sources(msa_root)
    if not sources.clean:
        raise ValueError(f"generated physical source retains forbidden calls {sources.forbidden_tokens}")
    source_digest = hashlib.sha256(
        f"{sources.instantiation_source}\n{sources.binding_source}\n{sources.stable_selection_source}".encode()
    ).hexdigest()[:16]
    if cache_root is None:
        cache_root = Path.home() / ".cache" / "shuttle" / "projected_selection"
    cache_dir = cache_root / source_digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    instantiation_path = cache_dir / "shuttle_projected_contract_fold_inst.cu"
    binding_path = cache_dir / "shuttle_projected_contract_fold_binding.cu"
    shared_object = cache_dir / "shuttle_projected_contract_fold.so"
    _write_if_changed(instantiation_path, sources.instantiation_source)
    _write_if_changed(binding_path, sources.binding_source)

    csrc = msa_root / "python" / "fmha_sm100" / "csrc"
    for name in ("fmha_sm100_params.h", "tvm_ffi_utils.h", "gmem_bounds_check.h"):
        destination = cache_dir / name
        if not destination.exists() or destination.read_bytes() != (csrc / name).read_bytes():
            shutil.copy2(csrc / name, destination)

    python_root = str((msa_root / "python").resolve())
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    jit = importlib.import_module("fmha_sm100.jit")
    nvcc = Path(jit._get_cuda_home()) / "bin" / "nvcc"
    flags = jit._get_nvcc_flags(cache_dir)
    instantiation_object = cache_dir / "instantiation.o"
    binding_object = cache_dir / "binding.o"
    ninja_source = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {flags}

rule nvcc_compile
  command = $nvcc $nvcc_flags -c $in -o $out

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda

build {instantiation_object}: nvcc_compile {instantiation_path}
build {binding_object}: nvcc_compile {binding_path}
build {shared_object}: nvcc_link {instantiation_object} {binding_object}
"""
    _write_if_changed(cache_dir / "build.ninja", ninja_source)
    if not shared_object.exists():
        result = subprocess.run(
            ["ninja", "-j1"],
            cwd=cache_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                "direct projected Contract/Fold compilation failed:\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    tvm_ffi = importlib.import_module("tvm_ffi")
    module = tvm_ffi.load_module(str(shared_object))
    selection_digest = hashlib.sha256(sources.stable_selection_source.encode()).hexdigest()[:16]
    cpp_extension = importlib.import_module("torch.utils.cpp_extension")
    selection_module = cpp_extension.load_inline(
        name=f"shuttle_stable_selection_{selection_digest}",
        cpp_sources=STABLE_SELECTION_CPP_SOURCE,
        cuda_sources=STABLE_SELECTION_CUDA_SOURCE,
        extra_cuda_cflags=("-O3",),
        with_cuda=True,
        verbose=False,
    )
    return getattr(module, GENERATED_ENTRY_NAME), selection_module.stable_topk_select, sources


def _verify_templates(csrc: Path) -> dict[str, str]:
    observed = {}
    for name, expected in TEMPLATE_SHA256.items():
        digest = hashlib.sha256((csrc / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"pinned MSA source mismatch for {name}: {digest} != {expected}")
        observed[name] = digest
    return observed


def _write_if_changed(path: Path, content: str) -> None:
    if not path.exists() or path.read_text() != content:
        path.write_text(content)
