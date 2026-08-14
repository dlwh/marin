// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-backward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=GENERIC
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-composed.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=GENERIC
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=GENERIC

// This opt-in descriptor boundary does not claim PTX assembly, device
// execution, performance, or production acceptance.

// CALL-LABEL: module @jit_forward
// CALL-NOT: shuttle.
// CALL-NOT: stablehlo.
// CALL-COUNT-1: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// CALL-SAME: api_version = 4 : i32
// CALL-SAME: backend_config = {bundle_bytes = "SHUTGPU\00
// CALL-SAME: bundle_schema_version = 2 : i64
// CALL-SAME: completion = "stream_ordered"
// CALL-SAME: device_schema_version = 3 : i64
// CALL-SAME: invocation_abi_schema_version = 3 : i64
// CALL-SAME: transport_schema_version = 2 : i64
// CALL-NOT: loaded_kernel
// CALL-NOT: temporary_address
// CALL-NOT: executor_state
// CALL-NOT: runtime_stream
// CALL-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]
// CALL-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>]
// CALL: return
// CALL-NOT: shuttle.
// CALL-NOT: stablehlo.

// GENERIC-COUNT-1: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// GENERIC-SAME: api_version = 4 : i32
// GENERIC-SAME: backend_config = {bundle_bytes = "SHUTGPU\00
// GENERIC: return
// GENERIC-NOT: shuttle.
// GENERIC-NOT: stablehlo.
