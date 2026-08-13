// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=FORWARD-MAPS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=FORWARD-FOLDS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=FORWARD-SEMANTICS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=BACKWARD-MAPS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=BACKWARD-FOLDS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=BACKWARD-SEMANTICS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=COMPOSED-MAPS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=COMPOSED-FOLDS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=COMPOSED-SEMANTICS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=FORWARD-LARGE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-backward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=BACKWARD-LARGE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-composed.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=COMPOSED-LARGE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=FORWARD-SMALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-backward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=BACKWARD-SMALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-composed.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=COMPOSED-SMALL

// These fixtures constrain a generic connected Map/Fold graph.  The compiler
// must not select behavior from function names, fixture digests, task counts,
// or a LayerNorm semantic label.

// FORWARD-MAPS-COUNT-29: "shuttle.map"(
// FORWARD-MAPS-NOT: "shuttle.map"(
// FORWARD-FOLDS-COUNT-2: "shuttle.fold"(
// FORWARD-FOLDS-NOT: "shuttle.fold"(
// FORWARD-SEMANTICS: "arith.subf"(
// FORWARD-SEMANTICS: excluded = []

// BACKWARD-MAPS-COUNT-63: "shuttle.map"(
// BACKWARD-MAPS-NOT: "shuttle.map"(
// BACKWARD-FOLDS-COUNT-10: "shuttle.fold"(
// BACKWARD-FOLDS-NOT: "shuttle.fold"(
// BACKWARD-SEMANTICS-DAG: "arith.subf"(
// BACKWARD-SEMANTICS-DAG: "arith.negf"(
// BACKWARD-SEMANTICS: excluded = []

// COMPOSED-MAPS-COUNT-70: "shuttle.map"(
// COMPOSED-MAPS-NOT: "shuttle.map"(
// COMPOSED-FOLDS-COUNT-10: "shuttle.fold"(
// COMPOSED-FOLDS-NOT: "shuttle.fold"(
// COMPOSED-SEMANTICS-DAG: "arith.subf"(
// COMPOSED-SEMANTICS-DAG: "arith.negf"(
// COMPOSED-SEMANTICS: excluded = []

// FORWARD-LARGE-LABEL: module @jit_forward
// FORWARD-LARGE-NOT: shuttle.
// FORWARD-LARGE-NOT: stablehlo.
// FORWARD-LARGE-COUNT-1: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// FORWARD-LARGE-SAME: api_version = 4 : i32
// FORWARD-LARGE-SAME: backend_config = {bundle_bytes = "SHUTGPU\00
// FORWARD-LARGE-SAME: bundle_schema_version = 3 : i64
// FORWARD-LARGE-SAME: device_schema_version = 4 : i64
// FORWARD-LARGE-SAME: invocation_abi_schema_version = 4 : i64
// FORWARD-LARGE-SAME: transport_schema_version = 3 : i64
// FORWARD-LARGE-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]
// FORWARD-LARGE-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>]
// FORWARD-LARGE: return
// FORWARD-LARGE-NOT: shuttle.
// FORWARD-LARGE-NOT: stablehlo.

// BACKWARD-LARGE-LABEL: module @jit_backward
// BACKWARD-LARGE-COUNT-1: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// BACKWARD-LARGE-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>]
// BACKWARD-LARGE-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]
// BACKWARD-LARGE: return

// COMPOSED-LARGE-LABEL: module @jit_composed
// COMPOSED-LARGE-COUNT-1: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// COMPOSED-LARGE-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>]
// COMPOSED-LARGE-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]
// COMPOSED-LARGE: return

// FORWARD-SMALL-LABEL: module @jit_forward
// FORWARD-SMALL-COUNT-1: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// FORWARD-SMALL-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]
// FORWARD-SMALL-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>]

// BACKWARD-SMALL-LABEL: module @jit_backward
// BACKWARD-SMALL-COUNT-1: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// BACKWARD-SMALL-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>]
// BACKWARD-SMALL-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]

// COMPOSED-SMALL-LABEL: module @jit_composed
// COMPOSED-SMALL-COUNT-1: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// COMPOSED-SMALL-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>]
// COMPOSED-SMALL-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]
