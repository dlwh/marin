// RUN: shuttle-test-opt %s --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-plan-source-event-lowering '--shuttle-build-gpu-executable-bundle=target=sm_90' --shuttle-verify-gpu-executable-bundle | FileCheck %s

module {
  func.func @main() -> (tensor<2x3xi32>, tensor<2x3xi1>) {
    %indices = stablehlo.iota dim = 1 : tensor<2x3xi32>
    %predicate = stablehlo.constant dense<true> : tensor<2x3xi1>
    return %indices, %predicate : tensor<2x3xi32>, tensor<2x3xi1>
  }
}

// CHECK-DAG: "shuttle.device_module"
// CHECK-DAG: "shuttle.device_module"
// CHECK-DAG: "shuttle.invocation_slot"() <{{.*}}required_bytes = 24 : i64{{.*}}tensor_type = tensor<2x3xi32>
// CHECK-DAG: "shuttle.invocation_slot"() <{{.*}}required_bytes = 6 : i64{{.*}}tensor_type = tensor<2x3xi1>
