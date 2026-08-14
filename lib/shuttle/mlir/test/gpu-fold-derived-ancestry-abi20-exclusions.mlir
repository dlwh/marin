// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/abi20-fold-derived-ancestry-out-of-component.mlir --shuttle-annotate-source --shuttle-form-structural-regions --mlir-print-op-generic | FileCheck %s

// CHECK: "shuttle.region"()
// CHECK: "stablehlo.constant"()
// CHECK-SAME: value = dense<1.000000e+00> : tensor<f32>
// CHECK: "shuttle.yield"()
// CHECK: }) {{.*}} : () -> ()
// CHECK-NEXT: %{{[0-9]+}} = "stablehlo.broadcast_in_dim"({{.*}}broadcast_dimensions = array<i64: 0, 1, 2>{{.*}}(tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
