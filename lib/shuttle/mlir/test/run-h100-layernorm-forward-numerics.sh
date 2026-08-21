#!/bin/sh
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 GPU_TRANSPORT_V2_TEST" >&2
  exit 2
fi

SHUTTLE_RUN_H100_NUMERICS=1 exec "$1" \
  --gtest_fail_if_no_test_selected \
  --gtest_filter=GpuTransportV2Test.H100RunnerMatchesIndependentLayerNormForwardOracle
