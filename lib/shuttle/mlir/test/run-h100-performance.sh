#!/bin/sh
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 GPU_TRANSPORT_TEST PERFORMANCE_VALIDATOR" >&2
  exit 2
fi

if [ "${SHUTTLE_RUN_H100_PERFORMANCE:-0}" != 1 ]; then
  if [ "${SHUTTLE_H100_PERF_ACCEPTANCE:-0}" = 1 ] || [ "${SHUTTLE_H100_PERF_EXPLORATORY:-0}" = 1 ]; then
    echo "comparison mode requires SHUTTLE_RUN_H100_PERFORMANCE=1" >&2
    exit 2
  fi
  echo "skipping H100 performance harness; set SHUTTLE_RUN_H100_PERFORMANCE=1"
  exit 0
fi

: "${SHUTTLE_H100_PERF_REPORT:?set SHUTTLE_H100_PERF_REPORT}"
: "${SHUTTLE_COMPILER_IDENTITY:?set SHUTTLE_COMPILER_IDENTITY}"
if [ "${SHUTTLE_H100_PERF_ACCEPTANCE:-0}" = 1 ]; then
  : "${SHUTTLE_H100_XLA_REPORT:?set SHUTTLE_H100_XLA_REPORT in acceptance mode}"
  : "${SHUTTLE_H100_TE_REPORT:?set SHUTTLE_H100_TE_REPORT in acceptance mode}"
  : "${SHUTTLE_H100_PERF_ACCEPTANCE_REPORT:?set SHUTTLE_H100_PERF_ACCEPTANCE_REPORT in acceptance mode}"
fi
if [ "${SHUTTLE_H100_PERF_EXPLORATORY:-0}" = 1 ]; then
  if [ "${SHUTTLE_H100_PERF_ACCEPTANCE:-0}" = 1 ]; then
    echo "acceptance and exploratory modes are mutually exclusive" >&2
    exit 2
  fi
  : "${SHUTTLE_H100_XLA_HOST_REPORT:?set SHUTTLE_H100_XLA_HOST_REPORT in exploratory mode}"
  : "${SHUTTLE_H100_TE_STATUS:?set SHUTTLE_H100_TE_STATUS in exploratory mode}"
  : "${SHUTTLE_H100_PERF_EXPLORATORY_REPORT:?set SHUTTLE_H100_PERF_EXPLORATORY_REPORT in exploratory mode}"
fi

SHUTTLE_RUN_H100_NUMERICS=1 \
SHUTTLE_RUN_H100_PERFORMANCE=1 \
SHUTTLE_H100_PERF_REPORT="$SHUTTLE_H100_PERF_REPORT" \
SHUTTLE_COMPILER_IDENTITY="$SHUTTLE_COMPILER_IDENTITY" \
  "$1" --gtest_fail_if_no_test_selected \
  --gtest_filter='GpuTransportV2Test.H100RunnerMatchesIndependentRmsNormForwardAndVjpOracle:GpuTransportV2Test.H100RunnerMatchesIndependentRmsNormContractOracle:GpuTransportV2Test.H100RunnerWritesCounterbalancedPerformanceReport'

validated_report=${SHUTTLE_H100_PERF_VALIDATED_REPORT:-"$SHUTTLE_H100_PERF_REPORT.validated.json"}
python3 "$2" --shuttle "$SHUTTLE_H100_PERF_REPORT" --output "$validated_report"

if [ "${SHUTTLE_H100_PERF_ACCEPTANCE:-0}" = 1 ]; then
  python3 "$2" --acceptance \
    --shuttle "$SHUTTLE_H100_PERF_REPORT" \
    --xla "$SHUTTLE_H100_XLA_REPORT" \
    --te "$SHUTTLE_H100_TE_REPORT" \
    --output "$SHUTTLE_H100_PERF_ACCEPTANCE_REPORT"
fi

if [ "${SHUTTLE_H100_PERF_EXPLORATORY:-0}" = 1 ]; then
  python3 "$2" --exploratory \
    --shuttle "$SHUTTLE_H100_PERF_REPORT" \
    --xla "$SHUTTLE_H100_XLA_HOST_REPORT" \
    --te-status "$SHUTTLE_H100_TE_STATUS" \
    --output "$SHUTTLE_H100_PERF_EXPLORATORY_REPORT"
fi
