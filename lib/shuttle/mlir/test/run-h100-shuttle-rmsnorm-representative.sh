#!/bin/sh
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$#" -lt 1 ]; then
  echo "usage: $0 GPU_TRANSPORT_TEST --case-root PATH --raw-root PATH --policy POLICY --boundary BOUNDARY --repeat N" >&2
  exit 2
fi

gpu_transport_test=$1
shift

case_root=
raw_root=
policy=
boundary=
repeat=

while [ "$#" -gt 0 ]; do
  if [ "$#" -lt 2 ]; then
    echo "missing value for $1" >&2
    exit 2
  fi
  case "$1" in
    --case-root) case_root=$2 ;;
    --raw-root) raw_root=$2 ;;
    --policy) policy=$2 ;;
    --boundary) boundary=$2 ;;
    --repeat) repeat=$2 ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift 2
done

if [ -z "$case_root" ] || [ -z "$raw_root" ] || [ -z "$policy" ] || [ -z "$boundary" ] || [ -z "$repeat" ]; then
  echo "case root, raw root, policy, boundary, and repeat are required" >&2
  exit 2
fi

case "$policy" in
  source_ordered|fast) ;;
  *) echo "unsupported policy: $policy" >&2; exit 2 ;;
esac
case "$boundary" in
  forward|backward_recompute|composed) ;;
  *) echo "unsupported boundary: $boundary" >&2; exit 2 ;;
esac
case "$repeat" in
  0|1) ;;
  *) echo "unsupported repeat: $repeat" >&2; exit 2 ;;
esac

case "$boundary" in
  forward) representative_fixture=jax-0.10.1-bf16-projected_parameter_broadcast-2048x4096-forward.mlir ;;
  backward_recompute) representative_fixture=jax-0.10.1-bf16-projected_parameter_broadcast-2048x4096-backward.mlir ;;
  composed) representative_fixture=jax-0.10.1-bf16-projected_parameter_broadcast-2048x4096-composed.mlir ;;
esac

runfiles_root=
consider_runfiles_root() {
  candidate=$1
  if [ ! -d "$candidate" ]; then
    return
  fi
  candidate=$(CDPATH= cd "$candidate" 2>/dev/null && pwd -P) || return
  if [ ! -r "$candidate/shuttle_mlir/test/Inputs/$representative_fixture" ]; then
    return
  fi
  if [ -n "$runfiles_root" ] && [ "$runfiles_root" != "$candidate" ]; then
    echo "ambiguous representative runfiles roots: $runfiles_root and $candidate" >&2
    exit 2
  fi
  runfiles_root=$candidate
}

if [ -z "${TEST_SRCDIR:-}" ]; then
  if [ -n "${RUNFILES_DIR:-}" ]; then
    if [ ! -d "$RUNFILES_DIR" ]; then
      echo "RUNFILES_DIR is not a directory: $RUNFILES_DIR" >&2
      exit 2
    fi
    consider_runfiles_root "$RUNFILES_DIR"
  fi
  consider_runfiles_root "$PWD"
  consider_runfiles_root "$PWD/.."
  gpu_transport_dir=$(dirname "$gpu_transport_test")
  consider_runfiles_root "$gpu_transport_dir/../../.."
  if [ -z "$runfiles_root" ]; then
    echo "unable to derive representative runfiles root for $representative_fixture" >&2
    exit 2
  fi
  TEST_SRCDIR=$runfiles_root
  export TEST_SRCDIR
fi

SHUTTLE_RUN_H100_REPRESENTATIVE=1 \
SHUTTLE_REP_CASE_ROOT="$case_root" \
SHUTTLE_REP_RAW_ROOT="$raw_root" \
SHUTTLE_REP_POLICY="$policy" \
SHUTTLE_REP_BOUNDARY="$boundary" \
SHUTTLE_REP_REPEAT="$repeat" \
  "$gpu_transport_test" \
  --gtest_fail_if_no_test_selected \
  --gtest_filter=GpuTransportV2Test.H100RepresentativeRmsNormRunnerWritesOneAuthenticatedRawCell
