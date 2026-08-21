#!/bin/sh
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$#" -lt 7 ] || [ "$6" != "--python" ]; then
  echo "usage: $0 MODULE RUNNER GPU_TRANSPORT_TEST R12_RESULT R12_RAW --python PYTHON [MATRIX_ARGUMENTS...]" >&2
  exit 2
fi

module=$1
runner=$2
gpu_transport_test=$3
r12_result=$4
r12_raw=$5
python=$7
shift 7

exec "$python" "$module" \
  --runner "$runner" \
  --gpu-transport-test "$gpu_transport_test" \
  --r12-result "$r12_result" \
  --r12-policy-raw "$r12_raw" \
  "$@"
