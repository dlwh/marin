#!/bin/sh
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$#" -lt 6 ] || [ "$5" != "--python" ]; then
  echo "usage: $0 MODULE RUNNER R12_RESULT R12_RAW --python PYTHON [MATRIX_ARGUMENTS...]" >&2
  exit 2
fi

module=$1
runner=$2
r12_result=$3
r12_raw=$4
python=$6
shift 6

exec "$python" "$module" \
  --runner "$runner" \
  --r12-result "$r12_result" \
  --r12-policy-raw "$r12_raw" \
  "$@"
