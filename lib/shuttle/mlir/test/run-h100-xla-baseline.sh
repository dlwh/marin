#!/bin/sh
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 XLA_BASELINE_PRODUCER" >&2
  exit 2
fi

if [ "${SHUTTLE_RUN_H100_XLA_BASELINE:-0}" != 1 ]; then
  echo "skipping ordinary-JAX H100 baseline; set SHUTTLE_RUN_H100_XLA_BASELINE=1"
  exit 0
fi

: "${SHUTTLE_H100_BASELINE_PYTHON:?set SHUTTLE_H100_BASELINE_PYTHON}"
: "${SHUTTLE_H100_XLA_HOST_REPORT:?set SHUTTLE_H100_XLA_HOST_REPORT}"
: "${SHUTTLE_H100_TE_STATUS:?set SHUTTLE_H100_TE_STATUS}"

if [ ! -x "$SHUTTLE_H100_BASELINE_PYTHON" ]; then
  echo "SHUTTLE_H100_BASELINE_PYTHON must be an executable Python with stock JAX GPU support" >&2
  exit 2
fi

"$SHUTTLE_H100_BASELINE_PYTHON" "$1" \
  --output "$SHUTTLE_H100_XLA_HOST_REPORT" \
  --te-status-output "$SHUTTLE_H100_TE_STATUS"
