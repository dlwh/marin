// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

struct OutputView {
  float *data;
  int64_t count;
  int64_t stride;
};

extern "C" __global__ void shuttle_stage0(const float *input, float *scratch,
                                           int32_t zero) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 16)
    scratch[index] = input[index] * 2.0f + static_cast<float>(zero);
}

extern "C" __global__ void shuttle_stage1(float *scratch, OutputView output,
                                           uint32_t tag) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= output.count)
    return;
  scratch[index] += 1.0f;
  output.data[index * output.stride] =
      scratch[index] + static_cast<float>(tag);
}
