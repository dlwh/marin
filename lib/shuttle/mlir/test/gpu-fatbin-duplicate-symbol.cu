// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_IMAGE_FACTOR
#error "define SHUTTLE_IMAGE_FACTOR for each independent image"
#endif
#ifndef SHUTTLE_IMAGE_BIAS
#error "define SHUTTLE_IMAGE_BIAS for each independent image"
#endif

extern "C" __global__ void shuttle_shared_symbol(const float *input,
                                                 float *output) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    *output = *input * SHUTTLE_IMAGE_FACTOR + SHUTTLE_IMAGE_BIAS;
}
