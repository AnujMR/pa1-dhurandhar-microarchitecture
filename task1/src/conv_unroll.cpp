// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_naive(const float* in, float* out, const float* ker,
                int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        // Process 4 output pixels per iteration
        for (; ox + 3 < W; ox += 4) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride;
                const float* k_row  = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    float w = k_row[kx];
                    acc0 += in_row[ox + 0 + kx] * w;
                    acc1 += in_row[ox + 1 + kx] * w;
                    acc2 += in_row[ox + 2 + kx] * w;
                    acc3 += in_row[ox + 3 + kx] * w;
                }
            }
            out[oy * W + ox + 0] = acc0;
            out[oy * W + ox + 1] = acc1;
            out[oy * W + ox + 2] = acc2;
            out[oy * W + ox + 3] = acc3;
        }
    }
}
