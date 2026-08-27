// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

// void conv_tile(const float* in, float* out, const float* ker,
//                int H, int W, int K) {
//     // TODO(student): replace this placeholder with your tiled/blocked implementation.
//     conv_naive(in, out, ker, H, W, K);
// }

#define TILE_SIZE 25

void conv_tile(const float *in, float *out, const float *ker,
               int H, int W, int K)
{

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int ty = 0; ty < H; ty += TILE_SIZE)
    {
        for (int tx = 0; tx < W; tx += TILE_SIZE)
        {
            for (int oy = ty; oy < ty + TILE_SIZE && oy < H; ++oy)
            {
                for (int ox = tx; ox < tx + TILE_SIZE && ox < W; ++ox)
                {

                    float acc = 0.0f;

                    for (int ky = 0; ky < K; ++ky)
                    {
                        for (int kx = 0; kx < K; ++kx)
                        {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }

                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}