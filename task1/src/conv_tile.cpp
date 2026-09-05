// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#include <algorithm>
#include <cstdlib>

namespace
{

    int get_tile_size(int fallback)
    {
        if (const char *env = std::getenv("TILE"))
        {
            const int v = std::atoi(env);
            if (v > 0)
                return v;
        }
        return fallback;
    }

}

void conv_tile(const float *in, float *out, const float *ker,
               int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    static const int TILE = get_tile_size(64); // default tile size

    for (int ty = 0; ty < H; ty += TILE)
    {
        const int y_end = std::min(ty + TILE, H);
        for (int tx = 0; tx < W; tx += TILE)
        {
            const int x_end = std::min(tx + TILE, W);

            for (int oy = ty; oy < y_end; ++oy)
            {
                for (int ox = tx; ox < x_end; ++ox)
                {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ++ky)
                    {
                        const float *in_row = in + (oy + ky) * in_stride + ox;
                        const float *ker_row = ker + ky * K;
                        for (int kx = 0; kx < K; ++kx)
                        {
                            acc += in_row[kx] * ker_row[kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}