// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

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

void conv_optimized(const float* in, float* out, const float* ker,int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    static const int TILE = get_tile_size(64);

    for (int ty = 0; ty < H; ty += TILE)
    {
        const int y_end = std::min(ty + TILE, H);

        for (int tx = 0; tx < W; tx += TILE)
        {
            const int x_end = std::min(tx + TILE, W);

            for (int oy = ty; oy < y_end; ++oy)
            {
                int ox = tx;

                for (; ox + 15 < x_end; ox += 16)
                {
                    __m256 vacc0 = _mm256_setzero_ps();
                    __m256 vacc1 = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky)
                    {
                        const float* in_row =
                            in + (oy + ky) * in_stride + ox;

                        const float* ker_row =
                            ker + ky * K;

                        for (int kx = 0; kx < K; ++kx)
                        {
                            __m256 vin0 =
                                _mm256_loadu_ps(in_row + kx);

                            __m256 vin1 =
                                _mm256_loadu_ps(in_row + kx + 8);

                            __m256 vk =
                                _mm256_set1_ps(ker_row[kx]);

                            vacc0 =
                                _mm256_fmadd_ps(vin0, vk, vacc0);

                            vacc1 =
                                _mm256_fmadd_ps(vin1, vk, vacc1);
                        }
                    }

                    _mm256_storeu_ps(
                        out + oy * W + ox,
                        vacc0);

                    _mm256_storeu_ps(
                        out + oy * W + ox + 8,
                        vacc1);
                }

                for (; ox + 7 < x_end; ox += 8)
                {
                    __m256 vacc =
                        _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky)
                    {
                        const float* in_row =
                            in + (oy + ky) * in_stride + ox;

                        const float* ker_row =
                            ker + ky * K;

                        for (int kx = 0; kx < K; ++kx)
                        {
                            __m256 vin =
                                _mm256_loadu_ps(in_row + kx);

                            __m256 vk =
                                _mm256_set1_ps(ker_row[kx]);

                            vacc =
                                _mm256_fmadd_ps(
                                    vin, vk, vacc);
                        }
                    }

                    _mm256_storeu_ps(
                        out + oy * W + ox,
                        vacc);
                }

                for (; ox < x_end; ++ox)
                {
                    float acc = 0.0f;

                    for (int ky = 0; ky < K; ++ky)
                    {
                        for (int kx = 0; kx < K; ++kx)
                        {
                            acc +=
                                in[(oy + ky) * in_stride +
                                   (ox + kx)]
                                * ker[ky * K + kx];
                        }
                    }

                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}


// void conv_optimized(const float* in, float* out, const float* ker,int H, int W, int K)
// {
//     const int p = K / 2;
//     const int in_stride = W + 2 * p;

//     static const int TILE = get_tile_size(64);

//     for (int ty = 0; ty < H; ty += TILE)
//     {
//         const int y_end = std::min(ty + TILE, H);
//         for (int tx = 0; tx < W; tx += TILE)
//         {
//             const int x_end = std::min(tx + TILE, W);
//             for (int oy = ty; oy < y_end; ++oy)
//             {
//                 int ox = tx;
//                 for (; ox + 31 < x_end; ox += 32)
//                 {
//                     __m256 vacc0 = _mm256_setzero_ps();
//                     __m256 vacc1 = _mm256_setzero_ps();
//                     __m256 vacc2 = _mm256_setzero_ps();
//                     __m256 vacc3 = _mm256_setzero_ps();
//                     for (int ky = 0; ky < K; ++ky)
//                     {
//                         const float* in_row =in + (oy + ky) * in_stride + ox;
//                         const float* ker_row =ker + ky * K;
//                         for (int kx = 0; kx < K; ++kx)
//                         {
//                             __m256 vk =_mm256_set1_ps(ker_row[kx]);
//                             __m256 vin0 = _mm256_loadu_ps(in_row + kx);
//                             __m256 vin1 =_mm256_loadu_ps(in_row + kx + 8);
//                             __m256 vin2 =_mm256_loadu_ps(in_row + kx + 16);
//                             __m256 vin3 =_mm256_loadu_ps(in_row + kx + 24);
//                             vacc0 =_mm256_fmadd_ps(vin0, vk, vacc0);
//                             vacc1 =_mm256_fmadd_ps(vin1, vk, vacc1);
//                             vacc2 = _mm256_fmadd_ps(vin2, vk, vacc2);
//                             vacc3 =_mm256_fmadd_ps(vin3, vk, vacc3);
//                         }
//                     }
//                     _mm256_storeu_ps(out + oy * W + ox,vacc0);
//                     _mm256_storeu_ps(out + oy * W + ox + 8,vacc1);
//                     _mm256_storeu_ps(out + oy * W + ox + 16,vacc2);
//                     _mm256_storeu_ps(out + oy * W + ox + 24,vacc3);
//                 }

//                 for (; ox + 15 < x_end; ox += 16)
//                 {
//                     __m256 vacc0 = _mm256_setzero_ps();
//                     __m256 vacc1 = _mm256_setzero_ps();
//                     for (int ky = 0; ky < K; ++ky)
//                     {
//                         const float* in_row =in + (oy + ky) * in_stride + ox;
//                         const float* ker_row =ker + ky * K;
//                         for (int kx = 0; kx < K; ++kx)
//                         {
//                             __m256 vk =_mm256_set1_ps(ker_row[kx]);
//                             __m256 vin0 =_mm256_loadu_ps(in_row + kx);
//                             __m256 vin1 =_mm256_loadu_ps(in_row + kx + 8);
//                             vacc0 =_mm256_fmadd_ps(vin0, vk, vacc0);
//                             vacc1 =_mm256_fmadd_ps(vin1, vk, vacc1);
//                         }
//                     }
//                     _mm256_storeu_ps(out + oy * W + ox,vacc0);
//                     _mm256_storeu_ps(out + oy * W + ox + 8,vacc1);
//                 }
//                 for (; ox + 7 < x_end; ox += 8)
//                 {
//                     __m256 vacc =_mm256_setzero_ps();
//                     for (int ky = 0; ky < K; ++ky)
//                     {
//                         const float* in_row =in + (oy + ky) * in_stride + ox;
//                         const float* ker_row =ker + ky * K;
//                         for (int kx = 0; kx < K; ++kx)
//                         {
//                             __m256 vin =_mm256_loadu_ps(in_row + kx);
//                             __m256 vk =_mm256_set1_ps(ker_row[kx]);
//                             vacc =_mm256_fmadd_ps(vin, vk, vacc);
//                         }
//                     }
//                     _mm256_storeu_ps(out + oy * W + ox,vacc);
//                 }

//                 for (; ox < x_end; ++ox)
//                 {
//                     float acc = 0.0f;
//                     for (int ky = 0; ky < K; ++ky)
//                     {
//                         for (int kx = 0; kx < K; ++kx)
//                         {
//                             acc +=in[(oy + ky) * in_stride +(ox + kx)]* ker[ky * K + kx];
//                         }
//                     }
//                     out[oy * W + ox] = acc;
//                 }
//             }
//         }
//     }
// }

