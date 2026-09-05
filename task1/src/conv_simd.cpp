// for 256 bits registers, we can use the following code:
// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>
#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy)
    {
        int ox = 0;
        for (; ox + 15 < W; ox += 16)
        {
            __m256 vacc0 = _mm256_setzero_ps();
            __m256 vacc1 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky)
            {
                const float* in_row =&in[(oy + ky) * in_stride + ox];
                const float* ker_row =&ker[ky * K];
                for (int kx = 0; kx < K; ++kx)
                {
                    __m256 vin0 =_mm256_loadu_ps(in_row + kx);
                    __m256 vin1 =_mm256_loadu_ps(in_row + kx + 8);
                    __m256 vk =_mm256_set1_ps(ker_row[kx]);
                    vacc0 =_mm256_fmadd_ps(vin0, vk, vacc0);
                    vacc1 =_mm256_fmadd_ps(vin1, vk, vacc1);
                }
            }
            _mm256_storeu_ps(out + oy * W + ox, vacc0);
            _mm256_storeu_ps(out + oy * W + ox + 8, vacc1);
        }

        for (; ox + 7 < W; ox += 8)
        {
            __m256 vacc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky)
            {
                const float* in_row =&in[(oy + ky) * in_stride + ox];
                const float* ker_row =&ker[ky * K];
                for (int kx = 0; kx < K; ++kx)
                {
                    __m256 vin =_mm256_loadu_ps(in_row + kx);
                    __m256 vk =_mm256_set1_ps(ker_row[kx]);
                    vacc =_mm256_fmadd_ps(vin, vk, vacc);
                }
            }

            _mm256_storeu_ps(out + oy * W + ox, vacc);
        }

        for (; ox < W; ++ox)
        {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky)
            {
                for (int kx = 0; kx < K; ++kx)
                {
                    acc +=in[(oy + ky) * in_stride + (ox + kx)]* ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}


// For 128 bits registers, we can use the following code:
// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
// #include <immintrin.h>
// #include "convolution.h"

// void conv_simd(const float* in, float* out, const float* ker,int H, int W, int K)
// {
//     const int p = K / 2;
//     const int in_stride = W + 2 * p;

//     for (int oy = 0; oy < H; ++oy)
//     {
//         int ox = 0;
//         for (; ox + 7 < W; ox += 8)
//         {
//             __m128 vacc0 = _mm_setzero_ps();
//             __m128 vacc1 = _mm_setzero_ps();

//             for (int ky = 0; ky < K; ++ky)
//             {
//                 const float* in_row =&in[(oy + ky) * in_stride + ox];
//                 const float* ker_row =&ker[ky * K];

//                 for (int kx = 0; kx < K; ++kx)
//                 {
//                     __m128 vin0 =_mm_loadu_ps(in_row + kx);
//                     __m128 vin1 =_mm_loadu_ps(in_row + kx + 4);
//                     __m128 vk =_mm_set1_ps(ker_row[kx]);

//                     vacc0 =_mm_fmadd_ps(vin0, vk, vacc0);
//                     vacc1 =_mm_fmadd_ps(vin1, vk, vacc1);
//                 }
//             }

//             _mm_storeu_ps(out + oy * W + ox, vacc0);
//             _mm_storeu_ps(out + oy * W + ox + 4, vacc1);
//         }

//         for (; ox + 3 < W; ox += 4)
//         {
//             __m128 vacc = _mm_setzero_ps();

//             for (int ky = 0; ky < K; ++ky)
//             {
//                 const float* in_row =&in[(oy + ky) * in_stride + ox];
//                 const float* ker_row =&ker[ky * K];

//                 for (int kx = 0; kx < K; ++kx)
//                 {
//                     __m128 vin =_mm_loadu_ps(in_row + kx);
//                     __m128 vk =_mm_set1_ps(ker_row[kx]);

//                     vacc =_mm_fmadd_ps(vin, vk, vacc);
//                 }
//             }

//             _mm_storeu_ps(out + oy * W + ox, vacc);
//         }

//         for (; ox < W; ++ox)
//         {
//             float acc = 0.0f;

//             for (int ky = 0; ky < K; ++ky)
//             {
//                 for (int kx = 0; kx < K; ++kx)
//                 {
//                     acc +=in[(oy + ky) * in_stride + (ox + kx)]* ker[ky * K + kx];
//                 }
//             }

//             out[oy * W + ox] = acc;
//         }
//     }
// }
