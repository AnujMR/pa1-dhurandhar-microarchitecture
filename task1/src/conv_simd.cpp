// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>
#include "convolution.h"
#include <immintrin.h>

void conv_simd(const float* in, float* out, const float* ker,int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    // const int p = K / 2;
    // const int in_stride = W + 2 * p;  // padded row stride
    // for (int oy = 0; oy < H; ++oy) 
    // {
    //     for (int ox = 0; ox < W; ++ox) 
    //     {
    //         float acc = 0.0f;
    //         for (int ky = 0; ky < K; ++ky) 
    //         {
    //             for (int kx = 0; kx < K; ++kx) 
    //             {
    //                 acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
    //             }
    //         }
    //         out[oy * W + ox] = acc;
    //     }
    // }

   
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        // AVX2: calculate 8 output pixels simultaneously


        // for 256bits keep ox+=8 and for 128bits keep ox+=4
        for (; ox + 7 < W; ox += 8) 
        {

            // for 256bits size register, we can process 8 floats at a time
            __m256 vacc = _mm256_setzero_ps();

            // for 128bits size register, we can process 4 floats at a time
            // __m128 vacc = _mm_setzero_ps();

            for (int ky = 0; ky < K; ++ky) 
            {
                const float* in_row = &in[(oy + ky) * in_stride + ox];
                const float* ker_row = &ker[ky * K];
                for (int kx = 0; kx < K; ++kx) 
                {
                    // Load 8 consecutive input pixels for 256 bits register
                    __m256 vin = _mm256_loadu_ps(in_row + kx);

                    // Load 4 consecutive input pixels for 128bits register
                    // __m128 vin = _mm_loadu_ps(in_row + kx);


                    // Broadcast one kernel value to all 8 lanes
                    __m256 vk = _mm256_set1_ps(ker_row[kx]);


                    // Broadcast one kernel value to all 4 lanes
                    // __m128 vk = _mm_set1_ps(ker_row[kx]);

                    // vacc += vin * vk for 256bits register
                    vacc = _mm256_fmadd_ps(vin, vk, vacc);

                    // vacc += vin * vk for 128bits register
                    // vacc = _mm_add_ps(vacc, _mm_mul_ps(vin, vk));
                }
            }
            // Store 8 output pixels
            _mm256_storeu_ps(out + oy * W + ox, vacc);

            //  Store 4 output pixels
            // _mm_storeu_ps(out + oy * W + ox, vacc);
        }

        // Handle remaining pixels when W is not divisible by 8/4
        for (; ox < W; ++ox) 
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
