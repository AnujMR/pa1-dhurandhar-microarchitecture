// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics

#include <immintrin.h>
#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,int M, int N, int K, int lda, int ldb, int ldc) 
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            // for 256bits size register, we can process 8 floats at a time
            __m256 vacc = _mm256_setzero_ps();

            // for 128bits size register, we can process 4 floats at a time
            // __m128 vacc = _mm_setzero_ps();

            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;

            int k = 0;

            for (; k + 7 < K; k += 8)
            {
                
                // Load 8 B values for 256 bits register
                __m256 vb = _mm256_loadu_ps(b + k);

                // Load 4 B values for 128 bits register
                // __m128 vb = _mm_loadu_ps(b + k);

                // Load 8 A values for 256 bits register
                __m256 va = _mm256_loadu_ps(a + k);

                // Load 4 A values for 128 bits register
                // __m128 va = _mm_loadu_ps(a + k);

                // vacc += va * vb for 256bits register
                vacc = _mm256_fmadd_ps(va, vb, vacc);

                // vacc += va * vb for 128bits register
                // vacc = _mm_fmadd_ps(va, vb, vacc);
            }

            // for 256bits register, we can process 8 floats at a time
            __m256 temp = _mm256_hadd_ps(vacc, vacc);
            temp = _mm256_hadd_ps(temp, temp);

            __m128 low = _mm256_castps256_ps128(temp);
            __m128 high = _mm256_extractf128_ps(temp, 1);

            __m128 sum = _mm_add_ps(low, high);

            float acc = _mm_cvtss_f32(sum);

            // for 128bits register, we can process 4 floats at a time
            // __m128 temp = _mm_hadd_ps(vacc, vacc);
            // temp = _mm_hadd_ps(temp, temp);

            // float acc = _mm_cvtss_f32(temp);


            // Handle remaining elements when K is not divisible by 8/4
            for (; k < K; ++k)
            {
                acc += a[k] * b[k];
            }

            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}