// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics

#include <immintrin.h>
#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,int M, int N, int K, int lda, int ldb, int ldc) 
{
    for (int i = 0; i < M; ++i)
    {
        int j = 0;

        // for 256bits keep j+=8 and for 128bits keep j+=4
        for (; j + 3 < N; j += 4)
        {
            // for 256bits size register, we can process 8 floats at a time
            // __m256 vacc = _mm256_setzero_ps();

            // for 128bits size register, we can process 4 floats at a time
            __m128 vacc = _mm_setzero_ps();

            for (int k = 0; k < K; ++k)
            {
                
                // Load 8 B values for 256 bits register
                // __m256 vb = _mm256_set_ps(
                //     B[(j + 7) * ldb + k],
                //     B[(j + 6) * ldb + k],
                //     B[(j + 5) * ldb + k],
                //     B[(j + 4) * ldb + k],
                //     B[(j + 3) * ldb + k],
                //     B[(j + 2) * ldb + k],
                //     B[(j + 1) * ldb + k],
                //     B[j * ldb + k]
                // );


                // Load 4 B values for 128 bits register
                __m128 vb = _mm_set_ps(
                    B[static_cast<long>(j + 3) * ldb + k],
                    B[static_cast<long>(j + 2) * ldb + k],
                    B[static_cast<long>(j + 1) * ldb + k],
                    B[static_cast<long>(j) * ldb + k]
                );

                // Broadcast one A value to all 8 lanes
                // __m256 va = _mm256_set1_ps(A[i * lda + k]);

                // Broadcast one A value to all 4 lanes
                __m128 va = _mm_set1_ps(A[static_cast<long>(i) * lda + k]);

                // vacc += va * vb for 256bits register
                // vacc = _mm256_fmadd_ps(va, vb, vacc);

                // vacc += va * vb for 128bits register
                vacc = _mm_add_ps(vacc, _mm_mul_ps(va, vb));
            }

            // Store 8 output elements
            // _mm256_storeu_ps(C + i * ldc + j, vacc);

            // Store 4 output elements
            _mm_storeu_ps(C + static_cast<long>(i) * ldc + j, vacc);
        }

        // Handle remaining columns when N is not divisible by 8/4
        for (; j < N; ++j)
        {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                acc += A[static_cast<long>(i) * lda + k] * B[static_cast<long>(j) * ldb + k];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}