// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics

#include <immintrin.h>
#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.

    for (int i = 0; i < M; i += 4) {

        int rows = (M - i >= 4) ? 4 : (M - i);

        for (int j = 0; j < N; ++j) {

            const float* b = B + static_cast<long>(j) * ldb;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int p = 0;

            // SIMD LOOP
            for (; p + 7 < K; p += 8) {

                __m256 bv = _mm256_loadu_ps(b + p);

                if (rows >= 1) {
                    __m256 av0 = _mm256_loadu_ps(A + static_cast<long>(i) * lda + p);
                    acc0 = _mm256_fmadd_ps(av0, bv, acc0);
                }

                if (rows >= 2) {
                    __m256 av1 = _mm256_loadu_ps(A + static_cast<long>(i + 1) * lda + p);
                    acc1 = _mm256_fmadd_ps(av1, bv, acc1);
                }

                if (rows >= 3) {
                    __m256 av2 = _mm256_loadu_ps(A + static_cast<long>(i + 2) * lda + p);
                    acc2 = _mm256_fmadd_ps(av2, bv, acc2);
                }

                if (rows >= 4) {
                    __m256 av3 = _mm256_loadu_ps(A + static_cast<long>(i + 3) * lda + p);
                    acc3 = _mm256_fmadd_ps(av3, bv, acc3);
                }
            }

            // horizontal reduction
            auto hsum256 = [](__m256 v) -> float {
                __m128 lo = _mm256_castps256_ps128(v);
                __m128 hi = _mm256_extractf128_ps(v, 1);

                __m128 sum = _mm_add_ps(lo, hi);

                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);

                return _mm_cvtss_f32(sum);
            };

            float result0 = hsum256(acc0);
            float result1 = hsum256(acc1);
            float result2 = hsum256(acc2);
            float result3 = hsum256(acc3);

            for (; p < K; ++p) {

                result0 += A[static_cast<long>(i) * lda + p] * b[p];

                if (rows >= 2)
                    result1 += A[static_cast<long>(i + 1) * lda + p] * b[p];

                if (rows >= 3)
                    result2 += A[static_cast<long>(i + 2) * lda + p] * b[p];

                if (rows >= 4)
                    result3 += A[static_cast<long>(i + 3) * lda + p] * b[p];
            }

            C[static_cast<long>(i) * ldc + j] = result0;

            if (rows >= 2)
                C[static_cast<long>(i + 1) * ldc + j] = result1;

            if (rows >= 3)
                C[static_cast<long>(i + 2) * ldc + j] = result2;

            if (rows >= 4)
                C[static_cast<long>(i + 3) * ldc + j] = result3;
        }
    }
    //matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
