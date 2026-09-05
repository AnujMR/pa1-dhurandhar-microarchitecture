// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
                        constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int BLOCK_K = 64;

    constexpr int PREFETCH_DISTANCE = 16;

    auto hsum256 = [](__m256 v) -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);

        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);

        return _mm_cvtss_f32(sum);
    };

    //cache blocking
    for (int ii = 0; ii < M; ii += BLOCK_M) {
        for (int jj = 0; jj < N; jj += BLOCK_N) {
            for (int kk = 0; kk < K; kk += BLOCK_K) {

                const int i_end =
                    (ii + BLOCK_M < M) ? ii + BLOCK_M : M;

                const int j_end =
                    (jj + BLOCK_N < N) ? jj + BLOCK_N : N;

                const int k_end =
                    (kk + BLOCK_K < K) ? kk + BLOCK_K : K;

                // 4 rows
                for (int i = ii; i < i_end; i += 4) {

                    const int rows =
                        (i_end - i >= 4) ? 4 : (i_end - i);

                    // 2 output columns 
                    for (int j = jj; j < j_end; j += 2) {

                        const int cols =
                            (j_end - j >= 2) ? 2 : (j_end - j);

                        __m256 acc00 = _mm256_setzero_ps();
                        __m256 acc01 = _mm256_setzero_ps();

                        __m256 acc10 = _mm256_setzero_ps();
                        __m256 acc11 = _mm256_setzero_ps();

                        __m256 acc20 = _mm256_setzero_ps();
                        __m256 acc21 = _mm256_setzero_ps();

                        __m256 acc30 = _mm256_setzero_ps();
                        __m256 acc31 = _mm256_setzero_ps();

                        const float* b0 = B + static_cast<long>(j) * ldb;

                        const float* b1 = (cols >= 2)
                                ? B + static_cast<long>(j + 1) * ldb
                                : nullptr;

                        int p = kk;

                        for (; p + 7 < k_end; p += 8) {

                            //prefetch
                            if (p + PREFETCH_DISTANCE < k_end) {

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(
                                        b0 + p + PREFETCH_DISTANCE),
                                    _MM_HINT_T2);

                                if (cols >= 2) {
                                    _mm_prefetch(
                                        reinterpret_cast<const char*>(
                                            b1 + p + PREFETCH_DISTANCE),
                                        _MM_HINT_T2);
                                }

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(
                                        A + static_cast<long>(i) * lda +
                                        p + PREFETCH_DISTANCE),
                                    _MM_HINT_T2);

                                if (rows >= 2) {
                                    _mm_prefetch(
                                        reinterpret_cast<const char*>(
                                            A + static_cast<long>(i + 1) * lda +
                                            p + PREFETCH_DISTANCE),
                                        _MM_HINT_T2);
                                }

                                if (rows >= 3) {
                                    _mm_prefetch(
                                        reinterpret_cast<const char*>(
                                            A + static_cast<long>(i + 2) * lda +
                                            p + PREFETCH_DISTANCE),
                                        _MM_HINT_T2);
                                }

                                if (rows >= 4) {
                                    _mm_prefetch(
                                        reinterpret_cast<const char*>(
                                            A + static_cast<long>(i + 3) * lda +
                                            p + PREFETCH_DISTANCE),
                                        _MM_HINT_T2);
                                }
                            }

                            __m256 bv0 = _mm256_loadu_ps(b0 + p);

                            __m256 bv1 = (cols >= 2)
                                    ? _mm256_loadu_ps(b1 + p)
                                    : _mm256_setzero_ps();

                            __m256 av0 = _mm256_loadu_ps(A + static_cast<long>(i) * lda + p);

                            acc00 = _mm256_fmadd_ps(av0, bv0, acc00);

                            if (cols >= 2)
                                acc01 = _mm256_fmadd_ps(av0, bv1, acc01);

                            if (rows >= 2) {
                                __m256 av1 = _mm256_loadu_ps(A + static_cast<long>(i + 1) * lda + p);

                                acc10 = _mm256_fmadd_ps(av1, bv0, acc10);

                                if (cols >= 2)
                                    acc11 = _mm256_fmadd_ps(av1, bv1, acc11);
                            }

                            if (rows >= 3) {
                                __m256 av2 = _mm256_loadu_ps(A + static_cast<long>(i + 2) * lda + p);

                                acc20 = _mm256_fmadd_ps(av2, bv0, acc20);

                                if (cols >= 2)
                                    acc21 = _mm256_fmadd_ps(av2, bv1, acc21);
                            }

                            if (rows >= 4) {
                                __m256 av3 = _mm256_loadu_ps(A + static_cast<long>(i + 3) * lda + p);

                                acc30 = _mm256_fmadd_ps(av3, bv0, acc30);

                                if (cols >= 2)
                                    acc31 = _mm256_fmadd_ps(av3, bv1, acc31);
                            }
                        }

                        float r00 = hsum256(acc00);
                        float r01 = hsum256(acc01);

                        float r10 = hsum256(acc10);
                        float r11 = hsum256(acc11);

                        float r20 = hsum256(acc20);
                        float r21 = hsum256(acc21);

                        float r30 = hsum256(acc30);
                        float r31 = hsum256(acc31);

                        for (; p < k_end; ++p) {

                            float av0 = A[static_cast<long>(i) * lda + p];

                            r00 += av0 * b0[p];

                            if (cols >= 2)
                                r01 += av0 * b1[p];

                            if (rows >= 2) {
                                float av1 = A[static_cast<long>(i + 1) * lda + p];

                                r10 += av1 * b0[p];

                                if (cols >= 2)
                                    r11 += av1 * b1[p];
                            }

                            if (rows >= 3) {
                                float av2 = A[static_cast<long>(i + 2) * lda + p];

                                r20 += av2 * b0[p];

                                if (cols >= 2)
                                    r21 += av2 * b1[p];
                            }

                            if (rows >= 4) {
                                float av3 = A[static_cast<long>(i + 3) * lda + p];

                                r30 += av3 * b0[p];

                                if (cols >= 2)
                                    r31 += av3 * b1[p];
                            }
                        }

                        // accumulate k tile into C
                        const long c0 = static_cast<long>(i) * ldc + j;

                        const long c1 = static_cast<long>(i + 1) * ldc + j;

                        const long c2 = static_cast<long>(i + 2) * ldc + j;

                        const long c3 = static_cast<long>(i + 3) * ldc + j;

                        if (kk == 0) {
                            C[c0] = r00;

                            if (cols >= 2)
                                C[c0 + 1] = r01;

                            if (rows >= 2) {
                                C[c1] = r10;

                                if (cols >= 2)
                                    C[c1 + 1] = r11;
                            }

                            if (rows >= 3) {
                                C[c2] = r20;

                                if (cols >= 2)
                                    C[c2 + 1] = r21;
                            }

                            if (rows >= 4) {
                                C[c3] = r30;

                                if (cols >= 2)
                                    C[c3 + 1] = r31;
                            }

                        } 
                        else {
                            C[c0] += r00;

                            if (cols >= 2)
                                C[c0 + 1] += r01;

                            if (rows >= 2) {
                                C[c1] += r10;

                                if (cols >= 2)
                                    C[c1 + 1] += r11;
                            }

                            if (rows >= 3) {
                                C[c2] += r20;

                                if (cols >= 2)
                                    C[c2 + 1] += r21;
                            }

                            if (rows >= 4) {
                                C[c3] += r30;

                                if (cols >= 2)
                                    C[c3 + 1] += r31;
                            }
                        }
                    }
                }
            }
        }
    }
}