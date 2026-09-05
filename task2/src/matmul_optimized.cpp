#include <immintrin.h>
#include "matmul.h"

#define BLOCK_M 64
#define BLOCK_N 32
#define PREFETCH_DISTANCE 16

static inline float hsum256(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K,
                      int lda, int ldb, int ldc) {

    for (int ii = 0; ii < M; ii += BLOCK_M) {
        int i_end = (ii + BLOCK_M < M) ? ii + BLOCK_M : M;

        for (int jj = 0; jj < N; jj += BLOCK_N) {
            int j_end = (jj + BLOCK_N < N) ? jj + BLOCK_N : N;

            for (int i = ii; i < i_end; i += 4) {

                int rows = (i_end - i >= 4) ? 4 : (i_end - i);

                const float* a0 = A + (long)i * lda;
                const float* a1 = rows >= 2 ? A + (long)(i + 1) * lda : nullptr;
                const float* a2 = rows >= 3 ? A + (long)(i + 2) * lda : nullptr;
                const float* a3 = rows >= 4 ? A + (long)(i + 3) * lda : nullptr;

                int j = jj;

                for (; j + 1 < j_end; j += 2) {

                    const float* b0 = B + (long)j * ldb;
                    const float* b1 = B + (long)(j + 1) * ldb;

                    __m256 acc00 = _mm256_setzero_ps();
                    __m256 acc01 = _mm256_setzero_ps();
                    __m256 acc10 = _mm256_setzero_ps();
                    __m256 acc11 = _mm256_setzero_ps();
                    __m256 acc20 = _mm256_setzero_ps();
                    __m256 acc21 = _mm256_setzero_ps();
                    __m256 acc30 = _mm256_setzero_ps();
                    __m256 acc31 = _mm256_setzero_ps();

                    for (int p = 0; p + 7 < K; p += 8) {

                        if (p + PREFETCH_DISTANCE < K) {
                            _mm_prefetch(
                                (const char*)(a0 + p + PREFETCH_DISTANCE),
                                _MM_HINT_T2);

                            _mm_prefetch(
                                (const char*)(b0 + p + PREFETCH_DISTANCE),
                                _MM_HINT_T2);

                            _mm_prefetch(
                                (const char*)(b1 + p + PREFETCH_DISTANCE),
                                _MM_HINT_T2);

                            if (rows >= 2)
                                _mm_prefetch(
                                    (const char*)(a1 + p + PREFETCH_DISTANCE),
                                    _MM_HINT_T2);

                            if (rows >= 3)
                                _mm_prefetch(
                                    (const char*)(a2 + p + PREFETCH_DISTANCE),
                                    _MM_HINT_T2);

                            if (rows >= 4)
                                _mm_prefetch(
                                    (const char*)(a3 + p + PREFETCH_DISTANCE),
                                    _MM_HINT_T2);
                        }

                        __m256 av0 = _mm256_loadu_ps(a0 + p);
                        __m256 bv0 = _mm256_loadu_ps(b0 + p);
                        __m256 bv1 = _mm256_loadu_ps(b1 + p);

                        acc00 = _mm256_fmadd_ps(av0, bv0, acc00);
                        acc01 = _mm256_fmadd_ps(av0, bv1, acc01);

                        if (rows >= 2) {
                            __m256 av1 = _mm256_loadu_ps(a1 + p);
                            acc10 = _mm256_fmadd_ps(av1, bv0, acc10);
                            acc11 = _mm256_fmadd_ps(av1, bv1, acc11);
                        }

                        if (rows >= 3) {
                            __m256 av2 = _mm256_loadu_ps(a2 + p);
                            acc20 = _mm256_fmadd_ps(av2, bv0, acc20);
                            acc21 = _mm256_fmadd_ps(av2, bv1, acc21);
                        }

                        if (rows >= 4) {
                            __m256 av3 = _mm256_loadu_ps(a3 + p);
                            acc30 = _mm256_fmadd_ps(av3, bv0, acc30);
                            acc31 = _mm256_fmadd_ps(av3, bv1, acc31);
                        }
                    }

                    float s00 = hsum256(acc00);
                    float s01 = hsum256(acc01);
                    float s10 = hsum256(acc10);
                    float s11 = hsum256(acc11);
                    float s20 = hsum256(acc20);
                    float s21 = hsum256(acc21);
                    float s30 = hsum256(acc30);
                    float s31 = hsum256(acc31);

                    for (int p = (K / 8) * 8; p < K; ++p) {
                        s00 += a0[p] * b0[p];
                        s01 += a0[p] * b1[p];

                        if (rows >= 2) {
                            s10 += a1[p] * b0[p];
                            s11 += a1[p] * b1[p];
                        }

                        if (rows >= 3) {
                            s20 += a2[p] * b0[p];
                            s21 += a2[p] * b1[p];
                        }

                        if (rows >= 4) {
                            s30 += a3[p] * b0[p];
                            s31 += a3[p] * b1[p];
                        }
                    }

                    C[(long)i * ldc + j] = s00;
                    C[(long)i * ldc + j + 1] = s01;

                    if (rows >= 2) {
                        C[(long)(i + 1) * ldc + j] = s10;
                        C[(long)(i + 1) * ldc + j + 1] = s11;
                    }

                    if (rows >= 3) {
                        C[(long)(i + 2) * ldc + j] = s20;
                        C[(long)(i + 2) * ldc + j + 1] = s21;
                    }

                    if (rows >= 4) {
                        C[(long)(i + 3) * ldc + j] = s30;
                        C[(long)(i + 3) * ldc + j + 1] = s31;
                    }
                }

                if (j < j_end) {
                    const float* b = B + (long)j * ldb;

                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();

                    int p = 0;

                    for (; p + 7 < K; p += 8) {
                        if (p + PREFETCH_DISTANCE < K) {
                            _mm_prefetch(
                                (const char*)(b + p + PREFETCH_DISTANCE),
                                _MM_HINT_T2);
                        }

                        __m256 bv = _mm256_loadu_ps(b + p);

                        acc0 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(a0 + p), bv, acc0);

                        if (rows >= 2)
                            acc1 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(a1 + p), bv, acc1);

                        if (rows >= 3)
                            acc2 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(a2 + p), bv, acc2);

                        if (rows >= 4)
                            acc3 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(a3 + p), bv, acc3);
                    }

                    float s0 = hsum256(acc0);
                    float s1 = hsum256(acc1);
                    float s2 = hsum256(acc2);
                    float s3 = hsum256(acc3);

                    for (; p < K; ++p) {
                        s0 += a0[p] * b[p];

                        if (rows >= 2) s1 += a1[p] * b[p];
                        if (rows >= 3) s2 += a2[p] * b[p];
                        if (rows >= 4) s3 += a3[p] * b[p];
                    }

                    C[(long)i * ldc + j] = s0;

                    if (rows >= 2) C[(long)(i + 1) * ldc + j] = s1;
                    if (rows >= 3) C[(long)(i + 2) * ldc + j] = s2;
                    if (rows >= 4) C[(long)(i + 3) * ldc + j] = s3;
                }
            }
        }
    }
}
