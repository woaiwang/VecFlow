#include "vecflow/distance.hpp"

#include <immintrin.h>

namespace vecflow {

// Set at static initialization time via CPUID.
const bool g_cpu_has_avx2 = __builtin_cpu_supports("avx2");

namespace detail::avx2 {

float l2_sqr_kernel(const float* x, const float* y, size_t dim) {
    if (dim == 0) {
        return 0.0f;
    }

    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        __m256 x8 = _mm256_load_ps(&x[i]);
        __m256 y8 = _mm256_load_ps(&y[i]);
        __m256 diff = _mm256_sub_ps(x8, y8);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }

    // Horizontal reduction
    __m256 h = _mm256_hadd_ps(acc, acc);
    h = _mm256_hadd_ps(h, h);
    __m128 lo = _mm256_castps256_ps128(h);
    __m128 hi = _mm256_extractf128_ps(h, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    float result = _mm_cvtss_f32(sum128);

    // Scalar tail for remaining elements
    for (; i < dim; ++i) {
        float diff = x[i] - y[i];
        result += diff * diff;
    }

    return result;
}

float cosine_kernel(const float* x, const float* y, size_t dim) {
    if (dim == 0) {
        return 0.0f;
    }

    __m256 dot_acc = _mm256_setzero_ps();
    __m256 nx_acc = _mm256_setzero_ps();
    __m256 ny_acc = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        __m256 x8 = _mm256_load_ps(&x[i]);
        __m256 y8 = _mm256_load_ps(&y[i]);
        dot_acc = _mm256_fmadd_ps(x8, y8, dot_acc);
        nx_acc = _mm256_fmadd_ps(x8, x8, nx_acc);
        ny_acc = _mm256_fmadd_ps(y8, y8, ny_acc);
    }

    // Horizontal reduction helper
    auto reduce = [](__m256 v) -> float {
        __m256 h = _mm256_hadd_ps(v, v);
        h = _mm256_hadd_ps(h, h);
        __m128 lo = _mm256_castps256_ps128(h);
        __m128 hi = _mm256_extractf128_ps(h, 1);
        return _mm_cvtss_f32(_mm_add_ps(lo, hi));
    };

    float dot = reduce(dot_acc);
    float norm_x_sq = reduce(nx_acc);
    float norm_y_sq = reduce(ny_acc);

    // Scalar tail
    for (; i < dim; ++i) {
        dot += x[i] * y[i];
        norm_x_sq += x[i] * x[i];
        norm_y_sq += y[i] * y[i];
    }

    if (norm_x_sq == 0.0f || norm_y_sq == 0.0f) {
        return 0.0f;
    }

    float similarity = dot / std::sqrt(norm_x_sq * norm_y_sq);
    return 1.0f - similarity;
}

} // namespace detail::avx2
} // namespace vecflow
