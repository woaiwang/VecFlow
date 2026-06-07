// Truly scalar kernels compiled WITHOUT -march=native or AVX2 flags.
// This ensures a fair comparison against hand-tuned AVX2 intrinsics.
#include <cmath>

extern "C" {

float bench_scalar_l2(const float* x, const float* y, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = x[i] - y[i];
        sum += diff * diff;
    }
    return sum;
}

float bench_scalar_cosine(const float* x, const float* y, size_t dim) {
    float dot = 0.0f;
    float nx = 0.0f;
    float ny = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += x[i] * y[i];
        nx += x[i] * x[i];
        ny += y[i] * y[i];
    }
    if (nx == 0.0f || ny == 0.0f) return 0.0f;
    float similarity = dot / std::sqrt(nx * ny);
    return 1.0f - similarity;
}

} // extern "C"
