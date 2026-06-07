#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "vecflow/types.hpp"

namespace vecflow {

/// Distance metric selector.
enum class DistanceType {
    L2Sqr,  ///< Squared Euclidean distance: sum((x[i] - y[i])^2)
    Cosine  ///< Cosine distance: 1 - dot(x,y) / (|x|*|y|)
};

/// Set at static initialization in src/distance_avx2.cpp.
/// True if the host CPU supports AVX2 + FMA.  The public distance
/// functions (compute_l2_sqr, compute_cosine) use this to dispatch
/// between the portable scalar kernel and the hand-tuned AVX2 kernel.
extern const bool g_cpu_has_avx2;

/// AVX2 kernels (defined in src/distance_avx2.cpp, compiled with -mavx2 -mfma).
/// These assume the input pointers are 32-byte aligned.
namespace detail::avx2 {
float l2_sqr_kernel(const float* x, const float* y, size_t dim);
float cosine_kernel(const float* x, const float* y, size_t dim);
} // namespace detail::avx2

namespace detail {

template <typename T>
[[nodiscard]] float l2_sqr_kernel(const T* x, const T* y, size_t dim) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "Distance kernels only support float and double");

    if (dim == 0) {
        return 0.0f;
    }

    T sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        T diff = x[i] - y[i];
        sum += diff * diff;
    }
    return static_cast<float>(sum);
}

template <typename T>
[[nodiscard]] float cosine_kernel(const T* x, const T* y, size_t dim) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "Distance kernels only support float and double");

    if (dim == 0) {
        return 0.0f;
    }

    T dot = 0;
    T norm_x_sq = 0;
    T norm_y_sq = 0;

    for (size_t i = 0; i < dim; ++i) {
        dot += x[i] * y[i];
        norm_x_sq += x[i] * x[i];
        norm_y_sq += y[i] * y[i];
    }

    if (norm_x_sq == 0 || norm_y_sq == 0) {
        return 0.0f;
    }

    float similarity = static_cast<float>(dot) /
                       std::sqrt(static_cast<float>(norm_x_sq) *
                                 static_cast<float>(norm_y_sq));
    return 1.0f - similarity;
}

} // namespace detail

/// Squared Euclidean distance between two vectors.
/// Automatically dispatches to AVX2 on supported CPUs (float only).
/// @tparam T  float or double
/// @throws std::invalid_argument if dimensions mismatch
template <typename T>
[[nodiscard]] float compute_l2_sqr(const Vector<T>& a, const Vector<T>& b) {
    if (a.data.size() != b.data.size()) {
        throw std::invalid_argument(
            "Vector dimension mismatch in compute_l2_sqr");
    }
    if constexpr (std::is_same_v<T, float>) {
        if (g_cpu_has_avx2) {
            return detail::avx2::l2_sqr_kernel(
                a.data.data(), b.data.data(), a.data.size());
        }
    }
    return detail::l2_sqr_kernel(a.data.data(), b.data.data(), a.data.size());
}

/// Cosine distance between two vectors (0 = identical, 1 = orthogonal, 2 = opposite).
/// Automatically dispatches to AVX2 on supported CPUs (float only).
/// @tparam T  float or double
/// @throws std::invalid_argument if dimensions mismatch
template <typename T>
[[nodiscard]] float compute_cosine(const Vector<T>& a, const Vector<T>& b) {
    if (a.data.size() != b.data.size()) {
        throw std::invalid_argument(
            "Vector dimension mismatch in compute_cosine");
    }
    if constexpr (std::is_same_v<T, float>) {
        if (g_cpu_has_avx2) {
            return detail::avx2::cosine_kernel(
                a.data.data(), b.data.data(), a.data.size());
        }
    }
    return detail::cosine_kernel(a.data.data(), b.data.data(), a.data.size());
}

/// Compute distance using the specified metric.
/// Convenience wrapper that dispatches to compute_l2_sqr or compute_cosine.
template <typename T>
[[nodiscard]] float compute_distance(const Vector<T>& a, const Vector<T>& b,
                                     DistanceType type) {
    switch (type) {
        case DistanceType::L2Sqr:
            return compute_l2_sqr(a, b);
        case DistanceType::Cosine:
            return compute_cosine(a, b);
    }
    return 0.0f;
}

} // namespace vecflow
