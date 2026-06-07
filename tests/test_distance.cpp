#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>

#include "vecflow/allocator.hpp"
#include "vecflow/distance.hpp"
#include "vecflow/types.hpp"

// ---------------------------------------------------------------
// L2Sqr — float
// ---------------------------------------------------------------

TEST(L2SqrTest, FloatBasic) {
    vecflow::Vector<float> a(0, {1.0f, 2.0f, 3.0f});
    vecflow::Vector<float> b(1, {4.0f, 5.0f, 6.0f});
    // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
    EXPECT_FLOAT_EQ(vecflow::compute_l2_sqr(a, b), 27.0f);
}

TEST(L2SqrTest, FloatZeroVectors) {
    vecflow::Vector<float> a(0, {0.0f, 0.0f, 0.0f});
    vecflow::Vector<float> b(1, {0.0f, 0.0f, 0.0f});
    EXPECT_FLOAT_EQ(vecflow::compute_l2_sqr(a, b), 0.0f);
}

TEST(L2SqrTest, FloatFractional) {
    vecflow::Vector<float> a(0, {1.5f, 2.5f});
    vecflow::Vector<float> b(1, {3.5f, 4.5f});
    // (1.5-3.5)^2 + (2.5-4.5)^2 = 4 + 4 = 8
    EXPECT_FLOAT_EQ(vecflow::compute_l2_sqr(a, b), 8.0f);
}

TEST(L2SqrTest, FloatSingleElement) {
    vecflow::Vector<float> a(0, {5.0f});
    vecflow::Vector<float> b(1, {3.0f});
    EXPECT_FLOAT_EQ(vecflow::compute_l2_sqr(a, b), 4.0f);
}

TEST(L2SqrTest, FloatIdenticalVectors) {
    vecflow::Vector<float> a(0, {3.14f, 2.71f, 1.41f});
    vecflow::Vector<float> b(1, {3.14f, 2.71f, 1.41f});
    EXPECT_FLOAT_EQ(vecflow::compute_l2_sqr(a, b), 0.0f);
}

TEST(L2SqrTest, FloatZeroDim) {
    vecflow::Vector<float> a(0, {});
    vecflow::Vector<float> b(1, {});
    EXPECT_FLOAT_EQ(vecflow::compute_l2_sqr(a, b), 0.0f);
}

// ---------------------------------------------------------------
// L2Sqr — double
// ---------------------------------------------------------------

TEST(L2SqrTest, DoubleBasic) {
    vecflow::Vector<double> a(0, {1.0, 2.0, 3.0});
    vecflow::Vector<double> b(1, {4.0, 5.0, 6.0});
    // Same as float case: 27
    EXPECT_DOUBLE_EQ(vecflow::compute_l2_sqr(a, b), 27.0);
}

// ---------------------------------------------------------------
// Cosine — float
// ---------------------------------------------------------------

TEST(CosineTest, FloatOrthogonal) {
    vecflow::Vector<float> a(0, {1.0f, 0.0f});
    vecflow::Vector<float> b(1, {0.0f, 1.0f});
    // dot=0, norms=1 → distance=1
    EXPECT_FLOAT_EQ(vecflow::compute_cosine(a, b), 1.0f);
}

TEST(CosineTest, FloatIdentical) {
    vecflow::Vector<float> a(0, {1.0f, 0.0f});
    vecflow::Vector<float> b(1, {1.0f, 0.0f});
    // dot=1, norms=1 → distance=0
    EXPECT_FLOAT_EQ(vecflow::compute_cosine(a, b), 0.0f);
}

TEST(CosineTest, FloatOpposite) {
    vecflow::Vector<float> a(0, {1.0f, 2.0f, 3.0f});
    vecflow::Vector<float> b(1, {-1.0f, -2.0f, -3.0f});
    // dot=-14, |a|^2=14, |b|^2=14 → distance = 1 - (-14/14) = 2
    EXPECT_FLOAT_EQ(vecflow::compute_cosine(a, b), 2.0f);
}

TEST(CosineTest, FloatGeneral) {
    vecflow::Vector<float> a(0, {1.0f, 2.0f, 3.0f});
    vecflow::Vector<float> b(1, {2.0f, 3.0f, 4.0f});
    // dot=20, |a|^2=14, |b|^2=29 → 1 - 20/sqrt(406)
    // sqrt(406) ≈ 20.14944, 20/20.14944 ≈ 0.992583
    float expected = 1.0f - 20.0f / std::sqrt(14.0f * 29.0f);
    EXPECT_NEAR(vecflow::compute_cosine(a, b), expected, 1e-5f);
}

TEST(CosineTest, FloatZeroVectorA) {
    vecflow::Vector<float> a(0, {0.0f, 0.0f, 0.0f});
    vecflow::Vector<float> b(1, {1.0f, 2.0f, 3.0f});
    EXPECT_FLOAT_EQ(vecflow::compute_cosine(a, b), 0.0f);
}

TEST(CosineTest, FloatZeroVectorB) {
    vecflow::Vector<float> a(0, {1.0f, 2.0f, 3.0f});
    vecflow::Vector<float> b(1, {0.0f, 0.0f, 0.0f});
    EXPECT_FLOAT_EQ(vecflow::compute_cosine(a, b), 0.0f);
}

TEST(CosineTest, FloatBothZero) {
    vecflow::Vector<float> a(0, {0.0f, 0.0f});
    vecflow::Vector<float> b(1, {0.0f, 0.0f});
    EXPECT_FLOAT_EQ(vecflow::compute_cosine(a, b), 0.0f);
}

TEST(CosineTest, FloatZeroDim) {
    vecflow::Vector<float> a(0, {});
    vecflow::Vector<float> b(1, {});
    EXPECT_FLOAT_EQ(vecflow::compute_cosine(a, b), 0.0f);
}

// ---------------------------------------------------------------
// Cosine — double
// ---------------------------------------------------------------

TEST(CosineTest, DoubleOrthogonal) {
    vecflow::Vector<double> a(0, {1.0, 0.0});
    vecflow::Vector<double> b(1, {0.0, 1.0});
    // dot=0 → distance=1
    EXPECT_DOUBLE_EQ(vecflow::compute_cosine(a, b), 1.0);
}

// ---------------------------------------------------------------
// Dispatch via compute_distance
// ---------------------------------------------------------------

TEST(DispatchTest, L2SqrViaDispatch) {
    vecflow::Vector<float> a(0, {1.0f, 2.0f, 3.0f});
    vecflow::Vector<float> b(1, {4.0f, 5.0f, 6.0f});
    EXPECT_FLOAT_EQ(
        vecflow::compute_distance(a, b, vecflow::DistanceType::L2Sqr), 27.0f);
}

TEST(DispatchTest, CosineViaDispatch) {
    vecflow::Vector<float> a(0, {1.0f, 0.0f});
    vecflow::Vector<float> b(1, {0.0f, 1.0f});
    EXPECT_FLOAT_EQ(
        vecflow::compute_distance(a, b, vecflow::DistanceType::Cosine), 1.0f);
}

// ---------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------

TEST(ErrorTest, L2SqrDimensionMismatch) {
    vecflow::Vector<float> a(0, {1.0f, 2.0f});
    vecflow::Vector<float> b(1, {1.0f, 2.0f, 3.0f});
    EXPECT_THROW(vecflow::compute_l2_sqr(a, b), std::invalid_argument);
}

TEST(ErrorTest, CosineDimensionMismatch) {
    vecflow::Vector<float> a(0, {1.0f, 2.0f});
    vecflow::Vector<float> b(1, {1.0f, 2.0f, 3.0f});
    EXPECT_THROW(vecflow::compute_cosine(a, b), std::invalid_argument);
}

// ---------------------------------------------------------------
// AVX2 vs Scalar correctness
//
// Use aligned buffers (not raw stack arrays) so _mm256_load_ps
// gets 32-byte aligned pointers.
// ---------------------------------------------------------------

using AlignedBuf = std::vector<float, vecflow::AlignedAllocator<float, 32>>;

TEST(AVX2CorrectnessTest, L2SqrExactDim8) {
    constexpr size_t dim = 8;
    AlignedBuf x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    AlignedBuf y = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    float scalar_r = vecflow::detail::l2_sqr_kernel(x.data(), y.data(), dim);
    float avx2_r = vecflow::detail::avx2::l2_sqr_kernel(x.data(), y.data(), dim);

    EXPECT_FLOAT_EQ(avx2_r, scalar_r);
}

TEST(AVX2CorrectnessTest, L2SqrOddDim) {
    constexpr size_t dim = 13;
    AlignedBuf x(dim);
    AlignedBuf y(dim);
    for (size_t i = 0; i < dim; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(dim - i);
    }

    float scalar_r = vecflow::detail::l2_sqr_kernel(x.data(), y.data(), dim);
    float avx2_r = vecflow::detail::avx2::l2_sqr_kernel(x.data(), y.data(), dim);

    EXPECT_FLOAT_EQ(avx2_r, scalar_r);
}

TEST(AVX2CorrectnessTest, L2SqrLargeRandom) {
    constexpr size_t dim = 128;
    AlignedBuf x(dim);
    AlignedBuf y(dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (size_t i = 0; i < dim; ++i) {
        x[i] = dist(rng);
        y[i] = dist(rng);
    }

    float scalar_r = vecflow::detail::l2_sqr_kernel(x.data(), y.data(), dim);
    float avx2_r = vecflow::detail::avx2::l2_sqr_kernel(x.data(), y.data(), dim);

    EXPECT_NEAR(avx2_r, scalar_r, 1e-5f);
}

TEST(AVX2CorrectnessTest, L2SqrZeroVectors) {
    constexpr size_t dim = 16;
    AlignedBuf x(dim, 0.0f);
    AlignedBuf y(dim, 0.0f);

    float scalar_r = vecflow::detail::l2_sqr_kernel(x.data(), y.data(), dim);
    float avx2_r = vecflow::detail::avx2::l2_sqr_kernel(x.data(), y.data(), dim);

    EXPECT_FLOAT_EQ(avx2_r, scalar_r);
}

TEST(AVX2CorrectnessTest, L2SqrZeroDim) {
    float scalar_r = vecflow::detail::l2_sqr_kernel<float>(nullptr, nullptr, 0);
    float avx2_r = vecflow::detail::avx2::l2_sqr_kernel(nullptr, nullptr, 0);

    EXPECT_FLOAT_EQ(avx2_r, scalar_r);
}

TEST(AVX2CorrectnessTest, CosineExactDim8) {
    constexpr size_t dim = 8;
    AlignedBuf x = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    AlignedBuf y = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};

    float scalar_r = vecflow::detail::cosine_kernel(x.data(), y.data(), dim);
    float avx2_r = vecflow::detail::avx2::cosine_kernel(x.data(), y.data(), dim);

    EXPECT_FLOAT_EQ(avx2_r, scalar_r);
}

TEST(AVX2CorrectnessTest, CosineOddDim) {
    constexpr size_t dim = 11;
    AlignedBuf x(dim);
    AlignedBuf y(dim);
    for (size_t i = 0; i < dim; ++i) {
        x[i] = static_cast<float>(i + 1);
        y[i] = static_cast<float>(dim - i);
    }

    float scalar_r = vecflow::detail::cosine_kernel(x.data(), y.data(), dim);
    float avx2_r = vecflow::detail::avx2::cosine_kernel(x.data(), y.data(), dim);

    EXPECT_NEAR(avx2_r, scalar_r, 1e-5f);
}

TEST(AVX2CorrectnessTest, CosineLargeRandom) {
    constexpr size_t dim = 128;
    AlignedBuf x(dim);
    AlignedBuf y(dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < dim; ++i) {
        x[i] = dist(rng);
        y[i] = dist(rng);
    }

    float scalar_r = vecflow::detail::cosine_kernel(x.data(), y.data(), dim);
    float avx2_r = vecflow::detail::avx2::cosine_kernel(x.data(), y.data(), dim);

    EXPECT_NEAR(avx2_r, scalar_r, 1e-5f);
}

TEST(AVX2CorrectnessTest, CosineZeroVector) {
    constexpr size_t dim = 32;
    AlignedBuf x(dim, 0.0f);
    AlignedBuf y(dim);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < dim; ++i) {
        y[i] = dist(rng);
    }

    float scalar_r = vecflow::detail::cosine_kernel(x.data(), y.data(), dim);
    float avx2_r = vecflow::detail::avx2::cosine_kernel(x.data(), y.data(), dim);

    EXPECT_FLOAT_EQ(avx2_r, scalar_r);
}
