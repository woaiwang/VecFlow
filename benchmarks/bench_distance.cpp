// Manual microbenchmark for distance kernels.
// Google Benchmark has MinGW compatibility issues (segfault in
// RunSpecifiedBenchmarks), so we use std::chrono directly.
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "vecflow/allocator.hpp"
#include "vecflow/distance.hpp"
#include "vecflow/types.hpp"

using AlignedBuf = std::vector<float, vecflow::AlignedAllocator<float, 32>>;

// Scalar kernels defined in bench_scalar.cpp — compiled without
// -march=native, so they represent true portable performance.
extern "C" {
float bench_scalar_l2(const float* x, const float* y, size_t dim);
float bench_scalar_cosine(const float* x, const float* y, size_t dim);
}

namespace {

struct BenchmarkResult {
    std::string name;
    double ns_per_op;       // nanoseconds per call
    size_t iterations;
};

// Generate random aligned float vectors for benchmarking
std::pair<AlignedBuf, AlignedBuf> generate_random_vectors(size_t dim) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    AlignedBuf x(dim);
    AlignedBuf y(dim);
    for (size_t i = 0; i < dim; ++i) {
        x[i] = dist(rng);
        y[i] = dist(rng);
    }
    return {std::move(x), std::move(y)};
}

// Run a kernel repeatedly and measure average time
BenchmarkResult run_benchmark(const std::string& name,
                              const AlignedBuf& x,
                              const AlignedBuf& y,
                              size_t dim,
                              float (*kernel)(const float*, const float*,
                                              size_t)) {
    // Warmup
    for (int i = 0; i < 10; ++i) {
        volatile float r = kernel(x.data(), y.data(), dim);
        (void)r;
    }

    // Determine iteration count for ~1 second of measurement
    constexpr double target_seconds = 1.0;
    auto start = std::chrono::high_resolution_clock::now();
    size_t iters = 0;
    auto end = start;
    while (std::chrono::duration<double>(end - start).count() < target_seconds) {
        for (int batch = 0; batch < 1000; ++batch) {
            volatile float r = kernel(x.data(), y.data(), dim);
            (void)r;
        }
        iters += 1000;
        end = std::chrono::high_resolution_clock::now();
    }

    double elapsed =
        std::chrono::duration<double, std::nano>(end - start).count();
    return {name, elapsed / static_cast<double>(iters), iters};
}

void print_separator() {
    std::cout << std::string(58, '-') << "\n";
}

void print_header() {
    print_separator();
    std::cout << std::left << std::setw(28) << "Benchmark"
              << std::right << std::setw(12) << "Time (ns)"
              << std::setw(16) << "Iterations"
              << "\n";
    print_separator();
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(28) << r.name
              << std::right << std::setw(12) << std::fixed
              << std::setprecision(1) << r.ns_per_op
              << std::setw(16) << r.iterations << "\n";
}

} // namespace

int main() {
    constexpr size_t dim = 128;
    auto [x, y] = generate_random_vectors(dim);

    std::cout << "\nVecFlow Distance Kernel Micro-Benchmark\n";
    std::cout << "Dimension: " << dim << "\n\n";

    print_header();

    // L2Sqr benchmarks
    auto l2_scalar =
        run_benchmark("BM_L2Sqr_Scalar", x, y, dim, bench_scalar_l2);
    auto l2_avx2 =
        run_benchmark("BM_L2Sqr_AVX2", x, y, dim,
                      vecflow::detail::avx2::l2_sqr_kernel);

    print_result(l2_scalar);
    print_result(l2_avx2);

    // Cosine benchmarks
    auto cos_scalar =
        run_benchmark("BM_Cosine_Scalar", x, y, dim, bench_scalar_cosine);
    auto cos_avx2 =
        run_benchmark("BM_Cosine_AVX2", x, y, dim,
                      vecflow::detail::avx2::cosine_kernel);

    print_result(cos_scalar);
    print_result(cos_avx2);

    print_separator();

    // Speedup summary
    double l2_speedup = l2_scalar.ns_per_op / l2_avx2.ns_per_op;
    double cos_speedup = cos_scalar.ns_per_op / cos_avx2.ns_per_op;

    std::cout << "\nSpeedup Summary:\n";
    std::cout << "  L2Sqr  AVX2 vs Scalar: " << std::fixed
              << std::setprecision(2) << l2_speedup << "x\n";
    std::cout << "  Cosine AVX2 vs Scalar: " << std::fixed
              << std::setprecision(2) << cos_speedup << "x\n\n";

    return 0;
}
