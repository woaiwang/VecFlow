# Scalar kernels compiled with -O0 and no SIMD flags.
# Deliberately NOT linked to vecflow_core (which pulls in -march=native
# and -ffast-math). This gives a true scalar baseline for fair comparison.
add_library(bench_scalar STATIC bench_scalar.cpp)
target_include_directories(bench_scalar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Baseline scalar: -O2 without -march=native.  The compiler may use SSE2
# (mandatory for x86-64 float ABI) but won't emit AVX/AVX2.  This is the
# realistic "user who didn't compile with -march=native" performance.
# -fno-tree-vectorize ensures the inner loop stays scalar.
target_compile_options(bench_scalar PRIVATE
    -O2
    -fno-tree-vectorize
    -fno-fast-math
    -fno-unroll-loops
)

function(vecflow_add_benchmark BENCH_NAME)
    add_executable(${BENCH_NAME} ${BENCH_NAME}.cpp)
    target_link_libraries(${BENCH_NAME}
        PRIVATE
            vecflow_core
            vecflow_simd
            bench_scalar
    )
endfunction()

vecflow_add_benchmark(bench_distance)
