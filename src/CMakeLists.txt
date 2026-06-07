add_library(vecflow_simd STATIC distance_avx2.cpp)
target_link_libraries(vecflow_simd PUBLIC vecflow_core)
target_compile_options(vecflow_simd PRIVATE -mavx2 -mfma)
