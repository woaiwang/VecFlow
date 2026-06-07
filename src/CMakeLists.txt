# 文件路径：VecFlow/src/CMakeLists.txt

# 目前 vecflow_core 是纯头文件库，src/ 暂无编译单元
# Week3 引入 SIMD 实现文件时，在这里添加：
#
# add_library(vecflow_simd STATIC distance_avx2.cpp)
# target_link_libraries(vecflow_simd PUBLIC vecflow_core)