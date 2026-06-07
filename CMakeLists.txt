# 文件路径：VecFlow/CMakeLists.txt

# ---------------------------------------------------------------
# [1] 最低版本要求
#     3.14：FetchContent 正式可用
#     3.20：CMake Presets 支持（后续会用到）
#     写 3.14 是对旧环境的最大兼容，Ubuntu 20.04 默认是 3.16，安全
# ---------------------------------------------------------------
cmake_minimum_required(VERSION 3.14)

# ---------------------------------------------------------------
# [2] 项目声明
#     VERSION：语义化版本，影响 CPack 打包和 install() 规则
#     LANGUAGES：显式声明只用 CXX，CMake 不会去找 C 编译器
# ---------------------------------------------------------------
project(VecFlow
    VERSION 0.1.0
    DESCRIPTION "A lightweight SIMD-accelerated Approximate Nearest Neighbor Search library"
    LANGUAGES CXX
)

# ---------------------------------------------------------------
# [3] 强制 C++17 标准（全局生效）
#     STANDARD_REQUIRED ON：找不到 C++17 编译器直接报错，而不是静默降级
#     EXTENSIONS OFF：禁用 GNU 扩展（-std=c++17 而非 -std=gnu++17）
#                     保证代码在 MSVC/Clang/GCC 上行为一致
# ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------
# [4] 默认构建类型
#     如果用户没有指定 -DCMAKE_BUILD_TYPE，默认用 Release
#     注意：多配置生成器（Visual Studio/Xcode）不受此影响
# ---------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "[VecFlow] No build type specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------
# [5] 编译选项
#     为不同构建类型分别设置 flags
#     INTERFACE 表示这些选项会传递给链接此 target 的目标
# ---------------------------------------------------------------
add_library(vecflow_compile_options INTERFACE)

target_compile_options(vecflow_compile_options INTERFACE
    # GCC/Clang 通用选项
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall               # 开启常见警告
        -Wextra             # 开启额外警告
        -Wpedantic          # 严格遵循标准
        -Wno-unused-parameter  # 关掉"未使用参数"警告（开发期常见）
    >
    # Release 模式：激进优化
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<CONFIG:Release>>:
        -O3                 # 最高优化级别
        -DNDEBUG            # 禁用 assert()
        -march=native       # 针对当前机器 CPU 指令集优化（开启 AVX2 等）
        -ffast-math         # 放宽浮点精度换取速度（Week3 SIMD 时需要）
    >
    # Debug 模式：调试信息
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<CONFIG:Debug>>:
        -O0                 # 不优化，方便 GDB 调试
        -g3                 # 最详细的调试符号（包含宏展开信息）
    >
)

# ---------------------------------------------------------------
# [6] 可选功能开关（通过 -DVECFLOW_XXX=ON 开启）
#     这是工程化的标准做法：功能可插拔，不影响核心构建
# ---------------------------------------------------------------
option(VECFLOW_BUILD_TESTS      "Build unit tests"      ON)
option(VECFLOW_BUILD_BENCHMARKS "Build benchmarks"      ON)
option(VECFLOW_ENABLE_ASAN      "Enable AddressSanitizer (Debug only)" OFF)

# ---------------------------------------------------------------
# [7] AddressSanitizer（内存检测工具）
#     只在 Debug 模式 + 显式开启时启用
#     用法：cmake -B build -DCMAKE_BUILD_TYPE=Debug -DVECFLOW_ENABLE_ASAN=ON
# ---------------------------------------------------------------
if(VECFLOW_ENABLE_ASAN)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "[VecFlow] ASan is most useful with Debug build type")
    endif()
    message(STATUS "[VecFlow] AddressSanitizer ENABLED")
    target_compile_options(vecflow_compile_options INTERFACE
        -fsanitize=address      # 内存越界、use-after-free 检测
        -fsanitize=undefined    # 未定义行为检测（整数溢出等）
        -fno-omit-frame-pointer # 保留栈帧指针，让报错堆栈可读
    )
    target_link_options(vecflow_compile_options INTERFACE
        -fsanitize=address
        -fsanitize=undefined
    )
endif()

# ---------------------------------------------------------------
# [8] 核心库：vecflow_core
#     INTERFACE 类型：纯头文件库，不产生 .a/.so 文件
#     任何链接它的 target 都会自动获得 include/ 路径和编译选项
# ---------------------------------------------------------------
add_library(vecflow_core INTERFACE)
target_include_directories(vecflow_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>  # 为将来的 install() 规则预留
)
target_link_libraries(vecflow_core INTERFACE vecflow_compile_options)

# ---------------------------------------------------------------
# [9] FetchContent：自动下载第三方依赖
#     EXCLUDE_FROM_ALL：不把 GTest 的 target 加入默认 build 目标
#     GIT_SHALLOW ON：只拉取最新一层 commit，速度更快
#
#     ⚠️ 注意：FetchContent 在 cmake configure 阶段就会触发下载
#     首次运行需要网络，下载完会缓存在 build/_deps/ 目录
# ---------------------------------------------------------------
include(FetchContent)

# --- Google Test ---
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0         # 固定版本，保证可复现构建
    GIT_SHALLOW    ON
    EXCLUDE_FROM_ALL
)

# --- Google Benchmark ---
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
    GIT_SHALLOW    ON
    EXCLUDE_FROM_ALL
)

# 阻止 GTest 把自己的 install() 规则安装到我们的系统
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
# 阻止 benchmark 安装自己
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
# 阻止 benchmark 下载并构建 GTest（它自己的测试），节省时间
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

# 触发实际下载和配置（按需）
if(VECFLOW_BUILD_TESTS)
    FetchContent_MakeAvailable(googletest)
endif()

if(VECFLOW_BUILD_BENCHMARKS)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

# ---------------------------------------------------------------
# [10] 子目录（顺序有依赖关系，不能随意调换）
# ---------------------------------------------------------------
add_subdirectory(src)

if(VECFLOW_BUILD_TESTS)
    enable_testing()            # 开启 CTest 支持（必须在 add_subdirectory 之前）
    add_subdirectory(tests)
endif()

if(VECFLOW_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ---------------------------------------------------------------
# [11] 打印构建摘要（每次 cmake configure 时显示）
# ---------------------------------------------------------------
message(STATUS "========================================")
message(STATUS " VecFlow v${PROJECT_VERSION} Configuration")
message(STATUS "========================================")
message(STATUS " Build type     : ${CMAKE_BUILD_TYPE}")
message(STATUS " C++ standard   : C++${CMAKE_CXX_STANDARD}")
message(STATUS " Compiler       : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " Build tests    : ${VECFLOW_BUILD_TESTS}")
message(STATUS " Build benchmarks: ${VECFLOW_BUILD_BENCHMARKS}")
message(STATUS " ASan enabled   : ${VECFLOW_ENABLE_ASAN}")
message(STATUS "========================================")