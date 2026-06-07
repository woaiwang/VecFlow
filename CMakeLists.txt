cmake_minimum_required(VERSION 3.14)

project(VecFlow
    VERSION 0.1.0
    DESCRIPTION "A lightweight SIMD-accelerated Approximate Nearest Neighbor Search library"
    LANGUAGES CXX
)

# ---------------------------------------------------------------
# C++17 (required, no GNU extensions)
# ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------
# Default build type
# ---------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "[VecFlow] No build type specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------
# Compile options
# ---------------------------------------------------------------
add_library(vecflow_compile_options INTERFACE)

target_compile_options(vecflow_compile_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    >
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<CONFIG:Release>>:
        -O3
        -DNDEBUG
        -march=native
        -ffast-math
    >
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<CONFIG:Debug>>:
        -O0
        -g3
    >
)

# ---------------------------------------------------------------
# Options
# ---------------------------------------------------------------
option(VECFLOW_BUILD_TESTS "Build unit tests" ON)
option(VECFLOW_BUILD_BENCHMARKS "Build benchmarks" ON)
option(VECFLOW_ENABLE_ASAN  "Enable AddressSanitizer (Debug only)" OFF)

# ---------------------------------------------------------------
# AddressSanitizer
# ---------------------------------------------------------------
if(VECFLOW_ENABLE_ASAN)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "[VecFlow] ASan is most useful with Debug build type")
    endif()
    message(STATUS "[VecFlow] AddressSanitizer ENABLED")
    target_compile_options(vecflow_compile_options INTERFACE
        -fsanitize=address
        -fsanitize=undefined
        -fno-omit-frame-pointer
    )
    target_link_options(vecflow_compile_options INTERFACE
        -fsanitize=address
        -fsanitize=undefined
    )
endif()

# ---------------------------------------------------------------
# Core library (header-only)
# ---------------------------------------------------------------
add_library(vecflow_core INTERFACE)
target_include_directories(vecflow_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vecflow_core INTERFACE vecflow_compile_options)

# ---------------------------------------------------------------
# FetchContent — Google Test only
# ---------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    ON
    EXCLUDE_FROM_ALL
)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

if(VECFLOW_BUILD_TESTS)
    FetchContent_MakeAvailable(googletest)
endif()


# ---------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------
add_subdirectory(src)

if(VECFLOW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(VECFLOW_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

add_subdirectory(examples)

# ---------------------------------------------------------------
# Summary
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
