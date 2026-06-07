# 文件路径：VecFlow/tests/CMakeLists.txt

# ---------------------------------------------------------------
# 辅助函数：为每个测试文件创建一个独立的可执行目标
# 用法：vecflow_add_test(test_distance)
#       → 编译 test_distance.cpp，链接 GTest，注册到 CTest
# ---------------------------------------------------------------
function(vecflow_add_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME}
        PRIVATE
            vecflow_core        # 我们的头文件库（获得 include/ 路径）
            vecflow_simd        # SIMD 实现（AVX2 kernels）
            GTest::gtest_main   # GTest 主函数（不需要自己写 main）
    )
    # 将测试注册到 CTest，这样 ctest 命令才能发现它
    add_test(
        NAME    ${TEST_NAME}
        COMMAND ${TEST_NAME}
    )
    # 设置测试超时（秒），防止死循环挂起 CI
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 30)
endfunction()

# ---------------------------------------------------------------
# 注册所有测试（每新增一个 .cpp 就加一行）
# ---------------------------------------------------------------
vecflow_add_test(test_types)
vecflow_add_test(test_distance)
vecflow_add_test(test_flat_index)
vecflow_add_test(test_hnsw)