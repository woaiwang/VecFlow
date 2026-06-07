add_executable(word_similarity word_similarity.cpp)
target_link_libraries(word_similarity
    PRIVATE
        vecflow_core
        vecflow_simd
)
