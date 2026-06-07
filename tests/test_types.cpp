#include <gtest/gtest.h>

#include "vecflow/types.hpp"

TEST(VectorTest, ConstructAndAccess) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    vecflow::Vector<float> vec(42, data);

    EXPECT_EQ(vec.id, 42);
    EXPECT_EQ(vec.data.size(), 4);
    EXPECT_FLOAT_EQ(vec.data[0], 1.0f);
    EXPECT_FLOAT_EQ(vec.data[3], 4.0f);
}

TEST(VectorTest, MoveSemantics) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    vecflow::Vector<float> vec(1, std::move(data));

    EXPECT_EQ(vec.id, 1);
    EXPECT_EQ(vec.data.size(), 3);
    EXPECT_FLOAT_EQ(vec.data[1], 2.0f);
}

TEST(VectorTest, EmptyData) {
    vecflow::Vector<double> vec(0, {});

    EXPECT_EQ(vec.id, 0);
    EXPECT_TRUE(vec.data.empty());
}

TEST(VectorTest, IntType) {
    std::vector<int> data = {10, 20, 30};
    vecflow::Vector<int> vec(7, data);

    EXPECT_EQ(vec.id, 7);
    EXPECT_EQ(vec.data[1], 20);
}

TEST(SearchResultTest, ConstructAndAccess) {
    vecflow::SearchResult result{5, 0.75f};

    EXPECT_EQ(result.id, 5);
    EXPECT_FLOAT_EQ(result.distance, 0.75f);
}

TEST(SearchResultTest, ZeroDistance) {
    vecflow::SearchResult result{0, 0.0f};

    EXPECT_EQ(result.id, 0);
    EXPECT_FLOAT_EQ(result.distance, 0.0f);
}

TEST(SearchResultTest, LargeId) {
    vecflow::SearchResult result{99999, 3.14f};

    EXPECT_EQ(result.id, 99999);
    EXPECT_FLOAT_EQ(result.distance, 3.14f);
}
