#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "vecflow/flat_index.hpp"

// ---------------------------------------------------------------
// Construction
// ---------------------------------------------------------------

TEST(FlatIndexTest, ConstructWithValidDim) {
    vecflow::FlatIndex index(3);
    EXPECT_EQ(index.size(), 0);
    EXPECT_EQ(index.dim(), 3);
}

TEST(FlatIndexTest, ConstructWithZeroDim) {
    EXPECT_THROW(vecflow::FlatIndex(0), std::invalid_argument);
}

TEST(FlatIndexTest, ConstructWithCosineType) {
    vecflow::FlatIndex index(3, vecflow::DistanceType::Cosine);
    EXPECT_EQ(index.dim(), 3);
}

// ---------------------------------------------------------------
// add()
// ---------------------------------------------------------------

TEST(FlatIndexTest, AddSingleVector) {
    vecflow::FlatIndex index(3);
    index.add({0, {1.0f, 2.0f, 3.0f}});
    EXPECT_EQ(index.size(), 1);
}

TEST(FlatIndexTest, AddMultipleVectors) {
    vecflow::FlatIndex index(2);
    index.add({0, {0.0f, 0.0f}});
    index.add({1, {1.0f, 1.0f}});
    index.add({2, {2.0f, 2.0f}});
    EXPECT_EQ(index.size(), 3);
}

TEST(FlatIndexTest, AddDimensionMismatch) {
    vecflow::FlatIndex index(3);
    EXPECT_THROW(index.add({0, {1.0f, 2.0f}}), std::invalid_argument);
}

// ---------------------------------------------------------------
// search() — L2Sqr, exact ranking
// ---------------------------------------------------------------

TEST(FlatIndexTest, SearchL2SqrExactRanking) {
    vecflow::FlatIndex index(2);
    index.add({0, {0.0f, 0.0f}});  // dist to [0,0] = 0
    index.add({1, {0.0f, 1.0f}});  // dist to [0,0] = 1
    index.add({2, {3.0f, 4.0f}});  // dist to [0,0] = 25

    std::vector<float> query = {0.0f, 0.0f};
    auto results = index.search(query, 3);

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].id, 0);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
    EXPECT_EQ(results[1].id, 1);
    EXPECT_FLOAT_EQ(results[1].distance, 1.0f);
    EXPECT_EQ(results[2].id, 2);
    EXPECT_FLOAT_EQ(results[2].distance, 25.0f);
}

TEST(FlatIndexTest, SearchTop2Of3) {
    vecflow::FlatIndex index(2);
    index.add({0, {0.0f, 0.0f}});  // dist = 0
    index.add({1, {0.0f, 1.0f}});  // dist = 1
    index.add({2, {3.0f, 4.0f}});  // dist = 25

    std::vector<float> query = {0.0f, 0.0f};
    auto results = index.search(query, 2);

    // Only top-2 returned, sorted by distance ascending
    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].id, 0);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
    EXPECT_EQ(results[1].id, 1);
    EXPECT_FLOAT_EQ(results[1].distance, 1.0f);
}

// ---------------------------------------------------------------
// search() — Cosine distance
// ---------------------------------------------------------------

TEST(FlatIndexTest, SearchCosineRanking) {
    vecflow::FlatIndex index(2, vecflow::DistanceType::Cosine);
    index.add({0, {1.0f, 0.0f}});   // cos to [1,0] = 0
    index.add({30, {0.0f, 1.0f}});   // cos to [1,0] = 1
    index.add({99, {1.0f, 1.0f}});   // cos to [1,0] = 1 - 1/sqrt(2) ≈ 0.2929

    std::vector<float> query = {1.0f, 0.0f};
    auto results = index.search(query, 3);

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].id, 0);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
    EXPECT_EQ(results[2].id, 30);
    EXPECT_FLOAT_EQ(results[2].distance, 1.0f);
}

// ---------------------------------------------------------------
// Edge case: k > number of stored vectors
// ---------------------------------------------------------------

TEST(FlatIndexTest, SearchKGreaterThanN) {
    vecflow::FlatIndex index(2);
    index.add({10, {1.0f, 2.0f}});
    index.add({20, {3.0f, 4.0f}});

    std::vector<float> query = {0.0f, 0.0f};
    // Request 5 results from an index with only 2 vectors
    auto results = index.search(query, 5);

    ASSERT_EQ(results.size(), 2);
    // Both vectors returned, sorted by distance
    EXPECT_EQ(results[0].id, 10);
    EXPECT_FLOAT_EQ(results[0].distance, 5.0f);   // 1^2 + 2^2
    EXPECT_EQ(results[1].id, 20);
    EXPECT_FLOAT_EQ(results[1].distance, 25.0f);   // 3^2 + 4^2
}

// ---------------------------------------------------------------
// Edge case: k == 0
// ---------------------------------------------------------------

TEST(FlatIndexTest, SearchKZero) {
    vecflow::FlatIndex index(2);
    index.add({0, {1.0f, 2.0f}});

    std::vector<float> query = {0.0f, 0.0f};
    auto results = index.search(query, 0);

    EXPECT_TRUE(results.empty());
}

// ---------------------------------------------------------------
// Edge case: search on empty index
// ---------------------------------------------------------------

TEST(FlatIndexTest, SearchOnEmptyIndex) {
    vecflow::FlatIndex index(2);

    std::vector<float> query = {1.0f, 2.0f};
    auto results = index.search(query, 3);

    EXPECT_TRUE(results.empty());
}

// ---------------------------------------------------------------
// Edge case: single vector, query identical
// ---------------------------------------------------------------

TEST(FlatIndexTest, SearchSingleVectorIdenticalQuery) {
    vecflow::FlatIndex index(3);
    index.add({42, {3.14f, 2.71f, 1.41f}});

    std::vector<float> query = {3.14f, 2.71f, 1.41f};
    auto results = index.search(query, 1);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, 42);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
}

// ---------------------------------------------------------------
// Edge case: query dimension mismatch
// ---------------------------------------------------------------

TEST(FlatIndexTest, SearchDimensionMismatch) {
    vecflow::FlatIndex index(3);
    index.add({0, {1.0f, 2.0f, 3.0f}});

    std::vector<float> bad_query = {1.0f, 2.0f};  // only 2 dims
    EXPECT_THROW(index.search(bad_query, 1), std::invalid_argument);
}

// ---------------------------------------------------------------
// API-First scenario: simulating real user usage (like Quick Start)
// ---------------------------------------------------------------

TEST(FlatIndexTest, ApiFirstUsageScenario) {
    // User story: build a 3D vector index, insert some vectors,
    // then find the 2 nearest neighbors of a query.

    vecflow::FlatIndex index(/*dim=*/3);

    // Step 1: Insert vectors with meaningful ids
    index.add({100, std::vector<float>{1.0f, 2.0f, 3.0f}});
    index.add({200, std::vector<float>{4.0f, 5.0f, 6.0f}});
    index.add({300, std::vector<float>{2.0f, 3.0f, 4.0f}});

    EXPECT_EQ(index.size(), 3);
    EXPECT_EQ(index.dim(), 3);

    // Step 2: Query — find the 2 nearest neighbors to [2, 3, 4]
    // Expected L2 distances:
    //   id=100 [1,2,3]: (2-1)^2+(3-2)^2+(4-3)^2 = 1+1+1 = 3
    //   id=200 [4,5,6]: (2-4)^2+(3-5)^2+(4-6)^2 = 4+4+4 = 12
    //   id=300 [2,3,4]: (2-2)^2+(3-3)^2+(4-4)^2 = 0+0+0 = 0
    // Ranking: 300 (dist=0), 100 (dist=3)
    std::vector<float> query = {2.0f, 3.0f, 4.0f};
    auto results = index.search(query, /*k=*/2);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].id, 300);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
    EXPECT_EQ(results[1].id, 100);
    EXPECT_FLOAT_EQ(results[1].distance, 3.0f);
}
