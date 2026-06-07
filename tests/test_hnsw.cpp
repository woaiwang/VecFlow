#include <gtest/gtest.h>

#include <random>
#include <unordered_set>
#include <vector>

#include "vecflow/flat_index.hpp"
#include "vecflow/hnsw.hpp"

// ---------------------------------------------------------------
// Construction
// ---------------------------------------------------------------

TEST(HNSWTest, ConstructWithValidParams) {
    vecflow::HNSWIndex idx(128);
    EXPECT_EQ(idx.size(), 0);
    EXPECT_EQ(idx.dim(), 128);
}

TEST(HNSWTest, ConstructWithZeroDim) {
    EXPECT_THROW(vecflow::HNSWIndex(0), std::invalid_argument);
}

TEST(HNSWTest, ConstructWithMSmallerThan2) {
    EXPECT_THROW(vecflow::HNSWIndex(3, /*M=*/1), std::invalid_argument);
}

// ---------------------------------------------------------------
// add() edge cases
// ---------------------------------------------------------------

TEST(HNSWTest, AddDimensionMismatch) {
    vecflow::HNSWIndex idx(3);
    EXPECT_THROW(idx.add({0, {1.0f, 2.0f}}), std::invalid_argument);
}

TEST(HNSWTest, AddSingleVector) {
    vecflow::HNSWIndex idx(3);
    idx.add({0, {1.0f, 2.0f, 3.0f}});
    EXPECT_EQ(idx.size(), 1);
}

TEST(HNSWTest, AddMultipleVectors) {
    vecflow::HNSWIndex idx(3);
    idx.add({0, {0.0f, 0.0f, 0.0f}});
    idx.add({1, {1.0f, 1.0f, 1.0f}});
    idx.add({2, {2.0f, 2.0f, 2.0f}});
    EXPECT_EQ(idx.size(), 3);
}

// ---------------------------------------------------------------
// search() edge cases
// ---------------------------------------------------------------

TEST(HNSWTest, SearchOnEmptyIndex) {
    vecflow::HNSWIndex idx(3);
    auto results = idx.search({1.0f, 2.0f, 3.0f}, 3);
    EXPECT_TRUE(results.empty());
}

TEST(HNSWTest, SearchKZero) {
    vecflow::HNSWIndex idx(3);
    idx.add({0, {1.0f, 2.0f, 3.0f}});
    auto results = idx.search({1.0f, 2.0f, 3.0f}, 0);
    EXPECT_TRUE(results.empty());
}

TEST(HNSWTest, SearchDimensionMismatch) {
    vecflow::HNSWIndex idx(3);
    idx.add({0, {1.0f, 2.0f, 3.0f}});
    EXPECT_THROW(static_cast<void>(idx.search({1.0f, 2.0f}, 1)),
                 std::invalid_argument);
}

TEST(HNSWTest, SearchKGreaterThanN) {
    vecflow::HNSWIndex idx(2);
    idx.add({10, {1.0f, 2.0f}});
    idx.add({20, {3.0f, 4.0f}});
    auto results = idx.search({0.0f, 0.0f}, 5);
    ASSERT_EQ(results.size(), 2);
}

// ---------------------------------------------------------------
// Basic correctness: small dataset, exact search
// ---------------------------------------------------------------

TEST(HNSWTest, SearchSingleVectorIdenticalQuery) {
    vecflow::HNSWIndex idx(3);
    idx.add({42, {3.14f, 2.71f, 1.41f}});
    auto results = idx.search({3.14f, 2.71f, 1.41f}, 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, 42);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
}

TEST(HNSWTest, SearchTop2Of3) {
    vecflow::HNSWIndex idx(2);
    idx.add({0, {0.0f, 0.0f}});
    idx.add({1, {0.0f, 1.0f}});
    idx.add({2, {3.0f, 4.0f}});

    auto results = idx.search({0.0f, 0.0f}, 2);
    ASSERT_EQ(results.size(), 2);
    // The closest two should be id=0 (dist 0) and id=1 (dist 1)
    // Id=2 (dist 25) should not appear
    std::unordered_set<size_t> ids;
    for (const auto& r : results) {
        ids.insert(r.id);
    }
    EXPECT_TRUE(ids.count(0) > 0);
    EXPECT_TRUE(ids.count(1) > 0);
    EXPECT_TRUE(ids.count(2) == 0);
}

// ---------------------------------------------------------------
// Recall@K — main regression test
// ---------------------------------------------------------------

TEST(HNSWTest, RecallAtK) {
    constexpr size_t dim = 128;
    constexpr size_t num_vectors = 1000;
    constexpr size_t num_queries = 10;
    constexpr size_t k = 10;
    constexpr size_t M = 16;
    constexpr size_t ef_construction = 200;
    constexpr size_t ef_search = 128;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    vecflow::FlatIndex flat(dim);
    vecflow::HNSWIndex hnsw(dim, M, ef_construction, ef_search);

    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> data(dim);
        for (size_t j = 0; j < dim; ++j) {
            data[j] = dist(rng);
        }
        vecflow::Vector<float> vec(i, std::move(data));
        flat.add(vec);
        hnsw.add(vec);
    }

    float total_recall = 0.0f;
    for (size_t q = 0; q < num_queries; ++q) {
        std::vector<float> query(dim);
        for (size_t j = 0; j < dim; ++j) {
            query[j] = dist(rng);
        }

        auto flat_results = flat.search(query, k);
        auto hnsw_results = hnsw.search(query, k);

        std::unordered_set<size_t> ground_truth;
        for (const auto& r : flat_results) {
            ground_truth.insert(r.id);
        }

        size_t hits = 0;
        for (const auto& r : hnsw_results) {
            if (ground_truth.count(r.id) > 0) {
                ++hits;
            }
        }

        float recall = static_cast<float>(hits) / static_cast<float>(k);
        total_recall += recall;
    }

    float avg_recall = total_recall / static_cast<float>(num_queries);
    EXPECT_GE(avg_recall, 0.85f)
        << "Average Recall@10: " << avg_recall
        << ", expected >= 0.85";
}
