#pragma once

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <vector>

#include "vecflow/distance.hpp"
#include "vecflow/index.hpp"

namespace vecflow {

/// Brute-force exact KNN index (O(N) per query).
///
/// FlatIndex stores all vectors and scans them linearly on every search.
/// It serves as the correctness baseline for approximate indexes (HNSW).
class FlatIndex : public IIndex {
public:
    /// @param dim   vector dimension (must be > 0)
    /// @param type  distance metric (L2Sqr or Cosine)
    /// @throws std::invalid_argument if dim == 0
    explicit FlatIndex(size_t dim,
                       DistanceType type = DistanceType::L2Sqr)
        : dim_(dim), distance_type_(type) {
        if (dim == 0) {
            throw std::invalid_argument("FlatIndex: dimension must be > 0");
        }
    }

    /// @throws std::invalid_argument if vec dimension mismatches
    void add(const Vector<float>& vec) override {
        if (vec.data.size() != dim_) {
            throw std::invalid_argument("FlatIndex::add: dimension mismatch");
        }
        vectors_.push_back(vec);
    }

    /// @throws std::invalid_argument if query dimension mismatches
    [[nodiscard]] std::vector<SearchResult>
    search(const std::vector<float>& query, size_t k) const override {
        if (query.size() != dim_) {
            throw std::invalid_argument(
                "FlatIndex::search: query dimension mismatch");
        }
        if (k == 0 || vectors_.empty()) {
            return {};
        }

        Vector<float> query_vec(0, query);

        return (distance_type_ == DistanceType::L2Sqr)
                   ? search_impl(query_vec, k, [](const Vector<float>& a,
                                                   const Vector<float>& b) {
                         return compute_l2_sqr(a, b);
                     })
                   : search_impl(query_vec, k, [](const Vector<float>& a,
                                                   const Vector<float>& b) {
                         return compute_cosine(a, b);
                     });
    }

    [[nodiscard]] size_t size() const override { return vectors_.size(); }

    [[nodiscard]] size_t dim() const override { return dim_; }

private:
    struct HeapEntry {
        size_t id;
        float distance;

        bool operator<(const HeapEntry& other) const {
            return distance < other.distance;
        }
    };

    template <typename DistanceFunc>
    [[nodiscard]] std::vector<SearchResult>
    search_impl(const Vector<float>& query_vec, size_t k,
                DistanceFunc distance_func) const {
        std::priority_queue<HeapEntry> heap;

        for (const auto& vec : vectors_) {
            float d = distance_func(query_vec, vec);

            if (heap.size() < k) {
                heap.push({vec.id, d});
            } else if (d < heap.top().distance) {
                heap.pop();
                heap.push({vec.id, d});
            }
        }

        std::vector<SearchResult> results;
        results.reserve(heap.size());
        while (!heap.empty()) {
            const auto& top = heap.top();
            results.push_back({top.id, top.distance});
            heap.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

    size_t dim_;
    std::vector<Vector<float>> vectors_;
    DistanceType distance_type_;
};

} // namespace vecflow
