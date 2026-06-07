#pragma once

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>

#include "vecflow/distance.hpp"
#include "vecflow/index.hpp"

namespace vecflow {

// Forward declare test accessor (used only in test builds)
struct HNSWTestAccessor;

/// Hierarchical Navigable Small World index for approximate nearest-neighbor search.
///
/// Provides sub-linear (O(log N)) query time with tunable recall via
/// the construction-time parameters M and ef_construction.
///
/// Implements the HNSW algorithm from Malkov & Yashunin (2018):
/// - Multi-layer navigable small-world graph
/// - Standard heuristic neighbor selection for edge diversity
/// - Exponential-decay random level assignment
class HNSWIndex : public IIndex {
    friend struct HNSWTestAccessor;

public:
    /// @param dim              vector dimension (must be > 0)
    /// @param M                max edges per node at layers >= 1 (default 16, must be >= 2)
    /// @param ef_construction  candidate queue size during build (default 200, higher = better recall)
    /// @param ef               candidate queue size during search (default 16, higher = better recall)
    /// @param type             distance metric (L2Sqr or Cosine)
    /// @throws std::invalid_argument if dim == 0 or M < 2
    HNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 200,
              size_t ef = 16, DistanceType type = DistanceType::L2Sqr)
        : dim_(dim),
          M_(M),
          M_max0_(2 * M),
          ef_construction_(ef_construction),
          ef_(ef),
          distance_type_(type),
          entry_point_(0),
          max_level_global_(-1),
          mL_(1.0f / std::log(static_cast<float>(M))),
          rng_(std::random_device{}()) {
        if (dim == 0) {
            throw std::invalid_argument("HNSWIndex: dimension must be > 0");
        }
        if (M < 2) {
            throw std::invalid_argument("HNSWIndex: M must be >= 2");
        }
    }

    void add(const Vector<float>& vec) override {
        if (vec.data.size() != dim_) {
            throw std::invalid_argument("HNSWIndex::add: dimension mismatch");
        }

        size_t internal_idx = vectors_.size();
        vectors_.push_back(vec);

        int l = random_level();
        max_level_.push_back(l);
        adjacency_.push_back(
            std::vector<std::vector<size_t>>(static_cast<size_t>(l) + 1));

        if (internal_idx == 0) {
            entry_point_ = 0;
            max_level_global_ = l;
            return;
        }

        if (distance_type_ == DistanceType::L2Sqr) {
            add_impl(internal_idx, l, [](const Vector<float>& a,
                                         const Vector<float>& b) {
                return compute_l2_sqr(a, b);
            });
        } else {
            add_impl(internal_idx, l, [](const Vector<float>& a,
                                         const Vector<float>& b) {
                return compute_cosine(a, b);
            });
        }
    }

    [[nodiscard]] std::vector<SearchResult>
    search(const std::vector<float>& query, size_t k) const override {
        if (query.size() != dim_) {
            throw std::invalid_argument(
                "HNSWIndex::search: query dimension mismatch");
        }
        if (k == 0 || vectors_.empty()) {
            return {};
        }

        Vector<float> query_vec(0, query);

        if (distance_type_ == DistanceType::L2Sqr) {
            return search_impl(query_vec, k, [](const Vector<float>& a,
                                                const Vector<float>& b) {
                return compute_l2_sqr(a, b);
            });
        } else {
            return search_impl(query_vec, k, [](const Vector<float>& a,
                                                const Vector<float>& b) {
                return compute_cosine(a, b);
            });
        }
    }

    [[nodiscard]] size_t size() const override { return vectors_.size(); }

    [[nodiscard]] size_t dim() const override { return dim_; }

private:
    // --- Random level assignment ---

    int random_level() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        if (r == 0.0f) {
            r = 1e-6f;
        }
        return static_cast<int>(std::floor(-std::log(r) * mL_));
    }

    // --- Single-layer greedy search ---

    template <typename DistanceFunc>
    [[nodiscard]] std::vector<std::pair<float, size_t>>
    search_layer(const Vector<float>& query_vec, size_t ep, size_t ef,
                 int level, DistanceFunc distance_func) const {
        if (ef == 0 || vectors_.empty()) {
            return {};
        }

        ensure_visited_capacity();
        visited_tag_++;

        using Candidate = std::pair<float, size_t>;

        std::priority_queue<Candidate, std::vector<Candidate>,
                            std::greater<Candidate>>
            candidates;

        std::priority_queue<Candidate> results;

        visited_[ep] = visited_tag_;
        float d_entry =
            static_cast<float>(distance_func(query_vec, vectors_[ep]));
        candidates.emplace(d_entry, ep);
        results.emplace(d_entry, ep);

        while (!candidates.empty()) {
            auto [c_dist, c_idx] = candidates.top();
            candidates.pop();

            if (c_dist > results.top().first) {
                break;
            }

            const auto& neighbors = adjacency_[c_idx][static_cast<size_t>(level)];
            for (size_t neighbor : neighbors) {
                if (visited_[neighbor] == visited_tag_) {
                    continue;
                }
                visited_[neighbor] = visited_tag_;

                float d = static_cast<float>(
                    distance_func(query_vec, vectors_[neighbor]));

                float worst_in_results = results.top().first;
                if (d < worst_in_results || results.size() < ef) {
                    candidates.emplace(d, neighbor);
                    results.emplace(d, neighbor);
                    if (results.size() > ef) {
                        results.pop();
                    }
                }
            }
        }

        std::vector<std::pair<float, size_t>> output;
        output.reserve(results.size());
        while (!results.empty()) {
            output.push_back(results.top());
            results.pop();
        }
        std::reverse(output.begin(), output.end());
        return output;
    }

    // --- HNSW standard heuristic for diverse neighbor selection ---

    template <typename DistanceFunc>
    [[nodiscard]] std::vector<size_t>
    select_neighbors(DistanceFunc distance_func, const Vector<float>& query,
                     const std::vector<std::pair<float, size_t>>& candidates,
                     size_t max_conn) const {
        std::vector<size_t> result;
        std::vector<size_t> discard;
        result.reserve(max_conn);

        for (const auto& [dist_c, idx_c] : candidates) {
            if (result.size() >= max_conn) {
                break;
            }

            bool should_keep = true;
            for (size_t idx_r : result) {
                float d_c_to_r =
                    distance_func(vectors_[idx_c], vectors_[idx_r]);
                if (d_c_to_r < dist_c) {
                    should_keep = false;
                    break;
                }
            }

            if (should_keep) {
                result.push_back(idx_c);
            } else {
                discard.push_back(idx_c);
            }
        }

        if (result.size() < max_conn) {
            for (size_t idx_d : discard) {
                if (result.size() >= max_conn) {
                    break;
                }
                result.push_back(idx_d);
            }
        }

        return result;
    }

    // --- Full multi-layer insertion ---

    template <typename DistanceFunc>
    void add_impl(size_t internal_idx, int l, DistanceFunc distance_func) {
        const auto& query_vec = vectors_[internal_idx];

        size_t current_ep = entry_point_;
        for (int level = max_level_global_; level > l; --level) {
            auto results =
                search_layer(query_vec, current_ep, 1, level, distance_func);
            if (!results.empty()) {
                current_ep = results[0].second;
            }
        }

        for (int level = std::min(l, max_level_global_); level >= 0; --level) {
            auto candidates =
                search_layer(query_vec, current_ep, ef_construction_, level,
                             distance_func);

            size_t max_conn = (level == 0) ? M_max0_ : M_;
            auto neighbors =
                select_neighbors(distance_func, query_vec, candidates, max_conn);

            for (size_t n : neighbors) {
                adjacency_[internal_idx][static_cast<size_t>(level)].push_back(
                    n);
                adjacency_[n][static_cast<size_t>(level)].push_back(
                    internal_idx);
            }

            for (size_t n : neighbors) {
                auto& n_adj = adjacency_[n][static_cast<size_t>(level)];
                if (n_adj.size() > max_conn) {
                    std::vector<std::pair<float, size_t>> n_candidates;
                    n_candidates.reserve(n_adj.size());
                    for (size_t neighbor_of_n : n_adj) {
                        float d = distance_func(vectors_[n],
                                                vectors_[neighbor_of_n]);
                        n_candidates.emplace_back(d, neighbor_of_n);
                    }
                    std::sort(n_candidates.begin(), n_candidates.end());

                    auto pruned =
                        select_neighbors(distance_func, vectors_[n],
                                         n_candidates, max_conn);
                    n_adj = std::move(pruned);
                }
            }

            if (!candidates.empty()) {
                current_ep = candidates[0].second;
            }
        }

        if (l > max_level_global_) {
            max_level_global_ = l;
            entry_point_ = internal_idx;
        }
    }

    // --- Top-level search (layer descent + full search at layer 0) ---

    template <typename DistanceFunc>
    [[nodiscard]] std::vector<SearchResult>
    search_impl(const Vector<float>& query_vec, size_t k,
                DistanceFunc distance_func) const {
        size_t current_ep = entry_point_;

        for (int level = max_level_global_; level > 0; --level) {
            auto results =
                search_layer(query_vec, current_ep, 1, level, distance_func);
            if (!results.empty()) {
                current_ep = results[0].second;
            }
        }

        auto results =
            search_layer(query_vec, current_ep, ef_, 0, distance_func);

        std::vector<SearchResult> output;
        size_t n = std::min(k, results.size());
        output.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            output.push_back(
                {vectors_[results[i].second].id, results[i].first});
        }
        return output;
    }

    // --- Visited tag capacity ---

    void ensure_visited_capacity() const {
        if (visited_.size() < vectors_.size()) {
            visited_.resize(vectors_.size(), 0);
        }
    }

    // --- Data members ---

    size_t dim_;
    size_t M_;
    size_t M_max0_;
    size_t ef_construction_;
    size_t ef_;
    DistanceType distance_type_;

    std::vector<Vector<float>> vectors_;
    std::vector<int> max_level_;
    std::vector<std::vector<std::vector<size_t>>> adjacency_;

    size_t entry_point_;
    int max_level_global_;

    float mL_;
    mutable std::mt19937 rng_;

    mutable std::vector<int> visited_;
    mutable int visited_tag_ = 0;
};

} // namespace vecflow
