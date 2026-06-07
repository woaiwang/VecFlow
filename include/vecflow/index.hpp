#pragma once

#include <cstddef>
#include <vector>

#include "vecflow/types.hpp"

namespace vecflow {

/// Abstract interface for nearest-neighbor search indexes.
///
/// All indexes support two operations:
/// - add(): insert a vector into the index (not thread-safe)
/// - search(): find the k nearest neighbors of a query (thread-safe, read-only)
class IIndex {
public:
    virtual ~IIndex() = default;

    /// Insert a vector into the index.
    /// @param vec  the vector to insert (id + aligned data)
    virtual void add(const Vector<float>& vec) = 0;

    /// Find the k nearest neighbors of a query vector.
    /// @param query  raw float query (must match index dimension)
    /// @param k      number of results to return
    /// @return       up to k results sorted by distance ascending
    [[nodiscard]] virtual std::vector<SearchResult>
    search(const std::vector<float>& query, size_t k) const = 0;

    /// Number of vectors currently in the index.
    [[nodiscard]] virtual size_t size() const = 0;

    /// Dimension of vectors stored in the index.
    [[nodiscard]] virtual size_t dim() const = 0;
};

} // namespace vecflow
