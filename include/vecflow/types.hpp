#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

#include "vecflow/allocator.hpp"

namespace vecflow {

/// A user-labeled vector with 32-byte aligned storage.
/// @tparam T  element type (float or double)
template <typename T>
struct Vector {
    /// Aligned storage type — guarantees 32-byte alignment for AVX2.
    using AlignedData = std::vector<T, AlignedAllocator<T, 32>>;

    size_t id;         ///< User-assigned identifier
    AlignedData data;  ///< Aligned element buffer

    /// Construct from an already-aligned container (zero-copy via move).
    Vector(size_t id, AlignedData d) : id(id), data(std::move(d)) {}

    /// Construct from a plain std::vector (copies elements into aligned storage).
    Vector(size_t id, const std::vector<T>& d)
        : id(id), data(d.begin(), d.end()) {}

    /// Construct from a brace-enclosed initializer list.
    Vector(size_t id, std::initializer_list<T> il)
        : id(id), data(il.begin(), il.end()) {}
};

/// Result of a nearest-neighbor search.
struct SearchResult {
    size_t id;     ///< User-assigned identifier of the matched vector
    float distance; ///< Distance to the query (lower is closer)
};

} // namespace vecflow
