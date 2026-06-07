#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace vecflow {

// ---------------------------------------------------------------
// AlignedAllocator: std::allocator_traits-compatible allocator.
//
// On POSIX: uses std::aligned_alloc / std::free.
// On Windows: uses _aligned_malloc / _aligned_free (MinGW/MSVC).
//
// Guarantees Alignment-byte alignment for vector data, enabling
// _mm256_load_ps (aligned load) in the AVX2 kernel.
// ---------------------------------------------------------------
template <typename T, size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

    [[nodiscard]] T* allocate(size_t n) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        size_t size = n * sizeof(T);
        // aligned allocation requires size to be a multiple of alignment
        size_t aligned_size = (size + Alignment - 1) & ~(Alignment - 1);

        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(aligned_size, Alignment);
#else
        ptr = std::aligned_alloc(Alignment, aligned_size);
#endif
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
};

template <typename T1, size_t A1, typename T2, size_t A2>
bool operator==(const AlignedAllocator<T1, A1>&,
                const AlignedAllocator<T2, A2>&) {
    return A1 == A2;
}

template <typename T1, size_t A1, typename T2, size_t A2>
bool operator!=(const AlignedAllocator<T1, A1>& a,
                const AlignedAllocator<T2, A2>& b) {
    return !(a == b);
}

} // namespace vecflow
