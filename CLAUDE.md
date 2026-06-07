# CLAUDE.md

## Project Vibe

VecFlow is a **pedagogical-first, embeddable** C++17 library for SIMD-accelerated Approximate Nearest Neighbor (ANN) search. The primary audience is learners who want to understand how modern ANN indexes work under the hood — every component is written to be readable and hackable, not to win a benchmark at all costs.

**Core identity:**
- **Teaching-oriented**: Code clarity trumps micro-optimization. Inline comments explain *why*, not *what*.
- **Header-only core**: The entire public API lives in `include/vecflow/`. User code only needs `#include "vecflow/vecflow.hpp"`. No shared library, no runtime dependency.
- **Zero external dependencies**: GTest and Google Benchmark are fetched at configure-time by CMake FetchContent; they are not required by the library itself.
- **SIMD as a learning vehicle**: Hand-written AVX2 intrinsics (in `src/distance_avx2.cpp`) demonstrate real-world SIMD optimization, with a portable scalar fallback in the header-only path.
- **C++17 baseline**: Deliberately targeting C++17 — not C++20/23 — because that's what most production codebases use today.

## Architecture (Cheat Sheet)

```
User Code (#include vecflow/vecflow.hpp)
  └── IIndex interface (virtual add/search)
        ├── FlatIndex   → exact KNN, O(n), correctness baseline
        └── HNSWIndex   → approximate, O(log n), production index
              └── DistanceDispatcher → runtime CPU dispatch (scalar vs AVX2)
                    ├── scalar_kernel (header-only, portable)
                    └── avx2_kernel  (src/distance_avx2.cpp, compiled)

Data types: Vector<float>, SearchResult, LayeredGraph
```

Key files:
- `include/vecflow/types.hpp` — Core data structures
- `include/vecflow/distance.hpp` — Distance dispatcher + scalar kernels
- `include/vecflow/flat_index.hpp` — Brute-force exact KNN
- `include/vecflow/hnsw.hpp` — HNSW graph index
- `include/vecflow/vecflow.hpp` — Umbrella header
- `src/distance_avx2.cpp` — AVX2 SIMD kernels (compiled separately)
- `tests/` — GTest unit tests
- `benchmarks/` — Google Benchmark microbenchmarks

## C++ Coding Style

### Must Use (Modern C++17)

- **`std::optional<T>`** for functions that may not return a value. Never use `T*` with null as sentinel.
- **`std::string_view`** for all non-owning string parameters. Never `const char*` or `const std::string&` for read-only string access.
- **`[[nodiscard]]`** on every function where discarding the return value is a bug (e.g., `search()`, `add()`, all pure functions).
- **Structured bindings** when iterating maps or returning multiple values: `auto [id, dist] = ...;`
- **`if constexpr`** for compile-time branching instead of SFINAE or tag dispatch where the condition is known at compile time.
- **`static_assert`** for compile-time validation of template parameters.
- **`std::unique_ptr` / `std::shared_ptr`** for ownership. Never raw `new` / `delete`.
- **Range-based for loops** unless the index is semantically meaningful.
- **`auto`** when the type is obvious from context or excessively verbose.

### Must NOT Use

- **No C-style casts**: Use `static_cast`, `reinterpret_cast`, `const_cast`. A single `(float)x` in the codebase fails review.
- **No macros** except for include guards (`#pragma once` preferred) and platform detection (`#ifdef __AVX2__`). No function-like macros, ever.
- **No `printf` / `scanf`**: Use `std::cout` / `std::format` (or `fmt::print` if we add fmtlib later).
- **No raw C arrays**: Use `std::array<T, N>` for fixed-size or `std::vector<T>` for dynamic.
- **No `void*`**: If you think you need it, use `std::any`, `std::variant`, or a template.
- **No `NULL`**: Use `nullptr`.
- **No manual memory management**: No `malloc`/`free`, no raw `new`/`delete`. Use RAII types.
- **No `using namespace std;`** in header files. In `.cpp` files, it's tolerated but discouraged.

### Memory Safety

- All heap allocations must be owned by an RAII type (`std::unique_ptr`, `std::vector`, etc.).
- For 32-byte aligned heap allocations, use `vecflow::AlignedAllocator<T, 32>` (defined in `include/vecflow/allocator.hpp`). Never use raw `std::aligned_alloc` directly — it requires manual size rounding and `std::free` pairing that the allocator handles correctly across platforms.
- CI runs AddressSanitizer + UndefinedBehaviorSanitizer on every push. A change that introduces an ASan failure is a hard blocker.
- Thread safety contract: `search()` is thread-safe (read-only). `add()` is NOT thread-safe — call it from a single thread during index build. Document this explicitly rather than hiding behind a mutex.

### Naming Conventions

- **Types / Classes / Enums**: `PascalCase` → `FlatIndex`, `SearchResult`, `DistanceType`
- **Functions / Methods**: `snake_case` → `compute_l2()`, `search()`, `add()`
- **Variables**: `snake_case` → `num_vectors`, `ef_construction`
- **Private members**: trailing underscore → `dim_`, `vectors_`, `graph_`
- **Template parameters**: `PascalCase` or descriptive single-word → `typename T`, `typename DistanceFunc`

### Header Hygiene

- Every header must be self-contained (includes everything it needs).
- Use `#pragma once` for include guards.
- Include order: standard library → third-party → project headers. Separate groups with a blank line.
- Avoid heavy includes in headers; prefer forward declarations where possible.

## Build & Test Commands

### Quick Start (first time)

```bash
# Configure (downloads GTest + Google Benchmark via FetchContent, ~1-2 min)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)
```

### Build Variants

```bash
# Debug build (no optimization, full debug symbols)
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j$(nproc)

# Release build with -O3 -march=native -ffast-math
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# ASan + UBSan build (catches memory bugs)
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DVECFLOW_ENABLE_ASAN=ON
cmake --build build-asan -j$(nproc)

# Build without tests/benchmarks (library-only)
cmake -B build-lib -DVECFLOW_BUILD_TESTS=OFF -DVECFLOW_BUILD_BENCHMARKS=OFF
cmake --build build-lib
```

### Running Tests

```bash
# Run all tests (from build dir)
cd build && ctest --output-on-failure

# Run all tests (from project root, pointing at build dir)
ctest --test-dir build --output-on-failure

# Run a specific test binary with verbose output
./build/tests/test_hnsw --gtest_verbose
./build/tests/test_distance
./build/tests/test_flat_index

# Run ASan instrumented tests
cd build-asan && ctest --output-on-failure
```

### Running Benchmarks

```bash
# Run the distance kernel benchmark
./build/benchmarks/bench_distance
```

### CMake Options Reference

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel` |
| `VECFLOW_BUILD_TESTS` | `ON` | Build GTest unit tests |
| `VECFLOW_BUILD_BENCHMARKS` | `ON` | Build Google Benchmark microbenchmarks |
| `VECFLOW_ENABLE_ASAN` | `OFF` | Enable AddressSanitizer + UBSan (Debug only) |

## When Fixing Bugs

1. First, reproduce the failure: `cmake --build build && cd build && ctest --output-on-failure`
2. If it's a memory issue, rebuild with ASan: `cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DVECFLOW_ENABLE_ASAN=ON && cmake --build build-asan && cd build-asan && ctest --output-on-failure`
3. If a test binary is available, run it directly with `--gtest_verbose` or `--gtest_filter=TestName.*` for isolation.
4. Do not write fixes that introduce macros, C-style code, or raw pointers — the coding style section is non-negotiable.

## Important Engineering Rules (Lessons Learned)

These rules capture hard-won lessons from the development cycle. Violating any of them will cause segfaults, recall regressions, or silent correctness bugs.

### SIMD & Memory Alignment

- **`_mm256_load_ps` requires 32-byte alignment.** Passing a non-aligned pointer causes a segfault (not a performance degradation — a hard crash). Every pointer passed to the AVX2 kernel must originate from `AlignedAllocator<T, 32>`.
- **In production code**, always use `AlignedAllocator` (via `Vector<T>::AlignedData` or the `AlignedBuf` alias). The `Vector<T>` constructors accept plain `std::vector<T>` and `std::initializer_list<T>`, performing the copy into aligned storage transparently.
- **In test/benchmark code**, stack arrays (`float x[128]`) are NOT guaranteed 32-byte aligned. Use `alignas(32) float x[128]` for stack buffers, or `std::vector<float, AlignedAllocator<float, 32>>` for heap buffers.
- **Platform portability**: `std::aligned_alloc` is not available in MinGW. `AlignedAllocator` abstracts this — use `_aligned_malloc`/`_aligned_free` on Windows, `std::aligned_alloc`/`std::free` on POSIX.
- **Size rounding**: `std::aligned_alloc` requires the allocation size to be a multiple of alignment. `AlignedAllocator::allocate()` rounds `n * sizeof(T)` up automatically. If you ever bypass the allocator, you must do the same rounding.

### HNSW Graph & Recall

- **Any change to `search_layer`, `select_neighbors`, `add()`, or `search()` MUST keep the Recall@K test green.** The test `HNSWTest.RecallAtK` in `tests/test_hnsw.cpp` compares HNSW against FlatIndex ground truth on 1000 random 128-dim vectors. The threshold is ≥ 85% Recall@10. A dip below this means the change is wrong, not that the threshold should be lowered.
- **Never replace the heuristic selection with simple top-K.** The HNSW standard heuristic (`select_neighbors` in `hnsw.hpp`) prunes candidates that are closer to an already-selected neighbor than to the query. This diversity mechanism is essential for high recall — the "just pick the M closest" shortcut creates hub nodes that degrade graph navigability.
- **The visited tag mechanism is delicate.** `search_layer` relies on `visited_tag_` increment + `visited_[node] == visited_tag_` checks to provide O(1) per-search deduplication without clearing the array. If you add a new `search_layer` call path, make sure `visited_tag_` is incremented before the search (it already is inside `search_layer`).
- **`ef=1` on upper layers is intentional.** During insertion and search, the layer descent uses `ef=1` because upper layers are exponentially sparse — finding a single closest node is sufficient to route to the next layer down. Don't increase this "for safety"; it wastes cycles with zero recall benefit.

### API-First Development

- **New features start with a test or example, not with implementation.** Before writing any algorithm code, write a test that demonstrates the expected behavior (with concrete input/output values) or an example that shows the end-to-end user experience. This forces you to think about the API contract before the internals.
- **FlatIndex is the correctness baseline.** Any search result can be validated by comparing against `FlatIndex::search()` on the same data. Use this in every HNSW test.
- **The `HNSWTestAccessor` friend struct is for white-box testing only.** It exists so tests can manually construct specific graph topologies to verify `search_layer` correctness. Production code must never use it — all production insertion goes through `add()`.
- **Examples in `examples/` are user-facing documentation.** They must compile with a single `#include <vecflow/vecflow.hpp>` and produce readable output. Keep them under 80 lines.

### Troubleshooting Quick Reference

Problems encountered during development and their root causes:

| Symptom | Root Cause | Fix |
|---|---|---|
| `cmake: command not found` | Portable cmake not in PATH | `export PATH="$HOME/.local/cmake-3.28.3-windows-x86_64/bin:$PATH"` |
| FetchContent git clone fails with "CMD does not support UNC paths" | Project on `\\wsl.localhost\...` UNC path; cmd.exe can't use UNC as CWD | Build from a local path: `cmake -B /tmp/build` |
| `search_layer` returns only the entry point (4 tests fail with size()==1) | `visited_tag_` starts at 0, collides with `visited_.resize(n, 0)` default fill — all nodes appear "already visited" | Increment `visited_tag_` at start of every `search_layer` call |
| `'vectors_' is private within this context` | `HNSWTestAccessor` defined in global namespace, but `friend struct` is in `namespace vecflow` — two different types | Define `HNSWTestAccessor` inside `namespace vecflow { ... }` |
| `'aligned_alloc' is not a member of 'std'` | MinGW libstdc++ doesn't expose `std::aligned_alloc` | Use `_aligned_malloc`/`_aligned_free` on `_WIN32`, `std::aligned_alloc`/`std::free` on POSIX (abstracted in `AlignedAllocator`) |
| Segfault in AVX2 correctness test after upgrading `_mm256_loadu_ps` → `_mm256_load_ps` | Test data in stack arrays (`float x[128]`) is not 32-byte aligned | Use `alignas(32)` on stack arrays, or `std::vector<float, AlignedAllocator<float, 32>>` for heap buffers |
| Segfault in benchmark after same AVX2 upgrade | `generate_random_vectors()` returned plain `std::vector<float>` — data not aligned | Return `AlignedBuf` (aligned vector) from benchmark helpers |
| Google Benchmark `RunSpecifiedBenchmarks` segfaults on MinGW | Likely thread-pool / static-init incompatibility between Google Benchmark and MinGW's pthreads | Replace with manual `std::chrono`-based benchmark (see `benchmarks/bench_distance.cpp`) |
| Scalar benchmark as fast as AVX2 (0.9x instead of 7x speedup) | GCC `-O3 -march=native` auto-vectorizes the scalar loop into SSE/AVX instructions | Isolate scalar kernel in a separate .cpp file compiled without `-march=native`; add `-fno-tree-vectorize` to prevent auto-vectorization |
| `cannot convert 'const double*' to 'const float*'` in `compute_l2_sqr<Vector<double>>` | AVX2 kernel only takes `const float*`, but the dispatch didn't guard against `T=double` | Add `if constexpr (std::is_same_v<T, float>)` before the AVX2 dispatch in `compute_l2_sqr` and `compute_cosine` |
| `'true_type' in namespace 'std' does not name a type` | `allocator.hpp` was missing `#include <type_traits>`; worked transitively when included after other headers but failed when included first via umbrella header | Add `#include <type_traits>` to `allocator.hpp` (every header must be self-contained) |
