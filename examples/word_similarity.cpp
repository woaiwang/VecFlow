#include <vecflow/vecflow.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // -----------------------------------------------------------
    // Step 1: Build a semantic word-vector index (dim=3)
    // -----------------------------------------------------------
    vecflow::HNSWIndex index(/*dim=*/3);

    const std::vector<std::string> words = {"king", "man", "woman", "queen",
                                            "apple"};
    const std::vector<std::vector<float>> vectors = {
        {0.9f, 0.1f, 0.2f},  // king
        {0.8f, 0.1f, 0.1f},  // man
        {0.1f, 0.8f, 0.1f},  // woman
        {0.2f, 0.9f, 0.2f},  // queen
        {0.1f, 0.1f, 0.9f},  // apple
    };

    for (size_t i = 0; i < words.size(); ++i) {
        index.add({i, vectors[i]});
    }

    std::cout << "Inserted " << words.size() << " word vectors.\n\n";

    // -----------------------------------------------------------
    // Step 2: Classic word analogy: king - man + woman ≈ ?
    // -----------------------------------------------------------
    const auto& king = vectors[0];
    const auto& man = vectors[1];
    const auto& woman = vectors[2];

    std::vector<float> query(3);
    for (size_t d = 0; d < 3; ++d) {
        query[d] = king[d] - man[d] + woman[d];
    }

    std::cout << "Query: vector(\"king\") - vector(\"man\") + vector(\"woman\")\n";
    std::cout << "  = [" << query[0] << ", " << query[1] << ", " << query[2]
              << "]\n\n";

    // -----------------------------------------------------------
    // Step 3: Search for the 2 nearest neighbors
    // -----------------------------------------------------------
    constexpr size_t k = 2;
    auto results = index.search(query, k);

    std::cout << "Top-" << k << " nearest neighbors:\n\n";
    std::cout << std::left << std::setw(10) << "Word"
              << std::right << std::setw(12) << "Distance"
              << "  (expected)\n";
    std::cout << std::string(40, '-') << "\n";

    for (const auto& r : results) {
        std::string expected;
        if (r.id == 3) {  // queen
            expected = "  ← queen (king - man + woman)";
        }
        std::cout << std::left << std::setw(10) << words[r.id]
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(12) << r.distance << expected << "\n";
    }

    std::cout << "\n";

    // -----------------------------------------------------------
    // Verify the expected result
    // -----------------------------------------------------------
    if (!results.empty() && results[0].id == 3) {
        std::cout << "✓ The closest match is \"queen\" — the analogy holds!\n";
        return 0;
    }
    std::cout << "✗ Unexpected result.\n";
    return 1;
}
