#include <unordered_map>
#include <vector>
#include <cstdint>
#include <iostream>
#include "types.hpp"
#include "config.hpp"

namespace enyo {

// Assuming these are your Move and entry_type types


enum entry_type {
    exact,
    alpha,
    beta
};


struct ttentry_t {
    uint64_t hash_key{};
    double score{};
    enyo::Move move{};
    int depth{};
    enyo::entry_type flag{exact};
};

class Cache {
public:
    // Constructor
    explicit Cache(size_t size = (16 * 1024 * 1024)) : table_size(size) {
        if constexpr (config::use_tt) {
            // Initialize the transposition table with the specified size
            transposition_table.reserve(size);
        } else {
            table_size = 0;
        }
    }

    // Destructor (if needed)

    // Insert an entry into the transposition table
    void insert(uint64_t poskey, enyo::Move move, double score, int depth, entry_type hash_flag) {
        if constexpr (!config::use_tt)
            return;

        // Insert the entry into the unordered_multimap
        transposition_table.emplace(poskey, ttentry_t{poskey, score, move, depth, hash_flag});
    }

    // Find principal variation for a given hash key
    std::vector<ttentry_t> find_pv(uint64_t hash_key) const {
        std::vector<ttentry_t> pv;

        if constexpr (!config::use_tt)
            return pv;

        // Find all entries with the specified hash key in the unordered_multimap
        auto range = transposition_table.equal_range(hash_key);
        for (auto it = range.first; it != range.second; ++it) {
            pv.push_back(it->second);
        }

        return pv;
    }

private:
    // Unordered_multimap to store entries with the same hash key
    std::unordered_multimap<uint64_t, ttentry_t> transposition_table;

    // Additional members as needed
    size_t table_size{};
};

}

// Example usage
int main() {
    enyo::Cache cache;
    uint64_t hash_key = 123456789; // Replace with your hash key

    // Insert some entries
    cache.insert(hash_key, enyo::Move{}, 10.5, 1, enyo::exact);
    cache.insert(hash_key, enyo::Move{}, 15.7, 2, enyo::exact);
    cache.insert(hash_key, enyo::Move{}, 20.3, 3, enyo::exact);

    // Find principal variation
    auto pv = cache.find_pv(hash_key);

    // Display the principal variation
    for (const auto& entry : pv) {
        std::cout << "Hash Key: " << entry.hash_key << " Depth: " << entry.depth << " Score: " << entry.score << "\n";
    }

    return 0;
}

