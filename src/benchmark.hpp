#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace enyo {

class Uci;

inline constexpr int default_benchmark_depth = 11;
inline constexpr int benchmark_hash_size_mb = 16;

struct BenchmarkResult {
    uint64_t nodes {};
    uint64_t nodes_per_second {};
    std::chrono::milliseconds elapsed {};
    size_t positions {};
};

BenchmarkResult run_search_benchmark(Uci & uci, int depth);

} // namespace enyo
