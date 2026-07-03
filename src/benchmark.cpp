#include "benchmark.hpp"

#include "uci.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <string_view>

namespace enyo {

namespace {

constexpr std::array<std::string_view, 24> benchmark_fens = {
    // Openings.
    "startpos",
    "rn1qkbnr/ppp1pppp/3p4/8/8/3P4/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 7 12",
    "r3k2r/pppqppbp/2np1np1/8/8/2NP1NP1/PPPQPPBP/R3K2R w KQkq - 0 1",
    "r3k2r/pppqppbp/2np1np1/8/1p6/P1NP1NP1/1PPQPPBP/R3K2R w KQkq - 0 1",

    // Middlegames, both tactical and quiet.
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "1r3rk1/p1q1pp1p/1np1b1p1/2Q5/8/1PNB1P2/P1P3PP/2KR3R b - - 0 17",
    "4rrk1/2p3pp/p1b2q2/1p1p1p2/3P1P2/P1P1R1P1/1P3QBP/R5K1 b - - 0 19",
    "2r1r1k1/1b1nbppp/pq1pp3/1p6/3NP3/PPN1B3/1PQ1BPPP/2RR2K1 b - - 0 18",
    "r3q1k1/pp3ppp/2pb1nn1/8/2B1P3/2N1B3/PPP2PPP/R2Q1RK1 b - - 0 14",
    "2r2rk1/1bqnbpp1/p2ppn1p/1p6/3NP3/1PN1B3/PBQ1BPPP/2RR2K1 w - - 0 17",
    "r3kb1r/3n1pp1/p6p/2pPp2q/Pp2N3/3B2PP/1PQ2P2/R3K2R w KQkq - 0 1",
    "2k4R/1p6/4p3/3pP3/8/2P1KB2/3Q4/1q4r1 b - - 16 53",
    "7R/1p6/8/4Q3/8/8/1k3K2/1q1r4 b - - 3 62",

    // Endgames.
    "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",
    "8/8/5k1p/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1",
    "1r6/2K5/R6p/1P5P/3b2k1/8/8/8 w - - 0 118",
    "8/3k4/1P6/2K1P3/8/8/4b3/6B1 b - - 0 70",
    "8/8/8/8/1k6/2n1KP2/p7/5R2 b - - 0 77",
    "8/8/8/3n4/1k6/5P2/4K3/R7 b - - 0 79",
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "8/8/2k5/8/8/5Q2/1pq2PKP/8 b - - 13 60",
};

} // namespace

BenchmarkResult run_search_benchmark(Uci & uci, int depth)
{
    uci.prepare_benchmark();

    const auto start = std::chrono::steady_clock::now();
    uint64_t nodes = 0;
    for (const auto fen : benchmark_fens)
        nodes += uci.benchmark_position(fen, depth);

    const auto elapsed = std::max(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start),
        std::chrono::milliseconds{1});
    const auto nodes_per_second = static_cast<uint64_t>(
        static_cast<long double>(nodes) * 1000.0L
        / static_cast<long double>(elapsed.count()));

    return {
        .nodes = nodes,
        .nodes_per_second = nodes_per_second,
        .elapsed = elapsed,
        .positions = benchmark_fens.size(),
    };
}

} // namespace enyo
