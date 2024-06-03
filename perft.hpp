#pragma once

#include "board.hpp"
#include "movegen.hpp"
#include <cstdint>
#include <chrono>

namespace enyo {

// for debug of perft depth 1
/*

If I have e.g 1x d1b3 too many:

  divide -f 6k1/8/4p3/3B4/8/1Q6/4K3/8 b - - 4 10 -d1

  e6d5 1
  e6d5 1            WTF? 2x e6d5
  g8h7 1
  g8g7 1
  g8f7 1
  g8h8 1
  g8f8 1

7 positions reported
Make the move, and use this FEN w/ stockfish:

stockfish
position fen 6k1/8/4p3/3B4/8/1Q6/4K3/8 b - - 4 10
go perft 1

g8h8: 1
e6d5: 1
g8f7: 1
g8g7: 1
g8h7: 1
g8f8: 1

Nodes searched: 6

*/

constexpr bool debug_perft = false;

static inline uint64_t print_leaf_nodes(Board & b)
{
    auto moves = b.side == white
        ? generate_legal_moves<white, false, false>(b)
        : generate_legal_moves<black, false, false>(b);
    for (auto const move : moves) {
        fmt::print("  {} {}\n", move, 1);
    }
    fmt::print("\n{}\n", moves.size());
    return moves.size();
}

template <Color Us, bool Verbose>
inline uint64_t perft_impl(Board & b, int depth)
{
    constexpr auto Them = ~Us;
    uint64_t total_leaf_nodes = 0;
    auto moves = generate_legal_moves<Us, false, false>(b);
    for (auto const move : moves) {
#if 1
        if (depth == 1) return moves.size();
#else
        // for movegen debug: (leaf nodes are printed first, them branch)
        if (depth == 1) {
            print_leaf_nodes(b);
            return moves.size();
        }
#endif

        apply_move<Us>(b, move);
        auto const leaf_nodes = perft_impl<Them, false>(b, depth - 1);
        if constexpr (Verbose) fmt::print("{} {}\n", move, leaf_nodes);
        total_leaf_nodes += leaf_nodes;
        revert_move<Us>(b);
    }
    return total_leaf_nodes;
}

template <bool Verbose = false>
uint64_t perft(Board & b, int depth)
{
    if (depth == 1) {
        return print_leaf_nodes(b);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t const total_leaf_nodes = b.side == white
        ? perft_impl<white, Verbose>(b, depth)
        : perft_impl<black, Verbose>(b, depth);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

    if constexpr (Verbose) {
        if constexpr (debug_perft) {
            fmt::print("\n{}\n", total_leaf_nodes);
        } else {
            fmt::print("\n{} nodes in {:.6f} sec, ", total_leaf_nodes, duration / 1000.0);
            auto nodes_per_second = static_cast<int>(static_cast<double>(total_leaf_nodes) * 1000.0 / duration);
            fmt::print("{:0.6f} Mnps\n", nodes_per_second / 1000000.0);
        }
    }

    return total_leaf_nodes;
}


} // enyo

