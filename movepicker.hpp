#pragma once

#include <iostream>
#include "board.hpp"
#include "search.hpp"
#include "movelist.hpp"
#include "movegen.hpp"
#include "types.hpp"
#include "thread.hpp"

namespace enyo {

enum SearchType { QSEARCH, ABSEARCH };

static inline int piece_value(PieceType pt) {
    switch (pt) {
        case pawn:   return 100;
        case knight: return 320;
        case bishop: return 330;
        case rook:   return 500;
        case queen:  return 900;
        case king:   return 0;
        default:     return 0;
    }
}

inline static constexpr int mvvlva(Move move) {
    constexpr std::array<std::array<int, piece_type_nb>, piece_type_nb> MVV_LVA = {{
        // attacker ... ->
        //no_pt    P     N     B     R     Q   K    // victim:
        {   0,     0,    0,    0,    0,    0,  0 }, // no_piece_type
        {   0,  1500, 1400, 1300, 1200, 1100,  0 }, // P
        {   0,  2500, 2400, 2300, 2200, 2100,  0 }, // N
        {   0,  3500, 3400, 3300, 3200, 3100,  0 }, // B
        {   0,  4500, 4400, 4300, 4200, 4100,  0 }, // R
        {   0,  5500, 5400, 5300, 5200, 5100,  0 }, // Q
        {   0,     0,    0,    0,    0,    0,  0 }, // K
    }};
    return MVV_LVA
        [static_cast<size_t>(move.dst_piece())]
        [static_cast<size_t>(move.src_piece())];
}


enum MoveScores : int {
    TT_SCORE            = 10'000'000,
    PV_SCORE            = 9'000'000,
    PROMOTE_SCORE       = 8'000'000,
    CAPTURE_SCORE       = 7'000'000,
    KILLER1_SCORE       = 6'000'000,
    KILLER2_SCORE       = 5'000'000,
    CASTLE_SCORE        = 4'500'000,
    COUNTER_SCORE       = 4'000'000,
    DRAW_SCORE          = 0,
    NEGATIVE_SCORE      = -10'000'000
};


static inline bool is_castle(Move move) {
    const auto src_sq = move.src_sq();
    const auto dst_sq = move.dst_sq();
    const auto src_file = 1ULL << src_sq;
    const auto dst_file = 1ULL << dst_sq;
    return ((move.src_piece() == king)
        && ((src_file & file_e) && (dst_file & (file_c | file_g))));
}

template <Color Us, SearchType ST>
static inline std::vector<enyo::Move> prioritize_moves(
    Worker& worker,
    const Movelist& moves,
    Move tt_move = 0,
    int ply = MAX_PLY)
{
    constexpr bool debug = false;
    const auto& board = worker.si.board;

    std::vector<ScoredMove> scored_moves;
    scored_moves.reserve(moves.size());

    for (const auto& move : moves) {
        int score = DRAW_SCORE;
        if (move == tt_move) {
            score = TT_SCORE;
        } else if (move.dst_piece() != no_piece_type) {
            score = CAPTURE_SCORE + mvvlva(move);
        } else if (move.flags() & Move::Flags::Promote) {
            score = (move.promo_piece() == queen) ? PROMOTE_SCORE : DRAW_SCORE;
        } else if constexpr (ST == QSEARCH) {
            // Skip non-capturing, non-promoting moves in QSEARCH
            continue;
        } else {
            if (move == worker.killers[0]) {
                score = KILLER1_SCORE;
            } else if (move == worker.killers[1]) {
                score = KILLER2_SCORE;
            } else if (is_castle(move)) {
                score = CASTLE_SCORE;
            } else {
                auto range = board.pv_table | std::views::take(ply);
                if (auto it = std::ranges::find(range, move); it != range.end()) {
                    score = PV_SCORE - static_cast<int>(std::distance(range.begin(), it));
                }
            }
        }

        scored_moves.emplace_back(ScoredMove{score, move});
    }

    std::sort(scored_moves.begin(), scored_moves.end());

    if constexpr (debug) {
        fmt::print("{} moves:{}\n", ST == QSEARCH ? "QS" : "AB", moves.size());
        for (const auto& [score, move] : scored_moves) {
            if (score) fmt::print("  {}: {}\n", move, score);
        }
    }

    std::vector<Move> prioritized_moves;
    prioritized_moves.reserve(scored_moves.size());
    std::ranges::transform(scored_moves,
        std::back_inserter(prioritized_moves), [](const ScoredMove& scored_move) {
            return scored_move.move;
        }
    );

    return prioritized_moves;
}


} // enyo ns
