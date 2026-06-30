#pragma once

#include <iostream>
#include "board.hpp"
#include "search.hpp"
#include "movelist.hpp"
#include "movegen.hpp"
#include "types.hpp"
#include "thread.hpp"
#include "see.hpp"

namespace enyo {

enum SearchType { QSEARCH, ABSEARCH };

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
    COUNTER_SCORE       = 4'500'000,
    CASTLE_SCORE        = 4'000'000,
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

// Fixed-size buffer for the prioritize_moves result. 256 = max legal
// moves in any position (Movelist::max_size). Avoids two heap
// allocations per call (the old code returned a std::vector<Move>
// built from a separate std::vector<ScoredMove>).
struct PrioritizedMoves {
    // Deliberately uninitialized (requires Move's trivial default ctor):
    // only [0, size) is ever read, and prioritize_moves writes each of
    // those slots before publishing size.
    std::array<Move, 256> moves;
    std::size_t           size = 0;

    using iterator       = Move *;
    using const_iterator = const Move *;

    iterator       begin()        { return moves.data(); }
    iterator       end()          { return moves.data() + size; }
    const_iterator begin() const  { return moves.data(); }
    const_iterator end()   const  { return moves.data() + size; }

    Move operator[](std::size_t i) const { return moves[i]; }
    Move & operator[](std::size_t i)     { return moves[i]; }
};

template <Color Us, SearchType ST>
static inline void prioritize_moves(
    PrioritizedMoves & out,
    Worker& worker,
    const Movelist& moves,
    Move tt_move = 0,
    int ply = MAX_PLY,
    const Move * killers = nullptr,
    Move countermove = Move{},
    const typename Worker::CmhPieceTable * cmh_slice = nullptr)
{
    constexpr bool debug = false;
    auto & board = worker.si.board;

    // Stack-allocated scratch. Same capacity as before but no heap
    // traffic. Previously two std::vector allocations per call.
    std::array<ScoredMove, 256> scored;
    std::size_t n = 0;

    for (const auto move : moves) {
        int score = DRAW_SCORE;
        // In QSEARCH, reject a quiet (non-capture, non-promotion) tt_move
        // before it can inherit TT_SCORE and bypass the capture/promo
        // filter below. Previously the `move == tt_move` branch fired
        // first, letting a quiet tt_move into qsearch and forcing the
        // next node into the in-check all-evasions branch — a recursion
        // explosion that burned the hard-time budget. See
        // odonata-bot vs EnyoBot JCKXOwix replay experiment.
        //
        // Narrow fix: only reject *quiet* tt_move in QSEARCH. tt_move
        // captures still get TT_SCORE (and, as before, bypass the SEE
        // filter) — changing that is a separate question.
        if constexpr (ST == QSEARCH) {
            if (move == tt_move
                && move.dst_piece() == no_piece_type
                && (move.flags() & Move::Flags::promote) == 0)
                continue;
        }
        if (move == tt_move) {
            score = TT_SCORE;
        } else if (move.dst_piece() != no_piece_type) {
            if constexpr (ST == QSEARCH) {
                if (!see_ge<Us>(board, move))
                    continue;
            }
            score = CAPTURE_SCORE + mvvlva(move);
        } else if (move.flags() & Move::Flags::promote) {
            score = (move.promo_piece() == queen) ? PROMOTE_SCORE : DRAW_SCORE;
        } else if constexpr (ST == QSEARCH) {
            continue;
        } else {
            if (killers && move == killers[0]) {
                score = KILLER1_SCORE;
            } else if (killers && move == killers[1]) {
                score = KILLER2_SCORE;
            } else if (countermove && move == countermove) {
                score = COUNTER_SCORE;
            } else if (is_castle(move)) {
                score = CASTLE_SCORE;
            } else {
                score = worker.history[Us][move.src_sq()][move.dst_sq()];
                if (cmh_slice) {
                    score += (*cmh_slice)[static_cast<size_t>(move.src_piece())][move.dst_sq()];
                }
                auto range = board.pv_table | std::views::take(ply);
                if (auto it = std::ranges::find(range, move); it != range.end()) {
                    score += PV_SCORE - static_cast<int>(std::distance(range.begin(), it));
                }
            }
        }

        scored[n++] = ScoredMove{score, move};
    }

    std::ranges::sort(
        scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(n),
        [](const auto & a, const auto & b) { return a.score > b.score; });

    if constexpr (debug) {
        fmt::print("{} moves:{}\n", ST == QSEARCH ? "QS" : "AB", moves.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (scored[i].score) fmt::print("  {}: {}\n", scored[i].move, scored[i].score);
        }
    }

    out.size = n;
    for (std::size_t i = 0; i < n; ++i)
        out.moves[i] = scored[i].move;
}


} // enyo ns
