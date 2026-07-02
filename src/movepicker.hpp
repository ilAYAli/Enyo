#pragma once

#include <limits>

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

struct PrioritizedMoves {
    struct Entry {
        int score;
        Move move;
    };

    void clear() {
        size = 0;
        picked = 0;
        best_score = std::numeric_limits<int>::min();
        best_is_unique = false;
        sorted = false;
    }

    void add(Move move, int score) {
        assert(size < entries.size());
        entries[size] = Entry {score, move};
        if (size == 0 || score > best_score) {
            best_score = score;
            best_index = size;
            best_is_unique = true;
        } else if (score == best_score) {
            best_is_unique = false;
        }
        ++size;
    }

    Move next() {
        if (picked == size)
            return Move {};

        // Keep the original input untouched so a later fallback sort retains
        // its exact tie order. A unique maximum is first under either path.
        if (picked == 0 && best_is_unique) {
            ++picked;
            return entries[best_index].move;
        }

        sort();
        return entries[picked++].move;
    }

    Move operator[](std::size_t index) const {
        assert(index < picked);
        if (!sorted) {
            assert(index == 0 && best_is_unique);
            return entries[best_index].move;
        }
        return entries[index].move;
    }

    std::size_t count() const {
        return size;
    }

private:
    void sort() {
        if (sorted)
            return;

        std::ranges::sort(
            entries.begin(),
            entries.begin() + static_cast<std::ptrdiff_t>(size),
            [](const Entry & lhs, const Entry & rhs) {
                return lhs.score > rhs.score;
            });
        sorted = true;
    }

    std::array<Entry, MAX_MOVES> entries;
    std::size_t size = 0;
    std::size_t picked = 0;
    std::size_t best_index = 0;
    int best_score = std::numeric_limits<int>::min();
    bool best_is_unique = false;
    bool sorted = false;
};

template <Color Us, SearchType ST>
static inline void prioritize_moves(
    PrioritizedMoves & out,
    Worker& worker,
    const Movelist& moves,
    Move tt_move = 0,
    const Move * killers = nullptr,
    Move countermove = Move{},
    const typename Worker::CmhPieceTable * cmh_slice = nullptr)
{
    auto & board = worker.si.board;
    out.clear();

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
            // Losing captures (SEE < 0) go below killers, counters and
            // quiets: searching QxP-defended before good quiet moves
            // wastes nodes at every ABSEARCH node. MVV-LVA still orders
            // within each of the two capture bands, nudged by observed
            // cutoff success (capture history, ±512 at /32 so it mostly
            // reorders within a victim band).
            const int capthist = worker.capture_history
                [Us]
                [static_cast<size_t>(move.src_piece())]
                [move.dst_sq()]
                [static_cast<size_t>(move.dst_piece())] / 32;
            if (ST == ABSEARCH && !see_ge<Us>(board, move))
                score = NEGATIVE_SCORE + mvvlva(move) + capthist;
            else
                score = CAPTURE_SCORE + mvvlva(move) + capthist;
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
            }
        }

        out.add(move, score);
    }

}


} // enyo ns
