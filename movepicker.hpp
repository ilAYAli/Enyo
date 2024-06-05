#pragma once

#include "board.hpp"
#include "search.hpp"
#include "movelist.hpp"
#include "movegen.hpp"
#include "types.hpp"
#include "thread.hpp"

namespace enyo {


enum SearchType { QSEARCH, ABSEARCH };

enum MoveScores : int {
    TT_SCORE            = 10'000'000,
    CAPTURE_SCORE       = 7'000'000,
    KILLER_ONE_SCORE    = 6'000'000,
    KILLER_TWO_SCORE    = 5'000'000,
    COUNTER_SCORE       = 4'000'000,
    NEGATIVE_SCORE      = -10'000'000
};

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


template <Color Us>
std::vector<enyo::Move> prioritize_moves(Worker & worker, Movelist const & moves, Move tt_move = 0)
{
    constexpr bool debug = false;

    auto & b = worker.si.board;
    auto get_pv_move = [&](Move m) {
        for (int i = 0; i < MAX_PLY; i++) {
            auto const entry = b.pv_table[i];
            if (entry == Move::no_move)
                return 0;
            if (b.pv_table[i] == m) { // todo: use 'i' here?
                return 10000 - i;
            }
        }
        return 0;
    };

    auto get_killer_score = [](Worker const & w, Move m) {
        if (w.killers[0] == m) {
            return 900;
        }
        if (w.killers[1] == m) {
            return 800;
        }
        return 0;
    };

    auto is_castle = [](Move m) {
        return
             (m.src_sq() == e1 || m.src_sq() == e8) &&
            ((m.dst_sq() == c1 || m.dst_sq() == c8) || (m.dst_sq() == g1 || m.dst_sq() == g8));
    };

    /*
    1) PV-move of the principal variation from the previous iteration of an iterative deepening framework for the leftmost path, often implicitly done by 2.
    2) Hash move from hash tables
    3) Winning captures/promotions
    4) Equal captures/promotions
    5) Killer moves (non capture), often with mate killers first
    6) Non-captures sorted by history heuristic and that like
    7) Losing captures
    */

    std::vector<ScoredMove> scored_moves;
    scored_moves.reserve(moves.size());
    for (const auto move : moves) {
        int score = 0;

        if (move == tt_move) {
            scored_moves.emplace_back(ScoredMove{12000, move});
            continue;
        }

        // PV: 10.000 - i
        score = get_pv_move(move);
        if (score) {
            scored_moves.emplace_back(ScoredMove{score, move});
            continue;
        }

        // promote: 8000
        if (move.flags() & Move::Flags::Promote) {
            if (move.promo_piece() == queen)
                scored_moves.emplace_back(ScoredMove{800, move});
            else
                scored_moves.emplace_back(ScoredMove{-100, move});
            continue;
        }
        // TODO: add "Hash move from hash tables"

        // captures: 1100 -> 5500
        score = mvvlva(move);
        if (score) {
            scored_moves.emplace_back(ScoredMove{score, move});
            continue;
        }

        // killers: 800-900
        score = get_killer_score(worker, move);
        if (score) {
            scored_moves.emplace_back(ScoredMove{score, move});
            continue;
        }

        // castle: 500
        if (is_castle(move)) {
            scored_moves.emplace_back(ScoredMove{500, move});
            continue;
        }

        //quiet moves: random 0->moves.size()
        scored_moves.emplace_back(ScoredMove{score, move});
    }

    std::sort(scored_moves.begin(), scored_moves.end());

    if constexpr (debug) {
        fmt::print("moves:{}\n", moves.size());
        for (auto const move: scored_moves) {
            fmt::print("  {}: {}\n", move.move, move.score);
        }
    }

    std::vector<Move> prioritized_moves;
    for (auto const & scored_move : scored_moves) {
        prioritized_moves.push_back(scored_move.move);
    }

    return prioritized_moves;
}



template <Color Us, SearchType ST>
class MovePicker {
public:
    MovePicker(Worker &worker, const Stack *ss, Move move)
    : legal(generate_legal_moves<Us>(worker.si.board))
    , sorted(legal.size())
    , worker_(worker)
    , ss_(ss)
    , available_tt_move_(move)
    { }

    void score() {
        for (auto i = 0U; i < legal.size(); i++) {
            sorted[i] = ScoredMove{
                score_move(legal[i]),
                legal[i]
            };
        }
        std::sort(std::begin(sorted), std::end(sorted));
    }

    [[nodiscard]] Move next_move() {
        switch (pick_) {
            case Pick::tt:
                pick_ = Pick::score;

                if (available_tt_move_) {
                    auto it = std::find_if(std::begin(sorted), std::end(sorted),
                    [&](const ScoredMove& sm) {
                        return sm.move == available_tt_move_;
                    });
                    if (it != std::end(sorted)) {
                        it->score = TT_SCORE;
                        tt_move_ = available_tt_move_;
                        return available_tt_move_;
                    }
                }

                [[fallthrough]];
            case Pick::score:
                pick_ = Pick::captures;

                score();
                //for (auto [move, score] : prioritized) {
                //    fmt::print("  move: {}, score: {}\n", move, score);
                //}
                [[fallthrough]];
            case Pick::captures: {
                while (played_ < legal.size()) {
                    auto index = played_;
                    for (auto i = 1 + index; i < sorted.size(); i++) {
                        if (sorted[i].score > sorted[index].score) {
                            index = i;
                        }
                    }

                    if (sorted[index].score < CAPTURE_SCORE) {
                        break;
                    }

                    std::swap(sorted[index], sorted[played_]);

                    if (sorted[played_].move != tt_move_) {
                        return sorted[played_++].move;
                    }

                    played_++;
                }

                if constexpr (ST == QSEARCH) {
                    return 0;
                }

                pick_ = Pick::KILLERS_1;
                [[fallthrough]];
            }
            case Pick::KILLERS_1:
                pick_ = Pick::KILLERS_2;

                if (killer_move_1_ != 0) {
                    return killer_move_1_;
                }

                [[fallthrough]];
            case Pick::KILLERS_2:
                pick_ = Pick::COUNTER;

                if (killer_move_2_ != 0) {
                    return killer_move_2_;
                }

                [[fallthrough]];
            case Pick::COUNTER:
                pick_ = Pick::QUIET;

                if (counter_move_ != 0) {
                    return counter_move_;
                }

                [[fallthrough]];
            case Pick::QUIET:
                while (played_ < sorted.size()) {
                    auto index = played_;
                    for (auto i = 1 + index; i < sorted.size(); i++) {
                        if (sorted[i].score > sorted[index].score) {
                            index = i;
                        }
                    }

                    std::swap(sorted[index], sorted[played_]);

                    if (sorted[played_].move != tt_move_ &&
                        sorted[played_].move != killer_move_1_ &&
                        sorted[played_].move != killer_move_2_ &&
                        sorted[played_].move != counter_move_) {
                        assert(sorted[played_].score < COUNTER_SCORE);

                        return sorted[played_++].move;
                    }

                    played_++;
                }

                return 0;

            default:
                return 0;
        }
    }

    [[nodiscard]] int score_move(const Move move)
    {
        if constexpr (ST == QSEARCH) {
            if (move == available_tt_move_) {
                return TT_SCORE;
            } else if (move.dst_piece() != no_piece_type) {
                return CAPTURE_SCORE + mvvlva(move);
            }
        }

        if (worker_.killers[0] == move) {
            killer_move_1_ = move;
            return KILLER_ONE_SCORE;
        } else if (worker_.killers[1] == move) {
            killer_move_2_ = move;
            return KILLER_TWO_SCORE;
        }
        return 0;
    }


    const Movelist legal;
    std::vector<ScoredMove> sorted {};

private:
    enum class Pick { tt, score, captures, KILLERS_1, KILLERS_2, COUNTER, QUIET };

    const Worker & worker_;
    const Stack *ss_;

    size_t played_ = 0;

    Pick pick_ = Pick::tt;

    Move available_tt_move_ = 0;

    Move tt_move_ = 0;
    Move killer_move_1_ = 0;
    Move killer_move_2_ = 0;
    Move counter_move_ = 0;
};

} // enyo ns
