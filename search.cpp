#include "search.hpp"
#include <chrono>
#include <algorithm>
#include <thread>
#include <iostream>
#include <unistd.h>
#include "fmt/core.h"
#include "fmt/format.h"

#include "nnue.hpp"
#include "precalc/knight_attacks.hpp"
#include "probe.hpp"
#include "types.hpp"
#include "uci.hpp"
#include "util.hpp"
#include "movegen.hpp"
#include "movepicker.hpp"
#include "config.hpp"
#include "tt.hpp"
#include "hce.hpp"

using namespace enyo;
using namespace eventlog;

namespace enyo {

int lmr_reductions[MAX_PLY][MAX_MOVES];

void init_search() {
    constexpr int lmr_divisor = 224;
    constexpr int lmr_base = 111;
    lmr_reductions[0][0] = 0;
    for (int depth = 1; depth < MAX_PLY; depth++) {
        for (int move = 1; move < MAX_MOVES; move++) {
            lmr_reductions[depth][move] =
                static_cast<int>(
                std::max(0.0, std::log(depth)
                    * std::log(move) / double(lmr_divisor / 100.0)
                    + double(lmr_base / 100.0)
                ));
        }
    }
}

bool is_repetition(const Board & b, int draw)
{
    if (b.half_moves < 2)
        return false;

    const auto end = b.histply;
    const auto begin = end - b.half_moves;
    return static_cast<std::size_t>(
        std::ranges::count_if(
            std::begin(b.history) + begin,
            std::begin(b.history) + end,
            [&](auto const & entry) {
                return entry.hash == b.hash;
            }
        )
    ) > static_cast<size_t>(draw);
}

template <Color Us, bool UseNNUE = true>
Value evaluate(Board & b, NNUE::Net * nnue)
{
    if constexpr (UseNNUE) {
        auto const score = static_cast<Value>(nnue->Evaluate(Us));
        if constexpr (Constexpr::debug_eval)
            fmt::print("<{}> move: {}, score: {}\n", Us, b.history[b.histply -1].move, score);
        return score;
    } else {
        auto mv = b.history[b.histply -1].move;
        if (mv.get_flags() == Move::Flags::Promote)
            return enyo::HCE_evaluation<Us>(b);
        const auto score = enyo::HCE_evaluation<Us>(b);
        if constexpr (Constexpr::debug_eval)
            fmt::print("<{}> move: {}, score: {}, repeat: {}, key: {:016X}, fen: {}\n",
                Us, b.history[b.histply -1].move, score, is_repetition(b), b.hash, b.fen());
        return score;
    }
}

// Static Exchange Evaluation
template <Color Us, NodeType Node>
inline bool see(Board & b, Move move, int threshold)
{
    const auto src = move.get_src();
    const auto dst = move.get_dst();
    const auto src_piece = move.get_src_piece();
    const auto dst_piece = move.get_dst_piece();
    constexpr auto Them = ~Us;
}

template <Color Us, NodeType Node>
Value qsearch(Board & b, Worker & worker, Stack * ss, int depth, int alpha, int beta)
{
    if (worker.time_expired())
        return Value::DRAW;

    constexpr bool pv_node = Node == NodeType::PV;
    constexpr Color Them = ~Us;

    auto & si = worker.si;
    if (ss->ply >= MAX_PLY) {
        return evaluate<Us, true>(b, &si.nnue);
    }

    if (is_repetition(b, 1 + pv_node)) {
        return Value::DRAW;
    }

    Move tt_move {};
    auto tt_value = Value::NONE;
    auto tthit = tt::ttable.probe(b.hash);
    ss->tthit = tthit.has_value();
    if (ss->tthit) {
        tt_value = tt::value_from(tthit->value, ss->ply);
        tt_move = tthit->move;

        if (Node != NodeType::PV && tt_value != Value::NONE) {
            if (tthit->flag == tt::type::ExactBound
            || (tthit->flag == tt::type::UpperBound && (tt_value <= alpha))
            || (tthit->flag == tt::type::LowerBound && (tt_value >= beta)))
                return tt_value;
        }
    }

    auto best_value = ss->eval = evaluate<Us, true>(b, &si.nnue);
    if (best_value >= beta) {
        if (!ss->tthit) {
            tt::ttable.store(
                b.hash,
                Move{},
                tt::value_to(best_value, ss->ply),
                tt::type::UpperBound,
                depth
            );
        }
        return best_value;
    }

    if (best_value > alpha)
        alpha = best_value;

#if 1
    MovePicker<Us, QSEARCH> mp(worker, ss, tt_move);
    //auto const moves = mp.filter_captures();
#else
    auto const moves = filter_captures(generate_legal_moves<Us>(b), tt_move);
#endif

    Move move {};
    Move best_move {};
    while ((move = mp.next_move())) {
        si.nodes++;

        apply_move<Us, true, true>(b, move, &si.nnue);
        auto score = -qsearch<Them, Node>(b, worker, ss + 1, depth +1, -beta, -alpha);
        revert_move<Us, true, true>(b, &si.nnue);

        if constexpr (Constexpr::debug_qsearch)
            fmt::print("qsearch: {} depth: {} move: {}, score: {}, repeat: {}, poskey: {:016X}, fen: {}\n",
                Us, depth, move, score, is_repetition(b), b.hash, b.fen());

        if (score > best_value) {
            best_value = score;
            if (score > alpha) {
                best_move = move;

                if (score < beta)
                    alpha = score;
                else
                    break;
            }
        }
    }
    if (!thread::pool.stop.load(std::memory_order_relaxed))
        tt::ttable.store(
            b.hash,
            best_move,
            tt::value_to(best_value, ss->ply),
            best_value >= beta
                ? tt::type::LowerBound
                : tt::type::UpperBound,
            depth
        );

    return best_value;
}

template <Color Us, NodeType NT>
Value negamax(int depth, Worker & worker, Stack * ss, Value alpha, Value beta)
{
    if (worker.time_expired())
        return Value::DRAW;

    constexpr Color Them = ~Us;
    constexpr bool pv_node = NT != NodeType::NonPV;
    auto & b = worker.si.board;
    auto & si = worker.si;
    Value value = Value::NONE;
    Move best_move {};
    Value best_value = -Value::INFINITE;

    worker.pvline.setlen(ss->ply);

    ss->in_check = is_check<Us>(b);
    ss->eval = Value::NONE;

    if (depth <= 0) {
        return qsearch<Us, NT != NodeType::NonPV ? NodeType::PV : NodeType::NonPV>(
            b, worker, ss, depth, alpha, beta
        );
    }

    if (NT != NodeType::Root) {
        if (is_repetition(b, 1 + pv_node)) {
            return Value::DRAW;
        }

        if (ss->ply >= MAX_PLY) {
            return !ss->in_check
                ? evaluate<Us, true>(b, &si.nnue)
                : Value::DRAW;
        }

        alpha = std::max(alpha, mated_in(ss->ply));
        beta = std::min(beta, mate_in(ss->ply));
        if (alpha >= beta) {
            return alpha;
        }
    }

    // tt lookup:
    Move tt_move {};
    auto tt_value = -Value::NONE;
    auto tte = tt::ttable.probe(b.hash);
    ss->tthit = tte.has_value();
    if (ss->tthit) {
        tt::ttable.hit++;
        tt_value = tt::value_from(tte->value, ss->ply);
        tt_move = tte->move;
        if (tt_value != Value::NONE && tte->depth >= depth) {
            if (tte->flag == tt::type::ExactBound
            || (tte->flag == tt::type::UpperBound && tt_value <= alpha)
            || (tte->flag == tt::type::LowerBound && tt_value >= beta)) {
                tt::ttable.cut++;
                return tt_value;
            }
        }
    }

    // tb lookup:
    if constexpr (Constexpr::use_syzygy) {
        auto & board = worker.si.board;
        const auto num_pieces = count_bits(board.color_bb[Us] | board.color_bb[Them]);
        constexpr auto cardinality = 6; // Todo:
        if (NT != NodeType::Root
            && num_pieces <= cardinality
            && !board.gamestate.can_castle(CastlingRights::any_castling)) {

            using namespace syzygy;
            Value tb_value = Value::DRAW;
            auto tb_flag = tt::type::NoneBound;
            if (auto status = WDL_probe(si.board) != syzygy::Status::Error) {
                switch (WDL_probe(si.board)) {
                    case Status::Win:
                        tb_value = mate_in(ss->ply);
                        tb_flag = tt::type::LowerBound;
                        break;
                    case Status::Loss:
                        tb_value = mated_in(ss->ply);
                        tb_flag = tt::type::UpperBound;
                        break;
                    default:
                        tb_value = Value::DRAW;
                        tb_flag = tt::type::ExactBound;
                        break;
                }
                value = tb_value;

                if (tb_flag == tt::type::ExactBound
                || (tb_flag == tt::type::LowerBound && value >= beta)
                || (tb_flag == tt::type::UpperBound && value <= alpha)) {
                    tt::ttable.store(
                        b.hash,
                        Move{},
                        tt::value_to(tt_value, ss->ply),
                        best_value >= beta
                            ? tt::type::LowerBound
                            : tt::type::UpperBound,
                        std::min(depth + cardinality, MAX_PLY)
                    );
                    return value;
                }

                if (NT != NodeType::NonPV) {
                    if (tte->flag == tt::type::LowerBound) {
                        best_value = value;
                        alpha = std::max(alpha, best_value);
                    } else {
                        best_value = value;
                    }
                }
            }
        }
    }

    // depth extension
    bool improving = true;
    if (ss->in_check) {
        improving = false;
        ss->eval = Value::DRAW;
        depth++;
        goto moves_loop;
    }

    // static eval:
    if (ss->tthit) {
        ss->eval = tt_value;
        if (ss->eval == Value::NONE)
            ss->eval =  evaluate<Us, true>(b, &si.nnue);

        if (tt_value != Value::NONE
            && (tte->flag != tt::type::NoneBound)
            & (tt_value > ss->eval ? tt::type::LowerBound : tt::type::UpperBound))
            ss->eval = tt_value;
    } else {
        ss->eval = evaluate<Us, true>(b, &si.nnue);
        tt::ttable.store(
            b.hash,
            Move{},
            ss->eval,
            tt::type::NoneBound,
            0
        );
    }

    improving = (ss - 2)->eval != Value::NONE
             && (ss - 2)->eval < ss->eval;

    if constexpr (true) { // Todo: check Elo
        if (NT == NodeType::NonPV && !ss->tthit)
            depth--;

        if (depth <= 0)
            return qsearch<Us, NodeType::PV>(b, worker, ss, depth, alpha, beta);
    }

    // IIR: Internal Iterative Reductions
    if constexpr (Constexpr::use_iir) {
        if (depth >= 3 && !ss->tthit)
            depth--;
    }

    // razoring:
    if constexpr (Constexpr::use_razoring) {
        constexpr int razor_depth = 3;
        constexpr int razor_margin = 63;
        constexpr int depth_factor = 182;
        if (depth < razor_depth
        && (ss->eval - razor_margin + (depth_factor * depth)) < alpha) {
            return qsearch<Us, NodeType::NonPV>(b, worker, ss, depth, alpha, beta);
        }
    }

    if constexpr (Constexpr::use_rfp) {
        if (std::abs(beta) <  Constexpr::mate_value - 2 * MAX_PLY)
            if (depth < 7 && ss->eval - 64 * depth + 71 * improving >= beta)
                return beta;
    }

    // todo: futility pruning

    // null move search
    if (Constexpr::use_nullmove) {
        const bool have_big_pieces = static_cast<bool>(
            b.pt_bb[Us][knight]
          | b.pt_bb[Us][bishop]
          | b.pt_bb[Us][rook]
          | b.pt_bb[Us][queen]
        );
        if (have_big_pieces
            && depth >= 3
            && (ss->eval >= beta)
            && (ss-1)->move != Move{}) {

            int R = 5 + std::min(4, depth / 5) + std::min(Value(3), (ss->eval - beta) / 214);
            apply_null_move<Us>(b);
            auto nullscore = -negamax<Them, NodeType::NonPV>(depth -R, worker, ss + 1, -beta, -beta + 1);
            ss->move = Move{};
            revert_null_move<Us>(b);

            if (nullscore >= beta) {
                if (nullscore >= (Constexpr::mate_value - MAX_PLY))
                    nullscore = beta;
                return beta;
            }
        }
    }

    // todo: Internal iterative reductions
    // todo: ProbCut

moves_loop:
    auto const lm = generate_legal_moves<Us>(b);
    if (lm.empty()) {
        if (ss->in_check) {
            auto prev_move = b.history[b.histply-1].move;
            if constexpr (false) {
                fmt::print("mated in {}, prev move: {}, depth: {}, ply: {}  fen: {}\n",
                    mate_in_moves(mated_in(ss->ply)), prev_move, depth, ss->ply, b.fen());
            }
            return mated_in(ss->ply);
        }
        return Value::DRAW;
    }
    auto const mp = prioritize_moves<Us>(worker, lm, tt_move);

    bool do_fullsearch = false;
    ss->move_count = 0;
    for (const auto move : mp) {
        si.nodes++;
        ss->move = move;
        ss->move_count++;

        const bool is_quiet = move.get_dst_piece() == no_piece_type && move.get_flags() != Move::Flags::Promote;

        // todo: Pruning at shallow depth
        // todo: Extensions
        int extension = 0;
        int new_depth = depth -1 + extension;

        // make move
        apply_move<Us, true, true>(b, move, &worker.si.nnue);

        const bool is_capture = move.get_dst_piece() != no_piece_type;

        // LMR: Late Move Reductions
        if (depth >= 3 && !ss->in_check && ss->move_count > 3 + 2 * pv_node) {
            int R = lmr_reductions[depth][ss->move_count];

            R += !improving;
            R -= NT != NodeType::NonPV;
            R -= is_capture;
            R = std::clamp(new_depth - R, 1, new_depth + 1);

            value = -negamax<Them, NodeType::NonPV>(R, worker, ss + 1, -alpha -1, -alpha);

            do_fullsearch = value > alpha && R < new_depth;
        } else {
            do_fullsearch = !pv_node || ss->move_count > 1;
        }

        if (do_fullsearch) {
            value = -negamax<Them, NodeType::NonPV>(new_depth, worker, ss + 1, -alpha -1, -alpha);
        }

        // PVS: Principal Variation Search
        if (NT != NodeType::NonPV && ((value > alpha && value < beta) || ss->move_count == 1)) {
            value = -negamax<Them, NodeType::PV>(new_depth, worker, ss + 1, -beta, -alpha);
        }

        // revert move
        revert_move<Us, true, true>(b, &worker.si.nnue);
        if (thread::pool.stop.load(std::memory_order_relaxed))
            return Value::DRAW;

        if (value > best_value) {
            best_value = value;

            if (value > alpha) {
                alpha = value;
                best_move = move;

                worker.pvline.setmove(move, ss->ply);

                if (value >= beta) {
                    if (is_quiet) {
                        worker.killers[1] = worker.killers[0];
                        worker.killers[0] = move;
                    }

                    tt::ttable.store(
                        b.hash,
                        best_move,
                        tt::value_to(best_value, ss->ply),
                        tt::type::LowerBound,
                        depth
                    );
                    return beta;
                }
            }
        }
    }


    if (!thread::pool.stop.load(std::memory_order_relaxed)) {
        tt::ttable.store(
            b.hash,
            best_move,
            tt::value_to(best_value, ss->ply),
            best_value >= beta
                ? tt::type::LowerBound
                : tt::type::UpperBound,
            depth
        );
    }

    return alpha;
}


namespace {

template <Color Us>
Value aspiration_window(Value prev_eval, int depth, Worker & worker, Stack * ss)
{
    constexpr Value initial_delta = static_cast<Value>(12);
    constexpr Value infinite = Value::INFINITE;
    constexpr Value mate_value = Value::MATE;
    constexpr auto aspiration_depth = 5;

    if (depth < aspiration_depth || std::abs(prev_eval) >= mate_value / 2) {
        return negamax<Us, NodeType::Root>(depth, worker, ss, -infinite, infinite);
    }

    Value alpha = prev_eval - initial_delta;
    Value beta = prev_eval + initial_delta;
    Value delta = initial_delta;
    Value score = Value::DRAW;

    while (true) {
        score = negamax<Us, NodeType::Root>(depth, worker, ss, alpha, beta);

        if (thread::pool.stop.load(std::memory_order_relaxed)) {
            return Value::DRAW;
        }

        if (score <= alpha) {
            beta = (alpha + beta) / 2;
            alpha = std::max(alpha - delta, -infinite);
        } else if (score >= beta) {
            beta = std::min(beta + delta, infinite);
        } else {
            break;
        }

        delta += delta / Value(2);
    }

    return score;
}

}

void search_position(Worker & worker)
{
    auto & si = worker.si;
    Stack stack[MAX_PLY + 5];
    stack[0].ply = 4;
    stack[1].ply = 3;
    stack[2].ply = 2;
    stack[3].ply = 1;
    for (int i = 0; i < MAX_PLY; i++)
        stack[i + 4].ply = i;
    Stack *ss = stack + 4;

    //tt::ttable.prepare();
    worker.pvline.clear();

    struct Mate {
        int moves { MAX_PLY };
        Move move {};
    } shortest_mate;

    Value value = Value::DRAW;
    uint64_t prev_nodes {};
    auto const max_depth = std::min(si.depth, MAX_PLY);
    for (auto depth = 1; depth <= max_depth; ++depth) {
        prev_nodes = thread::pool.get_nodes();
        si.nodes = 0;
        si.depth = depth;

        if constexpr (Constexpr::debug_threads)
            fmt::print("<{}> thread: {}, depth: {}\n", __func__, worker.id, depth);

        value = Constexpr::use_aspiration_window
            ? (si.board.side == white
                ? aspiration_window<white>(value, depth, worker, ss)
                : aspiration_window<black>(value, depth, worker, ss))
            : (si.board.side == white
                ? negamax<white, NodeType::Root>(depth, worker, ss)
                : negamax<black, NodeType::Root>(depth, worker, ss));

        if (worker.time_expired())
            break;

        if (worker.id)
            continue;

        const auto pvbm = worker.pvline.bestmove();
        auto mate_distance = mate_in_moves(value);
        if (pvbm) {
            if (mate_distance > 0 && mate_distance < shortest_mate.moves)
                shortest_mate = {mate_distance, pvbm};
            worker.bestmove = pvbm;
        }

        const std::string score_info = mate_distance
            ? fmt::format("mate {}", mate_distance)
            : fmt::format("cp {}", value);

        std::string info_string = fmt::format("info depth {} score {} nodes {} nps {} time {} hashfull {} pv {}",
            depth,
            score_info,
            thread::pool.get_nodes(),
            thread::pool.get_nps(),
            std::chrono::duration_cast<std::chrono::milliseconds>(si.elapsed_time).count(),
            tt::ttable.get_hashfull(),
            worker.pvline.str());

        Uci::log("{}\n", info_string);

        // not making progress:
        if (prev_nodes == thread::pool.get_nodes())
            break;

        if (shortest_mate.moves == 1)
            break;
    }
    if (worker.id == 0) {
        Uci::log("bestmove {}\n", shortest_mate.move
            ? shortest_mate.move
            : worker.bestmove);

    }
}

} // enyo ns
