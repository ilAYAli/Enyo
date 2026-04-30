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
#include "see.hpp"

using namespace enyo;
using namespace eventlog;

namespace enyo {

namespace {

void update_history_score(int16_t & entry, int bonus)
{
    constexpr int max_history = 16384;
    bonus = std::clamp(bonus, -max_history, max_history);
    entry += static_cast<int16_t>(bonus - (entry * std::abs(bonus)) / max_history);
}

}

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
        const auto mv = b.history[b.histply -1].move;
        if (mv.flags() == Move::Flags::promote)
            return enyo::HCE_evaluation<Us>(b);
        const auto score = enyo::HCE_evaluation<Us>(b);
        if constexpr (Constexpr::debug_eval)
            fmt::print("<{}> move: {}, score: {}, repeat: {}, key: {:016X}, fen: {}\n",
                Us, b.history[b.histply -1].move, score, is_repetition(b), b.hash, b.fen());
        return score;
    }
}

template <Color Us, NodeType Node>
Value qsearch(Board & b, Worker & worker, Stack * ss, int depth, int alpha, int beta)
{
    if (worker.time_expired()) {
        return Value::draw;
    }

    constexpr bool pv_node = Node == NodeType::PV;
    constexpr Color Them = ~Us;

    auto & si = worker.si;
    if (ss->ply >= MAX_PLY) {
        return evaluate<Us, true>(b, &si.nnue);
    }

    if (is_repetition(b, 1 + pv_node)) {
        return Value::draw;
    }

    Move tt_move {};
    auto tt_value = Value::none;
    auto tthit = tt::ttable.probe(b.hash);
    ss->tthit = tthit.has_value();
    if (ss->tthit) {
        tt_value = tt::value_from(tthit->value, ss->ply);
        tt_move = tthit->move;

        if (Node != NodeType::PV && tt_value != Value::none) {
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

    auto const lm = generate_legal_moves<Us>(b);
    auto const mp = prioritize_moves<Us, QSEARCH>(worker, lm, tt_move, depth);
    Move best_move {};
    for (auto move : mp) {
        if ((si.nodes & 1023U) == 0 && worker.time_expired())
            return Value::draw;

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
    if (!thread::pool.stop.load(std::memory_order_relaxed) && best_move)
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
    if (worker.time_expired()) {
        return Value::draw;
    }

    constexpr Color Them = ~Us;
    constexpr bool pv_node = NT != NodeType::NonPV;
    auto & b = worker.si.board;
    auto & si = worker.si;
    Value value = Value::none;
    Move best_move {};
    Value best_value = -Value::infinite;

    if (ss->ply >= MAX_PLY) {
        ss->in_check = is_check<Us>(b);
        return !ss->in_check
            ? evaluate<Us, true>(b, &si.nnue)
            : Value::draw;
    }

    worker.pvline.setlen(ss->ply);

    ss->in_check = is_check<Us>(b);
    ss->eval = Value::none;

    if (depth <= 0) {
        return qsearch<Us, NT != NodeType::NonPV ? NodeType::PV : NodeType::NonPV>(
            b, worker, ss, depth, alpha, beta
        );
    }

    if (NT != NodeType::Root) {
        if (is_repetition(b, 1 + pv_node)) {
            return Value::draw;
        }

        alpha = std::max(alpha, mated_in(ss->ply));
        beta = std::min(beta, mate_in(ss->ply));
        if (alpha >= beta) {
            return alpha;
        }
    }

    // tt lookup:
    Move tt_move {};
    auto tt_value = -Value::none;
    auto tte = tt::ttable.probe(b.hash);
    ss->tthit = tte.has_value();
    if (ss->tthit) {
        tt::ttable.hit++;
        tt_value = tt::value_from(tte->value, ss->ply);
        tt_move = tte->move;
        if (tt_value != Value::none && tte->depth >= depth) {
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
        const auto tb_max = static_cast<int>(syzygy::largest());
        if (NT != NodeType::Root
            && tb_max > 0
            && num_pieces <= tb_max
            && !board.gamestate.can_castle(CastlingRights::any_castling)) {

            const auto status = syzygy::WDL_probe(board);
            if (status != syzygy::Status::Error) {
                Value tb_value = Value::draw;
                auto tb_flag = tt::type::ExactBound;
                switch (status) {
                    case syzygy::Status::Win:
                        tb_value = Value::tb_win_in_max_ply;
                        tb_flag = tt::type::LowerBound;
                        break;
                    case syzygy::Status::Loss:
                        tb_value = Value::tb_loss_in_max_ply;
                        tb_flag = tt::type::UpperBound;
                        break;
                    default: // Draw
                        break;
                }

                if (tb_flag == tt::type::ExactBound
                || (tb_flag == tt::type::LowerBound && tb_value >= beta)
                || (tb_flag == tt::type::UpperBound && tb_value <= alpha)) {
                    tt::ttable.store(
                        b.hash,
                        Move{},
                        tt::value_to(tb_value, ss->ply),
                        tb_flag,
                        std::min(depth + tb_max, MAX_PLY)
                    );
                    return tb_value;
                }
            }
        }
    }

    // depth extension
    bool improving = true;
    if (ss->in_check) {
        improving = false;
        ss->eval = Value::draw;
        depth++;
        goto moves_loop;
    }

    // static eval:
    if (ss->tthit) {
        ss->eval = tt_value;
        if (ss->eval == Value::none)
            ss->eval =  evaluate<Us, true>(b, &si.nnue);

        if (tt_value != Value::none
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

    improving = (ss - 2)->eval != Value::none
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
        if ((si.nodes & 1023U) == 0 && worker.time_expired())
            return Value::draw;

        const bool have_big_pieces = static_cast<bool>(
            b.pt_bb[Us][knight]
          | b.pt_bb[Us][bishop]
          | b.pt_bb[Us][rook]
          | b.pt_bb[Us][queen]
        );
        const bool tt_rules_out_nmp = ss->tthit
            && tt_value != Value::none
            && (tte->flag & tt::type::UpperBound)
            && tt_value < beta;

        if (have_big_pieces
            && depth >= 3
            && ss->eval >= beta
            && (ss-1)->move != Move{}
            && !tt_rules_out_nmp
            && std::abs(beta) < Constexpr::mate_value - MAX_PLY) {

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
    auto const lm = (NT == NodeType::Root && !si.searchmoves.empty())
        ? si.searchmoves
        : generate_legal_moves<Us>(b);
    if (lm.empty()) {
        if (ss->in_check) {
            auto prev_move = b.history[b.histply-1].move;
            if constexpr (false) {
                fmt::print("mated in {}, prev move: {}, depth: {}, ply: {}  fen: {}\n",
                    mate_in_moves(mated_in(ss->ply)), prev_move, depth, ss->ply, b.fen());
            }
            return mated_in(ss->ply);
        }
        return Value::draw;
    }
#if 1
    const Move prev_move = (ss-1)->move;
    const Move cm = prev_move ? worker.countermove[Us][prev_move.src_sq()][prev_move.dst_sq()] : Move{};
    auto const mp = prioritize_moves<Us, ABSEARCH>(worker, lm, tt_move, depth, ss->killers, cm);
#else
    auto mp = MovePicker2<Us>(worker, lm, tt_move);
#endif

    bool do_fullsearch = false;
    ss->move_count = 0;
    for (const auto move : mp) {
    //while (const auto move = mp.next()) {
        if ((si.nodes & 1023U) == 0 && worker.time_expired()) {
            return Value::draw;
        }

        si.nodes++;
        ss->move = move;
        ss->move_count++;

       const bool is_quiet = move.dst_piece() == no_piece_type && move.flags() != Move::Flags::promote;
       const bool is_capture = move.dst_piece() != no_piece_type;

        // Futility pruning: skip quiet moves at shallow depth when static
        // eval plus a depth-scaled margin is still below alpha.
        if (!pv_node
            && !ss->in_check
            && is_quiet
            && depth <= 6
            && ss->move_count > 1
            && ss->eval != Value::none
            && ss->eval + 120 * depth <= alpha
            && std::abs(alpha) < Constexpr::mate_value - MAX_PLY) {
            continue;
        }

        // todo: Extensions
        int extension = 0;
        int new_depth = depth -1 + extension;

        // make move
        apply_move<Us, true, true>(b, move, &worker.si.nnue);

        // LMR: Late Move Reductions
        if (depth >= 3 && !ss->in_check && ss->move_count > 3 + 2 * pv_node) {
            int R = lmr_reductions[depth][ss->move_count];

            R += !improving;
            R -= NT != NodeType::NonPV;
            R -= is_capture;
            R = std::clamp(new_depth - R, 1, new_depth + 1);

            if (worker.time_expired()) {
                revert_move<Us, true, true>(b, &worker.si.nnue);
                return Value::draw;
            }

            value = -negamax<Them, NodeType::NonPV>(R, worker, ss + 1, -alpha -1, -alpha);

            do_fullsearch = value > alpha && R < new_depth;
        } else {
            do_fullsearch = !pv_node || ss->move_count > 1;
        }

        if (do_fullsearch) {
            if (worker.time_expired()) {
                revert_move<Us, true, true>(b, &worker.si.nnue);
                return Value::draw;
            }

            value = -negamax<Them, NodeType::NonPV>(new_depth, worker, ss + 1, -alpha -1, -alpha);
        }

        // PVS: Principal Variation Search
        if (NT != NodeType::NonPV && ((value > alpha && value < beta) || ss->move_count == 1)) {
            if (worker.time_expired()) {
                revert_move<Us, true, true>(b, &worker.si.nnue);
                return Value::draw;
            }

            value = -negamax<Them, NodeType::PV>(new_depth, worker, ss + 1, -beta, -alpha);
        }

        // revert move
        revert_move<Us, true, true>(b, &worker.si.nnue);
        if (thread::pool.stop.load(std::memory_order_relaxed))
            return Value::draw;

        if (value > best_value) {
            best_value = value;

            if (value > alpha) {
                alpha = value;
                best_move = move;

                worker.pvline.setmove(move, ss->ply);

                if constexpr (NT == NodeType::Root) {
                    eventlog::log<eventlog::Log::error>(
                        "ROOT setmove: depth={} move={} value={} alpha={} beta={} move_count={} table[0][0]={} len[0]={} pv='{}'\n",
                        depth,
                        move,
                        value,
                        alpha,
                        beta,
                        ss->move_count,
                        worker.pvline.table[0][0],
                        static_cast<int>(worker.pvline.len[0]),
                        worker.pvline.str());
                }

                if (value >= beta) {
                    if (is_quiet) {
                        if (move != ss->killers[0]) {
                            ss->killers[1] = ss->killers[0];
                            ss->killers[0] = move;
                        }

                        if (prev_move != Move{}) {
                            worker.countermove[Us][prev_move.src_sq()][prev_move.dst_sq()] = move;
                        }

                        const int bonus = std::min(1600, depth * depth * 32);
                        update_history_score(worker.history[Us][move.src_sq()][move.dst_sq()], bonus);
                        for (int i = 0; i < ss->move_count - 1; ++i) {
                            const auto prev = mp[static_cast<size_t>(i)];
                            if (prev.dst_piece() == no_piece_type && prev.flags() != Move::Flags::promote)
                                update_history_score(worker.history[Us][prev.src_sq()][prev.dst_sq()], -bonus / 2);
                        }
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


    if (!thread::pool.stop.load(std::memory_order_relaxed) && best_move) {
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
    constexpr Value infinite = Value::infinite;
    constexpr Value mate_value = Value::mate;
    constexpr auto aspiration_depth = 5;

    if (depth < aspiration_depth || std::abs(prev_eval) >= mate_value / 2) {
        return negamax<Us, NodeType::Root>(depth, worker, ss, -infinite, infinite);
    }

    Value alpha = prev_eval - initial_delta;
    Value beta = prev_eval + initial_delta;
    Value delta = initial_delta;
    Value score = Value::draw;

    while (true) {
        score = negamax<Us, NodeType::Root>(depth, worker, ss, alpha, beta);

        if (thread::pool.stop.load(std::memory_order_relaxed)) {
            return Value::draw;
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
    auto & board = si.board;
    Stack stack[MAX_PLY + 5];
    stack[0].ply = 4;
    stack[1].ply = 3;
    stack[2].ply = 2;
    stack[3].ply = 1;
    for (int i = 0; i <= MAX_PLY; i++)
        stack[i + 4].ply = i;
    Stack *ss = stack + 4;

    const auto legal_fallback = board.side == white
        ? generate_legal_moves<white>(board)
        : generate_legal_moves<black>(board);

    const auto is_legal_root_move = [&](Move move) {
        return std::ranges::find(legal_fallback, move) != legal_fallback.end();
    };

    tt::ttable.prepare();
    worker.pvline.clear();

    if (worker.id == 0) {
        eventlog::log<eventlog::Log::error>(
            "search_position start: fen={}, legal_moves={}, legal[0]={}\n",
            board.fen(),
            legal_fallback.size(),
            legal_fallback.empty() ? Move{} : legal_fallback[0]);
    }

    // Root DTZ probe: if the position is fully covered by a loaded tablebase,
    // skip search entirely and emit the tablebase's guaranteed-progress move.
    // This is what prevents the engine from blundering in "obviously drawn"
    // or "obviously won" endgames — Fathom knows the DTZ-optimal move; the
    // NNUE-driven search is blind to it once the horizon passes.
    if (worker.id == 0 && Constexpr::use_syzygy) {
        const auto num_pieces = count_bits(board.color_bb[white] | board.color_bb[black]);
        const auto tb_max = static_cast<int>(syzygy::largest());
        if (tb_max > 0
         && num_pieces <= tb_max
         && !board.gamestate.can_castle(CastlingRights::any_castling)) {
            syzygy::Status tb_status = syzygy::Status::Error;
            auto [tb_score, tb_move] = syzygy::DTZ_probe(board, tb_status);
            if (tb_status != syzygy::Status::Error
             && tb_move
             && is_legal_root_move(tb_move)) {
                const char* verdict =
                    tb_status == syzygy::Status::Win  ? "win"
                  : tb_status == syzygy::Status::Loss ? "loss"
                                                      : "draw";
                ucilog("info depth 1 score cp {} string tbhit {}\n", tb_score, verdict);
                ucilog("bestmove {}\n", tb_move);
                return;
            }
        }
    }

    struct Mate {
        int moves { MAX_PLY };
        Move move {};
    } shortest_mate;

    Value value = Value::draw;
    uint64_t prev_nodes {};
    auto const max_depth = std::min(si.depth, MAX_PLY);
    for (auto depth = 1; depth <= max_depth; ++depth) {
        if (worker.time_expired())
            break;

        // Soft limit: stop starting new iterations once the optimum budget is
        // used. The current-best move is already committed, so this gives us
        // cheap, principled "stop between iterations" behavior. The hard
        // limit (watchdog + time_expired) still aborts mid-iteration if we
        // blow through the emergency budget.
        if (worker.id == 0 && depth > 1 && si.soft_time_expired()) {
            thread::pool.stop = true;
            break;
        }

        prev_nodes = thread::pool.get_nodes();
        si.nodes = 0;
        si.depth = depth;

        if constexpr (Constexpr::debug_threads)
            fmt::print("<{}> thread: {}, depth: {}\n", __func__, worker.id, depth);

        if (worker.id == 0) {
            eventlog::log<eventlog::Log::error>(
                "ITER begin: depth={} table[0][0]={} len[0]={} worker.bestmove={} pv='{}'\n",
                depth,
                worker.pvline.table[0][0],
                static_cast<int>(worker.pvline.len[0]),
                worker.bestmove,
                worker.pvline.str());
        }

        value = Constexpr::use_aspiration_window
            ? (si.board.side == white
                ? aspiration_window<white>(value, depth, worker, ss)
                : aspiration_window<black>(value, depth, worker, ss))
            : (si.board.side == white
                ? negamax<white, NodeType::Root>(depth, worker, ss)
                : negamax<black, NodeType::Root>(depth, worker, ss));

        if (worker.id == 0) {
            eventlog::log<eventlog::Log::error>(
                "ITER end: depth={} value={} table[0][0]={} len[0]={} pv='{}'\n",
                depth,
                value,
                worker.pvline.table[0][0],
                static_cast<int>(worker.pvline.len[0]),
                worker.pvline.str());
        }

        if (worker.time_expired())
            break;

        if (worker.id)
            continue;

        const auto pvbm = worker.pvline.bestmove();
        auto mate_distance = mate_in_moves(value);
        const auto prev_bestmove = worker.bestmove;
        if (pvbm) {
            if (mate_distance > 0 && mate_distance < shortest_mate.moves)
                shortest_mate = {mate_distance, pvbm};
            worker.bestmove = pvbm;
        } else {
            eventlog::log<eventlog::Log::error>(
                "ERROR: pvbm empty at depth {}. pv_str='{}' len[0]={} table[0][0]={} prev_bestmove={} score={} fen={}\n",
                depth,
                worker.pvline.str(),
                static_cast<int>(worker.pvline.len[0]),
                worker.pvline.table[0][0],
                prev_bestmove,
                value,
                board.fen());
        }

        if (!is_legal_root_move(worker.bestmove)) {
            eventlog::log<eventlog::Log::error>(
                "ERROR: worker.bestmove={} is NOT legal at root, depth={}. pvbm={} prev_bestmove={} pv_str='{}' len[0]={} fen={}\n",
                worker.bestmove,
                depth,
                pvbm,
                prev_bestmove,
                worker.pvline.str(),
                static_cast<int>(worker.pvline.len[0]),
                board.fen());
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

        ucilog("{}\n", info_string);

        // not making progress:
        if (prev_nodes == thread::pool.get_nodes())
            break;

        if (shortest_mate.moves == 1) {
            eventlog::log<eventlog::Log::info>("Breaking search loop: found mate in 1, shortest_mate.move={}, worker.bestmove={}\n", 
                shortest_mate.move, worker.bestmove);
            break;
        }
    }
    
    eventlog::log<eventlog::Log::info>("After search loop: worker.id={}, shortest_mate.move={}, worker.bestmove={}\n",
        worker.id, shortest_mate.move, worker.bestmove);
    
    if (worker.id == 0) {
        if (si.has_searchmoves)
            ucilog("info string forced score {}\n", value);

        Move out = is_legal_root_move(shortest_mate.move) ? shortest_mate.move : worker.bestmove;
        const Move pre_fallback_out = out;

        eventlog::log<eventlog::Log::info>("Before bestmove output: out={}, is_legal={}\n",
            out, is_legal_root_move(out));

        if (!is_legal_root_move(out) && !legal_fallback.empty()) {
            eventlog::log<eventlog::Log::error>(
                "ERROR: bestmove fallback fired. pre_fallback_out={} worker.bestmove={} shortest_mate.move={} "
                "pvline.bestmove()={} pv_str='{}' len[0]={} legal_fallback[0]={} fen={}\n",
                pre_fallback_out,
                worker.bestmove,
                shortest_mate.move,
                worker.pvline.bestmove(),
                worker.pvline.str(),
                static_cast<int>(worker.pvline.len[0]),
                legal_fallback[0],
                board.fen());
            out = legal_fallback[0];
        }

        eventlog::log<eventlog::Log::info>("Outputting bestmove: {}\n", out);
        ucilog("bestmove {}\n", out);
        eventlog::log<eventlog::Log::info>("Successfully output bestmove\n");
    } else {
        eventlog::log<eventlog::Log::error>("ERROR: worker.id={} is not 0, not outputting bestmove!\n", worker.id);
    }
}

} // enyo ns
