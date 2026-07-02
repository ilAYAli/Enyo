#include "search.hpp"
#include <chrono>
#include <algorithm>
#include <thread>
#include <iostream>
#include <iterator>
#include <unistd.h>
#include "fmt/core.h"
#include "fmt/format.h"

#include "nnue.hpp"
#include "nnue_model.hpp"
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
#include "move_policy.hpp"

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

constexpr int corrhist_grain = 256;
constexpr int corrhist_max = 8192; // 32 cp * grain

// Exponentially-weighted average of (search score - static eval),
// depth-weighted so deeper results move the entry more.
template <Color Us>
void update_correction_history(Worker & worker, const Board & b, int depth, int diff)
{
    const int weight = std::min(depth + 1, 16);
    const auto nudge = [&](int16_t & entry) {
        const int updated = (entry * (256 - weight) + diff * corrhist_grain * weight) / 256;
        entry = static_cast<int16_t>(std::clamp(updated, -corrhist_max, corrhist_max));
    };
    // pawn_hash == 0 means no pawns; skip rather than share one entry.
    if (b.pawn_hash)
        nudge(worker.corrhist[Us][b.pawn_hash & (Worker::corrhist_size - 1)]);
    // hash ^ pawn_hash removes the pawn psq terms from the full zobrist:
    // a non-pawn placement key with no extra board bookkeeping.
    nudge(worker.nonpawn_corrhist[Us][(b.hash ^ b.pawn_hash) & (Worker::corrhist_size - 1)]);
}

// Returns a ply-aware draw score incorporating contempt. At even ply (engine's
// move) draws are penalised; at odd ply the penalty is mirrored so the parent
// node (engine) also sees draws as slightly negative after negation.
Value contempt_draw_score(const Stack * ss)
{
    const int c = cfgmgr.root_repetition_contempt;
    if (c <= 0)
        return Value::draw;
    return static_cast<Value>((ss->ply & 1) ? c : -c);
}

template <Color Us>
bool low_material_check_net(Board const & b)
{
    const int total_pieces = count_bits(b.color_bb[white] | b.color_bb[black]);
    return total_pieces <= 8
        && (b.pt_bb[Us][queen] | b.pt_bb[Us][rook]);
}

template <Color Us>
bool move_gives_check(Board & b, NNUE::Net * nnue, Move move)
{
    apply_move<Us, true, true>(b, move, nnue);
    const bool gives_check = is_check<~Us>(b);
    revert_move<Us, true, true>(b, nnue);
    return gives_check;
}

const char* tablebase_verdict(syzygy::Status status)
{
    switch (status) {
        case syzygy::Status::Win:  return "win";
        case syzygy::Status::Loss: return "loss";
        default:                   return "draw";
    }
}

void log_tablebase_root_hit(syzygy::Status status, int dtz)
{
    const char* verdict = tablebase_verdict(status);
    if (dtz >= 0)
        ucilog("info depth 1 string tbhit {} dtz {}\n", verdict, dtz);
    else
        ucilog("info depth 1 string tbhit {}\n", verdict);
}

const char* tablebase_score_uci(syzygy::Status status)
{
    switch (status) {
        case syzygy::Status::Win:  return "cp 20000";
        case syzygy::Status::Loss: return "cp -20000";
        default:                   return "cp 0";
    }
}

void log_tablebase_root_bestmove(syzygy::Status status, Move move)
{
    ucilog("info depth 1 score {} nodes 1 nps 0 time 0 hashfull {} pv {}\n",
        tablebase_score_uci(status), tt::ttable.get_hashfull(), move);
}

bool side_to_move_in_check(Board const & board)
{
    return board.side == white ? is_check<white>(board) : is_check<black>(board);
}

void log_terminal_root_bestmove(Board const & board)
{
    const int score = side_to_move_in_check(board)
        ? -static_cast<int>(Value::mate)
        : static_cast<int>(Value::draw);
    ucilog("info depth 0 score cp {} nodes 0 nps 0 time 0 hashfull {} pv 0000\n",
        score, tt::ttable.get_hashfull());
    ucilog("bestmove 0000\n");
}

bool is_tablebase_tt_value(Value value)
{
    const auto v = static_cast<int>(value);
    return (v >= static_cast<int>(Value::tb_win_in_max_ply) - MAX_PLY
            && v < static_cast<int>(Value::mate_in_max_ply))
        || (v <= static_cast<int>(Value::tb_loss_in_max_ply) + MAX_PLY
            && v > static_cast<int>(Value::mated_in_max_ply));
}

}

int lmr_reductions[MAX_PLY][MAX_MOVES];

void init_search() {
    const int lmr_divisor = cfgmgr.lmr_divisor;
    const int lmr_base = cfgmgr.lmr_base;
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
    cfgmgr.lmr_dirty = false;
}

bool is_repetition(const Board & b, int draw)
{
    if (b.half_moves < 2)
        return false;

    const int begin = b.histply - b.half_moves;
    int matches = 0;
    for (int index = b.histply - 1; index >= begin; index -= 2) {
        if (b.history[index].hash == b.hash
            && static_cast<size_t>(++matches) > static_cast<size_t>(draw))
            return true;
    }
    return false;
}

bool is_fifty_move_draw(const Board & b)
{
    return b.half_moves + static_cast<int>(b.gamestate.half_moves) >= 100;
}

std::string root_pv_string_until_terminal(Board const & root, QuadraticPV const & pvline)
{
    Board board { root };
    std::string out;
    const int len = pvline.len[0];
    for (int ply = 0; ply < len; ++ply) {
        const Move move = pvline.table[0][ply];
        if (!move)
            break;

        if (!out.empty())
            out.push_back(' ');
        fmt::format_to(std::back_inserter(out), "{}", move);

        if (board.side == white) apply_move<white>(board, move);
        else                     apply_move<black>(board, move);
        if (is_fifty_move_draw(board))
            break;
    }
    return out;
}

template <Color Us, bool UseNNUE = true>
Value evaluate(Board & b, NNUE::Net * nnue)
{
    if constexpr (UseNNUE) {
        if (Network::enabled && Network::INPUT_WEIGHTS != nullptr) {
            auto const score = static_cast<Value>(nnue->Evaluate2(b, Us));
            if constexpr (Constexpr::debug_eval)
                fmt::print("<{}> move: {}, score2: {}\n", Us, b.history[b.histply -1].move, score);
            return score;
        }
        auto const score = static_cast<Value>(nnue->Evaluate(Us));
        if constexpr (Constexpr::debug_eval)
            fmt::print("<{}> move: {}, score: {}\n", Us, b.history[b.histply -1].move, score);
        return score;
    } else {
        const auto mv = b.history[b.histply -1].move;
        if (mv.flags() == Move::Flags::promote)
            return static_cast<Value>(enyo::HCE_evaluation<Us>(b));
        const auto score = static_cast<Value>(enyo::HCE_evaluation<Us>(b));
        if constexpr (Constexpr::debug_eval)
            fmt::print("<{}> move: {}, score: {}, repeat: {}, key: {:016X}, fen: {}\n",
                Us, b.history[b.histply -1].move, score, is_repetition(b), b.hash, b.fen());
        return score;
    }
}

template <Color Us>
Value evaluate_runtime(Board & b, NNUE::Net * nnue)
{
    return cfgmgr.use_nnue
        ? evaluate<Us, true>(b, nnue)
        : evaluate<Us, false>(b, nnue);
}

template <Color Us>
Value static_root_score_after(Board & b, NNUE::Net * nnue, Move move)
{
    apply_move<Us, true, true>(b, move, nnue);
    const auto score = static_cast<Value>(-evaluate_runtime<~Us>(b, nnue));
    revert_move<Us, true, true>(b, nnue);
    return score;
}

Value static_root_score_after(Board & b, NNUE::Net * nnue, Move move)
{
    return b.side == white
        ? static_root_score_after<white>(b, nnue, move)
        : static_root_score_after<black>(b, nnue, move);
}

int tablebase_status_rank(syzygy::Status status)
{
    switch (status) {
        case syzygy::Status::Win:  return 2;
        case syzygy::Status::Draw: return 1;
        case syzygy::Status::Loss: return 0;
        default:                   return -1;
    }
}

struct TablebaseRootCandidate {
    syzygy::RootMove root;
    Value eval;
};

bool better_tablebase_root_candidate(
    const TablebaseRootCandidate & candidate,
    const TablebaseRootCandidate & current)
{
    const auto candidate_rank = tablebase_status_rank(candidate.root.status);
    const auto current_rank = tablebase_status_rank(current.root.status);
    if (candidate_rank != current_rank)
        return candidate_rank > current_rank;

    const bool candidate_has_dtz = candidate.root.dtz >= 0;
    const bool current_has_dtz = current.root.dtz >= 0;

    if (candidate.root.status == syzygy::Status::Win) {
        if (candidate.eval != current.eval)
            return candidate.eval > current.eval;
        if (candidate_has_dtz != current_has_dtz)
            return candidate_has_dtz;
        if (candidate_has_dtz && candidate.root.dtz != current.root.dtz)
            return candidate.root.dtz < current.root.dtz;
        return false;
    }

    if (candidate_has_dtz != current_has_dtz)
        return candidate_has_dtz;
    if (candidate_has_dtz && candidate.root.dtz != current.root.dtz)
        return candidate.root.dtz > current.root.dtz;
    return candidate.eval > current.eval;
}

std::vector<TablebaseRootCandidate> order_tablebase_root_candidates(
    Board & b,
    NNUE::Net * nnue,
    const std::vector<syzygy::RootMove> & moves)
{
    std::vector<TablebaseRootCandidate> candidates;
    candidates.reserve(moves.size());

    for (const auto tb_move : moves) {
        candidates.emplace_back(TablebaseRootCandidate {
            .root = tb_move,
            .eval = static_root_score_after(b, nnue, tb_move.move)
        });
    }

    std::ranges::stable_sort(candidates, [](const auto & lhs, const auto & rhs) {
        return better_tablebase_root_candidate(lhs, rhs);
    });

    return candidates;
}

Movelist tablebase_moves_with_status(
    const std::vector<TablebaseRootCandidate> & candidates,
    syzygy::Status status)
{
    Movelist moves;
    for (const auto & candidate : candidates) {
        if (candidate.root.status == status)
            moves.emplace(candidate.root.move);
    }
    return moves;
}

template <Color Us, NodeType Node>
Value qsearch(Board & b, Worker & worker, Stack * ss, int depth, int alpha, int beta)
{
    constexpr Color Them = ~Us;
    constexpr bool pv_node = Node != NodeType::NonPV;

    auto & si = worker.si;
    if constexpr (pv_node) {
        if (ss->ply < MAX_PLY)
            worker.pvline.setlen(ss->ply);
    }

    if (worker.time_expired()) {
        return Value::draw;
    }

    // negamax sets ss->in_check for the frame it hands us, but recursive
    // qsearch->qsearch calls use ss+1, which was never touched by negamax.
    // Compute it here so the in-check path below is reliable at every ply.
    // Mirrors negamax's MAX_PLY handling — returning a raw static eval
    // while in check would be as wrong here as it was in the stand-pat
    // branch below.
    ss->in_check = is_check<Us>(b);
    if (ss->ply >= MAX_PLY) {
        return ss->in_check
            ? Value::draw
            : evaluate_runtime<Us>(b, &si.nnue);
    }

    if (is_repetition(b)) {
        return contempt_draw_score(ss);
    }

    Move tt_move {};
    auto tt_value = Value::none;
    std::optional<tt::HashEntry> tthit;
    if (cfgmgr.use_tt)
        tthit = tt::ttable.probe(b.hash);
    ss->tthit = tthit.has_value();
    if (ss->tthit) {
        tt_value = tt::value_from(tthit->value, ss->ply);
        tt_move = tthit->move;

        const bool can_use_tt_cut =
            !si.suppress_tablebase_cutoffs || !is_tablebase_tt_value(tt_value);
        if (Node != NodeType::PV && tt_value != Value::none && can_use_tt_cut) {
            // Mask off the wasPv bit: entries stored from PV nodes carry
            // it in flag bit 2, and raw equality would never match.
            const int tthit_bound = tt::bound_of(tthit->flag);
            if (tthit_bound == tt::type::ExactBound
            || (tthit_bound == tt::type::UpperBound && (tt_value <= alpha))
            || (tthit_bound == tt::type::LowerBound && (tt_value >= beta)))
                return tt_value;
        }
    }

    // In-check qsearch must not stand-pat (the position is forcing, so the
    // static eval is not a valid lower bound), and the move iteration must
    // consider every legal evasion — including quiets — not just captures.
    // The QSEARCH movepicker path skips non-capturing / non-promoting moves
    // (movepicker.hpp:85), which would filter out legal king moves and
    // blocks. Use ABSEARCH prioritization on the in-check branch so all
    // legal evasions are searched.
    Value best_value;
    Movelist lm;
    if (ss->in_check) {
        ss->eval = Value::none;
        best_value = -Value::infinite;
        generate_legal_moves<Us>(b, lm);
        if (lm.empty())
            return mated_in(ss->ply);
    } else {
        best_value = ss->eval = evaluate_runtime<Us>(b, &si.nnue);
        if (best_value >= beta) {
            // No TT store here. A stand-pat cutoff is a static eval, not a
            // search result; now that the TT accepts move-less bound entries
            // storing it would go live, and doing so (even at depth 0)
            // regressed the hypersion blunder-replay gate. Re-adding this is
            // its own SPRT candidate, not a freebie.
            return best_value;
        }

        if (best_value > alpha)
            alpha = best_value;

        generate_legal_tactical_moves<Us>(b, lm);
    }

    PrioritizedMoves mp;
    if (ss->in_check)
        prioritize_moves<Us, ABSEARCH>(mp, worker, lm, tt_move);
    else
        prioritize_moves<Us, QSEARCH>(mp, worker, lm, tt_move);
    Move best_move {};
    while (const auto move = mp.next()) {
        if ((si.nodes & 1023U) == 0 && worker.time_expired())
            return Value::draw;

        // Delta pruning: even winning the captured piece outright plus a
        // safety margin can't lift stand-pat over alpha, so the subtree
        // can't matter. piece_value() is SEE-scale centipawns but the
        // NNUE eval runs at ~2.2x cp (a clean extra queen evaluates to
        // ~2000, startpos to ~+43), so scale piece values by 2 — mixing
        // the raw scales prunes winning captures and cost -119.7 Elo
        // (66d64ba). Promotions/ep (non-generic flags) are exempt: their
        // material swing exceeds piece_value(dst). Mate windows are
        // exempt: eval arithmetic is meaningless near mate bounds.
        constexpr int delta_margin = 400;
        if (!ss->in_check
            && move.flags() == Move::Flags::generic
            && std::abs(alpha) < Constexpr::mate_value - MAX_PLY
            && static_cast<int>(ss->eval)
               + 2 * piece_value(move.dst_piece())
               + delta_margin <= alpha)
            continue;

        // SEE pruning: a capture that loses material can't rescue a
        // position where stand-pat already failed to reach beta. Evasions
        // are exempt — when in check every legal move must be considered.
        if (!ss->in_check && !see_ge<Us>(b, move))
            continue;

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
                if constexpr (pv_node)
                    worker.pvline.setmove(move, ss->ply);

                if (score < beta)
                    alpha = score;
                else
                    break;
            }
        }
    }
    // qsearch's move list is captures-only (plus legal evasions when in
    // check). A fail-high (best_value >= beta) is a genuine lower bound
    // on the true score — a capture that beats beta refutes the position
    // regardless of what quiet moves would have done. But a fail-low is
    // an upper bound only *over the captures-only subset*; a quiet move
    // we never generated could still beat alpha. Storing it as
    // UpperBound with full depth is unsound — a later negamax probe at
    // `tte->depth >= depth` would cut on a value that didn't see the
    // quiet moves. Same class of bug as the qsearch-ExactBound attempt
    // we already reverted (see exactbound stash tags).
    //
    // Fix: only store the fail-high LowerBound from qsearch. Drop the
    // fail-low UpperBound path entirely. Loses some move-ordering hint
    // via tt_move but removes the unsoundness.
    if (!thread::pool.stop.load(std::memory_order_relaxed)
        && best_move
        && cfgmgr.use_tt
        && best_value >= beta) {
        tt::ttable.store(
            b.hash,
            best_move,
            tt::value_to(best_value, ss->ply),
            tt::type::LowerBound,
            depth
        );
    }

    return best_value;
}

template <Color Us, NodeType NT>
Value negamax(int depth, Worker & worker, Stack * ss, Value alpha, Value beta)
{
    if (ss->ply < MAX_PLY)
        worker.pvline.setlen(ss->ply);

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
            ? evaluate_runtime<Us>(b, &si.nnue)
            : Value::draw;
    }

    ss->in_check = is_check<Us>(b);
    ss->eval = Value::none;

    if (depth <= 0) {
        return qsearch<Us, NT != NodeType::NonPV ? NodeType::PV : NodeType::NonPV>(
            b, worker, ss, depth, alpha, beta
        );
    }

    // Age killers: clear the grandchild slots so killers stay local to
    // nearby subtrees. Children still share killers with their cousins
    // (that sharing is the point of the heuristic), but a killer set in
    // a distant part of the tree — or a previous ID iteration — no
    // longer outranks every quiet at this ply for the rest of the search.
    if (ss->ply < MAX_PLY - 1) {
        (ss + 2)->killers[0] = Move{};
        (ss + 2)->killers[1] = Move{};
        // Children accumulate their beta cutoffs here (ss->cutoffCnt++ on
        // fail-high); LMR reads it as "how refutable are this node's
        // children". Reset for the new subtree.
        (ss + 1)->cutoffCnt = 0;
    }

    if (NT != NodeType::Root) {
        if (is_repetition(b)) {
            return contempt_draw_score(ss);
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
    std::optional<tt::HashEntry> tte;
    if (cfgmgr.use_tt)
        tte = tt::ttable.probe(b.hash);
    ss->tthit = tte.has_value();
    ss->ttPv = NT != NodeType::NonPV;
    if (ss->tthit) {
        tt::ttable.hit++;
        tt_value = tt::value_from(tte->value, ss->ply);
        tt_move = tte->move;
        ss->ttPv = ss->ttPv || tt::was_pv_of(tte->flag);
        const int tte_bound = tt::bound_of(tte->flag);
        const bool can_use_tt_cut =
            !si.suppress_tablebase_cutoffs || !is_tablebase_tt_value(tt_value);
        const bool can_tt_cut =
            can_use_tt_cut
            // A singular verification search probes the same position it
            // is verifying; the stored entry is what's being questioned,
            // so it must not answer.
            && ss->excluded_move == Move{}
            && tt_value != Value::none
            && tte->depth >= depth
            && (tte_bound == tt::type::ExactBound
             || (tte_bound == tt::type::UpperBound && tt_value <= alpha)
             || (tte_bound == tt::type::LowerBound && tt_value >= beta));
        if constexpr (NT == NodeType::Root) {
            if (can_tt_cut && tt_move) {
                const auto & root_moves = !si.searchmoves.empty()
                    ? si.searchmoves
                    : worker.root_moves;
                if (std::ranges::find(root_moves, tt_move) != root_moves.end()) {
                    worker.pvline.setmove(tt_move, ss->ply, false);
                    worker.bestmove = tt_move;
                    tt::ttable.cut++;
                    return tt_value;
                }
            }
        } else {
            if (can_tt_cut) {
                tt::ttable.cut++;
                return tt_value;
            }
        }
    }

    // tb lookup:
    if constexpr (Constexpr::use_syzygy) {
        auto & board = worker.si.board;
        if (cfgmgr.use_syzygy) {
            const auto num_pieces = count_bits(board.color_bb[Us] | board.color_bb[Them]);
            const auto tb_max = static_cast<int>(syzygy::largest());
            if (NT != NodeType::Root
                && !worker.si.suppress_tablebase_cutoffs
                && ss->excluded_move == Move{}
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
                        // Only store exact (draw) results. Win/Loss bounds at
                        // the inflated depth would win every depth-preferred
                        // replacement fight in their single-slot bucket, so
                        // the real search result (with its tt_move) could
                        // never be stored for that position again this age —
                        // move ordering collapses in the TB region. Verified
                        // by uci_root.wdl_only_lost_tablebase_root_keeps_
                        // cutoffs_fast: ~4x node blowup with bound stores on.
                        // Re-probing the WDL table on revisit is cheap.
                        if (cfgmgr.use_tt && tb_flag == tt::type::ExactBound) {
                            tt::ttable.store(
                                b.hash,
                                Move{},
                                tt::value_to(tb_value, ss->ply),
                                tb_flag,
                                std::min(depth + tb_max, MAX_PLY)
                            );
                        }
                        return tb_value;
                    }
                }
            }
        }
    }

    // depth extension
    bool improving = true;
    Value static_eval = Value::none;
    if (ss->in_check) {
        improving = false;
        ss->eval = Value::draw;
        depth++;
        goto moves_loop;
    }

    // Static eval for this node. Downstream heuristics — improving,
    // razoring, reverse-futility, null-move threshold — all assume
    // ss->eval is a static positional evaluation. Previously the
    // tthit branch set ss->eval = tt_value directly, then a bitwise-
    // broken refinement fell through without correcting:
    //
    //   ss->eval = tt_value;
    //   if (ss->eval == Value::none) ss->eval = evaluate(...);
    //   if (tt_value != Value::none
    //       && (tte->flag != NoneBound)
    //       & (tt_value > ss->eval ? LowerBound : UpperBound))
    //       ss->eval = tt_value;
    //
    // The `&` was a bitwise AND between a bool and a bound enum
    // value; with LowerBound=1 / UpperBound=2 it was truthy only
    // when tt_value > ss->eval, so the refinement was a noop and
    // ss->eval kept the raw tt_value. Under instrumentation (see
    // search/instrumentation branch), ~15% of ss->eval values on
    // a startpos depth-8 search were Lower/Upper bounds feeding
    // pruning — a search bound, often a mate score or far out-of-
    // window value, passed as a static eval.
    //
    // Fix (Stockfish-style): compute raw static eval as the
    // baseline; TT-correct it only when the stored bound strictly
    // tightens the raw eval in the same direction.
    //   - ExactBound: always more informed than raw eval.
    //   - UpperBound with tt_value < raw: tighter upper bound.
    //   - LowerBound with tt_value > raw: tighter lower bound.
    // Other combinations would widen, not tighten, so raw wins.
    ss->eval = evaluate_runtime<Us>(b, &si.nnue);
    // pawn_hash == 0 means no pawns: every pawnless node would share one
    // entry, turning the correction into a thrashing global eval offset.
    {
        int correction = worker.nonpawn_corrhist[Us]
            [(b.hash ^ b.pawn_hash) & (Worker::corrhist_size - 1)] / 2;
        if (b.pawn_hash)
            correction += worker.corrhist[Us][b.pawn_hash & (Worker::corrhist_size - 1)];
        ss->eval = static_cast<Value>(
            static_cast<int>(ss->eval) + correction / corrhist_grain);
    }
    static_eval = ss->eval;
    if (ss->tthit && tt_value != Value::none) {
        const int b_ = tt::bound_of(tte->flag);
        if (b_ == tt::type::ExactBound
        || (b_ == tt::type::UpperBound && tt_value < ss->eval)
        || (b_ == tt::type::LowerBound && tt_value > ss->eval)) {
            ss->eval = tt_value;
        }
    }

    improving = (ss - 2)->eval != Value::none
             && (ss - 2)->eval < ss->eval;

    // NonPV + TT-miss unconditional depth reduction. Looks like a
    // "shortcut to qsearch at depth 1/2" hack but is load-bearing:
    // instrumentation at startpos d12 showed ~12k fires at d0-2 and
    // ~1.5k at d3+. Two variants SPRT'd in May 2026 on top of 8306e70:
    //   A — remove entirely:  -9.0 ±21 Elo over 500 games, LLR -1.23.
    //   B — gate depth >= 4: -54.4 ±38 Elo at n=175, LLR -1.67, killed.
    // B confirms the shallow fires are providing real selectivity; the
    // deeper fires on their own are harmful. Don't "simplify" this
    // without fresh instrumentation finding a concrete bug.
    if (NT == NodeType::NonPV && !ss->tthit)
        depth--;

    if (depth <= 0)
        return qsearch<Us, NodeType::PV>(b, worker, ss, depth, alpha, beta);

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
            && (tt::bound_of(tte->flag) & tt::type::UpperBound)
            && tt_value < beta;

        if (have_big_pieces
            && depth >= 3
            && ss->eval >= beta
            && (ss-1)->move != Move{}
            && ss->excluded_move == Move{}
            && !tt_rules_out_nmp
            && std::abs(beta) < Constexpr::mate_value - MAX_PLY) {

            int R = 5 + std::min(4, depth / 5) + std::min(Value(3), (ss->eval - beta) / 214);
            apply_null_move<Us>(b);
            auto nullscore = -negamax<Them, NodeType::NonPV>(depth -R, worker, ss + 1, -beta, -beta + 1);
            ss->move = Move{};
            revert_null_move<Us>(b);

            if (nullscore >= beta) {
                // Unproven mate scores from the reduced null search are
                // clamped to beta; otherwise fail-soft.
                if (nullscore >= (Constexpr::mate_value - MAX_PLY))
                    nullscore = beta;
                return nullscore;
            }
        }
    }

    if constexpr (Constexpr::use_probcut) {
        constexpr Value probcut_margin = static_cast<Value>(150);
        constexpr int probcut_reduction = 4;
        const Value probcut_beta = beta + probcut_margin;

        if (NT == NodeType::NonPV
            && depth >= 5
            && !ss->in_check
            // The excluded (TT) move is a tactical move candidate here,
            // so probcut inside a singular verification could "refute"
            // the position with the very move being verified.
            && ss->excluded_move == Move{}
            && ss->eval + probcut_margin >= beta
            && std::abs(beta) < Constexpr::mate_value - MAX_PLY) {
            Movelist probcut_lm;
            generate_legal_tactical_moves<Us>(b, probcut_lm);
            PrioritizedMoves probcut_mp;
            prioritize_moves<Us, QSEARCH>(
                probcut_mp, worker, probcut_lm, tt_move);

            while (const auto move = probcut_mp.next()) {
                if ((si.nodes & 1023U) == 0 && worker.time_expired())
                    return Value::draw;

                apply_move<Us, true, true>(b, move, &worker.si.nnue);
                Value probcut_value = -qsearch<Them, NodeType::NonPV>(
                    b, worker, ss + 1, 0, -probcut_beta, -probcut_beta + 1);

                if (probcut_value >= probcut_beta) {
                    probcut_value = -negamax<Them, NodeType::NonPV>(
                        depth - probcut_reduction,
                        worker,
                        ss + 1,
                        -probcut_beta,
                        -probcut_beta + 1);
                }

                revert_move<Us, true, true>(b, &worker.si.nnue);
                if (thread::pool.stop.load(std::memory_order_relaxed))
                    return Value::draw;

                if (probcut_value >= probcut_beta) {
                    // Remember the refuting capture so a revisit cuts on
                    // the TT instead of re-running the verification
                    // search. The verification ran at depth-4, so the
                    // bound is good for depth-3.
                    if (cfgmgr.use_tt) {
                        tt::ttable.store(
                            b.hash,
                            move,
                            tt::value_to(probcut_value, ss->ply),
                            tt::type::LowerBound,
                            depth - probcut_reduction + 1
                        );
                    }
                    return probcut_value;
                }
            }
        }
    }

moves_loop:
    Movelist lm;
    if constexpr (NT == NodeType::Root) {
        if (!si.searchmoves.empty())
            lm = si.searchmoves;
        else if (!worker.root_moves.empty())
            lm = worker.root_moves;
        else
            generate_legal_moves<Us>(b, lm);
    } else {
        generate_legal_moves<Us>(b, lm);
    }
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
    const Worker::CmhPieceTable * cmh_slice = prev_move
        ? &worker.cmh[Us][static_cast<size_t>(prev_move.src_piece())][prev_move.dst_sq()]
        : nullptr;
    const Move followup_move = (ss-2)->move;
    const Worker::CmhPieceTable * fmh_slice = followup_move
        ? &worker.fmh[Us][static_cast<size_t>(followup_move.src_piece())][followup_move.dst_sq()]
        : nullptr;
    PrioritizedMoves mp;
    prioritize_moves<Us, ABSEARCH>(mp, worker, lm, tt_move, ss->killers, cm, cmh_slice, fmh_slice);
#else
    auto mp = MovePicker2<Us>(worker, lm, tt_move);
#endif

    bool do_fullsearch = false;
    ss->move_count = 0;
    while (const auto move = mp.next()) {
        // Inside a singular verification search: the whole point is to ask
        // "how good is this position without the TT move?".
        if (move == ss->excluded_move)
            continue;

        if ((si.nodes & 1023U) == 0 && worker.time_expired()) {
            return Value::draw;
        }

        si.nodes++;
        ss->move = move;
        ss->move_count++;

        const bool is_quiet = move.dst_piece() == no_piece_type && move.flags() != Move::Flags::promote;
        const bool is_capture = move.dst_piece() != no_piece_type;
        const bool protect_quiet_check =
            is_quiet
            && low_material_check_net<Us>(b)
            && move_gives_check<Us>(b, &worker.si.nnue, move);

        // Futility pruning: skip quiet moves at shallow depth when static
        // eval plus a depth-scaled margin is still below alpha.
        if (!pv_node
            && !ss->in_check
            && is_quiet
            && !protect_quiet_check
            && depth <= 6
            && ss->move_count > 1
            && ss->eval != Value::none
            && ss->eval + 120 * depth <= alpha
            && std::abs(alpha) < Constexpr::mate_value - MAX_PLY) {
            continue;
        }

        // LMP: Late Move Pruning
        if (!pv_node
            && !ss->in_check
            && is_quiet
            && !protect_quiet_check
            && depth <= 5
            && best_value > -Constexpr::mate_value + MAX_PLY
            && ss->move_count > (improving ? 5 + depth * depth : 3 + depth * depth)) {
            continue;
        }

        // History pruning: at shallow depth, skip quiets the continuation
        // histories actively hate. Complements LMP, which counts moves
        // but ignores their quality.
        if (!pv_node
            && !ss->in_check
            && is_quiet
            && !protect_quiet_check
            && depth <= 4
            && ss->move_count > 1
            && best_value > -Constexpr::mate_value + MAX_PLY
            && (cmh_slice || fmh_slice)) {
            int ch = 0;
            if (cmh_slice)
                ch += (*cmh_slice)[static_cast<size_t>(move.src_piece())][move.dst_sq()];
            if (fmh_slice)
                ch += (*fmh_slice)[static_cast<size_t>(move.src_piece())][move.dst_sq()];
            if (ch < -4000 * depth)
                continue;
        }

        // SEE pruning: at shallow depth, skip captures that lose more
        // material than a depth-scaled bound. Move ordering already
        // banishes SEE-losing captures to the bottom band; this skips
        // their subtrees entirely instead of merely searching them last.
        // (Quiets can't be SEE-pruned here: see_ge on a non-capture is
        // `threshold <= 0` — it never evaluates the moved piece hanging.)
        if (!pv_node
            && !ss->in_check
            && is_capture
            && depth <= 5
            && ss->move_count > 1
            && best_value > -Constexpr::mate_value + MAX_PLY
            && !see_ge<Us>(b, move, -100 * depth)) {
            continue;
        }

        // Singular extension: if the TT move's stored lower bound is well
        // above what every other move can reach (verified by a reduced
        // search of this same position with the TT move excluded), the
        // move is forced-ish — search it one ply deeper. If instead the
        // verification proves another move also beats beta, two moves
        // refute this node: fail high without searching further (multicut).
        int extension = 0;
        if (NT != NodeType::Root
            && depth >= 8
            && move == tt_move
            // An extended TT move searches at new_depth == depth, so a
            // chain of singular nodes never loses depth; without a ply
            // cap the chain only terminates at MAX_PLY (observed: 10x
            // test-suite wall time). Cap extensions to twice the current
            // iteration depth, as Stockfish does.
            && ss->ply < 2 * si.depth
            && ss->excluded_move == Move{}
            && ss->tthit
            && tt_value != Value::none
            && std::abs(tt_value) < Constexpr::mate_value - 2 * MAX_PLY
            && !is_tablebase_tt_value(tt_value)
            && (tt::bound_of(tte->flag) & tt::type::LowerBound)
            && tte->depth >= depth - 3) {

            const auto singular_beta =
                static_cast<Value>(static_cast<int>(tt_value) - 2 * depth);
            const int singular_depth = (depth - 1) / 2;
            // The verification runs on this same stack frame; save the
            // move-loop state it will clobber.
            const int saved_move_count = ss->move_count;

            ss->excluded_move = move;
            const Value singular_value = negamax<Us, NodeType::NonPV>(
                singular_depth, worker, ss, singular_beta - 1, singular_beta);
            ss->excluded_move = Move{};
            ss->move_count = saved_move_count;
            ss->move = move;

            if (worker.time_expired())
                return Value::draw;

            if (singular_value < singular_beta) {
                extension = 1;
                // Double extension: every alternative missed the window
                // by a wide margin, so the TT move is not merely best but
                // the only move. Extending +2 grows depth along the path;
                // the per-path doubleExtensions cap bounds the growth on
                // top of the global ply < 2 * si.depth cap.
                if (!pv_node
                    && singular_value < singular_beta - Value(30)
                    && (ss - 1)->doubleExtensions <= 4) {
                    extension = 2;
                }
            } else if (singular_beta >= beta) {
                return singular_beta;
            } else if (tt_value >= beta) {
                // Not singular and the TT bound already beats beta: some
                // alternative also reaches singular_beta, so this node is
                // very likely to fail high regardless — search the TT
                // move shallower rather than deeper (negative extension).
                extension = -1;
            }
        }
        ss->doubleExtensions = (ss - 1)->doubleExtensions + (extension == 2);
        int new_depth = depth -1 + extension;

        // make move
        apply_move<Us, true, true>(b, move, &worker.si.nnue);
        bool child_pv_valid = false;

        // A root move that immediately creates a claimable repetition is a
        // voluntary draw. If the root static eval is not worse, treat that
        // draw as slightly worse than continuing play so Enyo does not pick
        // threefolds as equal-score tie breaks.
        const bool root_repetition =
            NT == NodeType::Root
            && cfgmgr.root_repetition_contempt > 0
            && ss->eval >= Value::draw
            && is_repetition(b);
        if (root_repetition) {
            worker.pvline.setlen(ss->ply + 1);
            value = static_cast<Value>(-cfgmgr.root_repetition_contempt);
        } else {
            // LMR: Late Move Reductions
            if (cfgmgr.use_lmr
                && depth >= 3
                && !ss->in_check
                && !protect_quiet_check
                && ss->move_count > 3 + 2 * pv_node) {
                int R = lmr_reductions[depth][ss->move_count];

                R += !improving;
                // Several children already failed high: this node's moves
                // keep getting refuted, so late ones deserve less effort.
                R += (ss + 1)->cutoffCnt > 3;
                R -= NT != NodeType::NonPV;
                R -= is_capture;
                // Less reduction at positions previously identified as on
                // a principal variation. ttPv flows: PV-node entries are
                // stored with the was_pv bit set, the probe surfaces it
                // into ss->ttPv, and we lighten LMR by one ply here.
                R -= ss->ttPv;
                if (is_quiet) {
                    // Full quiet-stat sum (butterfly + cmh + fmh), the
                    // same signals move ordering uses; divisor doubled
                    // since the sum spans up to 3x the single-table range.
                    int h = worker.history[Us][move.src_sq()][move.dst_sq()];
                    if (cmh_slice)
                        h += (*cmh_slice)[static_cast<size_t>(move.src_piece())][move.dst_sq()];
                    if (fmh_slice)
                        h += (*fmh_slice)[static_cast<size_t>(move.src_piece())][move.dst_sq()];
                    R -= std::clamp(h / 16384, -2, 2);
                }
                if constexpr (NT == NodeType::Root)
                    if (is_quiet && move.src_piece() == queen)
                        R = std::min(R, 1);
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
                child_pv_valid = true;
            }
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

                worker.pvline.setmove(move, ss->ply, child_pv_valid);

                // Publish the best root move incrementally. If the ID loop
                // is interrupted mid-iteration (hard-time fires after some
                // root moves have completed but before aspiration_window
                // returns), the post-loop emission would otherwise find
                // worker.bestmove still at its default sentinel and fall
                // back to legal_fallback[0]. Keeping this tied to every
                // alpha-improvement guarantees the published move is one
                // search actually scored.
                if constexpr (NT == NodeType::Root) {
                    worker.bestmove = move;
                }

                if (value >= beta) {
                    ss->cutoffCnt++;
                    // Fail-high: the true score is a lower bound. Only pull
                    // the correction up when the bound exceeds static eval;
                    // a bound below it carries no information. Singular
                    // verification results don't train the correction.
                    if (!ss->in_check
                        && !is_capture
                        && ss->excluded_move == Move{}
                        && static_eval != Value::none
                        && best_value > static_eval
                        && std::abs(best_value) < Constexpr::mate_value - 2 * MAX_PLY
                        && !is_tablebase_tt_value(best_value)) {
                        update_correction_history<Us>(worker, b, depth,
                            static_cast<int>(best_value) - static_cast<int>(static_eval));
                    }

                    // Capture history: reward the cutoff capture, punish
                    // captures tried before it (regardless of whether the
                    // cutoff move itself was a capture).
                    {
                        const int bonus = std::min(1600, depth * depth * 32);
                        if (is_capture) {
                            update_history_score(
                                worker.capture_history[Us]
                                    [static_cast<size_t>(move.src_piece())]
                                    [move.dst_sq()]
                                    [static_cast<size_t>(move.dst_piece())],
                                bonus);
                        }
                        for (int i = 0; i < ss->move_count - 1; ++i) {
                            const auto prev = mp[static_cast<size_t>(i)];
                            if (prev.dst_piece() != no_piece_type) {
                                update_history_score(
                                    worker.capture_history[Us]
                                        [static_cast<size_t>(prev.src_piece())]
                                        [prev.dst_sq()]
                                        [static_cast<size_t>(prev.dst_piece())],
                                    -bonus / 2);
                            }
                        }
                    }

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

                        Worker::CmhPieceTable * cmh_update = prev_move
                            ? &worker.cmh[Us][static_cast<size_t>(prev_move.src_piece())][prev_move.dst_sq()]
                            : nullptr;
                        if (cmh_update) {
                            update_history_score(
                                (*cmh_update)[static_cast<size_t>(move.src_piece())][move.dst_sq()],
                                bonus);
                        }

                        Worker::CmhPieceTable * fmh_update = followup_move
                            ? &worker.fmh[Us][static_cast<size_t>(followup_move.src_piece())][followup_move.dst_sq()]
                            : nullptr;
                        if (fmh_update) {
                            update_history_score(
                                (*fmh_update)[static_cast<size_t>(move.src_piece())][move.dst_sq()],
                                bonus);
                        }

                        for (int i = 0; i < ss->move_count - 1; ++i) {
                            const auto prev = mp[static_cast<size_t>(i)];
                            if (prev.dst_piece() == no_piece_type && prev.flags() != Move::Flags::promote) {
                                update_history_score(worker.history[Us][prev.src_sq()][prev.dst_sq()], -bonus / 2);
                                if (cmh_update) {
                                    update_history_score(
                                        (*cmh_update)[static_cast<size_t>(prev.src_piece())][prev.dst_sq()],
                                        -bonus / 2);
                                }
                                if (fmh_update) {
                                    update_history_score(
                                        (*fmh_update)[static_cast<size_t>(prev.src_piece())][prev.dst_sq()],
                                        -bonus / 2);
                                }
                            }
                        }
                    }

                    // A singular verification search must not overwrite
                    // the entry it is verifying with its reduced-depth,
                    // offset-window result.
                    if (cfgmgr.use_tt && ss->excluded_move == Move{}) {
                        tt::ttable.store(
                            b.hash,
                            best_move,
                            tt::value_to(best_value, ss->ply),
                            tt::type::LowerBound,
                            depth,
                            ss->ttPv
                        );
                    }
                    // Fail-soft: return the real score, not beta. The
                    // parent gets the same bound the TT just stored.
                    return best_value;
                }
            }
        }
    }


    // A fail-low here means every reply fell short — the opponent's
    // previous move earned that. Reward it so its own side orders it
    // earlier; the mirror of the fail-high malus applied to prior quiets
    // in the cutoff path above.
    if (NT != NodeType::Root
        && !best_move
        && ss->excluded_move == Move{}
        && !thread::pool.stop.load(std::memory_order_relaxed)) {
        const auto prev = (ss - 1)->move;
        if (prev
            && prev.dst_piece() == no_piece_type
            && prev.flags() != Move::Flags::promote) {
            const int bonus = std::min(1600, depth * depth * 32);
            update_history_score(
                worker.history[Them][prev.src_sq()][prev.dst_sq()], bonus);
        }
    }

    // Fail-low nodes leave best_move empty; best_value is still a genuine
    // upper bound over all legal moves (unlike qsearch's captures-only
    // subset), so store it. The TT keeps any previous move hint for the
    // same position when the stored move is empty. Singular verification
    // results (excluded_move set) stay out of the table entirely.
    if (!thread::pool.stop.load(std::memory_order_relaxed)
        && ss->excluded_move == Move{}
        && cfgmgr.use_tt) {
        tt::ttable.store(
            b.hash,
            best_move,
            tt::value_to(best_value, ss->ply),
            best_value >= beta
                ? tt::type::LowerBound
                : tt::type::UpperBound,
            depth,
            ss->ttPv
        );
    }

    // Fail-low (no best_move): the score is an upper bound, only pull the
    // correction down when the bound is below static eval. With a best_move
    // the score is exact within the window; update unconditionally. A
    // singular verification (excluded_move set) scores the position with
    // one move removed, so its result must not train the correction.
    if (!thread::pool.stop.load(std::memory_order_relaxed)
        && !ss->in_check
        && ss->excluded_move == Move{}
        && !(best_move && best_move.dst_piece() != no_piece_type)
        && static_eval != Value::none
        && !(!best_move && best_value >= static_eval)
        && std::abs(best_value) < Constexpr::mate_value - 2 * MAX_PLY
        && !is_tablebase_tt_value(best_value)) {
        update_correction_history<Us>(worker, b, depth,
            static_cast<int>(best_value) - static_cast<int>(static_eval));
    }

    // Fail-soft: on fail-low this returns the sharpest upper bound the
    // search proved (== what the TT stores) instead of clamping to alpha,
    // so parents and the aspiration loop re-search less. qsearch already
    // returns best_value; negamax was the odd one out.
    return best_value;
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

Move find_immediate_mate(
    Board const & root,
    Movelist const & candidates,
    Movelist const & legal_root)
{
    for (const auto move : candidates) {
        if (!move || std::ranges::find(legal_root, move) == legal_root.end())
            continue;

        Board child {root};
        if (root.side == white) {
            apply_move<white>(child, move);
            if (is_check<black>(child) && generate_legal_moves<black>(child).empty())
                return move;
        } else {
            apply_move<black>(child, move);
            if (is_check<white>(child) && generate_legal_moves<white>(child).empty())
                return move;
        }
    }

    return Move {};
}

Move apply_move_policy_root_guard(
    Board & board,
    Worker & worker,
    Movelist const & legal_fallback,
    Move current)
{
    const auto & model = move_policy::runtime_model();
    if (!model.loaded() || !current)
        return current;

    const auto is_legal_root_move = [&](Move move) {
        return std::ranges::find(legal_fallback, move) != legal_fallback.end();
    };
    if (!is_legal_root_move(current))
        return current;

    Movelist active;
    if (worker.si.has_searchmoves) {
        for (const auto move : worker.si.searchmoves) {
            if (is_legal_root_move(move))
                active.emplace(move);
        }
    } else {
        active = legal_fallback;
    }
    if (active.size() < 2)
        return current;

    try {
        const double current_policy = model.score(board, current);
        double best_policy = current_policy;
        Move candidate = current;
        for (const auto move : active) {
            if (move == current)
                continue;
            const double score = model.score(board, move);
            if (score > best_policy) {
                best_policy = score;
                candidate = move;
            }
        }

        const double margin = best_policy - current_policy;
        if (candidate == current || margin < model.threshold())
            return current;

        const auto current_eval = static_root_score_after(board, &worker.si.nnue, current);
        const auto candidate_eval = static_root_score_after(board, &worker.si.nnue, candidate);
        const int eval_drop = static_cast<int>(current_eval) - static_cast<int>(candidate_eval);
        if (eval_drop > cfgmgr.move_policy_max_eval_drop) {
            ucilog(
                "info string movepolicy reject {} over {} margin {:.2f} eval_drop {} max_drop {}\n",
                candidate,
                current,
                margin,
                eval_drop,
                cfgmgr.move_policy_max_eval_drop);
            return current;
        }

        ucilog(
            "info string movepolicy override {} over {} margin {:.2f} eval_drop {} max_drop {}\n",
            candidate,
            current,
            margin,
            eval_drop,
            cfgmgr.move_policy_max_eval_drop);
        return candidate;
    } catch (std::exception const & ex) {
        ucilog("info string movepolicy error: {}\n", ex.what());
        return current;
    }
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

    worker.root_moves = legal_fallback;
    if (legal_fallback.empty()) {
        if (worker.id == 0)
            log_terminal_root_bestmove(board);
        return;
    }

    const auto is_legal_root_move = [&](Move move) {
        return std::ranges::find(legal_fallback, move) != legal_fallback.end();
    };
    const auto is_active_root_move = [&](Move move) {
        if (!is_legal_root_move(move))
            return false;
        return !si.has_searchmoves
            || std::ranges::find(si.searchmoves, move) != si.searchmoves.end();
    };
    const auto active_root_fallback = [&]() {
        if (si.has_searchmoves) {
            for (const auto move : si.searchmoves) {
                if (is_legal_root_move(move))
                    return move;
            }
        }
        return legal_fallback.empty() ? Move{} : legal_fallback[0];
    };

    tt::ttable.prepare();
    worker.pvline.clear();

    if (worker.id == 0) {
        eventlog::log<eventlog::Log::debug>(
            "search_position start: fen={}, legal_moves={}, legal[0]={}\n",
            board.fen(),
            legal_fallback.size(),
            legal_fallback.empty() ? Move{} : legal_fallback[0]);
    }

    const auto& root_candidates = si.has_searchmoves ? si.searchmoves : legal_fallback;
    if (worker.id == 0) {
        const auto mate_move = find_immediate_mate(board, root_candidates, legal_fallback);
        if (mate_move && is_active_root_move(mate_move)) {
            thread::pool.stop = true;
            ucilog("info depth 1 score mate 1 nodes 1 nps 0 time 0 hashfull {} pv {}\n",
                tt::ttable.get_hashfull(), mate_move);
            ucilog("bestmove {}\n", mate_move);
            return;
        }
    }

    // Root WDL is the correctness gate. DTZ is useful as a fallback and
    // tie-break, but it is not distance-to-mate, so TB wins still go through
    // search after filtering out non-winning root moves.
    if (Constexpr::use_syzygy && cfgmgr.use_syzygy) {
        const auto num_pieces = count_bits(board.color_bb[white] | board.color_bb[black]);
        const auto tb_max = static_cast<int>(syzygy::largest());
        if (tb_max > 0
         && num_pieces <= tb_max
         && !board.gamestate.can_castle(CastlingRights::any_castling)) {
            syzygy::Status tb_status = syzygy::Status::Error;
            int tb_dtz = -1;
            auto [tb_score, tb_move] = syzygy::DTZ_probe(board, tb_status, &tb_dtz);
            (void)tb_score;
            if (tb_status == syzygy::Status::Error)
                tb_status = syzygy::WDL_probe(board);

            if (tb_status != syzygy::Status::Error) {
                const char* verdict = tablebase_verdict(tb_status);
                if (worker.id == 0)
                    log_tablebase_root_hit(tb_status, tb_dtz);
                const auto root_candidate_count = root_candidates.size();
                bool let_search_choose = false;
                bool wdl_filter_complete = false;

                const auto use_root_filter = [&](const Movelist & filtered, const char * description) {
                    if (filtered.empty())
                        return false;

                    si.searchmoves = filtered;
                    si.has_searchmoves = true;
                    if (tb_status == syzygy::Status::Win && num_pieces <= 5)
                        si.suppress_tablebase_cutoffs = true;
                    if (worker.id == 0)
                        ucilog("info string tbhit {} {} root moves {}/{}\n",
                            verdict, description, filtered.size(), root_candidate_count);
                    return true;
                };

                if (tb_status == syzygy::Status::Win) {
                    const auto filtered = syzygy::root_WDL_filter(board, root_candidates, &wdl_filter_complete);
                    if (wdl_filter_complete && !filtered.empty()) {
                        let_search_choose = use_root_filter(filtered, "winning");
                        if (worker.id == 0)
                            ucilog("info string tbhit {} WDL root filter complete {}/{}\n",
                                verdict, filtered.size(), root_candidate_count);
                    }
                }

                if (worker.id == 0 && !let_search_choose && tb_dtz >= 0) {
                    bool tb_root_complete = false;
                    const auto tb_root_moves = syzygy::root_moves(board, root_candidates, &tb_root_complete);
                    const bool has_dtz_ranking = std::ranges::all_of(tb_root_moves, [](const auto & move) {
                        return move.dtz >= 0;
                    });
                    if (tb_root_complete && !tb_root_moves.empty() && has_dtz_ranking) {
                        const auto ordered = order_tablebase_root_candidates(board, &si.nnue, tb_root_moves);
                        if (tb_status == syzygy::Status::Win && !wdl_filter_complete) {
                            const auto winning = tablebase_moves_with_status(ordered, syzygy::Status::Win);
                            let_search_choose = use_root_filter(winning, "winning");
                        } else {
                            const auto best = ordered.empty() ? Move{} : ordered.front().root.move;
                            if (best && is_active_root_move(best)) {
                                thread::pool.stop = true;
                                log_tablebase_root_bestmove(tb_status, best);
                                ucilog("bestmove {}\n", best);
                                return;
                            }
                        }
                    }
                }

                if (worker.id == 0 && !let_search_choose) {
                    if (tb_move && is_active_root_move(tb_move)) {
                        thread::pool.stop = true;
                        log_tablebase_root_bestmove(tb_status, tb_move);
                        ucilog("bestmove {}\n", tb_move);
                        return;
                    }

                    bool filter_complete = false;
                    const auto filtered = syzygy::root_WDL_filter(board, root_candidates, &filter_complete);
                    if (!filtered.empty() && filtered.size() < root_candidate_count) {
                        let_search_choose = use_root_filter(filtered, "WDL-filtered");
                    }
                    if (filter_complete && !filtered.empty()) {
                        ucilog("info string tbhit {} WDL root filter complete {}/{}\n",
                            verdict, filtered.size(), root_candidate_count);
                    }

                    // Fallback for Loss: WDL filter can't distinguish losing moves
                    // (all have the same rank). Probe each child individually for DTZ
                    // to find the move that maximises resistance.
                    if (tb_status == syzygy::Status::Loss && !let_search_choose) {
                        int best_child_dtz = -1;
                        Move best_resistance {};
                        for (const auto root_move : root_candidates) {
                            Board child {board};
                            if (board.side == white) apply_move<white>(child, root_move);
                            else                     apply_move<black>(child, root_move);
                            syzygy::Status child_status;
                            int child_dtz = -1;
                            syzygy::DTZ_probe(child, child_status, &child_dtz);
                            if (child_status == syzygy::Status::Win && child_dtz > best_child_dtz) {
                                best_child_dtz = child_dtz;
                                best_resistance = root_move;
                            }
                        }
                        if (best_resistance && is_active_root_move(best_resistance)) {
                            thread::pool.stop = true;
                            ucilog("info string tbhit loss best-resistance {} dtz {}\n",
                                best_resistance, best_child_dtz);
                            log_tablebase_root_bestmove(tb_status, best_resistance);
                            ucilog("bestmove {}\n", best_resistance);
                            return;
                        }
                    }
                }
            }
        }
    }

    struct Mate {
        int moves { MAX_PLY };
        Move move {};
    } shortest_mate;

    Value value = Value::draw;
    auto const max_depth = std::min(si.depth, MAX_PLY);

    // Instability extension state.
    // `bestmove_changed_last_iter` is re-set at the end of every
    // iteration based on the most recent search result.
    // `used_instability_extension` is sticky for the duration of
    // this `go`: once we've granted an extra iteration past soft,
    // we won't grant another in the same move regardless of
    // subsequent flips. The hard cap remains the ultimate bound.
    bool bestmove_changed_last_iter = false;
    bool used_instability_extension = false;
    Value prev_iter_value = Value::none;
    bool score_swung_last_iter = false;
    const int score_swing_cp = cfgmgr.tm_volatility_threshold;
    Move last_legal_bestmove {};
    bool score_info_emitted = false;

    // Root search publishes worker.bestmove while the current iteration is
    // still in progress. If time cuts an iteration short, report the last
    // completed iteration's move so UCI bestmove matches the last completed PV.
    const auto restore_completed_bestmove = [&] {
        if (worker.id == 0)
            worker.bestmove = is_active_root_move(last_legal_bestmove)
                ? last_legal_bestmove
                : Move{};
    };

    for (auto depth = 1; depth <= max_depth; ++depth) {
        if (worker.time_expired()) {
            restore_completed_bestmove();
            break;
        }

        // Soft limit: stop starting new iterations once the optimum budget is
        // used. The current-best move is already committed, so this gives us
        // cheap, principled "stop between iterations" behavior. The hard
        // limit (watchdog + time_expired) still aborts mid-iteration if we
        // blow through the emergency budget.
        //
        // Instability extension: if the last completed iteration either
        // changed the bestmove OR swung the score by >= 100 cp, grant
        // one more iteration past soft. A low-time guard prevents the
        // extension when the clock is thin: we need uci_time >=
        // 10 * uci_inc + 1000 ms to have enough slack for a deeper
        // search without risking a flag. Fires at most once per `go`.
        if (worker.id == 0 && depth > 1 && si.soft_time_expired()) {
            const int uci_time = si.board.side == white ? si.wtime : si.btime;
            const int uci_inc  = si.board.side == white
                ? (si.winc != -1 ? si.winc : 0)
                : (si.binc != -1 ? si.binc : 0);
            const bool enough_time = uci_inc > 0
                ? uci_time >= 10 * uci_inc + 1000
                : uci_time >= 5000;
            const bool may_extend =
                   (bestmove_changed_last_iter || score_swung_last_iter)
                && enough_time
                && !used_instability_extension;
            if (!may_extend) {
                thread::pool.stop = true;
                break;
            }
            // Extension fires exactly once per `go` — mark the sticky
            // flag so subsequent post-soft iterations can't extend
            // again even if bestmove keeps flipping.
            used_instability_extension = true;
        }

        si.nodes = 0;
        si.depth = depth;

        if constexpr (Constexpr::debug_threads)
            fmt::print("<{}> thread: {}, depth: {}\n", __func__, worker.id, depth);

#if 0
        if (worker.id == 0) {
            eventlog::log<eventlog::Log::error>(
                "ITER begin: depth={} table[0][0]={} len[0]={} worker.bestmove={} pv='{}'\n",
                depth,
                worker.pvline.table[0][0],
                static_cast<int>(worker.pvline.len[0]),
                worker.bestmove,
                worker.pvline.str());
        }
#endif

        // Snapshot the previous iteration's bestmove *before* the call.
        // Root negamax now publishes worker.bestmove incrementally on
        // every alpha-improvement (see NodeType::Root block below), so by
        // the time control returns, worker.bestmove already equals the
        // current iteration's pvbm. The TM instability check (further
        // down) needs the pre-iteration value to detect bestmove changes.
        const Move bestmove_before_iter = worker.bestmove;

        value = Constexpr::use_aspiration_window
            ? (si.board.side == white
                ? aspiration_window<white>(value, depth, worker, ss)
                : aspiration_window<black>(value, depth, worker, ss))
            : (si.board.side == white
                ? negamax<white, NodeType::Root>(depth, worker, ss)
                : negamax<black, NodeType::Root>(depth, worker, ss));

#if 0
        if (worker.id == 0) {
            eventlog::log<eventlog::Log::error>(
                "ITER end: depth={} value={} table[0][0]={} len[0]={} pv='{}'\n",
                depth,
                value,
                worker.pvline.table[0][0],
                static_cast<int>(worker.pvline.len[0]),
                worker.pvline.str());
        }
#endif

        if (worker.time_expired()) {
            restore_completed_bestmove();
            break;
        }

        if (worker.id)
            continue;

        if (is_active_root_move(worker.bestmove))
            last_legal_bestmove = worker.bestmove;

        const auto pvbm = worker.pvline.bestmove();
        auto mate_distance = mate_in_moves(value);
        if (pvbm) {
            if (!is_active_root_move(pvbm)) {
                eventlog::log<eventlog::Log::error>(
                    "pv bestmove={} is NOT legal/active at root, depth={}. "
                    "prev_bestmove={} last_legal_bestmove={} pv_str='{}' len[0]={} fen={}\n",
                    pvbm,
                    depth,
                    bestmove_before_iter,
                    last_legal_bestmove,
                    worker.pvline.str(),
                    static_cast<int>(worker.pvline.len[0]),
                    board.fen());
                worker.bestmove = last_legal_bestmove;
            } else {
                if (mate_distance > 0 && mate_distance < shortest_mate.moves)
                    shortest_mate = {mate_distance, pvbm};
                worker.bestmove = pvbm;
                last_legal_bestmove = pvbm;
            }
            // TM: record whether this iteration changed the bestmove,
            // for the next iteration's soft-time extension check. Only
            // meaningful from depth 2 onward (depth 1 has no previous).
            // Compare against the pre-iteration snapshot, not the
            // just-updated worker.bestmove — the latter is rewritten
            // mid-iteration by the Root incremental-publish path.
            bestmove_changed_last_iter =
                (depth > 1)
                && is_active_root_move(worker.bestmove)
                && worker.bestmove != bestmove_before_iter;
            // TM: track score volatility for the next iteration's
            // soft-time extension check. A sharp swing (>= 100 cp)
            // means the tree is re-evaluating a key branch — worth
            // one more iteration of confirmation. Mate-range scores
            // are excluded: the encoding (mate - ply) produces huge
            // swings that don't reflect evaluation instability.
            const bool either_mate =
                   std::abs(value) >= Value::mate_in_max_ply
                || std::abs(prev_iter_value) >= Value::mate_in_max_ply;
            score_swung_last_iter =
                   (depth > 1)
                && prev_iter_value != Value::none
                && !either_mate
                && std::abs(value - prev_iter_value) >= score_swing_cp;
            prev_iter_value = value;
        } else {
            eventlog::log<eventlog::Log::error>(
                "pvbm empty at depth {}. pv_str='{}' len[0]={} table[0][0]={} prev_bestmove={} score={} fen={}\n",
                depth,
                worker.pvline.str(),
                static_cast<int>(worker.pvline.len[0]),
                worker.pvline.table[0][0],
                bestmove_before_iter,
                value,
                board.fen());
        }

        if (!is_legal_root_move(worker.bestmove)) {
            eventlog::log<eventlog::Log::error>(
                "worker.bestmove={} is NOT legal at root, depth={}. pvbm={} prev_bestmove={} pv_str='{}' len[0]={} fen={}\n",
                worker.bestmove,
                depth,
                pvbm,
                bestmove_before_iter,
                worker.pvline.str(),
                static_cast<int>(worker.pvline.len[0]),
                board.fen());
            worker.bestmove = last_legal_bestmove;
        }

        const std::string score_info = mate_distance
            ? fmt::format("mate {}", mate_distance)
            : fmt::format("cp {}", value);

        si.update_elapsed_time();

        const auto info_pv = is_active_root_move(pvbm)
            ? root_pv_string_until_terminal(board, worker.pvline)
            : fmt::format("{}", is_active_root_move(worker.bestmove)
                ? worker.bestmove
                : active_root_fallback());

        std::string info_string = fmt::format("info depth {} score {} nodes {} nps {} time {} hashfull {}",
            depth,
            score_info,
            thread::pool.get_nodes(),
            thread::pool.get_nps(),
            std::chrono::duration_cast<std::chrono::milliseconds>(si.elapsed_time).count(),
            tt::ttable.get_hashfull());

        if (!info_pv.empty())
            info_string += fmt::format(" pv {}", info_pv);

        ucilog("{}\n", info_string);
        score_info_emitted = true;

        if (shortest_mate.moves == 1) {
            eventlog::log<eventlog::Log::info>("Breaking search loop: found mate in 1, shortest_mate.move={}, worker.bestmove={}\n", 
                shortest_mate.move, worker.bestmove);
            break;
        }
    }
    
    if (worker.id == 0) {
        si.update_elapsed_time();
        if (si.has_searchmoves)
            ucilog("info string forced score {}\n", value);

        Move out = is_active_root_move(shortest_mate.move) ? shortest_mate.move : worker.bestmove;

        if (!is_active_root_move(out) && is_active_root_move(last_legal_bestmove))
            out = last_legal_bestmove;

        if (!is_active_root_move(out)) {
            const auto fallback = active_root_fallback();
            eventlog::log<eventlog::Log::error>(
                "bestmove fallback fired. pre_fallback_out={} worker.bestmove={} shortest_mate.move={} "
                "last_legal_bestmove={} pvline.bestmove()={} pv_str='{}' len[0]={} legal_fallback[0]={} "
                "active_fallback={} fen={}\n",
                out,
                worker.bestmove,
                shortest_mate.move,
                last_legal_bestmove,
                worker.pvline.bestmove(),
                worker.pvline.str(),
                static_cast<int>(worker.pvline.len[0]),
                legal_fallback.empty() ? Move{} : legal_fallback[0],
                fallback,
                board.fen());
            out = fallback;
        }

        const bool mate_score = std::abs(value) >= Value::mate_in_max_ply;
        if (!mate_score)
            out = apply_move_policy_root_guard(board, worker, legal_fallback, out);

        if (!score_info_emitted) {
            ucilog("info depth 1 score cp 0 nodes {} nps {} time {} hashfull {} pv {}\n",
                thread::pool.get_nodes(),
                thread::pool.get_nps(),
                std::chrono::duration_cast<std::chrono::milliseconds>(si.elapsed_time).count(),
                tt::ttable.get_hashfull(),
                out);
        }
        ucilog("bestmove {}\n", out);
    }
}

} // enyo ns
