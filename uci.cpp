#include <cstdint>
#include <iostream>
#include <sstream>
#include <memory>
#include <thread>
#include <string>
#include <limits>
#include <fmt/format.h>

#include "config.hpp"
#include "search.hpp"
#include "thread.hpp"
#include "fen.hpp"
#include "uci.hpp"
#include "perft.hpp"
#include "movegen_helper.hpp"
#include "movegen.hpp"
#include "tt.hpp"
#include "version.hpp"
#include "eventlog.hpp"

using namespace enyo;
using namespace eventlog;

namespace {

[[maybe_unused]] std::vector<std::string> history_to_vec(const Board & b, int max_size = 0)
{
    auto const N = max_size
        ? std::min(b.histply, max_size)
        : MAX_PLY;
    std::vector<std::string> moves;
    moves.reserve(static_cast<size_t>(N));
    std::transform(b.history, b.history + N, std::back_inserter(moves), [](Undo const & undo) {
        return fmt::format("{}", undo.move);
    });

    return moves;
}


struct TimeAllocation {
    std::chrono::milliseconds soft{-1};  // optimum — between-iteration stop
    std::chrono::milliseconds hard{-1};  // maximum — emergency stop
};

// Compute soft/hard time budgets for one move.
//
// Sudden-death (no movestogo): allocate ~1/30 of remaining time as the soft
//   budget, capped at 1/5 as the hard budget. Increment is spent generously
//   since it's replenished next move.
// Classical (movestogo): divide remaining time across the mandated moves plus
//   a small buffer; the hard budget is 4x soft, capped at time/4.
// Both branches reserve a 50ms lag margin and clamp to a 100ms floor so the
// engine never tries to move instantly on critically low time.
TimeAllocation calculate_time_allocation(const SearchInfo & si, int uci_time, int uci_inc)
{
    using namespace std::chrono;

    constexpr milliseconds lag(50);
    constexpr milliseconds min_time(100);

    if (si.movetime != -1) {
        auto t = std::max(min_time, milliseconds(si.movetime) - lag);
        return { t, t };
    }
    if (uci_time == -1) {
        return { milliseconds(-1), milliseconds(-1) };
    }
    if (uci_time < min_time.count()) { // critically low on time
        return { milliseconds(uci_time), milliseconds(uci_time) };
    }

    const int time_budget = std::max(0, uci_time - static_cast<int>(lag.count()));

    milliseconds soft;
    milliseconds hard;
    if (si.movestogo > 0) {
        // Classical: spread the clock across the mandated moves, plus a small
        // buffer so we don't exactly hit zero on the last move.
        const int per_move = time_budget / (si.movestogo + 2) + uci_inc;
        soft = milliseconds(per_move);
        hard = std::min(milliseconds(time_budget / 4), soft * 4);
    } else {
        // Sudden-death: budget ~1/30 of the clock, hard cap at 1/5.
        soft = milliseconds(time_budget / 30 + uci_inc * 3 / 4);
        hard = std::min(milliseconds(time_budget / 5), soft * 5);
    }

    soft = std::max(min_time, soft);
    hard = std::max(soft, hard);
    return { soft, hard };
}

TimeAllocation handle_time_management(const Board& b, SearchInfo & si)
{
    bool have_time_limit = si.wtime != -1 || si.btime != -1 || si.movetime != -1;
    if (!have_time_limit) {
        si.stoptime      = std::chrono::high_resolution_clock::time_point::max();
        si.soft_stoptime = std::chrono::high_resolution_clock::time_point::max();
        return { std::chrono::milliseconds(-1), std::chrono::milliseconds(-1) };
    }

    int uci_time = si.wtime;
    int uci_inc = si.winc != -1 ? si.winc : 0;
    if (b.side == black) {
        uci_time = si.btime;
        uci_inc = si.binc != -1 ? si.binc : 0;
    }
    const auto alloc = calculate_time_allocation(si, uci_time, uci_inc);
    if (alloc.hard.count() == -1) {
        si.stoptime      = std::chrono::high_resolution_clock::time_point::max();
        si.soft_stoptime = std::chrono::high_resolution_clock::time_point::max();
    } else {
        si.stoptime      = si.starttime + alloc.hard;
        si.soft_stoptime = si.starttime + alloc.soft;
    }

    return alloc;
}

PieceType get_promo_piece(std::string const & token)
{
    if (token.length() == 5) {
        if (token[4] == 'q') {
            return PieceType::queen;
        } else if (token[4] == 'r') {
            return PieceType::rook;
        } else if (token[4] == 'b') {
            return PieceType::bishop;
        } else if (token[4] == 'n') {
            return PieceType::knight;
        }
    }
    return PieceType::no_piece_type;
}

} // anon ns


Uci::Uci(enyo::Board & board)
    : b(board)
{ }

int Uci::operator()(const std::string& command)
{
    if (command.length() == 0)
        return 1;

    eventlog::log<Log::uci>("{}\n", command);

    std::istringstream iss(command);
    std::string token;

    iss >> token;
    std::transform(std::begin(token), std::end(token), std::begin(token), ::tolower);
    if (token == "uci") {
        uci();
    } else if (token == "debug") {
        debug(iss);
    } else if (token == "isready") {
        isready();
    } else if (token == "ucinewgame") {
        newgame();
    } else if (token == "position") {
        position(iss);
    } else if (token == "go") {
        go(iss);
    } else if (token == "bench") {
        bench(iss);
    } else if (token == "setoption") {
        setoption(iss);
    } else if (token == "quit") {
        quit();
    } else if (token == "stop") {
        stop();
        ;
    } else { // non-UCI commands:
        auto is_comment = [](const std::string & str) {
            auto first_non_ws = std::find_if_not(str.begin(), str.end(), [](unsigned char c) {
                return std::isspace(c);
            });
            return (first_non_ws != str.end()) && (*first_non_ws == '#');
        };

        if (is_comment(token)) {
            return 0;
        } else if (token == "print" || token == "d") {
            fmt::print("{}\n", b.str());
        } else if (token == "hash") {
            fmt::print("hash {:016X}\n", b.hash);
        } else if (token == "eval") {
            std::string side_token;
            iss >> side_token;
            SearchInfo si(b, 1);
            const auto side = side_token == "white"
                ? white
                : side_token == "black"
                    ? black
                    : b.side;
            const auto score = static_cast<Value>(si.nnue.Evaluate(side));
            fmt::print("eval {}\n", score);
        } else if (token == "nnuecheck") {
            SearchInfo si(b, 1);
            const auto inc = static_cast<Value>(si.nnue.Evaluate(b.side));
            si.nnue.refresh(si.board);
            const auto full = static_cast<Value>(si.nnue.Evaluate(b.side));
            fmt::print("nnue inc {} full {}\n", inc, full);
        } else if (token == "pgn") {
            pgn();
        } else if (token == "move") { // non-UCI command
            std::string to;
            iss >> token;
            iss >> to;

            auto src = str2sq(token.substr(0, 2).c_str());
            auto dst = str2sq(token.substr(2, 4).c_str());
            auto pp = get_promo_piece(token);

            if (b.side == white) {
                constexpr Color Us = white;
                auto piece = get_piece_type<Us>(b, src);
                if (piece == PieceType::no_piece_type) {
                    fmt::print("Error, no {} piece at {}\n", Us, sq2str(src));
                    return 0;
                }
                auto m = resolve_move<Us>(b, piece, src, dst);
                if (pp != PieceType::no_piece_type)
                    m.set_promo_piece(pp);
                apply_move<Us>(b, m);
            } else {
                constexpr Color Us = black;
                auto piece = get_piece_type<Us>(b, src);
                if (piece == PieceType::no_piece_type) {
                    fmt::print("Error, no {} piece at {}\n", Us, sq2str(src));
                    return 0;
                }
                auto m = resolve_move<Us>(b, piece, src, dst);
                if (pp != PieceType::no_piece_type)
                    m.set_promo_piece(pp);
                apply_move<Us>(b, m);
            }
        } else {
            fmt::print("unknown command: '{}'\n", token);
        }
    }
    return 0;
}

void Uci::uci()
{
    auto & config = cfgmgr;

    fmt::print("id name {}\n", g_version);
    ucilog("id author Petter Wahlman\n\n");
    ucilog("{}", config.allopts());
    ucilog("\n");
    ucilog("uciok\n");
}

// setoption name Debug Log File value /tmp/foo
void Uci::setoption(std::istringstream& iss)
{
    std::string rest;
    std::getline(iss >> std::ws, rest);
    if (rest.empty())
        return;

    auto lower = rest;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    constexpr std::string_view name_prefix = "name ";
    if (!lower.starts_with(name_prefix))
        return;

    const auto value_pos = lower.find(" value ", name_prefix.size());
    const auto name_end = value_pos == std::string::npos ? rest.size() : value_pos;
    auto name = rest.substr(name_prefix.size(), name_end - name_prefix.size());
    auto value = value_pos == std::string::npos ? std::string{} : rest.substr(value_pos + 7);

    if (name.empty())
        return;

    if (!cfgmgr.setopt(name, value))
        return;

    auto lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    if (lower_name == "logfile" || lower_name == "debug log file")
        eventlog::reopen_logfile(cfgmgr.logfile, true);
}

void Uci::debug(std::istringstream& iss)
{
    std::string token;
    iss >> token;
    if (token == "on") {
        ucilog("info string debug: on\n");
        eventlog::uci_debug_log = true;
    } else if (token == "off") {
        ucilog("info string debug: off\n");
        eventlog::uci_debug_log = false;
    }
}

void Uci::isready()
{
    ucilog("readyok\n");
}

void Uci::newgame()
{
    b.set();
    tt::ttable.clear();
    thread::pool.kill();
}

void Uci::position(std::istringstream& iss)
{
    std::string token;
    iss >> token;

    if (token == "startpos") {
        if (iss >> token && token != "moves") {
            eventlog::log<Log::debug>("Error, unexpected token after startpos: {}\n", token);
            return;
        }
        b.set();
    }  else if (token == "kiwi") { // perft
        b.set("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    }  else if (token == "wac2") { // b3b2 (depth 23)
        //b.set("8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - -");
        //b.set("8/8/5k1p/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 0");
        b.set("8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1");
    }  else if (token == "lct1") { // endgame: d5d6 (depth 21)
        b.set("r3kb1r/3n1pp1/p6p/2pPp2q/Pp2N3/3B2PP/1PQ2P2/R3K2R w KQkq -");
    }  else if (token == "fine70") { // a1b1 (depth 29)
        b.set("8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w -");
    } else if (token == "fen") {
        std::string fen;
        while (iss >> token && token != "moves")
            fen += token + " ";
        if (!fen.empty()) {
            fen.pop_back();
            if (!fen.empty() && fen.front() == '"')
                fen.erase(fen.begin());
            if (!fen.empty() && fen.back() == '"')
                fen.pop_back();
            b.set(fen);
        }
    } else {
        eventlog::log<Log::error>("Error, unknown position type: {}\n", token);
        return;
    }

    while (iss >> token) {
        if (token == "moves")
            continue; // Skip "moves" token

        auto src = str2sq(token.substr(0, 2).c_str());
        auto dst = str2sq(token.substr(2, 4).c_str());
        auto pp = get_promo_piece(token);
        if (b.side == enyo::white) {
            const auto piece = get_piece_type<white>(b, src);
            auto m = resolve_move<white>(b, piece, src, dst);
            if (pp != PieceType::no_piece_type)
                m.set_promo_piece(pp);
            apply_move<white>(b, m);
        } else {
            const auto piece = get_piece_type<black>(b, src);
            auto m = resolve_move<black>(b, piece, src, dst);
            if (pp != PieceType::no_piece_type)
                m.set_promo_piece(pp);
            apply_move<black>(b, m);
        }
    }
}

void Uci::go(std::istringstream & iss)
{
    thread::pool.kill();

    enyo::SearchInfo si{};
    std::string token;
    std::vector<std::string> searchmoves;

    if (!(b.color_bb[white] | b.color_bb[black]))
        b.set();

    while (iss >> token) {
        if (token == "depth" && (iss >> si.depth)) {
        } else if (token == "wtime" && (iss >> si.wtime)) {
        } else if (token == "btime" && (iss >> si.btime)) {
        } else if (token == "winc" && (iss >> si.winc)) {
        } else if (token == "binc" && (iss >> si.binc)) {
        } else if (token == "movetime" && (iss >> si.movetime)) {
        } else if (token == "movestogo" && (iss >> si.movestogo)) {
        } else if (token == "perft" && (iss >> si.depth)) {
            perft<true>(b, si.depth);
            return;
        } else if (token == "searchmoves") {
            while (iss >> token)
                searchmoves.push_back(token);
            break;
        }
    }

    si.starttime = std::chrono::high_resolution_clock::now();
    const auto alloc = handle_time_management(b, si);
    si.board = b;
    si.nnue.refresh(si.board);

    if (!searchmoves.empty()) {
        Movelist filtered;
        const auto legal = b.side == white
            ? generate_legal_moves<white>(b)
            : generate_legal_moves<black>(b);

        for (const auto& move_str : searchmoves) {
            auto src = str2sq(move_str.substr(0, 2).c_str());
            auto dst = str2sq(move_str.substr(2, 2).c_str());
            auto pp = get_promo_piece(move_str);
            for (const auto move : legal) {
                if (move.src_sq() == src && move.dst_sq() == dst) {
                    if (pp == PieceType::no_piece_type || move.promo_piece() == pp)
                        filtered.emplace(move);
                }
            }
        }

        if (!filtered.empty()) {
            si.searchmoves = filtered;
            si.has_searchmoves = true;
        }
    }

#if 1
    fmt::print("info string threads:{},soft:{},hard:{},movetime:{},wtime:{},btime:{},winc:{},binc:{},movestogo:{},depth:{}\n",
        cfgmgr.num_threads, alloc.soft.count(), alloc.hard.count(), si.movetime, si.wtime, si.btime, si.winc, si.binc, si.movestogo, si.depth);
#endif

    thread::pool.init_threads(std::move(si), cfgmgr.num_threads);

    auto watchdog = alloc.hard.count() >= 0
        ? std::optional<std::jthread>(std::in_place, [deadline = std::chrono::high_resolution_clock::now() + alloc.hard](std::stop_token st) {
            while (!st.stop_requested()) {
                if (std::chrono::high_resolution_clock::now() >= deadline) {
                    thread::pool.stop = true;
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        })
        : std::nullopt;

    // Always wait for search threads to complete before returning from 'go' command.
    // This ensures bestmove is always printed before we start reading the next command.
    // The original logic only waited for non-interactive (piped) stdin without time controls,
    // but this caused the engine to not wait for bestmove output when time controls were set,
    // leading to race conditions where the main thread could process new commands or exit
    // before the search thread finished outputting the bestmove.
    thread::pool.wait();
}

// bench 0 0 5 current perft
// ===========================
// Total time (ms) : 461
// Nodes searched  : 11.906.0324
// Nodes/second    : 2.58.265.344
void Uci::bench(std::istringstream & iss)
{
    int depth = 5;
    std::string token;
    while (iss >> token) {
        if (token == "depth" && (iss >> depth))
            break;
    }
    perft<true>(b, depth);
}


void Uci::pgn()
{
    enyo::print_pgn(b);
}

void Uci::stop() {
    thread::pool.kill();
}

void Uci::quit() {
    thread::pool.kill();
    quitting = true;
}
