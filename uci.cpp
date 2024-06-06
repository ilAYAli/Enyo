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


std::chrono::milliseconds calculate_time_slot(const SearchInfo & si, int uci_time, int uci_inc)
{
    using namespace std::chrono;

    constexpr milliseconds lag(50);
    constexpr milliseconds min_time(100);

    milliseconds time_slot = min_time;
    if (si.movetime != -1) {
        time_slot = milliseconds(si.movetime) - lag;
    } else if (uci_time == -1) {
        time_slot = milliseconds(-1);
    } else if (uci_time < min_time.count()) { // critically low on time
        time_slot = milliseconds(uci_time);
    } else {
        uci_time -= static_cast<int>(lag.count());
        if (si.movestogo > 0) {
            time_slot = std::max(min_time, milliseconds(uci_time / si.movestogo + uci_inc) - lag);
        } else {
            time_slot = std::max(min_time, milliseconds(uci_time) - lag);
        }
    }
    return time_slot;
}

std::chrono::milliseconds handle_time_management(const Board& b, SearchInfo & si)
{
    bool have_time_limit = si.wtime != -1 || si.btime != -1 || si.movestogo != -1;
    if (!have_time_limit) {
        si.stoptime = std::chrono::high_resolution_clock::time_point::max();
        return std::chrono::milliseconds(-1);
    }

    int uci_time = si.wtime;
    int uci_inc = si.winc != -1 ? si.winc : 0;
    if (b.side == black) {
        uci_time = si.btime;
        uci_inc = si.binc != -1 ? si.binc : 0;
    }
    std::chrono::milliseconds time_slot = calculate_time_slot(si, uci_time, uci_inc);
    if (time_slot.count() == -1) {
        si.stoptime = std::chrono::high_resolution_clock::time_point::max();
    } else {
        std::chrono::milliseconds duration(time_slot);
        si.stoptime = si.starttime + duration;
    }

    return time_slot;
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
                apply_move<Us>(b, m);
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
    std::string token;
    std::string name;
    std::string value;

    iss >> token;
    std::transform(std::begin(token), std::end(token), token.begin(), ::tolower);
    if (token != "name")
        return;
    iss >> name;

    iss >> token;
    std::transform(std::begin(token), std::end(token), std::begin(token), ::tolower);
    if (token != "value")
        return;
    iss >> value;

    cfgmgr.setopt(name, value);
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
        b.set("8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - -");
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
        }
    }

    si.starttime = std::chrono::high_resolution_clock::now();
    auto time_slot = handle_time_management(b, si);
    if (b.histply < 25) // hack:
        time_slot = std::min(time_slot, std::chrono::milliseconds(3000));

    if (time_slot.count() != -1) {
        std::chrono::milliseconds duration(time_slot);
        si.stoptime = si.starttime + duration;
    } else {
        si.stoptime = std::chrono::high_resolution_clock::time_point::max();
    }
    si.board = b;

#if 1
    fmt::print("info string threads:{},slot:{},movetime:{},wtime:{},btime:{},winc:{},binc:{},movestogo:{},depth:{}\n",
        cfgmgr.num_threads,time_slot.count(), si.movetime, si.wtime, si.btime, si.winc, si.binc, si.movestogo, si.depth);
#endif

    thread::pool.init_threads(std::move(si), cfgmgr.num_threads);
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
    thread::pool.wait();
    exit(0);
}

