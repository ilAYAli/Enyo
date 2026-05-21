#include <cstdint>
#include <iostream>
#include <sstream>
#include <memory>
#include <thread>
#include <string>
#include <limits>
#include <cstdlib>
#include <filesystem>
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
#include "probe.hpp"
#include "nnue.hpp"
#include "nnue_model.hpp"

using namespace enyo;
using namespace eventlog;

namespace {

namespace fs = std::filesystem;

std::string expand_home_path(std::string path)
{
    if (path == "~" || path.starts_with("~/")) {
        if (const char * home = std::getenv("HOME")) {
            if (path.size() == 1)
                return home;
            return std::string(home) + path.substr(1);
        }
    }
    return path;
}

bool load_eval_file(const std::string & value)
{
    if (value.empty()) {
        Network::enabled = false;
        NNUE::Init("");
        ucilog("info string nnue_file empty; using embedded evaluator\n");
        return true;
    }

    const auto path = expand_home_path(value);
    std::error_code ec;
    if (!fs::exists(path, ec) || ec) {
        Network::enabled = false;
        NNUE::Init("");
        ucilog("info string WARNING: nnue_file '{}' not found/readable; using embedded evaluator\n", path);
        return false;
    }

    const auto size = fs::file_size(path, ec);
    if (ec) {
        Network::enabled = false;
        NNUE::Init("");
        ucilog("info string WARNING: nnue_file '{}' size check failed; using embedded evaluator\n", path);
        return false;
    }

    if (size == Network::NETWORK_SIZE) {
        if (Network::LoadNetwork(path.c_str())) {
            Network::enabled = true;
            ucilog("info string network loaded from '{}'\n", path);
            return true;
        }
        Network::enabled = false;
        NNUE::Init("");
        ucilog("info string WARNING: nnue_file '{}' matched network size but failed to load; using embedded evaluator\n", path);
        return false;
    }

    if (size >= NNUE::LEGACY_NETWORK_SIZE) {
        Network::enabled = false;
        NNUE::Init(path);
        ucilog("info string embedded evaluator loaded from '{}'\n", path);
        return true;
    }

    Network::enabled = false;
    NNUE::Init("");
    ucilog("info string WARNING: nnue_file '{}' has invalid size {} bytes; using embedded evaluator\n", path, size);
    return false;
}

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

constexpr bool is_no_increment_low_clock(int uci_time, int uci_inc)
{
    return uci_inc == 0 && uci_time >= 0 && uci_time <= 60'000;
}

constexpr int clock_move_overhead_ms(int uci_time, int uci_inc)
{
    if (uci_inc > 0)
        return 50;
    return is_no_increment_low_clock(uci_time, uci_inc) ? 500 : 250;
}

constexpr int last_resort_move_threshold_ms(int uci_time, int uci_inc)
{
    if (uci_inc > 0)
        return 50;
    return is_no_increment_low_clock(uci_time, uci_inc) ? 1000 : 250;
}

std::chrono::milliseconds safe_clock_hard_cap(int uci_time, int uci_inc)
{
    return std::chrono::milliseconds(std::max(0, uci_time - clock_move_overhead_ms(uci_time, uci_inc)));
}

std::chrono::milliseconds clock_managed_hard_cap(int uci_time, int uci_inc)
{
    using namespace std::chrono;

    constexpr milliseconds min_time(100);
    constexpr milliseconds max_rapid_move(30000);
    constexpr int rapid_clock_ms = 15 * 60 * 1000;

    const auto by_clock = milliseconds(std::max(0, uci_time) / 20 + std::max(0, uci_inc));
    const auto absolute = uci_time <= rapid_clock_ms
        ? max_rapid_move
        : milliseconds(std::max(0, uci_time) / 6);
    return std::max(min_time, std::min(by_clock, absolute));
}

void cap_clock_managed_allocation(TimeAllocation & alloc, int uci_time, int uci_inc)
{
    const auto hard_cap = std::min(
        clock_managed_hard_cap(uci_time, uci_inc),
        safe_clock_hard_cap(uci_time, uci_inc));
    alloc.hard = std::min(alloc.hard, hard_cap);
    alloc.soft = std::min(alloc.soft, alloc.hard);
}

// Compute soft/hard time budgets for one move.
//
// Sudden-death (no movestogo): no-increment games must be much more
// conservative because every move has GUI/network overhead and there is no
// refill. With increment, spend part of the increment because it returns.
// Classical (movestogo): divide remaining time across the mandated moves plus
//   a small buffer; the hard budget is 4x soft, capped at time/4.
// Both branches reserve move overhead: 50ms with increment, 250ms without,
// or 500ms when no-increment clock is low and lichess round-trip cost dominates.
// The hard budget is always capped to the current clock minus move overhead:
// increment is useful for the long-term spending rate, but it is only added
// after this move is made.
TimeAllocation calculate_time_allocation(const SearchInfo & si, int uci_time, int uci_inc)
{
    using namespace std::chrono;

    const bool no_increment_low_clock = is_no_increment_low_clock(uci_time, uci_inc);
    const milliseconds min_time(no_increment_low_clock ? 5 : (uci_inc > 0 ? 100 : 20));
    const milliseconds lag(clock_move_overhead_ms(uci_time, uci_inc));

    if (si.movetime != -1) {
        auto t = std::max(min_time, milliseconds(si.movetime) - milliseconds(50));
        return { t, t };
    }
    if (uci_time == -1) {
        return { milliseconds(-1), milliseconds(-1) };
    }
    if (uci_time <= last_resort_move_threshold_ms(uci_time, uci_inc)) {
        return { milliseconds(0), milliseconds(0) };
    }

    const int time_budget = uci_time - static_cast<int>(lag.count());
    if (time_budget < min_time.count()) {
        return { milliseconds(time_budget), milliseconds(time_budget) };
    }

    milliseconds soft;
    milliseconds hard;
    if (si.movestogo > 0) {
        // Classical: spread the clock across the mandated moves, plus a small
        // buffer so we don't exactly hit zero on the last move.
        const int per_move = time_budget / (si.movestogo + 2) + uci_inc;
        soft = milliseconds(per_move);
        hard = std::min(milliseconds(time_budget / 4), soft * 4);
    } else if (uci_inc == 0) {
        if (no_increment_low_clock) {
            soft = milliseconds(time_budget / 150);
            hard = std::min(milliseconds(time_budget / 30), soft * 2);
        } else {
            soft = milliseconds(time_budget / 100);
            hard = std::min(milliseconds(time_budget / 12), soft * 2);
        }
    } else {
        soft = milliseconds(time_budget / 30 + uci_inc * 3 / 4);
        hard = std::min(milliseconds(time_budget / 6), soft * 3);
    }

    soft = std::max(min_time, soft);
    hard = std::max(soft, hard);
    TimeAllocation alloc { soft, hard };
    if (si.movestogo == 0)
        cap_clock_managed_allocation(alloc, uci_time, uci_inc);
    else if (alloc.hard.count() >= 0) {
        alloc.hard = std::min(alloc.hard, safe_clock_hard_cap(uci_time, uci_inc));
        alloc.soft = std::min(alloc.soft, alloc.hard);
    }
    return alloc;
}

TimeAllocation handle_time_management(Board& b, SearchInfo & si)
{
    if (si.nodes_limit != 0) {
        si.stoptime      = std::chrono::high_resolution_clock::time_point::max();
        si.soft_stoptime = std::chrono::high_resolution_clock::time_point::max();
        return { std::chrono::milliseconds(-1), std::chrono::milliseconds(-1) };
    }

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
    auto alloc = calculate_time_allocation(si, uci_time, uci_inc);
    if (alloc.hard.count() != -1 && si.movetime == -1 && si.movestogo == 0 && uci_inc > 0 && uci_time >= 3000) {
        const auto legal_count = b.side == white
            ? generate_legal_moves<white>(b).size()
            : generate_legal_moves<black>(b).size();
        if (legal_count <= 4) {
            const auto budget = std::chrono::milliseconds(std::max(0, uci_time - 50));
            alloc.soft = std::min(budget / 2, std::max(alloc.soft * 8, alloc.hard));
            alloc.hard = std::min(budget * 3 / 4, std::max(alloc.hard, alloc.soft * 2));
        }
    }
    if (alloc.hard.count() != -1 && si.movetime == -1 && si.movestogo == 0)
        cap_clock_managed_allocation(alloc, uci_time, uci_inc);
    if (alloc.hard.count() == -1) {
        si.stoptime      = std::chrono::high_resolution_clock::time_point::max();
        si.soft_stoptime = std::chrono::high_resolution_clock::time_point::max();
    } else {
        si.stoptime      = si.starttime + alloc.hard;
        si.soft_stoptime = si.starttime + alloc.soft;
    }

    return alloc;
}

int active_clock_ms(Board const & b, SearchInfo const & si)
{
    return b.side == white ? si.wtime : si.btime;
}

int active_increment_ms(Board const & b, SearchInfo const & si)
{
    return b.side == white
        ? (si.winc != -1 ? si.winc : 0)
        : (si.binc != -1 ? si.binc : 0);
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

            // Non-UCI extension: `eval dump` emits one JSONL object
            // with the fields needed by the Python parity test.
            // tools/nnue/enyo_nnue.py consumes this to verify that
            // its feature-index and forward-pass implementations
            // match C++ byte-for-byte.
            //
            // `eval` is always evaluated from side-to-move's
            // perspective (the field "side" records which color that
            // is), regardless of any trailing token — `dump` does not
            // accept a side argument. This matches how NNUE is
            // actually called during search.
            if (side_token == "dump") {
                SearchInfo si(b, 1);
                const auto side = b.side;
                const auto score = static_cast<Value>(si.nnue.Evaluate(side));

                std::vector<int> w_feats, bl_feats;
                NNUE::feature_indices(b, white, w_feats);
                NNUE::feature_indices(b, black, bl_feats);

                auto join = [](const std::vector<int> & v) {
                    std::string s;
                    for (size_t i = 0; i < v.size(); ++i) {
                        if (i) s += ",";
                        s += std::to_string(v[i]);
                    }
                    return s;
                };

                fmt::print(
                    "{{\"fen\":\"{}\",\"side\":\"{}\","
                    "\"eval\":{},"
                    "\"white_features\":[{}],"
                    "\"black_features\":[{}]}}\n",
                    b.fen(),
                    side == white ? "white" : "black",
                    static_cast<int>(score),
                    join(w_feats),
                    join(bl_feats));
                fflush(stdout);
                return 0;
            }

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
        } else if (token == "evalnet") {
            // Phase 4 (arch port): evaluate the current board through
            // the new 1024-hidden NNUE, fresh accumulators.
            // Requires a .nn loaded via `setoption name nnue_file value ...`
            // or the `evalnet load` subcommand. Emits a one-line result.
            std::string sub;
            iss >> sub;
            if (sub == "load") {
                std::string path;
                iss >> path;
                const auto resolved = expand_home_path(path);
                const bool ok = Network::LoadNetwork(resolved.c_str());
                if (ok)
                    Network::enabled = true;
                fmt::print("evalnet load {} {}{}\n",
                           ok ? "ok" : "fail", resolved,
                           ok ? " (search routed through network)" : "");
                return 0;
            }
            if (sub == "enable") {
                if (Network::INPUT_WEIGHTS == nullptr) {
                    fmt::print("evalnet enable failed: no network loaded\n");
                    return 0;
                }
                Network::enabled = true;
                fmt::print("evalnet enable ok\n");
                return 0;
            }
            if (sub == "disable") {
                Network::enabled = false;
                fmt::print("evalnet disable ok (reverted to embedded net)\n");
                return 0;
            }
            if (Network::INPUT_WEIGHTS == nullptr) {
                fmt::print("evalnet error: no network loaded; try "
                           "'evalnet load <path>' first\n");
                return 0;
            }
            const int cp = Network::EvaluateFromScratch(b);
            fmt::print("evalnet {} cp (stm={})\n", cp,
                       b.side == white ? "white" : "black");
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

    ucilog("id name {}\n", g_version);
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
    if (lower_name == "nnue_file")
        load_eval_file(cfgmgr.nnue_file);
    if (lower_name == "hash") {
        // setoption previously just updated cfgmgr.hash_size; the TT
        // was allocated once at Transposition singleton construction
        // from the default, so the requested size never took effect.
        // Now we resize in-place. UCI spec requires setoption only
        // between go commands, so this is always idle-time safe.
        if (tt::ttable.set_size(cfgmgr.hash_size)) {
            ucilog("info string hash table resized to {} MB\n", tt::ttable.size_mb());
        } else {
            ucilog("info string WARNING: failed to resize hash table to {} MB; keeping {} MB\n",
                cfgmgr.hash_size,
                tt::ttable.size_mb());
            cfgmgr.hash_size = tt::ttable.size_mb();
        }
    }
#if ENYO_USE_SYZYGY
    if (lower_name == "use_syzygy") {
        ucilog("info string Syzygy probing {}\n", cfgmgr.use_syzygy ? "enabled" : "disabled");
        const auto path = syzygy::resolve_path(cfgmgr.syzygy_path);
        if (cfgmgr.use_syzygy && !path.empty() && syzygy::init(path))
            ucilog("info string Syzygy tablebases loaded from '{}' (up to {}-man)\n",
                path, syzygy::largest());
    }
    if (lower_name == "syzygypath" && cfgmgr.use_syzygy && !cfgmgr.syzygy_path.empty()) {
        const auto path = syzygy::resolve_path(cfgmgr.syzygy_path);
        if (syzygy::init(path))
            ucilog("info string Syzygy tablebases loaded from '{}' (up to {}-man)\n",
                path, syzygy::largest());
        else
            ucilog("info string warning: no Syzygy tablebases found at '{}'\n",
                path);
    }
#endif
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
        eventlog::log<Log::error>("unknown position type: {}\n", token);
        return;
    }

    while (iss >> token) {
        if (token == "moves")
            continue; // Skip "moves" token

        auto src = str2sq(token.substr(0, 2).c_str());
        auto dst = str2sq(token.substr(2, 4).c_str());
        auto pp = get_promo_piece(token);

        // Validate that the UCI move is legal in the current position.
        // Without this check, a malformed or corrupted `position` command
        // (e.g. from a buggy GUI) will silently leave Enyo searching a
        // bogus position. See: lichess-bot score_syzygy_moves leak.
        const auto legal = b.side == white
            ? generate_legal_moves<white>(b)
            : generate_legal_moves<black>(b);
        bool is_legal = false;
        for (const auto& mv : legal) {
            if (mv.src_sq() == src && mv.dst_sq() == dst
                && (pp == PieceType::no_piece_type || mv.promo_piece() == pp)) {
                is_legal = true;
                break;
            }
        }
        if (!is_legal) {
            eventlog::log<Log::error>(
                "Rejecting illegal move '{}' in position command; FEN='{}'\n",
                token, b.fen());
            fmt::print("info string rejecting illegal move '{}' in position; FEN='{}'\n",
                       token, b.fen());
            fflush(stdout);
            return;
        }

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
        } else if (token == "nodes") {
            int64_t nodes = 0;
            if (iss >> nodes && nodes > 0)
                si.nodes_limit = static_cast<uint64_t>(nodes);
        } else if (token == "perft" && (iss >> si.depth)) {
            perft<true>(b, si.depth);
            return;
        } else if (token == "searchmoves") {
            while (iss >> token)
                searchmoves.push_back(token);
            break;
        }
    }

    const auto legal = b.side == white
        ? generate_legal_moves<white>(b)
        : generate_legal_moves<black>(b);
    Movelist filtered;
    if (!searchmoves.empty()) {
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

    const auto active_clock = active_clock_ms(b, si);
    const auto last_resort_move_ms = last_resort_move_threshold_ms(active_clock, active_increment_ms(b, si));
    if (si.nodes_limit == 0
        && si.movetime == -1
        && si.depth == MAX_PLY
        && active_clock >= 0
        && active_clock <= last_resort_move_ms) {
        const auto& moves = si.has_searchmoves ? si.searchmoves : legal;
        if (moves.empty()) {
            eventlog::log<eventlog::Log::warning>(
                "emergency move: no legal move clock={} threshold={}\n",
                active_clock,
                last_resort_move_ms);
            ucilog("bestmove 0000\n");
        } else {
            eventlog::log<eventlog::Log::warning>(
                "emergency move: clock={} threshold={} move={}\n",
                active_clock,
                last_resort_move_ms,
                moves[0]);
            ucilog("bestmove {}\n", moves[0]);
        }
        return;
    }

    si.starttime = std::chrono::high_resolution_clock::now();
    const auto alloc = handle_time_management(b, si);
    si.board = b;
    si.nnue.refresh(si.board);

#if 1
    fmt::print(
        "info string threads:{},soft:{},hard:{},movetime:{},wtime:{},btime:{},"
        "winc:{},binc:{},movestogo:{},depth:{},nodes:{}\n",
        cfgmgr.num_threads,
        alloc.soft.count(),
        alloc.hard.count(),
        si.movetime,
        si.wtime,
        si.btime,
        si.winc,
        si.binc,
        si.movestogo,
        si.depth,
        si.nodes_limit);
#endif

    thread::pool.init_threads(std::move(si), cfgmgr.num_threads);

    auto watchdog = alloc.hard.count() >= 0
        ? std::optional<std::jthread>(std::in_place, [deadline = std::chrono::high_resolution_clock::now() + alloc.hard,
                                                      hard_ms = alloc.hard.count()](std::stop_token st) {
            while (!st.stop_requested()) {
                if (std::chrono::high_resolution_clock::now() >= deadline) {
                    thread::pool.stop = true;
                    eventlog::log<eventlog::Log::error>(
                        "watchdog fired at deadline ({}ms budget)\n", hard_ms);
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
