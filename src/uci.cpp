#include <array>
#include <cstdint>
#include <atomic>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <memory>
#include <thread>
#include <string>
#include <string_view>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <utility>
#include <vector>
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
#include "move_policy.hpp"
#include "pgn.hpp"

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

uint32_t rotr(uint32_t value, int shift)
{
    return (value >> shift) | (value << (32 - shift));
}

std::string sha256_bytes(std::vector<unsigned char> bytes)
{
    static constexpr std::array<uint32_t, 64> k = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };

    std::array<uint32_t, 8> h = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };

    const uint64_t bit_len = static_cast<uint64_t>(bytes.size()) * 8;
    bytes.push_back(0x80);
    while (bytes.size() % 64 != 56)
        bytes.push_back(0);
    for (int i = 7; i >= 0; --i)
        bytes.push_back(static_cast<unsigned char>((bit_len >> (i * 8)) & 0xff));

    for (size_t offset = 0; offset < bytes.size(); offset += 64) {
        std::array<uint32_t, 64> w{};
        for (size_t i = 0; i < 16; ++i) {
            const size_t j = offset + i * 4;
            w[i] = (static_cast<uint32_t>(bytes[j]) << 24)
                | (static_cast<uint32_t>(bytes[j + 1]) << 16)
                | (static_cast<uint32_t>(bytes[j + 2]) << 8)
                | static_cast<uint32_t>(bytes[j + 3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = h[0];
        uint32_t b = h[1];
        uint32_t c = h[2];
        uint32_t d = h[3];
        uint32_t e = h[4];
        uint32_t f = h[5];
        uint32_t g = h[6];
        uint32_t hh = h[7];

        for (size_t i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t temp1 = hh + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + maj;
            hh = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    return fmt::format(
        "{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
}

std::optional<std::string> sha256_file(const fs::path & path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::nullopt;

    std::vector<unsigned char> bytes{
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>()};
    if (in.bad())
        return std::nullopt;
    return sha256_bytes(std::move(bytes));
}

std::string proven_path(const fs::path & path)
{
    std::error_code ec;
    auto result = fs::weakly_canonical(path, ec);
    if (!ec)
        return result.string();
    result = fs::absolute(path, ec);
    return ec ? path.string() : result.lexically_normal().string();
}

[[noreturn]] void fail_eval_file(const std::string & path, std::string_view reason)
{
    const auto msg = fmt::format("info string ERROR: nnue_file '{}' {}\n", path, reason);
    fmt::print("{}", msg);
    std::fflush(stdout);
    eventlog::log<eventlog::Log::error>("{}", msg);
    std::exit(EXIT_FAILURE);
}

void log_legacy_evaluator(
    const char * source,
    const std::string & path = {},
    const std::string & sha256 = {})
{
    if (path.empty()) {
        ucilog(
            "info string evaluator=legacy-default source={} hidden={} input_buckets={} feature_channels=12\n",
            source, HIDDEN_SIZE, BUCKETS);
        return;
    }

    ucilog(
        "info string evaluator=legacy-default source={} path='{}' sha256={} hidden={} input_buckets={} feature_channels=12\n",
        source, path, sha256, HIDDEN_SIZE, BUCKETS);
}

bool load_eval_file(const std::string & value)
{
    if (value.empty()) {
        Network::enabled = false;
        NNUE::Init("");
        log_legacy_evaluator("embedded-default");
        return true;
    }

    const auto raw_path = fs::path(expand_home_path(value));
    const auto path = proven_path(raw_path);
    std::error_code ec;
    if (!fs::exists(raw_path, ec) || ec)
        fail_eval_file(path, "not found/readable");

    const auto size = fs::file_size(raw_path, ec);
    if (ec)
        fail_eval_file(path, "size check failed");

    const auto sha256 = sha256_file(raw_path);
    if (!sha256)
        fail_eval_file(path, "sha256 read failed");

    if (Network::IsSupportedNetworkSize(size)
        || Network::IsSupportedQuantizedNetworkSize(size)) {
        if (Network::LoadNetwork(path.c_str())) {
            Network::enabled = true;
            ucilog(
                "info string evaluator=native-nnue path='{}' sha256={} hidden={} input_buckets={} feature_channels={} output_buckets={} head_features={} dense_format={}\n",
                path, *sha256, Network::TRAINED_HIDDEN, Network::INPUT_BUCKETS,
                Network::FEATURE_CHANNELS, Network::OUTPUT_BUCKETS,
                Network::OUTPUT_HEAD_FEATURES,
                Network::DENSE_LAYER_FORMAT == Network::DenseLayerFormat::Quantized
                    ? "quantized"
                    : "float");
            return true;
        }
        fail_eval_file(path, "matched supported network size but failed to load");
    }

    if (NNUE::IsSupportedLegacyNetworkSize(size)) {
        Network::enabled = false;
        if (!NNUE::Init(path))
            fail_eval_file(path, "matched legacy network size but failed to load");
        log_legacy_evaluator("file", path, *sha256);
        return true;
    }

    fail_eval_file(path, fmt::format("has invalid size {} bytes", size));
}

bool load_move_policy_file(const std::string & value)
{
    if (value.empty()) {
        move_policy::clear_runtime_model();
        ucilog("info string move_policy_file empty; sidecar disabled\n");
        return true;
    }

    const auto path = expand_home_path(value);
    std::error_code ec;
    if (!fs::exists(path, ec) || ec) {
        move_policy::clear_runtime_model();
        ucilog("info string WARNING: move_policy_file '{}' not found/readable; sidecar disabled\n", path);
        return false;
    }

    std::string error;
    if (!move_policy::load_runtime_model(path, &error)) {
        ucilog("info string WARNING: move_policy_file '{}' failed to load: {}; sidecar disabled\n", path, error);
        return false;
    }

    const auto & model = move_policy::runtime_model();
    ucilog("info string move policy loaded from '{}' (feature_set={}, input_dim={}, threshold={})\n",
           path,
           move_policy::feature_set_name(model.feature_set()),
           model.input_dim(),
           model.threshold());
    return true;
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

std::optional<Move> parse_current_legal_move(Board const & board, std::string_view uci)
{
    return uci_to_move(board, uci);
}

} // anon ns


Uci::Uci(enyo::Board & board)
    : b(board)
{ }

void Uci::prepare_benchmark()
{
    thread::pool.kill();
    ensure_eval_loaded();
}

uint64_t Uci::benchmark_position(std::string_view fen, int depth)
{
    thread::pool.kill();
    b.set(std::string{fen});
    tt::ttable.clear();

    SearchInfo si{b, depth};
    si.starttime = std::chrono::high_resolution_clock::now();
    si.stoptime = std::chrono::high_resolution_clock::time_point::max();
    si.soft_stoptime = std::chrono::high_resolution_clock::time_point::max();

    thread::pool.init_threads(si, 1);
    return thread::pool.wait_and_get_nodes();
}

void Uci::ensure_eval_loaded()
{
    if (eval_loaded)
        return;
    eval_loaded = load_eval_file(cfgmgr.nnue_file);
}

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
                if (ok) {
                    fmt::print(
                        "evalnet load ok {} ({} input buckets, {} feature channels, {} output buckets, {} head features; search routed through network)\n",
                        resolved, Network::INPUT_BUCKETS, Network::FEATURE_CHANNELS,
                        Network::OUTPUT_BUCKETS, Network::OUTPUT_HEAD_FEATURES);
                } else {
                    fmt::print("evalnet load fail {}\n", resolved);
                }
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
        } else if (token == "movepolicy") {
            std::string sub;
            iss >> sub;
            if (sub == "load") {
                std::string path;
                std::getline(iss >> std::ws, path);
                path = expand_home_path(path);
                std::string error;
                const bool ok = move_policy::load_runtime_model(path, &error);
                const auto & policy = move_policy::runtime_model();
                if (ok) {
                    fmt::print(
                        "movepolicy load ok {} feature_set={} input_dim={} threshold={}\n",
                        path,
                        move_policy::feature_set_name(policy.feature_set()),
                        policy.input_dim(),
                        policy.threshold());
                } else {
                    fmt::print("movepolicy load fail {}: {}\n", path, error);
                }
                return 0;
            }
            if (sub == "status") {
                const auto & policy = move_policy::runtime_model();
                fmt::print(
                    "movepolicy {} feature_set={} input_dim={} threshold={}\n",
                    policy.loaded() ? "loaded" : "empty",
                    move_policy::feature_set_name(policy.feature_set()),
                    policy.input_dim(),
                    policy.threshold());
                return 0;
            }
            auto const & policy = move_policy::runtime_model();
            if (!policy.loaded()) {
                fmt::print("movepolicy error: no model loaded\n");
                return 0;
            }
            if (sub == "score") {
                std::string move_token;
                iss >> move_token;
                const auto move = parse_current_legal_move(b, move_token);
                if (!move) {
                    fmt::print("movepolicy score illegal {}\n", move_token);
                    return 0;
                }
                try {
                    fmt::print("movepolicy score {} {:.6f}\n",
                               move_token, policy.score(b, *move));
                } catch (std::exception const & ex) {
                    fmt::print("movepolicy score error: {}\n", ex.what());
                }
                return 0;
            }
            if (sub == "margin") {
                std::string best_token;
                std::string played_token;
                iss >> best_token >> played_token;
                const auto best = parse_current_legal_move(b, best_token);
                const auto played = parse_current_legal_move(b, played_token);
                if (!best || !played) {
                    fmt::print("movepolicy margin illegal {} {}\n", best_token, played_token);
                    return 0;
                }
                try {
                    const double margin = policy.margin(b, *best, *played);
                    fmt::print("movepolicy margin {} {} {:.6f} selected={}\n",
                               best_token,
                               played_token,
                               margin,
                               margin >= policy.threshold() ? "true" : "false");
                } catch (std::exception const & ex) {
                    fmt::print("movepolicy margin error: {}\n", ex.what());
                }
                return 0;
            }
            fmt::print("movepolicy error: expected load/status/score/margin\n");
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
        eval_loaded = load_eval_file(cfgmgr.nnue_file);
    if (lower_name == "move_policy_file")
        load_move_policy_file(cfgmgr.move_policy_file);
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
            ucilog("info depth 1 score cp 0 nodes 0 nps 0 time 0 hashfull {} pv 0000\n",
                tt::ttable.get_hashfull());
            ucilog("bestmove 0000\n");
        } else {
            eventlog::log<eventlog::Log::warning>(
                "emergency move: clock={} threshold={} move={}\n",
                active_clock,
                last_resort_move_ms,
                moves[0]);
            ucilog("info depth 1 score cp 0 nodes 0 nps 0 time 0 hashfull {} pv {}\n",
                tt::ttable.get_hashfull(), moves[0]);
            ucilog("bestmove {}\n", moves[0]);
        }
        return;
    }

    ensure_eval_loaded();

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

    std::atomic_bool watchdog_stop{false};
    auto watchdog = alloc.hard.count() >= 0
        ? std::optional<std::thread>(std::in_place, [deadline = std::chrono::high_resolution_clock::now() + alloc.hard,
                                                     hard_ms = alloc.hard.count(),
                                                     &watchdog_stop] {
            while (!watchdog_stop.load(std::memory_order_relaxed)) {
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
    if (watchdog) {
        watchdog_stop.store(true, std::memory_order_relaxed);
        watchdog->join();
    }
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
