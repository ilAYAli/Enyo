#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>

#include "fmt/core.h"
#include "nlohmann/json.hpp"

#include "bitboard.h"
#include "evaluate.h"
#include "misc.h"
#include "nnue/network.h"
#include "position.h"
#include "types.h"
#include "uci.h"

#ifdef USE_NEON
#undef USE_NEON
#endif

#include "board.hpp"
#include "nnue_model.hpp"

namespace Stockfish {

std::string UCIEngine::square(Square s) {
    return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

} // namespace Stockfish

namespace {

using Json = nlohmann::json;

struct Options {
    std::filesystem::path stockfishNet;
    std::filesystem::path enyoNet;
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path sidecar;
    std::optional<std::size_t> limit;
    std::optional<int> maxAbsCp;
    std::optional<int> minDeltaCp;
    std::size_t shardCount = 1;
    std::size_t shardIndex = 0;
    std::size_t progressEvery = 100000;
};

struct Counters {
    std::size_t rows = 0;
    std::size_t selected = 0;
    std::size_t written = 0;
    std::size_t skippedFen = 0;
    std::size_t skippedCheck = 0;
    std::size_t skippedCp = 0;
    std::size_t skippedDelta = 0;
};

struct Bitboards {
    std::array<std::uint64_t, 8> bb{};
};

[[noreturn]] void fail(std::string_view message) {
    fmt::print(stderr, "error: {}\n", message);
    std::exit(1);
}

void usage() {
    fmt::print(
        "usage: enyo_sf_static_datagen --stockfish-net PATH --input rows.jsonl --output labels.data [options]\n"
        "\n"
        "options:\n"
        "  --net PATH           alias for --stockfish-net\n"
        "  --enyo-net PATH      also evaluate accepted FENs with this Enyo .nn\n"
        "  --sidecar PATH       write accepted row metadata as JSONL\n"
        "  --limit N            stop after N input rows (0 means unlimited)\n"
        "  --max-abs-cp N       skip labels with |Stockfish cp| above N\n"
        "  --min-delta-cp N     require |Stockfish cp - Enyo cp| >= N\n"
        "  --shard-count N      total shard count (default: 1)\n"
        "  --shard-index N      zero-based shard index (default: 0)\n"
        "  --progress N         print progress every N selected rows (default: 100000)\n"
        "  --help               show this help\n");
}

std::size_t parse_size(std::string_view text, std::string_view name) {
    std::size_t value = 0;
    auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size())
        fail(fmt::format("invalid {}: {}", name, text));
    return value;
}

int parse_int(std::string_view text, std::string_view name) {
    int value = 0;
    auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size())
        fail(fmt::format("invalid {}: {}", name, text));
    return value;
}

Options parse_args(std::span<char*> args) {
    Options opts;

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view arg = args[i];

        auto require_value = [&](std::string_view option) -> std::string_view {
            if (++i >= args.size())
                fail(fmt::format("{} requires a value", option));
            return args[i];
        };

        if (arg == "--help") {
            usage();
            std::exit(0);
        } else if (arg == "--net" || arg == "--stockfish-net")
            opts.stockfishNet = require_value(arg);
        else if (arg == "--enyo-net")
            opts.enyoNet = require_value(arg);
        else if (arg == "--input")
            opts.input = require_value(arg);
        else if (arg == "--output")
            opts.output = require_value(arg);
        else if (arg == "--sidecar")
            opts.sidecar = require_value(arg);
        else if (arg == "--limit") {
            const auto limit = parse_size(require_value(arg), arg);
            if (limit)
                opts.limit = limit;
        }
        else if (arg == "--max-abs-cp")
            opts.maxAbsCp = parse_int(require_value(arg), arg);
        else if (arg == "--min-delta-cp")
            opts.minDeltaCp = parse_int(require_value(arg), arg);
        else if (arg == "--shard-count")
            opts.shardCount = parse_size(require_value(arg), arg);
        else if (arg == "--shard-index")
            opts.shardIndex = parse_size(require_value(arg), arg);
        else if (arg == "--progress")
            opts.progressEvery = parse_size(require_value(arg), arg);
        else
            fail(fmt::format("unknown option: {}", arg));
    }

    if (opts.stockfishNet.empty())
        fail("--stockfish-net is required");
    if (opts.input.empty())
        fail("--input is required");
    if (opts.output.empty())
        fail("--output is required");
    if (opts.shardCount == 0)
        fail("--shard-count must be > 0");
    if (opts.shardIndex >= opts.shardCount)
        fail("--shard-index must be less than --shard-count");
    if (opts.minDeltaCp && opts.enyoNet.empty())
        fail("--min-delta-cp requires --enyo-net");

    return opts;
}

void write_u8(std::ofstream& out, std::uint8_t value) {
    out.put(static_cast<char>(value));
}

void write_le16(std::ofstream& out, std::int16_t value) {
    const auto raw = static_cast<std::uint16_t>(value);
    write_u8(out, static_cast<std::uint8_t>(raw & 0xff));
    write_u8(out, static_cast<std::uint8_t>(raw >> 8));
}

void write_le64(std::ofstream& out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i)
        write_u8(out, static_cast<std::uint8_t>(value >> (8 * i)));
}

std::uint64_t byte_swap(std::uint64_t value) {
    value = ((value & 0x00ff00ff00ff00ffULL) << 8) | ((value >> 8) & 0x00ff00ff00ff00ffULL);
    value = ((value & 0x0000ffff0000ffffULL) << 16) | ((value >> 16) & 0x0000ffff0000ffffULL);
    return (value << 32) | (value >> 32);
}

int trailing_zero(std::uint64_t value) {
    if (!value)
        fail("missing required king bit");
    return __builtin_ctzll(value);
}

std::uint8_t result_to_u8(double result) {
    if (result > 0.5)
        return 2;
    if (result < 0.5)
        return 0;
    return 1;
}

double result_from_json(const Json& row, Stockfish::Color stm) {
    if (const auto it = row.find("result"); it != row.end()) {
        if (it->is_number_float() || it->is_number_integer())
            return std::clamp(it->get<double>(), 0.0, 1.0);
        if (it->is_string()) {
            const auto text = it->get<std::string>();
            if (text == "1-0")
                return 1.0;
            if (text == "0-1")
                return 0.0;
            if (text == "1/2-1/2" || text == "0.5")
                return 0.5;
        }
    }

    if (const auto it = row.find("wdl"); it != row.end()) {
        const double stmResult = std::clamp(it->get<double>(), 0.0, 1.0);
        return stm == Stockfish::WHITE ? stmResult : 1.0 - stmResult;
    }

    return 0.5;
}

Bitboards bitboards_from_position(const Stockfish::Position& pos) {
    using namespace Stockfish;

    Bitboards out;
    out.bb[0] = pos.pieces(WHITE);
    out.bb[1] = pos.pieces(BLACK);
    out.bb[2] = pos.pieces(PAWN);
    out.bb[3] = pos.pieces(KNIGHT);
    out.bb[4] = pos.pieces(BISHOP);
    out.bb[5] = pos.pieces(ROOK);
    out.bb[6] = pos.pieces(QUEEN);
    out.bb[7] = pos.pieces(KING);
    return out;
}

void write_chessboard(std::ofstream& out, Bitboards boards, Stockfish::Color stm, int score, double result) {
    if (score < -32768 || score > 32767)
        fail(fmt::format("score out of int16 range: {}", score));

    if (stm == Stockfish::BLACK) {
        for (auto& bb : boards.bb)
            bb = byte_swap(bb);
        std::swap(boards.bb[0], boards.bb[1]);
        score = -score;
        result = 1.0 - result;
    }

    const auto occ = boards.bb[0] | boards.bb[1];
    std::array<std::uint8_t, 16> pieces{};
    auto remaining = occ;
    int idx = 0;
    while (remaining) {
        const int square = trailing_zero(remaining);
        const auto bit = 1ULL << square;
        remaining &= remaining - 1;

        if (idx >= 32)
            fail("too many pieces for BulletFormat ChessBoard");

        const auto color = (bit & boards.bb[1]) ? 8 : 0;
        int piece = -1;
        for (int p = 0; p < 6; ++p) {
            if (bit & boards.bb[static_cast<std::size_t>(2 + p)]) {
                piece = p;
                break;
            }
        }
        if (piece < 0)
            fail("occupied square has no piece type");

        pieces[static_cast<std::size_t>(idx / 2)] |= static_cast<std::uint8_t>((color | piece) << (4 * (idx & 1)));
        ++idx;
    }

    write_le64(out, occ);
    for (const auto pieceByte : pieces)
        write_u8(out, pieceByte);
    write_le16(out, static_cast<std::int16_t>(score));
    write_u8(out, result_to_u8(result));
    write_u8(out, static_cast<std::uint8_t>(trailing_zero(boards.bb[0] & boards.bb[7])));
    write_u8(out, static_cast<std::uint8_t>(trailing_zero(boards.bb[1] & boards.bb[7]) ^ 56));
    write_u8(out, 0);
    write_u8(out, 0);
    write_u8(out, 0);
}

Stockfish::Value stockfish_static_value(const Stockfish::Eval::NNUE::Network& network,
                                        const Stockfish::Position& pos,
                                        Stockfish::Eval::NNUE::AccumulatorStack& accumulators,
                                        Stockfish::Eval::NNUE::AccumulatorCaches& caches) {
    using namespace Stockfish;

    auto [psqt, positional] = network.evaluate(pos, accumulators, caches);
    Value nnue = (125 * psqt + 131 * positional) / 128;

    const int nnueComplexity = std::abs(psqt - positional);
    nnue -= nnue * nnueComplexity / 18236;

    const int material = 534 * pos.count<PAWN>() + pos.non_pawn_material();
    int v = nnue * (77871 + material) / 77871;
    v -= v * pos.rule50_count() / 199;
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
    return v;
}

int stockfish_to_cp(Stockfish::Value value, const Stockfish::Position& pos) {
    using namespace Stockfish;

    const int material = pos.count<PAWN>() + 3 * pos.count<KNIGHT>() + 3 * pos.count<BISHOP>()
                       + 5 * pos.count<ROOK>() + 9 * pos.count<QUEEN>();
    const double m = std::clamp(material, 17, 78) / 58.0;

    constexpr std::array<double, 4> as = {-72.32565836, 185.93832038, -144.58862193, 416.44950446};
    const double a = (((as[0] * m + as[1]) * m + as[2]) * m) + as[3];

    return int(std::round(100 * int(value) / a));
}

std::string fen_from_row(const Json& row) {
    const auto it = row.find("fen");
    if (it == row.end() || !it->is_string())
        fail("row is missing string field 'fen'");
    return it->get<std::string>();
}

void write_sidecar(std::ofstream& sidecar,
                   const std::string& fen,
                   int sfCp,
                   std::optional<int> enyoCp,
                   double result) {
    Json row;
    row["fen"] = fen;
    row["sf_static_cp"] = sfCp;
    if (enyoCp) {
        row["enyo_cp"] = *enyoCp;
        row["delta_cp"] = sfCp - *enyoCp;
    }
    row["result"] = result;
    sidecar << row.dump() << '\n';
}

void create_parent_dir(const std::filesystem::path& path) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
}

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
    using Seconds = std::chrono::duration<double>;
    return std::chrono::duration_cast<Seconds>(std::chrono::steady_clock::now() - start).count();
}

void print_progress(const Counters& counters, std::chrono::steady_clock::time_point start) {
    const double seconds = std::max(0.001, elapsed_seconds(start));
    const double rate = double(counters.selected) / seconds;
    fmt::print(
        "rows={} selected={} written={} skipped_fen={} skipped_check={} skipped_cp={} skipped_delta={} rate={:.2f}/s\n",
        counters.rows,
        counters.selected,
        counters.written,
        counters.skippedFen,
        counters.skippedCheck,
        counters.skippedCp,
        counters.skippedDelta,
        rate);
}

} // namespace

int main(int argc, char** argv) {
    const auto opts = parse_args(std::span<char*>(argv, static_cast<std::size_t>(argc)));

    Stockfish::Bitboards::init();
    Stockfish::Position::init();

    const Stockfish::Eval::NNUE::EvalFile evalFile{EvalFileDefaultName, "None", ""};
    auto network = std::make_unique<Stockfish::Eval::NNUE::Network>(evalFile);

    network->load("", opts.stockfishNet.string());
    network->verify(opts.stockfishNet.string(), [](std::string_view msg) { fmt::print("{}\n", msg); });

    const bool useEnyoNet = !opts.enyoNet.empty();
    if (useEnyoNet) {
        if (!Network::LoadNetwork(opts.enyoNet.string().c_str()))
            fail(fmt::format("failed to load Enyo net: {}", opts.enyoNet.string()));
        Network::enabled = true;
    }

    std::ifstream input(opts.input);
    if (!input)
        fail(fmt::format("failed to open input: {}", opts.input.string()));

    create_parent_dir(opts.output);
    std::ofstream output(opts.output, std::ios::binary);
    if (!output)
        fail(fmt::format("failed to open output: {}", opts.output.string()));

    std::ofstream sidecar;
    if (!opts.sidecar.empty()) {
        create_parent_dir(opts.sidecar);
        sidecar.open(opts.sidecar);
        if (!sidecar)
            fail(fmt::format("failed to open sidecar: {}", opts.sidecar.string()));
    }

    Stockfish::StateInfo state;
    auto accumulators = std::make_unique<Stockfish::Eval::NNUE::AccumulatorStack>();
    auto caches = std::make_unique<Stockfish::Eval::NNUE::AccumulatorCaches>(*network);
    Counters counters;
    const auto start = std::chrono::steady_clock::now();

    std::string line;
    while (std::getline(input, line)) {
        if (opts.limit && counters.rows >= *opts.limit)
            break;

        ++counters.rows;
        if (line.empty())
            continue;
        if ((counters.rows - 1) % opts.shardCount != opts.shardIndex)
            continue;
        ++counters.selected;

        Json row;
        try {
            row = Json::parse(line);
        } catch (const Json::exception&) {
            ++counters.skippedFen;
            continue;
        }

        const auto fen = fen_from_row(row);
        Stockfish::Position pos;
        if (const auto err = pos.set(fen, false, &state); err.has_value()) {
            ++counters.skippedFen;
            continue;
        }

        if (pos.checkers()) {
            ++counters.skippedCheck;
            continue;
        }

        accumulators->reset();
        const auto value = stockfish_static_value(*network, pos, *accumulators, *caches);
        int cp = stockfish_to_cp(value, pos);
        if (pos.side_to_move() == Stockfish::BLACK)
            cp = -cp;

        if (opts.maxAbsCp && std::abs(cp) > *opts.maxAbsCp) {
            ++counters.skippedCp;
            continue;
        }

        std::optional<int> enyoCp;
        if (useEnyoNet) {
            const enyo::Board enyoBoard(fen);
            enyoCp = Network::EvaluateFromScratch(enyoBoard);
            if (opts.minDeltaCp && std::abs(cp - *enyoCp) < *opts.minDeltaCp) {
                ++counters.skippedDelta;
                continue;
            }
        }

        const double result = result_from_json(row, pos.side_to_move());
        write_chessboard(output, bitboards_from_position(pos), pos.side_to_move(), cp, result);
        if (sidecar)
            write_sidecar(sidecar, fen, cp, enyoCp, result);
        ++counters.written;

        if (opts.progressEvery && counters.selected % opts.progressEvery == 0)
            print_progress(counters, start);
    }

    fmt::print("done rows={} selected={} written={} skipped_fen={} skipped_check={} skipped_cp={} skipped_delta={} output={}\n",
               counters.rows,
               counters.selected,
               counters.written,
               counters.skippedFen,
               counters.skippedCheck,
               counters.skippedCp,
               counters.skippedDelta,
               opts.output.string());
}
