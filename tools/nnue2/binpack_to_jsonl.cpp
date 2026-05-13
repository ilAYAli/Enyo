#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <fmt/core.h>

#include "nnue_training_data_formats.h"

namespace {

struct Args {
    std::string input;
    std::string output;
    std::size_t limit = 0;
    std::size_t progress = 1'000'000;
    int max_abs_cp = 1600;
};

struct Stats {
    std::size_t seen = 0;
    std::size_t written = 0;
    std::size_t skipped_invalid_move = 0;
    std::size_t skipped_invalid_fen = 0;
    std::size_t skipped_extreme = 0;
};

[[noreturn]] void usage(std::string_view argv0) {
    fmt::print(stderr,
               "usage: {} --input FILE.binpack --output rows.jsonl "
               "[--limit N] [--max-abs-cp N] [--progress N]\n",
               argv0);
    std::exit(2);
}

std::size_t parse_size(std::string_view text, std::string_view name) {
    std::size_t value = 0;
    const auto* begin = text.data();
    const auto* end = begin + text.size();
    auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end)
        throw std::runtime_error(fmt::format("invalid {}: {}", name, text));
    return value;
}

int parse_int(std::string_view text, std::string_view name) {
    int value = 0;
    const auto* begin = text.data();
    const auto* end = begin + text.size();
    auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end)
        throw std::runtime_error(fmt::format("invalid {}: {}", name, text));
    return value;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key(argv[i]);
        auto require_value = [&](std::string_view) -> std::string_view {
            if (i + 1 >= argc)
                usage(argv[0]);
            ++i;
            return argv[i];
        };

        if (key == "--input")
            args.input = std::string(require_value(key));
        else if (key == "--output")
            args.output = std::string(require_value(key));
        else if (key == "--limit")
            args.limit = parse_size(require_value(key), key);
        else if (key == "--progress")
            args.progress = parse_size(require_value(key), key);
        else if (key == "--max-abs-cp")
            args.max_abs_cp = parse_int(require_value(key), key);
        else if (key == "--help" || key == "-h")
            usage(argv[0]);
        else
            throw std::runtime_error(fmt::format("unknown argument: {}", key));
    }

    if (args.input.empty() || args.output.empty())
        usage(argv[0]);
    if (args.max_abs_cp <= 0)
        throw std::runtime_error("--max-abs-cp must be positive");
    return args;
}

void write_json_string(std::ostream& out, std::string_view text) {
    out << '"';
    for (const char ch : text) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << ch;
            break;
        }
    }
    out << '"';
}

bool fen_material_is_sane(std::string_view fen) {
    int white_kings = 0;
    int black_kings = 0;
    int white_pieces = 0;
    int black_pieces = 0;
    int rank_files = 0;
    int ranks = 1;

    for (const char ch : fen) {
        if (ch == ' ')
            break;

        if (ch == '/') {
            if (rank_files != 8)
                return false;
            rank_files = 0;
            ++ranks;
            continue;
        }

        if (ch >= '1' && ch <= '8') {
            rank_files += ch - '0';
            if (rank_files > 8)
                return false;
            continue;
        }

        ++rank_files;
        if (rank_files > 8)
            return false;

        if (ch >= 'A' && ch <= 'Z') {
            ++white_pieces;
            if (ch == 'K')
                ++white_kings;
        } else if (ch >= 'a' && ch <= 'z') {
            ++black_pieces;
            if (ch == 'k')
                ++black_kings;
        } else {
            return false;
        }
    }

    return ranks == 8 && rank_files == 8 && white_kings == 1 && black_kings == 1 &&
           white_pieces <= 16 && black_pieces <= 16;
}

int stockfish_score_to_cp(std::int16_t score) {
    return static_cast<int>(std::lround(static_cast<double>(score) * 100.0 / 208.0));
}

double result_to_wdl(std::int16_t result) {
    if (result > 0)
        return 1.0;
    if (result < 0)
        return 0.0;
    return 0.5;
}

void write_row(std::ostream& out,
               const binpack::TrainingDataEntry& entry,
               const std::string& fen,
               int score_cp,
               const std::string& source) {
    out << "{\"fen\":";
    write_json_string(out, fen);
    out << ",\"score\":" << score_cp;
    out << ",\"wdl\":" << result_to_wdl(entry.result);
    out << ",\"source\":";
    write_json_string(out, source);
    out << ",\"source_type\":\"stockfish_binpack\"";
    out << ",\"teacher\":\"stockfish_binpack\"";
    out << ",\"teacher_score_raw\":" << entry.score;
    out << ",\"ply\":" << entry.ply;
    out << ",\"result\":" << entry.result;
    out << "}\n";
}

}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        binpack::CompressedTrainingDataEntryReader reader(args.input, std::ios::in | std::ios::binary);
        std::ofstream out(args.output);
        if (!out)
            throw std::runtime_error("failed to open output: " + args.output);

        Stats stats;
        while (reader.hasNext()) {
            const auto entry = reader.next();
            ++stats.seen;

            if (!entry.isValid()) {
                ++stats.skipped_invalid_move;
                continue;
            }

            const std::string fen = entry.pos.fen();
            if (!fen_material_is_sane(fen)) {
                ++stats.skipped_invalid_fen;
                continue;
            }

            const int score_cp = stockfish_score_to_cp(entry.score);
            if (std::abs(score_cp) > args.max_abs_cp) {
                ++stats.skipped_extreme;
                continue;
            }

            write_row(out, entry, fen, score_cp, args.input);
            ++stats.written;

            if (args.progress > 0 && stats.written % args.progress == 0)
                fmt::print(stderr, "written {} seen {}\n", stats.written, stats.seen);
            if (args.limit > 0 && stats.written >= args.limit)
                break;
        }

        fmt::print(stderr,
                   "{{\n"
                   "  \"input\": \"{}\",\n"
                   "  \"output\": \"{}\",\n"
                   "  \"seen\": {},\n"
                   "  \"written\": {},\n"
                   "  \"skipped_invalid_move\": {},\n"
                   "  \"skipped_invalid_fen\": {},\n"
                   "  \"skipped_extreme\": {}\n"
                   "}}\n",
                   args.input,
                   args.output,
                   stats.seen,
                   stats.written,
                   stats.skipped_invalid_move,
                   stats.skipped_invalid_fen,
                   stats.skipped_extreme);
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "error: {}\n", e.what());
        return 1;
    }
}
