#include "board.hpp"
#include "benchmark.hpp"
#include "uci.hpp"
#include "pgn.hpp"
#include "exepath.hpp"
#include "getopt.h"
#include "thread.hpp"
#include "tt.hpp"
#include "version.hpp"
#include "eventlog.hpp"

#include <iostream>
#include <cstdio>
#include <cctype>
#include <filesystem>
#include <string_view>

#include "fmt/core.h"

using namespace enyo;

namespace {

namespace fs = std::filesystem;

std::string lowercase(std::string value)
{
    std::ranges::transform(value, value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

enum class RunMode {
    uci,
    search_benchmark,
    perft_benchmark,
};

constexpr std::string_view perft_benchmark_fen =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10";

std::string get_default_config_file_path() {
    // Precedence:
    // 1) ~/.config/enyo/settings.json — user's personal config
    // 2) <exepath>/../settings.json   — shipped next to the binary as a
    //                                   fallback so a fresh clone (replay
    //                                   harness, SPRT worker, unpacked
    //                                   release) finds a working default.
    const char * home_directory = getenv("HOME");
    if (home_directory) {
        const std::string user_config = std::string(home_directory) + "/.config/enyo/settings.json";
        if (fs::exists(user_config))
            return user_config;
    }
    const auto exe = fs::path(get_exe_path());
    if (!exe.empty()) {
        const auto next_to_binary = exe.parent_path() / "settings.json";
        if (fs::exists(next_to_binary))
            return next_to_binary.string();
    }
    return "";
}

void read_input_from_stdin(Uci& uci) {
    std::string input;
    while (std::getline(std::cin, input)) {
        if (!input.empty()) {
            uci(input);
            if (uci.quitting)
                break;
        }
    }
}

void apply_configured_uci_options(Uci & uci)
{
    const auto apply = [&](std::string_view selected_name) {
        for (const auto& [name, value] : cfgmgr.configured_uci_options()) {
            if (lowercase(name) == selected_name)
                uci(fmt::format("setoption name {} value {}", name, value));
        }
    };

    apply("use_syzygy");
    for (const auto& [name, value] : cfgmgr.configured_uci_options()) {
        const auto lower_name = lowercase(name);
        if (lower_name == "hash") {
            cfgmgr.setopt(name, value);
            continue;
        }
        if (lower_name != "use_syzygy" && lower_name != "syzygypath")
            uci(fmt::format("setoption name {} value {}", name, value));
    }
    apply("syzygypath");
}

} // anon ns

// setoption name Ponder value false
int main(int argc, char **argv)
{
    fmt::print("id {}\n", g_version);

    Board b{"startpos"};
    Uci uci(b);

    const char* const short_opts = "hb:d:f:p:t:c:l:W:B:";
    const option long_opts[] = {
        { "help",    no_argument,        nullptr, 'h' },
        { "perft",   no_argument,        nullptr, 'b' },
        { "depth",   required_argument,  nullptr, 'd' },
        { "fen",     required_argument,  nullptr, 'f' },
        { "threads", required_argument,  nullptr, 't' },
        { "config",  required_argument,  nullptr, 'c' },
        { "logfile", required_argument,  nullptr, 'l' },
        { "pgn",     required_argument,  nullptr, 'p' },
        { "white",   required_argument,  nullptr, 'W' },
        { "black",   required_argument,  nullptr, 'B' },
        { nullptr,   no_argument,        nullptr, 0 }
    };

    int opt = 0;
    int depth = 0;
    std::string fen;
    std::string pgnfile;
    bool print_help = false;
    bool perft = false;

    std::string config_file_path;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'h':
                print_help = true;
                break;
            case 'b':
                perft = true;
                break;
            case 'd':
                depth = std::stoi(optarg);
                break;
            case 'f':
                fen = optarg;
                break;
            case 'p':
                pgnfile = optarg;
                break;
            case 't':
                cfgmgr.num_threads = std::stoi(optarg);
                break;
            case 'c':
                config_file_path = optarg;
                config_file_path.erase(
                    std::remove_if(std::begin(config_file_path),
                        std::end(config_file_path), ::isspace),
                        std::end(config_file_path)
                );
                break;
            case 'l':
                cfgmgr.logfile = optarg;
                break;
            case 'W':
                pgn.white_player = optarg;
                break;
            case 'B':
                pgn.black_player = optarg;
                break;
            default:
                fmt::print("Error, no such option: '{}'\n", opt);
                fmt::print("Usage: {} [options]\n", argv[0]);
                return 1;
        }
    }

    if (print_help) {
        fmt::print("Usage: {} [OPTIONS]\n", argv[0]);
        fmt::print("       {} bench [--depth=VALUE]\n", argv[0]);
        fmt::print("       {} bench perft [--depth=VALUE] [--fen=FEN]\n\n", argv[0]);
        fmt::print("Options:\n");
        fmt::print("  -h, --help            Show this help message\n");
        fmt::print("  -b, --perft           Run perft test\n");
        fmt::print("  -d, --depth=VALUE     Set search/perft depth\n");
        fmt::print("  -f, --fen=VALUE       Set position from FEN string\n");
        fmt::print("  -t, --threads=VALUE   Set number of threads\n");
        fmt::print("  -c, --config=VALUE    Path to settings.json config file\n");
        fmt::print("  -l, --logfile=VALUE   Set log file path\n");
        fmt::print("  -p, --pgn=VALUE       Load game from PGN file\n");
        fmt::print("  -W, --white=VALUE     Set white player name\n");
        fmt::print("  -B, --black=VALUE     Set black player name\n");
        return 0;
    }

    RunMode run_mode = RunMode::uci;
    if (optind < argc) {
        if (std::string_view{argv[optind]} != "bench") {
            fmt::print(stderr, "error: unknown command '{}'\n", argv[optind]);
            return 1;
        }
        run_mode = RunMode::search_benchmark;
        ++optind;
        if (optind < argc && std::string_view{argv[optind]} == "perft") {
            run_mode = RunMode::perft_benchmark;
            ++optind;
        }
        if (optind < argc) {
            fmt::print(stderr, "error: unexpected argument '{}'\n", argv[optind]);
            return 1;
        }
    }

    if (perft && run_mode != RunMode::uci) {
        fmt::print(stderr, "error: --perft cannot be combined with the bench command\n");
        return 1;
    }

    if (!config_file_path.empty()) {
        if (!cfgmgr.load_config(config_file_path)) {
            fmt::print("error: failed to load config file: '{}'\n", config_file_path);
            return 1;
        }
        fmt::print("Using config file: '{}'\n", config_file_path);
    } else {
        config_file_path = get_default_config_file_path();
        if (!config_file_path.empty() && !cfgmgr.load_config(config_file_path)) {
            fmt::print("error: failed to load config file: '{}'\n", config_file_path);
            return 1;
        }
        if (!config_file_path.empty())
            fmt::print("Using config file: '{}'\n", config_file_path);
    }

    if (run_mode == RunMode::search_benchmark) {
        if (!fen.empty()) {
            fmt::print(stderr, "error: --fen is only valid with 'bench perft'\n");
            return 1;
        }

        eventlog::init();
        for (const auto& [name, value] : cfgmgr.configured_uci_options()) {
            const auto lower_name = lowercase(name);
            if (lower_name != "threads"
                && lower_name != "hash"
                && lower_name != "use_syzygy"
                && lower_name != "syzygypath") {
                uci(fmt::format("setoption name {} value {}", name, value));
            }
        }

        cfgmgr.num_threads = 1;
        cfgmgr.hash_size = benchmark_hash_size_mb;
        cfgmgr.use_syzygy = false;
        if (!tt::ttable.set_size(benchmark_hash_size_mb)) {
            fmt::print(stderr, "error: failed to allocate {} MB benchmark hash\n",
                benchmark_hash_size_mb);
            return 1;
        }

        init_search();
        const int benchmark_depth = depth > 0 ? depth : default_benchmark_depth;
        const auto result = run_search_benchmark(uci, benchmark_depth);
        fmt::print("Benchmark depth: {}\n", benchmark_depth);
        fmt::print("Positions: {}\n", result.positions);
        fmt::print("Total time (ms): {}\n", result.elapsed.count());
        fmt::print("Nodes searched: {}\n", result.nodes);
        fmt::print("Nodes/second: {}\n", result.nodes_per_second);
        return 0;
    }

    if (run_mode == RunMode::perft_benchmark) {
        uci(fmt::format("position fen {}", fen.empty() ? perft_benchmark_fen : fen));
        return uci(fmt::format("go perft {}", depth > 0 ? depth : 5));
    }

    if (!pgnfile.empty()) {
        load_pgn(b, pgnfile);
    }

    eventlog::init();
    eventlog::log<eventlog::Log::info>("id {}\n", g_version);

    if (!fen.empty())
        uci(fmt::format("position fen {}", fen));

    if (perft)
        return uci(fmt::format("go perft {}", depth > 0 ? depth : 1));

    uci(fmt::format("setoption name Threads value {}", cfgmgr.num_threads));
    if (!cfgmgr.move_policy_file.empty())
        uci(fmt::format("setoption name move_policy_file value {}", cfgmgr.move_policy_file));

    apply_configured_uci_options(uci);

    init_search();

    read_input_from_stdin(uci);

    return 0;
}
