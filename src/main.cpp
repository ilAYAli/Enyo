#include "board.hpp"
#include "uci.hpp"
#include "pgn.hpp"
#include "exepath.hpp"
#include "probe.hpp"
#include "getopt.h"
#include "thread.hpp"
#include "version.hpp"
#include "eventlog.hpp"

#include <fcntl.h>
#include <unistd.h>


#include <iostream>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <nlohmann/json.hpp>

#include "fmt/core.h"

using json = nlohmann::json;
using namespace enyo;

namespace {

namespace fs = std::filesystem;

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
        fmt::print("Usage: {} [OPTIONS]\n\n", argv[0]);
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

    if (!config_file_path.empty()) {
        if (!cfgmgr.load_config(config_file_path)) {
            fmt::print("error: failed to load config file: '{}'\n", config_file_path);
            return 1;
        }
        fmt::print("Using config file: '{}'\n", config_file_path);
    } else {
        config_file_path = get_default_config_file_path();
        if (cfgmgr.load_config(config_file_path)) {
            fmt::print("Using config file: '{}'\n", config_file_path);
        }
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
    // Route Hash through the UCI handler so the TT is actually resized
    // (Uci::setoption invokes tt::ttable.set_size). Previously the
    // config only mutated cfgmgr.hash_size and the TT stayed at the
    // 64 MB default allocated at singleton construction.
    uci(fmt::format("setoption name Hash value {}", cfgmgr.hash_size));
    if (!cfgmgr.move_policy_file.empty())
        uci(fmt::format("setoption name move_policy_file value {}", cfgmgr.move_policy_file));

    for (const auto& [name, value] : cfgmgr.configured_uci_options())
        uci(fmt::format("setoption name {} value {}", name, value));

#if ENYO_USE_SYZYGY
    if (cfgmgr.use_syzygy && !cfgmgr.syzygy_path.empty()) {
        const auto path = syzygy::resolve_path(cfgmgr.syzygy_path);
        if (!syzygy::init(path)) {
            fmt::print(stderr,
                "Fatal: failed to initialize tablebases at '{}'\n"
                "Fix or remove the SyzygyPath setting before starting.\n",
                path);
            std::exit(1);
        }
    }
#endif

    init_search();

    read_input_from_stdin(uci);

    return 0;
}
