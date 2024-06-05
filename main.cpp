#include "board.hpp"
#include "uci.hpp"
#include "pgn.hpp"
#include "exepath.hpp"
#include "probe.hpp"
#include "getopt.h"
#include "thread.hpp"
#include "version.hpp"

#include <fcntl.h>
#include <unistd.h>


#include <iostream>
#include <cstdio>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "fmt/core.h"

using json = nlohmann::json;
using namespace enyo;

namespace {

namespace fs = std::filesystem;
void truncate(std::string filename) {
    if (fs::exists(filename))
        fs::resize_file(filename, 0);
}

std::string get_default_config_file_path() {
    const char * home_directory = getenv("HOME");
    if (home_directory) {
        const std::string default_config_path = std::string(home_directory) + "/.config/enyo/settings.json";
        if (fs::exists(default_config_path))
            return default_config_path;
    }
    return "";
}

void read_input_from_stdin(Uci& uci) {
    std::string input;
    while (std::getline(std::cin, input)) {
        if (!input.empty()) {
            uci(input);
        }
    }
}

void read_input_from_tty(Uci & uci) {
    int tty_fd = open("/dev/tty", O_RDONLY);
    if (tty_fd == -1) {
        std::cerr << fmt::format("Failed to open /dev/tty\n");
        return;
    }

    FILE * tty_file = fdopen(tty_fd, "r");
    if (tty_file == nullptr) {
        std::cerr << fmt::format("Failed to create file stream for /dev/tty\n");
        close(tty_fd);
        return;
    }

    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), tty_file) != nullptr) {
        std::string input(buffer);
        input.erase(input.find_last_not_of(" \n\r\t") + 1);
        if (!input.empty()) {
            uci(input);
            if (input == "quit")
                break;
        }
    }

    fclose(tty_file);
    close(tty_fd);
}

} // anon ns

// setoption name Ponder value false
int main(int argc, char **argv)
{
    fmt::print("id {}\n", g_version);

    Board b{"startpos"};
    Uci uci(b);

    const char* const short_opts = "t:c:l:f:p:w:b:h";
    const option long_opts[] = {
        { "threads", required_argument,  nullptr, 't' },
        { "config",  required_argument,  nullptr, 'c' },
        { "logfile", required_argument,  nullptr, 'l' },
        { "fen",     required_argument,  nullptr, 'f' },
        { "pgn",     required_argument,  nullptr, 'p' },
        { "white",   required_argument,  nullptr, 'w' },
        { "black",   required_argument,  nullptr, 'b' },
        { "help",    no_argument,        nullptr, 'h' },
        { nullptr,   no_argument,        nullptr, 0 }
    };

    int opt;
    std::string fen;
    std::string pgnfile;
    bool print_help = false;

    std::string config_file_path;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        switch (opt) {
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
            case 'f':
                fen = optarg;
                break;
            case 'p':
                pgnfile = optarg;
                break;
            case 'w':
                pgn.white_player = optarg;
                break;
            case 'b':
                pgn.black_player = optarg;
                break;
            case 'h':
                print_help = true;
                break;
            default:
                fmt::print("Error, no such option: '{}'\n", opt);
                fmt::print("Usage: {} [options]\n", argv[0]);
                return 1;
        }
    }

    if (print_help) {
        fmt::print("Usage: program_name [OPTIONS]\n");
        fmt::print("Options:\n");
        for (const option * param = long_opts; param->name != nullptr; ++param) {
            fmt::print("  -{}, --{}={}    {}\n",
                       static_cast<char>(param->val),
                       param->name,
                       (param->has_arg == required_argument ? "VALUE" : ""),
                       "Specify description here");
        }
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

    truncate(cfgmgr.logfile);

    if (!fen.empty())
        uci(fmt::format("position fen {}", fen));

    uci(fmt::format("setoption name Threads value {}", cfgmgr.num_threads));
    NNUE::Init("");

    if (!syzygy::init("./syzygy/wdl")) {
        fmt::print("info string error, failed to initialize tablebases\n");
        return 1;
    }

    init_search();

    read_input_from_stdin(uci);
    read_input_from_tty(uci);

    return 0;
}
