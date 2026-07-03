#pragma once

#include <fstream>
#include <sstream>
#include <thread>
#include <string>
#include <string_view>
#include "fmt/format.h"

#include "fen.hpp"
#include "eventlog.hpp"
#include "thread.hpp"
#include "movelist.hpp"
#include "version.hpp"
#include "eventlog.hpp"

extern bool UCILogEnabled;

using namespace eventlog;

namespace enyo {

class Uci {
public:
    explicit Uci(Board & board);

    void parse(std::string const & command);
    int operator()(const std::string & command);
    void prepare_benchmark();
    uint64_t benchmark_position(std::string_view fen, int depth);

    std::chrono::milliseconds time_limit {};
    std::thread main_search_thread {};
    Board & b;
    std::string white_player = "?";
    std::string black_player = "?";
    bool quitting = false;

private:
    void uci();
    void setoption(std::istringstream& iss);
    void debug(std::istringstream& iss);
    void isready();
    void newgame();
    void position(std::istringstream& iss);
    void go(std::istringstream& iss);
    void bench(std::istringstream& iss);
    void pgn();
    void stop();
    void quit();
    void ensure_eval_loaded();

    bool eval_loaded = false;
};

}
