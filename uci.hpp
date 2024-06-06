#pragma once

#include <fstream>
#include <sstream>
#include <thread>
#include <string>
#include "fmt/format.h"

#include "fen.hpp"
#include "eventlog.hpp"
#include "thread.hpp"
#include "movelist.hpp"
#include "version.hpp"
#include "eventlog.hpp"

extern bool UCILogEnabled;
static bool uci_debug_log = false;

using namespace eventlog;

namespace enyo {

class Uci {
public:
    explicit Uci(Board & board);

    void parse(std::string const & command);
    int operator()(const std::string & command);

    std::chrono::milliseconds time_limit {};
    std::thread main_search_thread {};
    Board & b;
    std::string white_player = "?";
    std::string black_player = "?";

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
};

}
