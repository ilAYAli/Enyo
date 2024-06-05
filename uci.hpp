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

extern bool UCILogEnabled;

static bool enable_debug = false;

class Uci {
public:
    explicit Uci(enyo::Board & board);

    void parse(std::string const & command);
    int operator()(const std::string & command);

    template<typename... T>
    static void log(fmt::format_string<T...> fmt = {}, T&&... args) {
        if constexpr (true) {
            const auto str = fmt::format(fmt, std::forward<T>(args)...);
            eventlog::debug("{}", str); // [tx]
            if (!enable_debug && str.starts_with("info string")) {
                return;
            }
            fmt::print("{}", str);
            fflush(stdout);
        }
    }

    std::chrono::milliseconds time_limit {};
    std::thread main_search_thread {};
    enyo::Board & b;
    //enyo::ThreadPool pool;
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

