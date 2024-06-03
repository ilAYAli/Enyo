#pragma once

#include <fstream>
#include "unistd.h"
#include "fmt/format.h"
#include "config.hpp"

//#ifdef PROFILE_BUILD
//    inline constexpr auto dbg_enabled = false;
//#else
//    inline constexpr auto dbg_enabled = false;
//#endif

inline constexpr auto dbg_enabled = true;
inline constexpr auto log_enabled = false;

namespace eventlog {

enum class Log {
    debug,
    verbose,
    info,
    warning,
    error,
    none,
};

inline auto logLevel = Log::none;

inline constexpr const char* levelToString(Log l) {
    switch (l) {
        case Log::verbose:  return "[verbose]";
        case Log::debug:    return "[debug]";
        case Log::info:     return "[info]";
        case Log::warning:  return "[warning]";
        case Log::error:    return "[error]";
        default:            return "[unknown]";
    }
}

// compile time: debug (debug on):
template <Log level = Log::debug, typename... T>
void debug(fmt::format_string<T...> fmt, T &&... args) {
    if constexpr (true) {
        auto const filename = fmt::format("{}", enyo::cfgmgr.logfile);
        auto const str = fmt::format(fmt, std::forward<T>(args)...);
        std::ofstream logFile(filename, std::ios::app);
        if (logFile.is_open()) {
            logFile << fmt::format("{}", str);
            logFile.close();
        }
    }
}

// runtime: system logging:
template<typename... T>
void log(Log level, fmt::format_string<T...> fmt = {}, T&&... args) {
    if (level < logLevel)
        return;
    auto const str = fmt::format(fmt, std::forward<T>(args)...);
    fmt::print("{}",  str);
}

}
