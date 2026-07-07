#pragma once

#include <atomic>
#include <fstream>
#include <mutex>
#include <string_view>

#include <fmt/format.h>
#include <unistd.h>

#include "config.hpp"

namespace eventlog {

// Severity ordering: lower = noisier tracing, higher = more critical.
// A message logs iff its level >= defaultLogLevel.
enum class Log {
    none = 0,
    debug,
    verbose,
    uci,
    info,
    warning,
    error,
    all,
};


constexpr inline auto defaultLogLevel = Log::uci;

inline std::string logFilename;
inline std::ofstream logFile;
inline std::mutex logMutex;

inline void reopen_logfile(const std::string& filename, bool truncate = false)
{
    std::lock_guard lock(logMutex);

    // settings.json is replayed through UCI after startup. Do not reopen the
    // same file then: doing so would either append forever or erase the startup
    // banner that was just written.
    if (logFile.is_open() && logFilename == filename)
        return;

    logFile.close();
    logFilename = filename;
    logFile.open(logFilename, truncate ? std::ios::trunc : std::ios::app);
}

inline void init()
{
    reopen_logfile(enyo::cfgmgr.logfile, true);
}

template <Log level = Log::info, typename... T>
inline void log(fmt::format_string<T...> fmtStr, T&&... args) {
    if constexpr (level >= defaultLogLevel) {
        const auto s = fmt::format(fmtStr, std::forward<T>(args)...);
        std::lock_guard lock(logMutex);
        if (logFile.is_open()) {
            if constexpr (level == Log::error) {
                constexpr std::string_view prefix = "ERROR: ";
                logFile.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
            } else if constexpr (level == Log::warning) {
                constexpr std::string_view prefix = "WARNING: ";
                logFile.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
            }
            logFile.write(s.data(), static_cast<std::streamsize>(s.size()));
            if constexpr (level >= Log::warning)
                logFile.flush();
        }
    }
}

static inline std::atomic<bool> uci_debug_log = false;
template<typename... T>
static void ucilog(fmt::format_string<T...> fmt = {}, T&&... args) {
    if constexpr (true) {
        const auto str = fmt::format(fmt, std::forward<T>(args)...);
        log<Log::info>(fmt, std::forward<T>(args)...);
        if (!uci_debug_log
            && str.starts_with("info string")
            && !str.starts_with("info string evaluator="))
            return;
        fmt::print("{}", str);
        fflush(stdout);
    }
}

} // namespace eventlog
