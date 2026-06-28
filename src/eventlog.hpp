#pragma once

#include <fstream>
#include <unistd.h>
#include <fmt/format.h>
#include "config.hpp"
#include <thread>
#include <sstream>
#include <mutex>
#include <atomic>
#include <string_view>


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
constexpr inline auto logfilesToKeep = 20;

namespace fs = std::filesystem;

inline void remove_old_logfiles(const std::string& baseFilename, std::size_t maxFiles) {
    fs::path basePath(baseFilename);
    std::string logFilePattern = basePath.stem().string() + "_";
    fs::path logDir = basePath.parent_path();

    if (!fs::exists(logDir))
        return;

    struct LogFile {
        fs::path path;
        fs::file_time_type timestamp;
    };

    std::vector<LogFile> logFiles;

    for (const auto& entry : fs::directory_iterator(logDir)) {
        if (entry.is_regular_file()) {
            const auto& path = entry.path();
            if (path.filename().string().find(logFilePattern) == 0) {
                std::error_code ec;
                const auto timestamp = fs::last_write_time(path, ec);
                if (!ec)
                    logFiles.push_back({path, timestamp});
            }
        }
    }

    std::sort(logFiles.begin(), logFiles.end(), [](const LogFile& a, const LogFile& b) {
        if (a.timestamp != b.timestamp)
            return a.timestamp < b.timestamp;
        return a.path.filename() < b.path.filename();
    });

    while (logFiles.size() > maxFiles) {
        std::error_code ec;
        fs::remove(logFiles.front().path, ec);
        logFiles.erase(logFiles.begin());
    }
}


inline std::string getLogFilename(const std::string& baseFilename) {
    std::ostringstream oss;
    oss << baseFilename.substr(0, baseFilename.find_last_of('.'))
        << "_" << getpid()
        << "_" << std::this_thread::get_id()
        << baseFilename.substr(baseFilename.find_last_of('.'));
    return oss.str();
}

inline std::string getDefaultLogFilename()
{
    return getLogFilename(enyo::cfgmgr.logfile);
}

inline std::string logFilename = getDefaultLogFilename();
inline std::ofstream logFile;
inline std::mutex logMutex;

inline void reopen_logfile(const std::string& filename, bool exact_path = false)
{
    std::lock_guard lock(logMutex);
    logFile.close();
    logFilename = exact_path ? filename : getLogFilename(filename);
    logFile.open(logFilename, std::ios::app);
}

inline void init() {
    remove_old_logfiles(enyo::cfgmgr.logfile, logfilesToKeep);
    reopen_logfile(enyo::cfgmgr.logfile);
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
