#pragma once

#include <fstream>
#include <unistd.h>
#include <fmt/format.h>
#include "config.hpp"
#include <thread>
#include <sstream>
#include <mutex>
#include <atomic>


namespace eventlog {

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


constexpr inline auto defaultLogLevel = Log::info;
constexpr inline auto logfilesToKeep = 20;

namespace fs = std::filesystem;

inline void remove_old_logfiles(const std::string& baseFilename, std::size_t maxFiles) {
    fs::path basePath(baseFilename);
    std::string logFilePattern = basePath.stem().string() + "_";
    fs::path logDir = basePath.parent_path();

    std::vector<fs::directory_entry> logFiles;

    for (const auto& entry : fs::directory_iterator(logDir)) {
        if (entry.is_regular_file()) {
            const auto& path = entry.path();
            if (path.filename().string().find(logFilePattern) == 0) {
                logFiles.push_back(entry);
            }
        }
    }

    std::sort(logFiles.begin(), logFiles.end(), [](const fs::directory_entry& a, const fs::directory_entry& b) {
        return fs::last_write_time(a) < fs::last_write_time(b);
    });

    while (logFiles.size() > maxFiles) {
        fs::remove(logFiles.front());
        logFiles.erase(logFiles.begin());
    }
}


inline void init() {
    remove_old_logfiles(enyo::cfgmgr.logfile, logfilesToKeep);
}

inline std::string getLogFilename(const std::string& baseFilename) {
    std::ostringstream oss;
    oss << baseFilename.substr(0, baseFilename.find_last_of('.'))
        << "_" << getpid()
        << "_" << std::this_thread::get_id()
        << baseFilename.substr(baseFilename.find_last_of('.'));
    return oss.str();
}

inline std::string logFilename = getLogFilename(enyo::cfgmgr.logfile);
inline std::ofstream logFile(logFilename, std::ios::app);

template <Log level = Log::info, typename... T>
inline void log(fmt::format_string<T...> fmtStr, T&&... args) {
    if constexpr (level <= defaultLogLevel) {
        if (logFile.is_open()) {
            constexpr std::size_t average_log_size = 64000;
            static thread_local std::vector<char> buffer(average_log_size);
            auto it = fmt::format_to(buffer.begin(), fmtStr, std::forward<T>(args)...);
            if (it > buffer.end()) {
                buffer.resize(it - buffer.begin());
                it = fmt::format_to(buffer.begin(), fmtStr, std::forward<T>(args)...);
            }
            logFile.write(buffer.data(), it - buffer.begin());
        }
    }
}

static inline std::atomic<bool> uci_debug_log = false;
template<typename... T>
static void ucilog(fmt::format_string<T...> fmt = {}, T&&... args) {
    if constexpr (true) {
        const auto str = fmt::format(fmt, std::forward<T>(args)...);
        log<Log::info>(fmt, std::forward<T>(args)...);
        if (!uci_debug_log && str.starts_with("info string"))
            return;
        fmt::print("{}", str);
        fflush(stdout);
    }
}

} // namespace eventlog
