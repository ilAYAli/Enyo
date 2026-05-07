#pragma once
// turn off threading with: setoption name threads value 0

#include "types.hpp"

#include <string>
#include <fstream>
#include <filesystem>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include "fmt/format.h"


#ifdef Release
    #define BREAKPOINT(message)
#else
    #ifdef __arm64__
        #define BREAKPOINT(message) do { \
            fmt::print("BREAKPOINT: {}\n", message); \
            asm volatile("BRK 0"); \
        } while (0)
    #elif defined(__GNUC__)
        #define BREAKPOINT(message) do { \
            fmt::print("BREAKPOINT: {}\n", message); \
            __asm__("int3"); \
        } while (0)
    #endif
#endif


using json = nlohmann::json;
namespace fs = std::filesystem;

namespace enyo {


namespace Constexpr {
    // toggles:
    static constexpr bool use_qsearch                = true;
    static constexpr bool use_nullmove               = true; // +195.2 Elo
    static constexpr bool use_aspiration_window      = true; // +274.6 Elo
    static constexpr bool use_razoring               = true; // +135.5 Elo
    static constexpr bool use_probcut                = true;
    static constexpr bool use_tt                     = true;
    // Experimental: gate the qsearch TT-cut only. Probe still happens (tt_move
    // is still used for ordering), but the early return on flag match is
    // skipped. Negamax TT-cut is unaffected. Intended to isolate whether the
    // depth-unguarded qsearch cut is the source of warm-TT depth-1 blowups in
    // check-rich endgames (see odonata-bot vs EnyoBot JCKXOwix move 60).
    static constexpr bool use_qsearch_tt_cut         = true;
    static constexpr bool use_syzygy                 = true;
    // testing:
    static constexpr bool use_iir                    = true; //
    // looses elo:
    static constexpr bool use_rfp                    = false; //
    //static constexpr bool use_pvs                    = false; //

    // constants:
    static constexpr int max_moves                   = 256; // max # moves in a given position
    static constexpr int value_none                  = 32767;
    static constexpr int infinite                    = 32766;
    static constexpr int mate_value                  = 30000;
    static constexpr int mate_in_ply                 = mate_value - MAX_PLY;
    static constexpr int mated_in_ply                = -mate_in_ply;
    static constexpr int draw_value                  = 0;

    // debug:
    #ifndef CONSTEXPR_ASSERT
        #define CONSTEXPR_ASSERT false
    #endif
    constexpr bool constexpr_assert                  = CONSTEXPR_ASSERT;
    static constexpr bool debug_eval                 = false;
    static constexpr bool debug_move                 = false;
    static constexpr bool debug_threads              = false;
    static constexpr bool debug_tt                   = false;
    static constexpr bool debug_qsearch              = false;
    static constexpr bool debug_threefold_repetition = false;
    static constexpr bool check_ksq                  = true;
}

inline constexpr Value mate_in(int ply)
{
    return static_cast<Value>(Value::mate - 1 - ply);
}

inline constexpr Value mated_in(int ply)
{
    return static_cast<Value>(-Constexpr::mate_value + 1 + ply);
}

inline constexpr int mate_in_moves(int v)
{
    constexpr auto mate_in_max_ply = Constexpr::mate_value - MAX_PLY;
    if ((std::abs(v) > Constexpr::mate_value) || (std::abs(v) < mate_in_max_ply))
        return 0;
    return (v > 0
        ? Constexpr::mate_value - v + 1
        : -Constexpr::mate_value - v) / 2;
}

class ConfigManager {
public:
    int num_threads         = 1;
    int hash_size           = 64;
    bool use_chess_960      = false;
    bool use_tt             = true;
    bool use_tt_exact_cutoff = true;
    bool use_tt_lower_cutoff = false;
    bool use_tt_upper_cutoff = false;
    bool use_lmr            = false;
    std::string nnue_file   = "nn-eba324f53044.nnue";
    std::string nnue2_file  = "";  // empty = disabled (use embedded nnue)
    std::string logfile     = "/tmp/enyo.log";
    std::string syzygy_path = "";
    std::vector<std::pair<std::string, std::string>> uci_options;

    static ConfigManager& instance() {
        static ConfigManager instance;
        return instance;
    }

    bool load_config(const std::string& filename) {
        if (fs::exists(filename) && fs::is_regular_file(filename)) {
            std::ifstream config_file(filename);
            if (config_file) {
                config_file >> option;
                update_config();
                return true;
            }
        }
        return false;
    }

    std::string allopts() const {
        return
            concat("Threads",       "spin", num_threads, int(1), int(64)) +
            concat("Hash",          "spin", hash_size, int(1), int(33554432)) +
            concat("use_tt",        "check", use_tt) +
            concat("use_tt_exact_cutoff", "check", use_tt_exact_cutoff) +
            concat("use_tt_lower_cutoff", "check", use_tt_lower_cutoff) +
            concat("use_tt_upper_cutoff", "check", use_tt_upper_cutoff) +
            //concat("UCI_Chess960",  "check", use_chess_960) +
            concat("nnue_file",     "string", nnue_file) +
            concat("nnue2_file",    "string", nnue2_file) +
            concat("logfile",       "string", logfile) +
            concat("SyzygyPath",    "string", syzygy_path) +
            concat("use_lmr",       "check", use_lmr);
    }

    bool setopt(const std::string& opt, const std::string& value)
    {
        std::string lc = opt;
        std::transform(std::begin(lc), std::end(lc), std::begin(lc), ::tolower);

        bool found = true;
        if (lc == "threads")
            num_threads = std::stoi(value);
        else if (lc == "hash")
            hash_size = std::stoi(value);
        else if (lc == "uci_chess960")
            use_chess_960 = true;
        else if (lc == "use_tt")
            use_tt = (value == "true");
        else if (lc == "use_tt_exact_cutoff")
            use_tt_exact_cutoff = (value == "true");
        else if (lc == "use_tt_lower_cutoff")
            use_tt_lower_cutoff = (value == "true");
        else if (lc == "use_tt_upper_cutoff")
            use_tt_upper_cutoff = (value == "true");
        else if (lc == "nnue_file")
            nnue_file = value;
        else if (lc == "nnue2_file")
            nnue2_file = value;
        else if (lc == "logfile" || lc == "debug log file")
            logfile = value;
        else if (lc == "syzygypath")
            syzygy_path = value;
        else if (lc == "use_lmr")
            use_lmr = (value == "true");
        else
            found = false;
        return found;
    }

    const std::vector<std::pair<std::string, std::string>>& configured_uci_options() const
    {
        return uci_options;
    }

private:
    // Config is nested under "constants" / "toggles". Any field may be absent,
    // in which case the member's in-class default is kept — so a user's
    // settings.json only needs to carry the fields they care about overriding.
    void update_config() {
        try {
            const json empty = json::object();
            const auto& c = option.contains("constants") ? option["constants"] : empty;
            const auto& t = option.contains("toggles")   ? option["toggles"]   : empty;

            num_threads = c.value("threads",   num_threads);
            hash_size   = c.value("Hash",      hash_size);
            nnue_file   = c.value("nnue_file", nnue_file);
            nnue2_file  = c.value("nnue2_file", nnue2_file);
            logfile     = c.value("logfile",   logfile);
            syzygy_path = c.value("syzygy_path", syzygy_path);

            use_tt              = t.value("use_tt",              use_tt);
            use_tt_exact_cutoff = t.value("use_tt_exact_cutoff", use_tt_exact_cutoff);
            use_tt_lower_cutoff = t.value("use_tt_lower_cutoff", use_tt_lower_cutoff);
            use_tt_upper_cutoff = t.value("use_tt_upper_cutoff", use_tt_upper_cutoff);
            use_lmr             = t.value("use_lmr",             use_lmr);

            uci_options.clear();
            if (option.contains("uci_options"))
                load_uci_options(option["uci_options"]);
        } catch (const json::exception& e) {
            fmt::print("Error: Unable to parse config JSON: '{}'\n", e.what());
        }
    }

    static std::string json_value_to_string(const json& value)
    {
        if (value.is_string())
            return value.get<std::string>();
        if (value.is_boolean())
            return value.get<bool>() ? "true" : "false";
        if (value.is_number_integer())
            return fmt::format("{}", value.get<int64_t>());
        if (value.is_number_unsigned())
            return fmt::format("{}", value.get<uint64_t>());
        if (value.is_number_float())
            return fmt::format("{}", value.get<double>());
        if (value.is_null())
            return "";
        return value.dump();
    }

    void load_uci_options(const json& options)
    {
        if (options.is_object()) {
            for (auto it = options.begin(); it != options.end(); ++it)
                uci_options.emplace_back(it.key(), json_value_to_string(it.value()));
            return;
        }
        if (options.is_array()) {
            for (const auto& entry : options) {
                if (!entry.is_object() || !entry.contains("name"))
                    continue;
                uci_options.emplace_back(
                    entry.value("name", ""),
                    json_value_to_string(entry.value("value", json{})));
            }
        }
    }

    template <typename T>
    std::string concat(
        const std::string& name,
        const std::string& type,
        const T& value,
        T min = T{},
        T max = T{}) const {
        return fmt::format("option name {} type {} default {}", name, type, value)
            + (type == "spin" ? fmt::format(" min {} max {}", min, max) : "") + '\n';
    }

    json option;
};


inline ConfigManager& cfgmgr = ConfigManager::instance();

} // enyo
