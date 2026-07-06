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

#ifndef ENYO_USE_SYZYGY
#define ENYO_USE_SYZYGY 0
#endif


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
    static constexpr bool use_nullmove               = true; // +195.2 Elo
    static constexpr bool use_aspiration_window      = true; // +274.6 Elo
    static constexpr bool use_razoring               = true; // +135.5 Elo
    static constexpr bool use_probcut                = true;
    static constexpr bool use_syzygy                 = ENYO_USE_SYZYGY != 0;
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
    int hash_size           = 1024;
    int root_repetition_contempt = 24;
    // Score swing (cp) between iterations that counts as instability for
    // the soft-time extension. TODO Track 1: SPRT 60/80/150 variants.
    int tm_volatility_threshold = 106;
    // LMR formula constants used by init_search() to precompute the
    // [depth][move_count] reduction table. Stored *100 so UCI spin can
    // express fractions (e.g. lmr_base 125 -> 1.25). Changing either
    // forces a rebuild of the table via init_search().
    int lmr_base = 142;
    int lmr_divisor = 106;
    // Set when either lmr_* knob is touched at runtime; consumers must
    // re-run init_search() and clear it. Kept out of setopt's return so
    // callers don't have to special-case the recompute.
    bool lmr_dirty = false;
    // Search defaults promoted from SPSA; UCI options may override them.
    int razor_depth = 3;
    int razor_margin = 53;
    int razor_depth_factor = 194;
    int rfp_depth = 7;
    int rfp_margin = 64;
    int rfp_improving = 71;
    int nmp_base = 7;
    int nmp_depth_div = 4;
    int nmp_eval_div = 195;
    int probcut_margin = 140;
    int futility_depth = 6;
    int futility_margin = 136;
    int lmp_depth = 5;
    int lmp_base = 5;
    int lmp_improving_base = 8;
    int se_depth = 11;
    int se_margin_mult = 4;
    int se_entry_slack = 2;
    int se_double_margin = 32;
    int se_double_cap = 4;
    int lmr_cutoff_cnt = 4;
    int lmr_hist_div = 6929;
    int hist_bonus_cap = 1656;
    int hist_bonus_scale = 33;
    int asp_delta = 16;
    int asp_depth = 5;
    bool use_chess_960      = false;
    bool use_syzygy         = true;
    std::string nnue_file     = "";  // empty = embedded default
    std::string move_policy_file = "";
    int move_policy_max_eval_drop = 80;
    std::string logfile     = "/tmp/enyo.log";
    std::string syzygy_path = "";
    std::vector<std::pair<std::string, std::string>> uci_options;

    static ConfigManager& instance() {
        static ConfigManager instance;
        return instance;
    }

    bool load_config(const std::string& filename) {
        uci_options.clear();
        if (!fs::exists(filename) || !fs::is_regular_file(filename))
            return false;

        std::ifstream config_file(filename);
        if (!config_file)
            return false;

        try {
            json config;
            config_file >> config;
            if (!config.is_object()
                || config.size() != 1
                || !config.contains("uci_options")
                || !config["uci_options"].is_object()) {
                fmt::print(stderr,
                    "error: config '{}' must contain only an object named 'uci_options'\n",
                    filename);
                return false;
            }

            ConfigManager validated = *this;
            for (auto it = config["uci_options"].begin();
                 it != config["uci_options"].end(); ++it) {
                if (!it.value().is_primitive()) {
                    fmt::print(stderr,
                        "error: UCI option '{}' in '{}' must be a scalar value\n",
                        it.key(), filename);
                    uci_options.clear();
                    return false;
                }
                const auto value = json_value_to_string(it.value());
                if (!validated.setopt(it.key(), value)) {
                    fmt::print(stderr,
                        "error: unknown UCI option '{}' in '{}'\n",
                        it.key(), filename);
                    uci_options.clear();
                    return false;
                }
                uci_options.emplace_back(it.key(), value);
            }
            return true;
        } catch (const json::exception& e) {
            fmt::print(stderr, "error: failed to parse config '{}': {}\n", filename, e.what());
            uci_options.clear();
            return false;
        } catch (const std::exception& e) {
            fmt::print(stderr, "error: invalid option in config '{}': {}\n", filename, e.what());
            uci_options.clear();
            return false;
        }
    }

    std::string allopts() const {
        return
            concat("Threads",       "spin", num_threads, int(1), int(64)) +
            concat("Hash",          "spin", hash_size, int(1), int(33554432)) +
            concat("root_repetition_contempt", "spin", root_repetition_contempt, int(0), int(1000)) +
            concat("tm_volatility_threshold", "spin", tm_volatility_threshold, int(1), int(10000)) +
            concat("lmr_base", "spin", lmr_base, int(1), int(500)) +
            concat("lmr_divisor", "spin", lmr_divisor, int(50), int(1000)) +
            concat("razor_depth", "spin", razor_depth, int(1), int(10)) +
            concat("razor_margin", "spin", razor_margin, int(1), int(500)) +
            concat("razor_depth_factor", "spin", razor_depth_factor, int(1), int(1000)) +
            concat("rfp_depth", "spin", rfp_depth, int(1), int(20)) +
            concat("rfp_margin", "spin", rfp_margin, int(1), int(300)) +
            concat("rfp_improving", "spin", rfp_improving, int(0), int(300)) +
            concat("nmp_base", "spin", nmp_base, int(1), int(10)) +
            concat("nmp_depth_div", "spin", nmp_depth_div, int(1), int(20)) +
            concat("nmp_eval_div", "spin", nmp_eval_div, int(50), int(1000)) +
            concat("probcut_margin", "spin", probcut_margin, int(50), int(500)) +
            concat("futility_depth", "spin", futility_depth, int(1), int(15)) +
            concat("futility_margin", "spin", futility_margin, int(20), int(500)) +
            concat("lmp_depth", "spin", lmp_depth, int(1), int(15)) +
            concat("lmp_base", "spin", lmp_base, int(0), int(20)) +
            concat("lmp_improving_base", "spin", lmp_improving_base, int(0), int(20)) +
            concat("se_depth", "spin", se_depth, int(4), int(16)) +
            concat("se_margin_mult", "spin", se_margin_mult, int(1), int(10)) +
            concat("se_entry_slack", "spin", se_entry_slack, int(0), int(10)) +
            concat("se_double_margin", "spin", se_double_margin, int(5), int(300)) +
            concat("se_double_cap", "spin", se_double_cap, int(0), int(16)) +
            concat("lmr_cutoff_cnt", "spin", lmr_cutoff_cnt, int(1), int(10)) +
            concat("lmr_hist_div", "spin", lmr_hist_div, int(1024), int(65536)) +
            concat("hist_bonus_cap", "spin", hist_bonus_cap, int(256), int(8192)) +
            concat("hist_bonus_scale", "spin", hist_bonus_scale, int(4), int(128)) +
            concat("asp_delta", "spin", asp_delta, int(4), int(100)) +
            concat("asp_depth", "spin", asp_depth, int(1), int(12)) +
            concat("use_syzygy",    "check", use_syzygy) +
            //concat("UCI_Chess960",  "check", use_chess_960) +
            concat("nnue_file",     "string", nnue_file) +
            concat("move_policy_file", "string", move_policy_file) +
            concat("move_policy_max_eval_drop", "spin", move_policy_max_eval_drop, int(0), int(1000)) +
            concat("logfile",       "string", logfile) +
            concat("SyzygyPath",    "string", syzygy_path);
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
        else if (lc == "root_repetition_contempt")
            root_repetition_contempt = std::max(0, std::stoi(value));
        else if (lc == "tm_volatility_threshold")
            tm_volatility_threshold = std::max(1, std::stoi(value));
        else if (lc == "lmr_base") {
            lmr_base = std::max(1, std::stoi(value));
            lmr_dirty = true;
        }
        else if (lc == "lmr_divisor") {
            lmr_divisor = std::max(50, std::stoi(value));
            lmr_dirty = true;
        }
        else if (lc == "razor_depth")
            razor_depth = std::max(1, std::stoi(value));
        else if (lc == "razor_margin")
            razor_margin = std::max(1, std::stoi(value));
        else if (lc == "razor_depth_factor")
            razor_depth_factor = std::max(1, std::stoi(value));
        else if (lc == "rfp_depth")
            rfp_depth = std::max(1, std::stoi(value));
        else if (lc == "rfp_margin")
            rfp_margin = std::max(1, std::stoi(value));
        else if (lc == "rfp_improving")
            rfp_improving = std::max(0, std::stoi(value));
        else if (lc == "nmp_base")
            nmp_base = std::max(1, std::stoi(value));
        else if (lc == "nmp_depth_div")
            nmp_depth_div = std::max(1, std::stoi(value));
        else if (lc == "nmp_eval_div")
            nmp_eval_div = std::max(50, std::stoi(value));
        else if (lc == "probcut_margin")
            probcut_margin = std::max(50, std::stoi(value));
        else if (lc == "futility_depth")
            futility_depth = std::max(1, std::stoi(value));
        else if (lc == "futility_margin")
            futility_margin = std::max(20, std::stoi(value));
        else if (lc == "lmp_depth")
            lmp_depth = std::max(1, std::stoi(value));
        else if (lc == "lmp_base")
            lmp_base = std::max(0, std::stoi(value));
        else if (lc == "lmp_improving_base")
            lmp_improving_base = std::max(0, std::stoi(value));
        else if (lc == "se_depth")
            se_depth = std::max(4, std::stoi(value));
        else if (lc == "se_margin_mult")
            se_margin_mult = std::max(1, std::stoi(value));
        else if (lc == "se_entry_slack")
            se_entry_slack = std::max(0, std::stoi(value));
        else if (lc == "se_double_margin")
            se_double_margin = std::max(5, std::stoi(value));
        else if (lc == "se_double_cap")
            se_double_cap = std::max(0, std::stoi(value));
        else if (lc == "lmr_cutoff_cnt")
            lmr_cutoff_cnt = std::max(1, std::stoi(value));
        else if (lc == "lmr_hist_div")
            lmr_hist_div = std::max(1024, std::stoi(value));
        else if (lc == "hist_bonus_cap")
            hist_bonus_cap = std::max(256, std::stoi(value));
        else if (lc == "hist_bonus_scale")
            hist_bonus_scale = std::max(4, std::stoi(value));
        else if (lc == "asp_delta")
            asp_delta = std::max(4, std::stoi(value));
        else if (lc == "asp_depth")
            asp_depth = std::max(1, std::stoi(value));
        else if (lc == "uci_chess960")
            use_chess_960 = true;
        else if (lc == "use_syzygy")
            use_syzygy = (value == "true");
        else if (lc == "nnue_file")
            nnue_file = value;
        else if (lc == "move_policy_file")
            move_policy_file = value;
        else if (lc == "move_policy_max_eval_drop")
            move_policy_max_eval_drop = std::max(0, std::stoi(value));
        else if (lc == "logfile" || lc == "debug log file")
            logfile = value;
        else if (lc == "syzygypath")
            syzygy_path = value;
        else
            found = false;
        return found;
    }

    const std::vector<std::pair<std::string, std::string>>& configured_uci_options() const
    {
        return uci_options;
    }

private:
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
};


inline ConfigManager& cfgmgr = ConfigManager::instance();

} // enyo
