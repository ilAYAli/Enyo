#pragma once
// turn off threading with: setoption name threads value 0

#include "types.hpp"

#include <string>
#include <fstream>
#include <filesystem>
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
    static constexpr bool use_tt                     = true;
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
    return static_cast<Value>(Value::MATE - 1 - ply);
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
    bool use_lmr            = false;
    std::string nnue_file   = "nn-eba324f53044.nnue";
    std::string logfile     = "/tmp/enyo.log";

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
            //concat("UCI_Chess960",  "check", use_chess_960) +
            concat("nnue_file",     "string", nnue_file) +
            concat("logfile",       "string", logfile) +
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
        else if (lc == "nue_file")
            nnue_file = value;
        else if (lc == "use_lmr")
            use_lmr = (value == "true");
        else
            found = false;
        return found;
    }

private:
    // needs to be in sync with the public API:
    void update_config() {
        try {
            num_threads     = option["Threads"].get<int>();
            hash_size       = option["Hash"].get<int>();
            nnue_file       = option["Nnue_file"].get<std::string>();
            logfile         = option["Logfile"].get<std::string>();
            use_lmr         = option["Use_lmr"];

        } catch (const json::exception& e) {
            fmt::print("Error: Unable to parse config JSON: '{}\n", e.what());
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
