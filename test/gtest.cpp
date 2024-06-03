// cppcheck-suppress knownConditionTrueFalse

#include <gtest/gtest.h>

#include "../board.hpp"
#include "../movegen.hpp"
#include "../uci.hpp"
#include "../perft.hpp"
//#include "../pos.hpp"
#include "../zobrist.hpp"

#include <ranges>
#include <nlohmann/json.hpp>

using namespace enyo;
using namespace nlohmann;

std::string get_default_config_file_path() {
    const char * home_directory = getenv("HOME");
    if (home_directory) {
        const std::string default_config_path = std::string(home_directory) + "/.config/enyo/settings json";
        if (fs::exists(default_config_path))
            return default_config_path;
    }
    return "";
}

bool load_config() {
    auto config_file_path = get_default_config_file_path();
    if (cfgmgr.load_config(config_file_path)) {
        fmt::print("Using config file: '{}'\n", config_file_path);
        return true;
    }
    return false;
}

constexpr Color get_side_to_move(std::string_view fen) {
    size_t pos = fen.find(' ');
    if (pos != std::string_view::npos) {
        char side = fen[pos + 1];
        return (side == 'w') ? Color::white : Color::black;
    }
    return white;
}


#if 0
TEST(check, invalid_king_move) {
    std::string fen = "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR w kq - 0 2";
    Board b(fen);

    //template <Color Us, bool UpdateZobrist = true, bool UpdateNNUE = true>
    auto moves = generate_legal_moves<white, false, false>(b);
      auto filtered_moves = moves | std::views::filter([](const Move &move) {
        return move.get_src() == e1; // Example condition
    });

    for (const auto &move : filtered_moves) {
        fmt::print("move: {}\n", move);
    }
}
#endif


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

