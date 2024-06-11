// cppcheck-suppress knownConditionTrueFalse

#include <gtest/gtest.h>

#include "types.hpp"
#include "board.hpp"
#include "movegen.hpp"
#include "movegen_helper.hpp"
#include "uci.hpp"
#include "perft.hpp"
#include "zobrist.hpp"
#include "see.hpp"

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

#if 1
TEST(check, static_exchange_evaluation) {
    std::string fen = "8/2q2k2/4rb2/4pR2/5P2/3N4/Q7/4K3 w - - 0 1";
    Board b(fen);

    auto move = resolve_move<white>(b, pawn, f4, e5);
    //constexpr auto score = piece_value(pawn) - piece_value(queen) + piece_value(pawn) - piece_value(bishop);
    fmt::print("SEE: {}\n", see<white>(b, move, 0));
    //fmt::print("result: {}\n", see<white>(b, move, 0));
}
TEST(check, static_exchange_evaluation1) {
    std::string fen = "3b3k/p1n5/1p6/8/8/1Q2B3/8/7K w - - 0 1";
    Board b(fen);

    auto move = resolve_move<white>(b, queen, b3, b6);
    constexpr auto score = piece_value(pawn) - piece_value(queen) + piece_value(pawn) - piece_value(bishop);
    ASSERT_EQ(score, see<white>(b, move, 0));
    fmt::print("result: {}\n", see<white>(b, move, 0));
}
TEST(check, static_exchange_evaluation2) {
    std::string fen = "k7/8/5n2/3p4/8/2N2B2/8/K7 w - - 0 1";
    Board b(fen);

    auto move = resolve_move<white>(b, knight, c3, d5);
    constexpr auto score = piece_value(pawn) - piece_value(knight) + piece_value(knight);
    fmt::print("expected score: {}\n", score);
    ASSERT_EQ(score, see<white>(b, move, 0));
    fmt::print("result: {}\n", see<white>(b, move, 0));
}
TEST(check, static_exchange_evaluation3) {
    std::string fen = "2K5/8/8/8/3pRrRr/8/8/2k5 w - - 0 1";
    Board b(fen);

    fmt::print("{}\n", b);
    auto move = resolve_move<white>(b, rook, e5, d5);
    fmt::print("move.dst_piece: {}\n", move.dst_piece());
    constexpr auto score = piece_value(pawn) - piece_value(rook) + piece_value(rook) - piece_value(rook);
    fmt::print("expected score: {}\n", score);
    ASSERT_EQ(score, see<white>(b, move, 0));
    fmt::print("result: {}\n", see<white>(b, move, 0));
}
#endif

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
