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
TEST(hash, recompute_matches_initial_position) {
    Board b{"startpos"};
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));
}

TEST(hash, apply_revert_restores_hash) {
    Board b{"startpos"};
    const auto initial_hash = b.hash;

    auto move = resolve_move<white>(b, pawn, e2, e4);
    apply_move<white>(b, move);
    const auto applied_recomputed = zobrist::generate_hash(b);
    EXPECT_EQ(b.hash, applied_recomputed) << fmt::format("after apply fen={} inc={:016X} rec={:016X}", b.fen(), b.hash, applied_recomputed);

    revert_move<white>(b);
    const auto reverted_recomputed = zobrist::generate_hash(b);
    EXPECT_EQ(b.hash, reverted_recomputed) << fmt::format("after revert fen={} inc={:016X} rec={:016X}", b.fen(), b.hash, reverted_recomputed);
    EXPECT_EQ(b.hash, initial_hash) << fmt::format("initial={:016X} reverted={:016X}", initial_hash, b.hash);
}

TEST(hash, enpassant_double_push_matches_recompute) {
    Board b{"8/8/8/8/3p1p2/8/4P3/4K2k w - - 0 1"};
    auto move = resolve_move<white>(b, pawn, e2, e4);
    apply_move<white>(b, move);

    ASSERT_NE(0, b.gamestate.enpassant_square);
    const auto recomputed = zobrist::generate_hash(b);
    EXPECT_EQ(b.hash, recomputed) << fmt::format("fen={} inc={:016X} rec={:016X} ep={}", b.fen(), b.hash, recomputed, sq2str(b.gamestate.enpassant_square));
}

TEST(hash, compare_single_push_vs_double_push_delta) {
    Board single{"8/8/8/8/3p1p2/8/4P3/4K2k w - - 0 1"};
    Board dbl{single};

    auto e2e3 = resolve_move<white>(single, pawn, e2, e3);
    auto e2e4 = resolve_move<white>(dbl, pawn, e2, e4);
    apply_move<white>(single, e2e3);
    apply_move<white>(dbl, e2e4);

    const auto single_recomputed = zobrist::generate_hash(single);
    const auto double_recomputed = zobrist::generate_hash(dbl);

    EXPECT_EQ(single.hash, single_recomputed);
    EXPECT_EQ(double_recomputed ^ single_recomputed, dbl.hash ^ single.hash)
        << fmt::format("single={:016X}/{:016X} double={:016X}/{:016X}", single.hash, single_recomputed, dbl.hash, double_recomputed);
}

TEST(hash, castling_rights_mask_hash_matches_recompute) {
    Board b{"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"};
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));

    auto move = resolve_move<white>(b, rook, h1, h2);
    apply_move<white>(b, move);
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));

    revert_move<white>(b);
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));
}
#endif

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
