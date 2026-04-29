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
TEST(check, see_pawn_takes_pawn_defended) {
    // Pxe5: e5 defended by {Qc7, Bf6, Re6}; attacked by {Pf4, Rf5, Nd3}.
    // LVA capture sequence yields cumulative (white POV): +100,-100,+330,-320,+500,-500 = +10.
    // Minimax via backward induction settles on +10.
    std::string fen = "8/2q2k2/4rb2/4pR2/5P2/3N4/Q7/4K3 w - - 0 1";
    Board b(fen);

    auto move = resolve_move<white>(b, pawn, f4, e5);
    ASSERT_EQ(10, see<white>(b, move, 0));
}
TEST(check, see_queen_takes_defended_pawn) {
    // Qxb6: b6 is defended by a7 pawn only (d8 bishop blocked by c7 knight;
    // c7 knight does not attack b6). Exchange: Qxp, axQ, Bxp = p - q + p.
    std::string fen = "3b3k/p1n5/1p6/8/8/1Q2B3/8/7K w - - 0 1";
    Board b(fen);

    auto move = resolve_move<white>(b, queen, b3, b6);
    constexpr auto score = piece_value(pawn) - piece_value(queen) + piece_value(pawn);
    ASSERT_EQ(score, see<white>(b, move, 0));
}
TEST(check, see_knight_takes_defended_pawn) {
    // Nxd5: d5 defended by f6 knight; attackers N, B. Score = p - n + n.
    std::string fen = "k7/8/5n2/3p4/8/2N2B2/8/K7 w - - 0 1";
    Board b(fen);

    auto move = resolve_move<white>(b, knight, c3, d5);
    constexpr auto score = piece_value(pawn) - piece_value(knight) + piece_value(knight);
    ASSERT_EQ(score, see<white>(b, move, 0));
}
TEST(check, see_rook_battery_xray) {
    // Rxd4: rank-4 battery "3pRrRr" → d4=p, e4=R, f4=r, g4=R, h4=r.
    // Exchange (with X-ray): Rxp, Rxr, Rxr, Rxr = p - r + r - r.
    std::string fen = "2K5/8/8/8/3pRrRr/8/8/2k5 w - - 0 1";
    Board b(fen);

    auto move = resolve_move<white>(b, rook, e4, d4);
    constexpr auto score = piece_value(pawn) - piece_value(rook) + piece_value(rook) - piece_value(rook);
    ASSERT_EQ(score, see<white>(b, move, 0));
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

TEST(fen, round_trips_fullmove_counter) {
    Board b{"1r3rk1/p1q1pp1p/1np1b1p1/2Q5/8/1PNB1P2/P1P3PP/2KR3R b - - 0 17"};

    EXPECT_EQ(b.fen(), "1r3rk1/p1q1pp1p/1np1b1p1/2Q5/8/1PNB1P2/P1P3PP/2KR3R b - - 0 17");
}
#endif

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
