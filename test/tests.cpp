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
#include "nnue.hpp"
#include "probe.hpp"
#include "search.hpp"
#include "thread.hpp"
#include "tt.hpp"
#include "pgn.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <ranges>
#include <vector>
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

bool init_test_syzygy(int min_pieces)
{
    std::vector<std::string> candidates;
    if (const char * env_path = std::getenv("ENYO_TEST_SYZYGY_PATH"); env_path && *env_path)
        candidates.emplace_back(env_path);
    if (!cfgmgr.syzygy_path.empty())
        candidates.push_back(cfgmgr.syzygy_path);
    candidates.emplace_back("~/code/cpp/chess/assets/tablebases");
    candidates.emplace_back("../assets/tablebases");

    for (const auto & path : candidates) {
        if (syzygy::init(path) && static_cast<int>(syzygy::largest()) >= min_pieces)
            return true;
    }
    return false;
}

int first_uci_cp_score(std::string const & out)
{
    const auto pos = out.find("score cp ");
    if (pos == std::string::npos)
        return std::numeric_limits<int>::min();
    return std::stoi(out.substr(pos + 9));
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

// get_fen computes the fullmove counter as full_moves + histply/2, which is
// only correct when the starting FEN was white-to-move. After parsing a
// black-to-move FEN, black's first reply (histply=1) already completes a
// full move and must tick the counter — histply/2 is still 0 and the FEN
// emits the wrong number, which poisons any downstream consumer that
// trusts the FEN (GUI display, PGN export, external reference engines).
TEST(fen, fullmove_counter_advances_correctly_from_black_to_move_start) {
    // White-to-move baseline: counter ticks after each black move (even ply).
    {
        Board b{"4k3/8/8/8/8/8/8/4K3 w - - 0 5"};
        EXPECT_EQ(b.fen(), "4k3/8/8/8/8/8/8/4K3 w - - 0 5");
        apply_move<white>(b, resolve_move<white>(b, king, e1, e2));
        EXPECT_EQ(b.fen(), "4k3/8/8/8/8/8/4K3/8 b - - 1 5");  // white moved; still move 5
        apply_move<black>(b, resolve_move<black>(b, king, e8, e7));
        EXPECT_EQ(b.fen(), "8/4k3/8/8/8/8/4K3/8 w - - 2 6");  // black completed move 5; now 6
    }
    // Black-to-move start: counter must tick after black's first move.
    {
        Board b{"4k3/8/8/8/8/8/8/4K3 b - - 0 5"};
        EXPECT_EQ(b.fen(), "4k3/8/8/8/8/8/8/4K3 b - - 0 5");
        apply_move<black>(b, resolve_move<black>(b, king, e8, e7));
        EXPECT_EQ(b.fen(), "8/4k3/8/8/8/8/8/4K3 w - - 1 6");  // black completed move 5; now 6
        apply_move<white>(b, resolve_move<white>(b, king, e1, e2));
        EXPECT_EQ(b.fen(), "8/4k3/8/8/8/8/4K3/8 b - - 2 6");  // white moved; still 6
        apply_move<black>(b, resolve_move<black>(b, king, e7, e8));
        EXPECT_EQ(b.fen(), "4k3/8/8/8/8/8/4K3/8 w - - 3 7");  // black completed move 6; now 7
    }
}

// get_fen() emits `b.half_moves + b.gamestate.half_moves`, treating
// b.half_moves as a delta from the FEN-parsed base. When a reset event
// (capture, pawn move, en-passant, promotion) zeroed b.half_moves but
// left b.gamestate.half_moves at the parsed base, the emitted FEN still
// showed the stale base instead of 0. Syzygy rule50 (which reads the
// same sum via board2pos) then fed Fathom an inflated counter and its
// WDL wrapper rejected legal probes under "nonzero rule50."
//
// Fix: reset_halfmove_clock() clears both fields together; Undo captures
// gamestate before apply mutates it, so revert rolls both back.
TEST(fen, halfmove_clock_resets_to_zero_with_nonzero_fen_base) {
    struct Case {
        const char * fen;
        square_t     src;
        square_t     dst;
        PieceType    src_piece;
        PieceType    promo;       // no_piece_type for non-promotion
        const char * post_fen;
        const char * label;
    };
    const Case cases[] = {
        {"4k3/8/8/8/8/8/4P3/4K3 w - - 7 12", e2, e4, pawn, no_piece_type,
         "4k3/8/8/8/4P3/8/8/4K3 b - - 0 12", "pawn double-push"},
        {"4k3/8/8/3p4/4P3/8/8/4K3 w - - 7 12", e4, d5, pawn, no_piece_type,
         "4k3/8/8/3P4/8/8/8/4K3 b - - 0 12", "pawn capture"},
        {"4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12", e5, d6, pawn, no_piece_type,
         "4k3/8/3P4/8/8/8/8/4K3 b - - 0 12", "en-passant"},
        {"4k3/P7/8/8/8/8/8/4K3 w - - 7 12", a7, a8, pawn, queen,
         "Q3k3/8/8/8/8/8/8/4K3 b - - 0 12", "promotion without capture"},
    };
    for (const auto & c : cases) {
        Board b{c.fen};
        auto m = resolve_move<white>(b, c.src_piece, c.src, c.dst);
        if (c.promo != no_piece_type)
            m.set_promo_piece(c.promo);
        apply_move<white>(b, m);
        EXPECT_EQ(b.fen(), c.post_fen) << c.label;
    }

    // Counter-check: castle and quiet king moves must still carry the
    // FEN base forward, incrementing rather than resetting.
    {
        Board b{"r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 7 12"};
        apply_move<white>(b, resolve_move<white>(b, king, e1, g1));
        EXPECT_EQ(b.fen(), "r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R4RK1 b kq - 8 12");
    }
    {
        Board b{"4k3/8/8/8/8/5N2/8/4K3 w - - 7 12"};
        apply_move<white>(b, resolve_move<white>(b, knight, f3, e5));
        EXPECT_EQ(b.fen(), "4k3/8/8/4N3/8/8/8/4K3 b - - 8 12");
    }
}

TEST(pgn, uci_to_algebra_formats_check_without_mutating_board) {
    Board b{"4k3/8/8/8/8/8/8/3R3K w - - 0 1"};
    const auto before = b.fen();

    auto move = uci_to_move(b, "d1d8");
    ASSERT_TRUE(move.has_value());
    EXPECT_EQ("Rd8+", move_to_algebra(b, *move));
    EXPECT_EQ("Rd8+", uci_to_algebra(b, "d1d8"));
    EXPECT_EQ(before, b.fen());
}

TEST(movegen, lichess_fjmp5yck_endgame_sequence_preserves_position) {
    Board b{"1r6/2K5/R6p/1P5P/3b2k1/8/8/8 w - - 0 118"};
    const auto initial_hash = b.hash;
    const auto initial_fen = b.fen();

    apply_move<white>(b, resolve_move<white>(b, king, c7, b8));
    EXPECT_EQ(b.fen(), "1K6/8/R6p/1P5P/3b2k1/8/8/8 b - - 0 118");

    apply_move<black>(b, resolve_move<black>(b, king, g4, h5));
    EXPECT_EQ(b.fen(), "1K6/8/R6p/1P5k/3b4/8/8/8 w - - 0 119");

    apply_move<white>(b, resolve_move<white>(b, rook, a6, d6));
    EXPECT_EQ(b.fen(), "1K6/8/3R3p/1P5k/3b4/8/8/8 b - - 1 119");

    apply_move<black>(b, resolve_move<black>(b, bishop, d4, e3));
    EXPECT_EQ(b.fen(), "1K6/8/3R3p/1P5k/8/4b3/8/8 w - - 2 120");

    apply_move<white>(b, resolve_move<white>(b, rook, d6, d5));
    EXPECT_EQ(b.fen(), "1K6/8/7p/1P1R3k/8/4b3/8/8 b - - 3 120");

    apply_move<black>(b, resolve_move<black>(b, king, h5, g4));
    EXPECT_EQ(b.fen(), "1K6/8/7p/1P1R4/6k1/4b3/8/8 w - - 4 121");

    revert_move<black>(b);
    revert_move<white>(b);
    revert_move<black>(b);
    revert_move<white>(b);
    revert_move<black>(b);
    revert_move<white>(b);

    EXPECT_EQ(b.fen(), initial_fen);
    EXPECT_EQ(b.hash, initial_hash);
}

// Castling bugs caught before this was deployed:
//   1. white O-O incorrectly SET white_ooo (copy-paste of "false" -> "true")
//   2. black O-O-O XOR'd the wrong zobrist indices (2/3 swapped)
//   3. opposite-side rights XOR'd unconditionally, injecting phantom bits
//      into the hash whenever the bit was already cleared
// Each assertion compares the incrementally-maintained hash against
// zobrist::generate_hash (which rebuilds the hash from the current state),
// and verifies the castling bits are correct in gamestate. Apply/revert
// is also checked to round-trip both state and hash.
TEST(movegen, castle_updates_rights_and_hash_white_kingside) {
    Board b{"r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1"};
    const auto before_hash = b.hash;
    const auto before_fen = b.fen();

    apply_move<white>(b, resolve_move<white>(b, king, e1, g1));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::white_oo));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::white_ooo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::black_oo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::black_ooo));
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));

    revert_move<white>(b);
    EXPECT_EQ(b.hash, before_hash);
    EXPECT_EQ(b.fen(), before_fen);
}

TEST(movegen, castle_updates_rights_and_hash_white_queenside) {
    Board b{"r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1"};
    const auto before_hash = b.hash;
    const auto before_fen = b.fen();

    apply_move<white>(b, resolve_move<white>(b, king, e1, c1));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::white_oo));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::white_ooo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::black_oo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::black_ooo));
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));

    revert_move<white>(b);
    EXPECT_EQ(b.hash, before_hash);
    EXPECT_EQ(b.fen(), before_fen);
}

TEST(movegen, castle_updates_rights_and_hash_black_kingside) {
    Board b{"r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R b KQkq - 0 1"};
    const auto before_hash = b.hash;
    const auto before_fen = b.fen();

    apply_move<black>(b, resolve_move<black>(b, king, e8, g8));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::black_oo));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::black_ooo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::white_oo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::white_ooo));
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));

    revert_move<black>(b);
    EXPECT_EQ(b.hash, before_hash);
    EXPECT_EQ(b.fen(), before_fen);
}

TEST(movegen, castle_updates_rights_and_hash_black_queenside) {
    Board b{"r3k2r/pppqbppp/2np1n2/4p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R b KQkq - 0 1"};
    const auto before_hash = b.hash;
    const auto before_fen = b.fen();

    apply_move<black>(b, resolve_move<black>(b, king, e8, c8));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::black_oo));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::black_ooo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::white_oo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::white_ooo));
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));

    revert_move<black>(b);
    EXPECT_EQ(b.hash, before_hash);
    EXPECT_EQ(b.fen(), before_fen);
}

// Catches unconditional XOR of opposite-side rights: if the "other" castle bit
// is already cleared when we castle, XOR'ing its zobrist key injects a phantom
// bit into the hash. All four FENs start with only the *castling* side's
// rights, so the only correct hash delta is the one for the side that moved.
TEST(movegen, castle_does_not_touch_cleared_opposite_side_rights) {
    struct Case {
        const char * fen;
        const char * post_castle_fen;
    };
    // Each case: one-sided pre-castle rights; exercising the side that still
    // has rights must not XOR or set the already-cleared opposite bit, and
    // apply/revert must round-trip hash and FEN exactly.
    const Case cases[] = {
        {"4k3/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w K - 0 1",
         "4k3/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R4RK1 b - - 1 1"},  // W OO, W-OOO already cleared
        {"4k3/pppq1ppp/2np1n2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w Q - 0 1",
         "4k3/pppq1ppp/2np1n2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/2KR3R b - - 1 1"},  // W OOO, W-OO already cleared
        {"r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/4K3 b k - 0 1",
         "r4rk1/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/4K3 w - - 1 2"},  // B OO, B-OOO already cleared
        {"r3k2r/pppqbppp/2np1n2/4p3/2B1P3/2NP1N2/PPPQ1PPP/4K3 b q - 0 1",
         "2kr3r/pppqbppp/2np1n2/4p3/2B1P3/2NP1N2/PPPQ1PPP/4K3 w - - 1 2"},    // B OOO, B-OO already cleared
    };
    for (const auto & c : cases) {
        Board b{c.fen};
        const auto expected_before = zobrist::generate_hash(b);
        ASSERT_EQ(b.hash, expected_before) << c.fen;
        if (b.side == white) {
            const auto dst = b.gamestate.can_castle(CastlingRights::white_oo) ? g1 : c1;
            apply_move<white>(b, resolve_move<white>(b, king, e1, dst));
            EXPECT_EQ(b.fen(), c.post_castle_fen) << "apply: " << c.fen;
            EXPECT_EQ(b.hash, zobrist::generate_hash(b)) << "apply: " << c.fen;
            revert_move<white>(b);
        } else {
            const auto dst = b.gamestate.can_castle(CastlingRights::black_oo) ? g8 : c8;
            apply_move<black>(b, resolve_move<black>(b, king, e8, dst));
            EXPECT_EQ(b.fen(), c.post_castle_fen) << "apply: " << c.fen;
            EXPECT_EQ(b.hash, zobrist::generate_hash(b)) << "apply: " << c.fen;
            revert_move<black>(b);
        }
        EXPECT_EQ(b.fen(), c.fen) << "revert: " << c.fen;
        EXPECT_EQ(b.hash, expected_before) << "revert: " << c.fen;
        EXPECT_EQ(b.hash, zobrist::generate_hash(b)) << "revert: " << c.fen;
    }
}

// When a rook is captured on its home square, the corresponding side's
// castling right must be cleared in both state and hash. apply_move_generic's
// switch keys on src_piece (the capturer), so historically only a rook
// *moving* from its home cleared the right — the capture path never did.
// Same reachable position via "rook captured" vs "rook moved and returned"
// then produces different 64-bit hashes: TT fragmentation.
TEST(movegen, captured_home_rook_clears_castling_rights) {
    struct Case {
        const char * fen;
        square_t     src;
        square_t     dst;
        const char * post_fen;   // expected FEN after the capture
        Color        mover;
    };
    const Case cases[] = {
        // White knight b6 captures black rook on a8 — black queenside right clears.
        {"r3k2r/2pq1ppp/pN1p1n2/4p3/8/2N5/PPPQ1PPP/R3K2R w KQkq - 0 1", b6, a8,
         "N3k2r/2pq1ppp/p2p1n2/4p3/8/2N5/PPPQ1PPP/R3K2R b KQk - 0 1", white},
        // White knight g6 captures black rook on h8 — black kingside right clears.
        {"r3k2r/ppp2pp1/3p1nNp/4p3/8/2N5/PPPQ1PPP/R3K2R w KQkq - 0 1", g6, h8,
         "r3k2N/ppp2pp1/3p1n1p/4p3/8/2N5/PPPQ1PPP/R3K2R b KQq - 0 1", white},
        // Black queen a2 captures white rook on a1 — white queenside right clears.
        {"r3k2r/pppq1ppp/2np1n2/4p3/8/2N5/qPPQ1PPP/R3K2R b KQkq - 0 1", a2, a1,
         "r3k2r/pppq1ppp/2np1n2/4p3/8/2N5/1PPQ1PPP/q3K2R w Kkq - 0 2", black},
        // Black queen h5 captures white rook on h1 — white kingside right clears.
        {"r3k2r/pppq1pp1/2np1n2/4p2q/8/2N5/PPPQ1PP1/R3K2R b KQkq - 0 1", h5, h1,
         "r3k2r/pppq1pp1/2np1n2/4p3/8/2N5/PPPQ1PP1/R3K2q w Qkq - 0 2", black},
    };
    for (const auto & c : cases) {
        Board b{c.fen};
        const auto before_hash = b.hash;
        ASSERT_EQ(b.hash, zobrist::generate_hash(b)) << c.fen;
        if (c.mover == white) {
            const auto pt = get_piece_type<white>(b, c.src);
            apply_move<white>(b, resolve_move<white>(b, pt, c.src, c.dst));
        } else {
            const auto pt = get_piece_type<black>(b, c.src);
            apply_move<black>(b, resolve_move<black>(b, pt, c.src, c.dst));
        }
        EXPECT_EQ(b.fen(), c.post_fen) << "apply: " << c.fen;
        EXPECT_EQ(b.hash, zobrist::generate_hash(b)) << "apply: " << c.fen;
        if (c.mover == white) revert_move<white>(b); else revert_move<black>(b);
        EXPECT_EQ(b.fen(), c.fen) << "revert: " << c.fen;
        EXPECT_EQ(b.hash, before_hash) << "revert: " << c.fen;
    }
}

// Promotion-capture variant of the above. Promotion captures flow through
// apply_promotion/revert_promotion, not apply_move_generic, so they need
// their own path for clearing the captured home rook's castling right.
// Same TT-fragmentation hazard as captured_home_rook_clears_castling_rights.
TEST(movegen, promotion_capture_home_rook_clears_castling_rights) {
    struct Case {
        const char * fen;
        square_t     src;
        square_t     dst;
        const char * post_fen;   // expected FEN after the promotion capture
        Color        mover;
    };
    const Case cases[] = {
        // White pawn b7 captures black rook on a8, promoting to queen — black queenside right clears.
        {"r3k2r/1P6/8/8/8/8/8/4K3 w kq - 0 1", b7, a8,
         "Q3k2r/8/8/8/8/8/8/4K3 b k - 0 1", white},
        // White pawn g7 captures black rook on h8, promoting to queen — black kingside right clears.
        {"r3k2r/6P1/8/8/8/8/8/4K3 w kq - 0 1", g7, h8,
         "r3k2Q/8/8/8/8/8/8/4K3 b q - 0 1", white},
        // Black pawn b2 captures white rook on a1, promoting to queen — white queenside right clears.
        {"4k3/8/8/8/8/8/1p6/R3K2R b KQ - 0 1", b2, a1,
         "4k3/8/8/8/8/8/8/q3K2R w K - 0 2", black},
        // Black pawn g2 captures white rook on h1, promoting to queen — white kingside right clears.
        {"4k3/8/8/8/8/8/6p1/R3K2R b KQ - 0 1", g2, h1,
         "4k3/8/8/8/8/8/8/R3K2q w Q - 0 2", black},
    };
    for (const auto & c : cases) {
        Board b{c.fen};
        const auto before_hash = b.hash;
        ASSERT_EQ(b.hash, zobrist::generate_hash(b)) << c.fen;
        if (c.mover == white) {
            auto m = resolve_move<white>(b, pawn, c.src, c.dst);
            m.set_promo_piece(PieceType::queen);
            apply_move<white>(b, m);
        } else {
            auto m = resolve_move<black>(b, pawn, c.src, c.dst);
            m.set_promo_piece(PieceType::queen);
            apply_move<black>(b, m);
        }
        EXPECT_EQ(b.fen(), c.post_fen) << "apply: " << c.fen;
        EXPECT_EQ(b.hash, zobrist::generate_hash(b)) << "apply: " << c.fen;
        if (c.mover == white) revert_move<white>(b); else revert_move<black>(b);
        EXPECT_EQ(b.fen(), c.fen) << "revert: " << c.fen;
        EXPECT_EQ(b.hash, before_hash) << "revert: " << c.fen;
    }
}

// Pawn double-push probes the two diagonal neighbours on the landing rank
// for an enemy pawn that would make en-passant possible. The neighbour
// lookup used (1ULL << (dst+1)) | (1ULL << (dst-1)) without masking to the
// landing rank, so a-file and h-file double-pushes wrapped onto the
// adjacent rank: e.g. white a2a4 (dst=a4=bit31) set dst+1=32=h5, and a
// black pawn sitting on h5 was then read as a "b4 neighbour" that armed
// en-passant. Result: bogus ep square in the post-move state, wrong FEN,
// and a poisoned zobrist (the ep term gets XOR'd in). Legal moves generated
// from the poisoned state can include an illegal en-passant capture.
TEST(movegen, double_push_does_not_wrap_ep_to_opposite_file) {
    struct Case {
        const char * fen;
        square_t     src;
        square_t     dst;
        const char * post_fen;   // expected FEN after the double push (no ep)
        Color        mover;
    };
    const Case cases[] = {
        // White a2-a4 with a black pawn on h5: pre-fix, (a4+1)=h5 hit the
        // black pawn and spuriously set ep_square=a3.
        {"4k3/8/8/7p/8/8/P7/4K3 w - - 0 1", a2, a4,
         "4k3/8/8/7p/P7/8/8/4K3 b - - 0 1", white},
        // White h2-h4 with a black pawn on a3: pre-fix, (h4-1)=a3 hit it.
        {"4k3/8/8/8/8/p7/7P/4K3 w - - 0 1", h2, h4,
         "4k3/8/8/8/7P/p7/8/4K3 b - - 0 1", white},
        // Black a7-a5 with a white pawn on h6: pre-fix, (a5+1)=h6 hit it.
        {"4k3/p7/7P/8/8/8/8/4K3 b - - 0 1", a7, a5,
         "4k3/8/7P/p7/8/8/8/4K3 w - - 0 2", black},
        // Black h7-h5 with a white pawn on a4: pre-fix, (h5-1)=a4 hit it.
        {"4k3/7p/8/8/P7/8/8/4K3 b - - 0 1", h7, h5,
         "4k3/8/8/7p/P7/8/8/4K3 w - - 0 2", black},
    };
    for (const auto & c : cases) {
        Board b{c.fen};
        const auto before_hash = b.hash;
        ASSERT_EQ(b.hash, zobrist::generate_hash(b)) << c.fen;
        if (c.mover == white) {
            apply_move<white>(b, resolve_move<white>(b, pawn, c.src, c.dst));
        } else {
            apply_move<black>(b, resolve_move<black>(b, pawn, c.src, c.dst));
        }
        EXPECT_EQ(b.fen(), c.post_fen) << "apply: " << c.fen;
        EXPECT_EQ(b.hash, zobrist::generate_hash(b)) << "apply: " << c.fen;
        EXPECT_EQ(b.gamestate.enpassant_square, 0) << "apply: " << c.fen;
        if (c.mover == white) revert_move<white>(b); else revert_move<black>(b);
        EXPECT_EQ(b.fen(), c.fen) << "revert: " << c.fen;
        EXPECT_EQ(b.hash, before_hash) << "revert: " << c.fen;
    }
}

// Castling is neither a capture nor a pawn move, so per FIDE the 50-move
// halfmove clock must *increment* across a castle, not reset. apply_move
// historically zeroed b.half_moves in the castle branch, which (a) emits a
// wrong FEN, (b) mis-counts the 50-move rule, and (c) truncates the
// is_repetition() lookback window — which uses b.half_moves as the scan
// depth into history — allowing repetitions that straddle a castle to go
// undetected.
TEST(movegen, castle_does_not_reset_halfmove_clock) {
    struct Case {
        const char * fen;
        square_t     src;
        square_t     dst;
        Color        mover;
        const char * label;
    };
    const Case cases[] = {
        {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 5 10", e1, g1, white, "white O-O"},
        {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 5 10", e1, c1, white, "white O-O-O"},
        {"r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 5 10", e8, g8, black, "black O-O"},
        {"r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 5 10", e8, c8, black, "black O-O-O"},
    };
    for (const auto & c : cases) {
        Board b{c.fen};
        const int before_counter = b.half_moves + b.gamestate.half_moves;
        if (c.mover == white)
            apply_move<white>(b, resolve_move<white>(b, king, c.src, c.dst));
        else
            apply_move<black>(b, resolve_move<black>(b, king, c.src, c.dst));
        const int after_counter = b.half_moves + b.gamestate.half_moves;
        EXPECT_EQ(after_counter, before_counter + 1) << c.label;
    }
}

TEST(search, repetition_detects_claimable_third_occurrence) {
    Board b{"startpos"};

    apply_move<white>(b, resolve_move<white>(b, knight, g1, f3));
    apply_move<black>(b, resolve_move<black>(b, knight, g8, f6));
    apply_move<white>(b, resolve_move<white>(b, knight, f3, g1));
    apply_move<black>(b, resolve_move<black>(b, knight, f6, g8));
    apply_move<white>(b, resolve_move<white>(b, knight, g1, f3));
    apply_move<black>(b, resolve_move<black>(b, knight, g8, f6));
    apply_move<white>(b, resolve_move<white>(b, knight, f3, g1));
    apply_move<black>(b, resolve_move<black>(b, knight, f6, g8));

    EXPECT_TRUE(is_repetition(b, 1));
    EXPECT_FALSE(is_repetition(b, 2));
}

TEST(search, hypersion_check_net_root_evasions_find_safer_defenses) {
    const struct {
        std::string_view fen;
        std::string_view bestmove;
    } cases[] = {
        {"2k4R/1p6/4p3/3pP3/8/2P1KB2/3Q4/1q4r1 b - - 16 53", "bestmove c8c7"},
        {"7R/1p6/8/4Q3/8/8/1k3K2/1q1r4 b - - 3 62", "bestmove b2c1"},
    };

    const int old_threads = cfgmgr.num_threads;
    cfgmgr.num_threads = 1;

    for (const auto & c : cases) {
        Board b{std::string{c.fen}};
        SearchInfo si{b, 18};
        si.board = b;
        si.nnue.refresh(si.board);
        si.starttime = std::chrono::high_resolution_clock::now();
        si.stoptime = si.starttime + std::chrono::hours(1);
        si.soft_stoptime = si.stoptime;

        tt::ttable.clear();
        testing::internal::CaptureStdout();
        thread::pool.init_threads(si, 1);
        thread::pool.wait();
        const auto out = testing::internal::GetCapturedStdout();

        EXPECT_NE(out.find(c.bestmove), std::string::npos) << c.fen << "\n" << out;
    }

    cfgmgr.num_threads = old_threads;
}

// Qsearch-in-check regression. Pre-fix, qsearch unconditionally
// stand-patted — even when the side to move was in check — and the
// QSEARCH movepicker filtered out non-capturing / non-promoting moves
// (movepicker.hpp:85). Both together meant that a child qsearch node
// where the side to move was in check and mated would return the
// static eval instead of mated_in(ply).
//
// Concretely: from 7k/8/5KQ1/8/8/8/8/8 w - - 0 1, white's winning reply
// is g6g7 (mate in 1 — black is then in check with no legal move). At
// depth 1, negamax evaluates each root move by calling -qsearch on the
// child. After g6g7, the child position has black in check and mated;
// pre-fix qsearch stand-patted and returned a material-ahead eval that
// back-propagated as a finite score, not mate. Non-mating white moves
// like g6e4 looked better because they avoided losing the queen to a
// capture in the child's qsearch. The engine played g6e4 at depth 1
// and only found g6g7 once normal search depth reached 2.
//
// Post-fix: at depth 1 the root sees one move scoring mate-in-1 and
// picks g6g7.
TEST(search, qsearch_in_check_finds_mate_in_one_at_depth_one) {
    cfgmgr.num_threads = 1;

    Board b{"7k/8/5KQ1/8/8/8/8/8 w - - 0 1"};
    SearchInfo si{b, 1};
    si.board = b;
    si.nnue.refresh(si.board);
    // Leave starttime == stoptime (default-constructed) so time_expired()
    // is permanently false for this test. Setting only starttime would
    // make stoptime (still epoch) strictly less than now and expire
    // every call immediately.

    thread::pool.stop = false;
    tt::ttable.clear();
    tt::ttable.prepare();

    Worker worker{si, 0};
    worker.pvline.clear();

    // Bypass search_position() (which depends on thread::pool being
    // initialized for get_nps/get_nodes reporting). Call negamax<Root>
    // directly at depth 1 — the only thing that matters for this
    // regression is whether the in-check child-qsearch path back-
    // propagates mate correctly.
    Stack stack[MAX_PLY + 5];
    stack[0].ply = 4;
    stack[1].ply = 3;
    stack[2].ply = 2;
    stack[3].ply = 1;
    for (int i = 0; i <= MAX_PLY; i++)
        stack[i + 4].ply = i;
    Stack * ss = stack + 4;

    const auto value = negamax<white, NodeType::Root>(
        1, worker, ss, -Value::infinite, Value::infinite);

    // Expect the mate-in-1 move g6-g7 and a mate score.
    const Move best = worker.pvline.bestmove();
    EXPECT_EQ(best.src_sq(), g6);
    EXPECT_EQ(best.dst_sq(), g7);
    EXPECT_GE(static_cast<int>(value), static_cast<int>(Value::mate_in_max_ply));
}

TEST(search, root_ignores_empty_tt_cutoff_move) {
    Board b{"8/3k4/1P6/2K1P3/8/8/4b3/6B1 b - - 0 70"};
    SearchInfo si{b, 1};
    si.board = b;
    si.nnue.refresh(si.board);

    thread::pool.stop = false;
    tt::ttable.clear();
    tt::ttable.prepare();
    tt::ttable.store(
        si.board.hash,
        Move{},
        tt::value_to(Value::draw, 0),
        tt::type::ExactBound,
        64);

    Worker worker{si, 0};
    worker.pvline.clear();

    Stack stack[MAX_PLY + 5];
    stack[0].ply = 4;
    stack[1].ply = 3;
    stack[2].ply = 2;
    stack[3].ply = 1;
    for (int i = 0; i <= MAX_PLY; i++)
        stack[i + 4].ply = i;
    Stack * ss = stack + 4;

    (void)negamax<black, NodeType::Root>(
        1, worker, ss, -Value::infinite, Value::infinite);

    EXPECT_NE(worker.pvline.bestmove(), Move{});
    EXPECT_GT(worker.si.nodes, 0U);
}

// --- NNUE incremental audit -----------------------------------------------
// For each move type (quiet / capture / promotion / en-passant / castle),
// apply the move with UpdateNNUE=true (the live incremental/refresh path),
// snapshot the resulting accumulator, then do a full refresh() on the same
// post-move board and compare every element. Matching means the live path
// computed the correct accumulator; the whole HIDDEN_SIZE * 2 vector must
// agree.

namespace {

bool accumulators_match(const NNUE::Accumulator & a, const NNUE::Accumulator & b) {
    for (size_t i = 0; i < a.white_acc.size(); ++i)
        if (a.white_acc[i] != b.white_acc[i]) return false;
    for (size_t i = 0; i < a.black_acc.size(); ++i)
        if (a.black_acc[i] != b.black_acc[i]) return false;
    return true;
}

void ensure_network_mock_weights() {
    static std::vector<Network::acc_t> weights(
        static_cast<size_t>(Network::N_FEATURES) * Network::N_HIDDEN);
    static std::vector<Network::acc_t> biases(Network::N_HIDDEN);
    static bool initialized = false;

    if (!initialized) {
        for (size_t f = 0; f < static_cast<size_t>(Network::N_FEATURES); ++f) {
            for (size_t h = 0; h < static_cast<size_t>(Network::N_HIDDEN); ++h) {
                uint64_t x = (f + 0x9e3779b97f4a7c15ULL)
                           ^ ((h + 0xbf58476d1ce4e5b9ULL) << 1);
                x ^= x >> 30;
                x *= 0xbf58476d1ce4e5b9ULL;
                x ^= x >> 27;
                x *= 0x94d049bb133111ebULL;
                x ^= x >> 31;
                const int v = static_cast<int>(x % 255) - 127;
                weights[f * Network::N_HIDDEN + h] = static_cast<Network::acc_t>(v);
            }
        }
        for (size_t h = 0; h < static_cast<size_t>(Network::N_HIDDEN); ++h)
            biases[h] = static_cast<Network::acc_t>(static_cast<int>(h % 23) - 11);
        initialized = true;
    }

    Network::SetWeights(weights.data(), biases.data());
}

bool network_accumulators_match(const Network::Accumulator & a,
                              const Network::Accumulator & b) {
    for (int view = 0; view < 2; ++view) {
        for (size_t i = 0; i < static_cast<size_t>(Network::N_HIDDEN); ++i)
            if (a.values[view][i] != b.values[view][i]) return false;
    }
    return true;
}

void materialize_network(Board & b, NNUE::Net & net) {
    net.ensure_network(b);
}

template <Color Us>
void audit_network_lazy_tree(Board & b, NNUE::Net & live, int depth)
{
    if (depth == 0) {
        materialize_network(b, live);
        NNUE::Net fresh;
        fresh.refresh(b);
        EXPECT_TRUE(network_accumulators_match(
            live.network_accumulator_stack[live.currentAccumulator],
            fresh.network_accumulator_stack[fresh.currentAccumulator]))
            << "active Network lazy tree diverged; fen after: " << b.fen();
        return;
    }

    auto moves = generate_legal_moves<Us>(b);
    int searched = 0;
    for (auto move : moves) {
        apply_move<Us, true, true>(b, move, &live);
        audit_network_lazy_tree<~Us>(b, live, depth - 1);
        revert_move<Us, true, true>(b, &live);
        if (++searched >= 12)
            break;
    }
}

struct ScopedNetworkEnabled {
    bool previous;

    ScopedNetworkEnabled()
        : previous(Network::enabled)
    {
        Network::enabled = true;
    }

    ~ScopedNetworkEnabled() {
        Network::enabled = previous;
    }
};

void expect_network_matches_refresh(const char * label, Board & b, Move move, Color us) {
    ensure_network_mock_weights();

    NNUE::Net live;
    live.refresh(b);

    if (us == white)
        apply_move<white, true, true>(b, move, &live);
    else
        apply_move<black, true, true>(b, move, &live);
    materialize_network(b, live);

    const auto after_live = live.network_accumulator_stack[live.currentAccumulator];

    NNUE::Net fresh;
    fresh.refresh(b);
    const auto after_fresh = fresh.network_accumulator_stack[fresh.currentAccumulator];

    EXPECT_TRUE(network_accumulators_match(after_live, after_fresh))
        << label << ": Network incremental accumulator diverged from full refresh after "
        << fmt::format("{}", move) << " (fen after: " << b.fen() << ")";
}

template <Color Us>
void expect_incremental_matches_refresh(const char * label, Board & b, Move move) {
    NNUE::Net live;
    live.refresh(b);

    apply_move<Us, true, true>(b, move, &live);

    NNUE::Accumulator after_live = live.accumulator_stack[live.currentAccumulator];

    NNUE::Net fresh;
    fresh.refresh(b);
    NNUE::Accumulator after_fresh = fresh.accumulator_stack[fresh.currentAccumulator];

    EXPECT_TRUE(accumulators_match(after_live, after_fresh))
        << label << ": incremental accumulator diverged from full refresh after "
        << fmt::format("{}", move) << " (fen after: " << b.fen() << ")";
}

} // anon ns

TEST(nnue_audit, quiet_pawn_push) {
    Board b{"startpos"};
    auto move = resolve_move<white>(b, pawn, e2, e4);
    expect_incremental_matches_refresh<white>("quiet_pawn_push", b, move);
}

TEST(nnue_audit, quiet_knight_move) {
    Board b{"startpos"};
    auto move = resolve_move<white>(b, knight, g1, f3);
    expect_incremental_matches_refresh<white>("quiet_knight_move", b, move);
}

TEST(nnue_audit, capture) {
    // Scandinavian: 1.e4 d5 2.exd5 — white pawn captures black pawn
    Board b{"rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"};
    auto move = resolve_move<white>(b, pawn, e4, d5);
    expect_incremental_matches_refresh<white>("capture", b, move);
}

TEST(nnue_audit, promotion_no_capture) {
    // White pawn on a7 promotes to queen on a8; no capture.
    Board b{"4k3/P7/8/8/8/8/8/4K3 w - - 0 1"};
    auto moves = generate_legal_moves<white>(b);
    Move promo{};
    for (auto m : moves) {
        if (m.flags() == Move::Flags::promote
         && m.src_sq() == a7 && m.dst_sq() == a8
         && m.promo_piece() == queen) {
            promo = m; break;
        }
    }
    ASSERT_TRUE(promo) << "could not find promotion move";
    expect_incremental_matches_refresh<white>("promotion_no_capture", b, promo);
}

TEST(nnue_audit, promotion_with_capture) {
    // White pawn on a7 captures on b8 and promotes to queen.
    Board b{"1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1"};
    auto moves = generate_legal_moves<white>(b);
    Move promo{};
    for (auto m : moves) {
        if (m.flags() == Move::Flags::promote
         && m.src_sq() == a7 && m.dst_sq() == b8
         && m.promo_piece() == queen) {
            promo = m; break;
        }
    }
    ASSERT_TRUE(promo) << "could not find capture-promotion move";
    expect_incremental_matches_refresh<white>("promotion_with_capture", b, promo);
}

TEST(nnue_audit, enpassant) {
    // After 1.e4 d5 2.e5 f5 — white can ep-capture on f6.
    Board b{"rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"};
    auto moves = generate_legal_moves<white>(b);
    Move ep{};
    for (auto m : moves) {
        if (m.flags() == Move::Flags::enpassant) { ep = m; break; }
    }
    ASSERT_TRUE(ep) << "could not find en-passant move";
    expect_incremental_matches_refresh<white>("enpassant", b, ep);
}

TEST(nnue_audit, castle_kingside) {
    Board b{"r3k2r/pppqppbp/2np1np1/8/8/2NP1NP1/PPPQPPBP/R3K2R w KQkq - 0 1"};
    auto moves = generate_legal_moves<white>(b);
    Move castle{};
    for (auto m : moves) {
        if (m.flags() == Move::Flags::castle && m.dst_sq() < m.src_sq()) {
            castle = m; break;
        }
    }
    ASSERT_TRUE(castle) << "could not find kingside castle move";
    expect_incremental_matches_refresh<white>("castle_kingside", b, castle);
}

TEST(nnue_audit, castle_queenside) {
    Board b{"r3k2r/pppqppbp/2np1np1/8/8/2NP1NP1/PPPQPPBP/R3K2R w KQkq - 0 1"};
    auto moves = generate_legal_moves<white>(b);
    Move castle{};
    for (auto m : moves) {
        if (m.flags() == Move::Flags::castle && m.dst_sq() > m.src_sq()) {
            castle = m; break;
        }
    }
    ASSERT_TRUE(castle) << "could not find queenside castle move";
    expect_incremental_matches_refresh<white>("castle_queenside", b, castle);
}

TEST(nnue_audit, king_move_non_castle) {
    // Plain king move: the king-bucket index changes, so the live path
    // triggers refresh_with_cache. This test locks in correct behavior.
    Board b{"4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"};
    auto move = resolve_move<white>(b, king, e1, e2);
    expect_incremental_matches_refresh<white>("king_move_non_castle", b, move);
}

// Apply A, revert A, apply B (sibling) — verify sibling's accumulator
// matches a fresh refresh. This is the exact pattern a move loop uses.
TEST(nnue_audit, apply_revert_apply_sibling) {
    Board b{"startpos"};
    NNUE::Net live;
    live.refresh(b);

    auto a = resolve_move<white>(b, pawn, e2, e4);
    apply_move<white, true, true>(b, a, &live);
    revert_move<white, true, true>(b, &live);

    auto c = resolve_move<white>(b, pawn, d2, d4);
    apply_move<white, true, true>(b, c, &live);
    NNUE::Accumulator after_c_live = live.accumulator_stack[live.currentAccumulator];

    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(accumulators_match(after_c_live, fresh.accumulator_stack[fresh.currentAccumulator]))
        << "sibling apply after revert diverged; fen: " << b.fen();
}

// Deeper mimic of search: apply A, apply B, revert B, apply C — then
// unwind. Verify accumulators are correct at every visible step.
TEST(nnue_audit, nested_apply_revert_sibling) {
    Board b{"startpos"};
    NNUE::Net live;
    live.refresh(b);

    auto a = resolve_move<white>(b, pawn, e2, e4);
    apply_move<white, true, true>(b, a, &live);

    auto b1 = resolve_move<black>(b, pawn, e7, e5);
    apply_move<black, true, true>(b, b1, &live);
    revert_move<black, true, true>(b, &live);

    auto b2 = resolve_move<black>(b, pawn, c7, c5);
    apply_move<black, true, true>(b, b2, &live);
    NNUE::Accumulator live_b2 = live.accumulator_stack[live.currentAccumulator];

    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(accumulators_match(live_b2, fresh.accumulator_stack[fresh.currentAccumulator]))
        << "nested sibling diverged at ply 2; fen: " << b.fen();

    revert_move<black, true, true>(b, &live);
    revert_move<white, true, true>(b, &live);
    NNUE::Net root_fresh;
    root_fresh.refresh(b);
    EXPECT_TRUE(accumulators_match(live.accumulator_stack[live.currentAccumulator],
                                   root_fresh.accumulator_stack[root_fresh.currentAccumulator]))
        << "full unwind diverged; fen: " << b.fen();
}

// Deep recursion mimicking real search: at each ply, verify the live
// accumulator matches a fresh refresh. Tries several siblings per ply.
TEST(nnue_audit, deep_recursion_slots_consistent) {
    Board b{"r3k2r/pppqppbp/2np1np1/8/8/2NP1NP1/PPPQPPBP/R3K2R w KQkq - 0 1"};
    NNUE::Net live;
    live.refresh(b);

    auto verify_slot = [&](const char * tag) {
        NNUE::Net fresh;
        fresh.refresh(b);
        ASSERT_TRUE(accumulators_match(live.accumulator_stack[live.currentAccumulator],
                                       fresh.accumulator_stack[fresh.currentAccumulator]))
            << tag << " slot diverged; fen: " << b.fen()
            << " currentAccumulator=" << live.currentAccumulator;
    };

    // ply 0 (root): white
    auto w1 = resolve_move<white>(b, knight, c3, e4);
    apply_move<white, true, true>(b, w1, &live); verify_slot("w1");

    // ply 1 (black), try sibling B1
    auto b1 = resolve_move<black>(b, knight, f6, e4);
    apply_move<black, true, true>(b, b1, &live); verify_slot("b1");

    // ply 2 (white)
    auto w2 = resolve_move<white>(b, knight, f3, e5);
    apply_move<white, true, true>(b, w2, &live); verify_slot("w2");
    revert_move<white, true, true>(b, &live); verify_slot("revert w2");

    // sibling w2'
    auto w2b = resolve_move<white>(b, pawn, d3, e4);
    apply_move<white, true, true>(b, w2b, &live); verify_slot("w2'");

    // unwind all
    revert_move<white, true, true>(b, &live); verify_slot("revert w2'");
    revert_move<black, true, true>(b, &live); verify_slot("revert b1");

    // ply 1 sibling b1'
    auto b1b = resolve_move<black>(b, pawn, d6, d5);
    apply_move<black, true, true>(b, b1b, &live); verify_slot("b1'");
    revert_move<black, true, true>(b, &live); verify_slot("revert b1'");

    revert_move<white, true, true>(b, &live); verify_slot("revert w1");
}

// Mixed move types in a nested tree: capture → promotion → king-move →
// castle. Each of these hits a different apply/revert path.
TEST(nnue_audit, mixed_move_types_nested) {
    // Position set up so white can: (a) capture b4 with pawn a3xb4,
    // and also has castling rights for later.
    Board b{"r3k2r/pppqppbp/2np1np1/8/1p6/P1NP1NP1/1PPQPPBP/R3K2R w KQkq - 0 1"};
    NNUE::Net live;
    live.refresh(b);

    auto verify = [&](const char * tag) {
        NNUE::Net fresh;
        fresh.refresh(b);
        ASSERT_TRUE(accumulators_match(live.accumulator_stack[live.currentAccumulator],
                                       fresh.accumulator_stack[fresh.currentAccumulator]))
            << tag << "; fen: " << b.fen() << " cur=" << live.currentAccumulator;
    };

    // capture
    auto cap = resolve_move<white>(b, pawn, a3, b4);
    apply_move<white, true, true>(b, cap, &live); verify("capture");

    // black quiet
    auto bq = resolve_move<black>(b, pawn, e7, e6);
    apply_move<black, true, true>(b, bq, &live); verify("black quiet");

    // white castle kingside
    auto moves = generate_legal_moves<white>(b);
    Move castle{};
    for (auto m : moves) {
        if (m.flags() == Move::Flags::castle && m.dst_sq() < m.src_sq()) {
            castle = m; break;
        }
    }
    ASSERT_TRUE(castle);
    apply_move<white, true, true>(b, castle, &live); verify("castle");

    // unwind
    revert_move<white, true, true>(b, &live); verify("revert castle");
    revert_move<black, true, true>(b, &live); verify("revert black quiet");
    revert_move<white, true, true>(b, &live); verify("revert capture");
}

// Apply A, revert A, apply B — verify slot N (after revert) still matches
// a from-scratch refresh on the reverted board. Catches bugs where
// revert_move corrupts the popped slot.
TEST(nnue_audit, revert_restores_slot) {
    Board start{"startpos"};
    NNUE::Net live;
    live.refresh(start);

    const NNUE::Accumulator before = live.accumulator_stack[live.currentAccumulator];

    auto a = resolve_move<white>(start, pawn, e2, e4);
    apply_move<white, true, true>(start, a, &live);
    revert_move<white, true, true>(start, &live);

    const NNUE::Accumulator after_revert = live.accumulator_stack[live.currentAccumulator];
    EXPECT_TRUE(accumulators_match(before, after_revert))
        << "revert corrupted slot; fen: " << start.fen();

    NNUE::Net fresh;
    fresh.refresh(start);
    EXPECT_TRUE(accumulators_match(after_revert, fresh.accumulator_stack[fresh.currentAccumulator]))
        << "post-revert slot != fresh refresh; fen: " << start.fen();
}

// Deep-sequence audit: apply eight moves, each time verifying the live
// accumulator matches a full refresh. Flushes out any delta-stacking bug
// that a single-move test would miss.
TEST(nnue_audit, opening_sequence) {
    Board b{"startpos"};
    NNUE::Net live;
    live.refresh(b);

    struct Step { Color us; PieceType pt; square_t from; square_t to; };
    const Step seq[] = {
        { white, pawn,   e2, e4 },
        { black, pawn,   e7, e5 },
        { white, knight, g1, f3 },
        { black, knight, b8, c6 },
        { white, bishop, f1, b5 },
        { black, pawn,   a7, a6 },
        { white, bishop, b5, a4 },
        { black, knight, g8, f6 },
    };

    int step_idx = 0;
    for (auto const & s : seq) {
        Move m = (s.us == white)
            ? resolve_move<white>(b, s.pt, s.from, s.to)
            : resolve_move<black>(b, s.pt, s.from, s.to);

        if (s.us == white) apply_move<white, true, true>(b, m, &live);
        else               apply_move<black, true, true>(b, m, &live);

        NNUE::Accumulator after_live = live.accumulator_stack[live.currentAccumulator];
        NNUE::Net fresh;
        fresh.refresh(b);
        NNUE::Accumulator after_fresh = fresh.accumulator_stack[fresh.currentAccumulator];
        EXPECT_TRUE(accumulators_match(after_live, after_fresh))
            << "opening_sequence step " << step_idx << " ("
            << fmt::format("{}", m) << ") diverged; fen after: " << b.fen();
        ++step_idx;
    }
}
#endif

TEST(network_audit, opening_sequence_matches_refresh) {
    Board b{"startpos"};
    ensure_network_mock_weights();
    NNUE::Net live;
    live.refresh(b);

    struct Step { Color us; PieceType pt; square_t from; square_t to; };
    const Step seq[] = {
        { white, pawn,   e2, e4 },
        { black, pawn,   e7, e5 },
        { white, knight, g1, f3 },
        { black, knight, b8, c6 },
        { white, bishop, f1, b5 },
        { black, pawn,   a7, a6 },
        { white, bishop, b5, a4 },
        { black, knight, g8, f6 },
    };

    int step_idx = 0;
    for (auto const & s : seq) {
        Move m = (s.us == white)
            ? resolve_move<white>(b, s.pt, s.from, s.to)
            : resolve_move<black>(b, s.pt, s.from, s.to);

        if (s.us == white)
            apply_move<white, true, true>(b, m, &live);
        else
            apply_move<black, true, true>(b, m, &live);
        materialize_network(b, live);

        NNUE::Net fresh;
        fresh.refresh(b);
        EXPECT_TRUE(network_accumulators_match(
            live.network_accumulator_stack[live.currentAccumulator],
            fresh.network_accumulator_stack[fresh.currentAccumulator]))
            << "Network opening_sequence step " << step_idx << " ("
            << fmt::format("{}", m) << ") diverged; fen after: " << b.fen();
        ++step_idx;
    }
}

TEST(network_audit, active_quiet_delta_matches_refresh) {
    Board b{"startpos"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);

    struct Step { Color us; PieceType pt; square_t from; square_t to; };
    const Step seq[] = {
        { white, pawn,   e2, e4 },
        { black, pawn,   e7, e5 },
        { white, knight, g1, f3 },
        { black, knight, b8, c6 },
        { white, bishop, f1, b5 },
        { black, pawn,   a7, a6 },
        { white, bishop, b5, a4 },
        { black, knight, g8, f6 },
    };

    int step_idx = 0;
    for (auto const & s : seq) {
        Move m = s.us == white
            ? resolve_move<white>(b, s.pt, s.from, s.to)
            : resolve_move<black>(b, s.pt, s.from, s.to);

        if (s.us == white)
            apply_move<white, true, true>(b, m, &live);
        else
            apply_move<black, true, true>(b, m, &live);
        materialize_network(b, live);

        NNUE::Net fresh;
        fresh.refresh(b);
        EXPECT_TRUE(network_accumulators_match(
            live.network_accumulator_stack[live.currentAccumulator],
            fresh.network_accumulator_stack[fresh.currentAccumulator]))
            << "active Network quiet step " << step_idx << " ("
            << fmt::format("{}", m) << ") diverged; fen after: " << b.fen();
        ++step_idx;
    }
}

TEST(network_audit, active_lazy_chain_matches_refresh) {
    Board b{"startpos"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);

    struct Step { Color us; PieceType pt; square_t from; square_t to; };
    const Step seq[] = {
        { white, pawn,   e2, e4 },
        { black, pawn,   e7, e5 },
        { white, knight, g1, f3 },
        { black, knight, b8, c6 },
        { white, bishop, f1, b5 },
        { black, pawn,   a7, a6 },
        { white, bishop, b5, a4 },
        { black, knight, g8, f6 },
    };

    for (auto const & s : seq) {
        Move m = s.us == white
            ? resolve_move<white>(b, s.pt, s.from, s.to)
            : resolve_move<black>(b, s.pt, s.from, s.to);

        if (s.us == white)
            apply_move<white, true, true>(b, m, &live);
        else
            apply_move<black, true, true>(b, m, &live);
    }
    materialize_network(b, live);

    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(network_accumulators_match(
        live.network_accumulator_stack[live.currentAccumulator],
        fresh.network_accumulator_stack[fresh.currentAccumulator]))
        << "active Network lazy chain diverged; fen after: " << b.fen();
}

TEST(network_audit, active_lazy_tree_matches_refresh) {
    Board b{"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);
    audit_network_lazy_tree<white>(b, live, 3);
}

TEST(network_audit, active_kiwipete_pv_with_castle_matches_refresh) {
    Board b{"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);

    const std::string moves =
        "e2a6 b4c3 d2c3 e6d5 e4d5 h3g2 f3g2 f6d5 e1g1";
    std::istringstream iss{moves};
    std::string move_str;
    while (iss >> move_str) {
        const std::string src_str{move_str.substr(0, 2)};
        const std::string dst_str{move_str.substr(2, 2)};
        const auto src = str2sq(src_str.c_str());
        const auto dst = str2sq(dst_str.c_str());
        const auto pt = b.pt_mb[src];
        if (b.side == white) {
            auto move = resolve_move<white>(b, pt, src, dst);
            apply_move<white, true, true>(b, move, &live);
        } else {
            auto move = resolve_move<black>(b, pt, src, dst);
            apply_move<black, true, true>(b, move, &live);
        }
    }

    materialize_network(b, live);
    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(network_accumulators_match(
        live.network_accumulator_stack[live.currentAccumulator],
        fresh.network_accumulator_stack[fresh.currentAccumulator]))
        << "active Network Kiwipete PV diverged; fen after: " << b.fen();
}

TEST(network_audit, active_promotion_capture_king_chain_matches_refresh) {
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    const std::vector<std::string> moves = {
        "e2a6", "b4c3", "b2c3", "h3g2", "d5e6",
        "g2h1n", "e6f7", "e7f7", "e5f7", "e8f7",
    };

    for (size_t prefix = 1; prefix <= moves.size(); ++prefix) {
        Board b{"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"};
        NNUE::Net live;
        live.refresh(b);

        for (size_t i = 0; i < prefix; ++i) {
            const auto & move_str = moves[i];
            const std::string src_str{move_str.substr(0, 2)};
            const std::string dst_str{move_str.substr(2, 2)};
            const auto src = str2sq(src_str.c_str());
            const auto dst = str2sq(dst_str.c_str());
            const auto pt = b.pt_mb[src];
            if (b.side == white) {
                auto move = resolve_move<white>(b, pt, src, dst);
                apply_move<white, true, true>(b, move, &live);
            } else {
                auto move = resolve_move<black>(b, pt, src, dst);
                apply_move<black, true, true>(b, move, &live);
            }
        }

        materialize_network(b, live);
        NNUE::Net fresh;
        fresh.refresh(b);
        EXPECT_TRUE(network_accumulators_match(
            live.network_accumulator_stack[live.currentAccumulator],
            fresh.network_accumulator_stack[fresh.currentAccumulator]))
            << "active Network promotion/king chain diverged at prefix "
            << prefix << "; fen after: " << b.fen();
    }
}

TEST(network_audit, active_nonrefreshing_king_loop_matches_refresh) {
    Board b{"8/8/2k5/8/8/5Q2/1pq2PKP/8 b - - 13 60"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);

    const std::vector<std::string> moves = {
        "c6c5", "f3e3", "c5c6",
    };

    for (const auto & move_str : moves) {
        const std::string src_str{move_str.substr(0, 2)};
        const std::string dst_str{move_str.substr(2, 2)};
        const auto src = str2sq(src_str.c_str());
        const auto dst = str2sq(dst_str.c_str());
        const auto pt = b.pt_mb[src];
        if (b.side == white) {
            auto move = resolve_move<white>(b, pt, src, dst);
            apply_move<white, true, true>(b, move, &live);
        } else {
            auto move = resolve_move<black>(b, pt, src, dst);
            apply_move<black, true, true>(b, move, &live);
        }
    }

    materialize_network(b, live);
    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(network_accumulators_match(
        live.network_accumulator_stack[live.currentAccumulator],
        fresh.network_accumulator_stack[fresh.currentAccumulator]))
        << "active Network non-refreshing king loop diverged; fen after: " << b.fen();
}

TEST(network_audit, active_quiet_sibling_matches_refresh) {
    Board b{"startpos"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);

    auto e2e4 = resolve_move<white>(b, pawn, e2, e4);
    apply_move<white, true, true>(b, e2e4, &live);
    revert_move<white, true, true>(b, &live);

    auto d2d4 = resolve_move<white>(b, pawn, d2, d4);
    apply_move<white, true, true>(b, d2d4, &live);
    materialize_network(b, live);

    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(network_accumulators_match(
        live.network_accumulator_stack[live.currentAccumulator],
        fresh.network_accumulator_stack[fresh.currentAccumulator]))
        << "active Network sibling quiet move diverged; fen after: " << b.fen();
}

TEST(network_audit, active_generic_capture_matches_refresh) {
    Board b{"rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);

    auto move = resolve_move<white>(b, pawn, e4, d5);
    apply_move<white, true, true>(b, move, &live);
    materialize_network(b, live);

    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(network_accumulators_match(
        live.network_accumulator_stack[live.currentAccumulator],
        fresh.network_accumulator_stack[fresh.currentAccumulator]))
        << "active Network generic capture diverged; fen after: " << b.fen();
}

TEST(network_audit, special_moves_match_refresh) {
    ScopedNetworkEnabled scoped;

    {
        Board b{"rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"};
        auto moves = generate_legal_moves<white>(b);
        Move ep{};
        for (auto m : moves) {
            if (m.flags() == Move::Flags::enpassant) { ep = m; break; }
        }
        ASSERT_TRUE(ep);
        expect_network_matches_refresh("enpassant", b, ep, white);
    }
    {
        Board b{"1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1"};
        auto moves = generate_legal_moves<white>(b);
        Move promo{};
        for (auto m : moves) {
            if (m.flags() == Move::Flags::promote
             && m.src_sq() == a7 && m.dst_sq() == b8
             && m.promo_piece() == queen) {
                promo = m; break;
            }
        }
        ASSERT_TRUE(promo);
        expect_network_matches_refresh("promotion_with_capture", b, promo, white);
    }
    {
        Board b{"r3k2r/pppqppbp/2np1np1/8/8/2NP1NP1/PPPQPPBP/R3K2R w KQkq - 0 1"};
        auto moves = generate_legal_moves<white>(b);
        Move castle{};
        for (auto m : moves) {
            if (m.flags() == Move::Flags::castle && m.dst_sq() < m.src_sq()) {
                castle = m; break;
            }
        }
        ASSERT_TRUE(castle);
        expect_network_matches_refresh("castle_kingside", b, castle, white);
    }
    {
        Board b{"4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"};
        auto move = resolve_move<white>(b, king, e1, e2);
        expect_network_matches_refresh("king_move_non_castle", b, move, white);
    }
}

// Fathom/Syzygy expects bitboards in A1=0 layout: a1=bit0, h1=bit7, a8=bit56.
// Enyo internally uses H1=0. board2pos() applies bbconv()/sqconv() to bridge
// the two. These tests pin down that conversion against hand-computed
// expected values for a handful of asymmetry-sensitive positions.
TEST(syzygy_bbconv, startpos_bitboards) {
    Board b{"startpos"};
    auto p = syzygy::board2pos(b);

    // Ranks 1 and 2 are all white, ranks 7 and 8 all black.
    EXPECT_EQ(p.white,   0x000000000000FFFFULL);
    EXPECT_EQ(p.black,   0xFFFF000000000000ULL);
    // Kings on e1, e8 → bits 4 and 60.
    EXPECT_EQ(p.kings,   (1ULL << 4) | (1ULL << 60));
    // Queens on d1, d8 → bits 3 and 59.
    EXPECT_EQ(p.queens,  (1ULL << 3) | (1ULL << 59));
    // Rooks on a1, h1, a8, h8.
    EXPECT_EQ(p.rooks,   (1ULL << 0) | (1ULL << 7) | (1ULL << 56) | (1ULL << 63));
    // Bishops on c1, f1, c8, f8.
    EXPECT_EQ(p.bishops, (1ULL << 2) | (1ULL << 5) | (1ULL << 58) | (1ULL << 61));
    // Knights on b1, g1, b8, g8.
    EXPECT_EQ(p.knights, (1ULL << 1) | (1ULL << 6) | (1ULL << 57) | (1ULL << 62));
    // Pawns span ranks 2 and 7.
    EXPECT_EQ(p.pawns,   0x00FF00000000FF00ULL);
    EXPECT_TRUE(p.turn);
}

TEST(syzygy_bbconv, asymmetric_a1_bishop) {
    // Only a white bishop on a1. If bbconv mirrored files incorrectly (a1↔h1)
    // this would come out as bit 7 instead of bit 0.
    Board b{"8/8/8/8/8/8/8/B6k w - - 0 1"};
    auto p = syzygy::board2pos(b);

    EXPECT_EQ(p.white,   1ULL << 0);
    EXPECT_EQ(p.bishops, 1ULL << 0);
    EXPECT_EQ(p.pawns, 0ULL);
}

TEST(syzygy_bbconv, asymmetric_h8_bishop) {
    // Black bishop on h8 (plus kings on a1, h1 to keep the FEN legal).
    Board b{"7b/8/8/8/8/8/8/K6k w - - 0 1"};
    auto p = syzygy::board2pos(b);

    // Black set: bishop on h8 (bit 63) + black king on h1 (bit 7).
    EXPECT_EQ(p.black,   (1ULL << 63) | (1ULL << 7));
    EXPECT_EQ(p.bishops, 1ULL << 63);
    // Kings: white on a1 (bit 0), black on h1 (bit 7).
    EXPECT_EQ(p.kings,   (1ULL << 0) | (1ULL << 7));
}

TEST(syzygy_bbconv, krvk_winning_position) {
    // Standard KR vs K sanity position: White K on e1, R on h1, black K on e8.
    Board b{"4k3/8/8/8/8/8/8/4K2R w - - 0 1"};
    auto p = syzygy::board2pos(b);

    EXPECT_EQ(p.white,   (1ULL << 4) | (1ULL << 7));   // e1 | h1
    EXPECT_EQ(p.black,   1ULL << 60);                  // e8
    EXPECT_EQ(p.kings,   (1ULL << 4) | (1ULL << 60));
    EXPECT_EQ(p.rooks,   1ULL << 7);
    EXPECT_EQ(p.queens,  0ULL);
    EXPECT_EQ(p.pawns,   0ULL);
    EXPECT_TRUE(p.turn);
}

TEST(syzygy_bbconv, ep_square_conversion) {
    // After 1.e4, the en-passant square is e3 (file e, rank 3) = A1-layout index
    // 4*8? No — rank 3 means rank_index=2, so bit = 2*8 + 4 = 20.
    Board b{"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"};
    auto p = syzygy::board2pos(b);

    // sqconv() maps the board's H1=0 square index to Fathom's A1=0 index.
    // We don't compare the raw square_t here — just the probe-ready uint8_t.
    EXPECT_EQ(p.ep, 20u);  // e3 in A1=0 layout
}

TEST(syzygy_bbconv, no_ep_square_stays_fathom_sentinel) {
    Board b{"startpos"};
    auto p = syzygy::board2pos(b);

    EXPECT_EQ(p.ep, 0u);
}

TEST(syzygy_root, filters_root_promotions_that_lose_wdl) {
    if (!init_test_syzygy(6))
        GTEST_SKIP() << "6-man Syzygy tablebases not available";

    Board b{"8/8/8/8/1k6/2n1KP2/p7/5R2 b - - 0 77"};
    const auto legal = generate_legal_moves<black>(b);
    const auto winning = syzygy::root_WDL_filter(b, legal);

    auto knight_move = uci_to_move(b, "c3b1");
    auto promotion = uci_to_move(b, "a2a1q");
    ASSERT_TRUE(knight_move);
    ASSERT_TRUE(promotion);

    const auto contains = [&](Move move) {
        return std::ranges::any_of(winning, [&](Move candidate) {
            return candidate == move;
        });
    };

    EXPECT_TRUE(contains(*knight_move));
    EXPECT_FALSE(contains(*promotion));
    EXPECT_EQ(b.fen(), "8/8/8/8/1k6/2n1KP2/p7/5R2 b - - 0 77");
}

TEST(syzygy_root, dtz_returns_move_for_lost_root_position) {
    if (!init_test_syzygy(3))
        GTEST_SKIP() << "Syzygy tablebases not available";

    Board b{"6k1/2Q5/5K2/8/8/8/8/8 b - - 10 96"};
    syzygy::Status status = syzygy::Status::Error;
    const auto [score, move] = syzygy::DTZ_probe(b, status);

    EXPECT_EQ(status, syzygy::Status::Loss);
    EXPECT_EQ(score, Value::tb_loss_in_max_ply);
    ASSERT_TRUE(move);
    const auto legal = generate_legal_moves<black>(b);
    EXPECT_TRUE(std::ranges::any_of(legal, [&](Move candidate) {
        return candidate == move;
    }));
}

TEST(syzygy_root, piece_count_boundaries) {
    if (!init_test_syzygy(6))
        GTEST_SKIP() << "6-man Syzygy tablebases not available";
    ASSERT_LT(syzygy::largest(), 7u);

    {
        Board b{"8/8/8/3n4/1k6/5P2/4K3/R7 b - - 0 79"};
        ASSERT_EQ(count_bits(b.color_bb[white] | b.color_bb[black]), 5);

        syzygy::Status status = syzygy::Status::Error;
        const auto [score, move] = syzygy::DTZ_probe(b, status);

        EXPECT_EQ(syzygy::WDL_probe(b), syzygy::Status::Loss);
        EXPECT_EQ(status, syzygy::Status::Loss);
        EXPECT_EQ(score, Value::tb_loss_in_max_ply);
        EXPECT_TRUE(move);
    }

    {
        Board b{"8/8/8/8/1k6/2n1KP2/p7/5R2 b - - 0 77"};
        ASSERT_EQ(count_bits(b.color_bb[white] | b.color_bb[black]), 6);

        syzygy::Status status = syzygy::Status::Error;
        const auto [score, move] = syzygy::DTZ_probe(b, status);
        const auto legal = generate_legal_moves<black>(b);
        bool filter_complete = false;
        const auto filtered = syzygy::root_WDL_filter(b, legal, &filter_complete);

        EXPECT_EQ(syzygy::WDL_probe(b), syzygy::Status::Win);
        if (move) {
            EXPECT_EQ(status, syzygy::Status::Win);
            EXPECT_EQ(score, Value::tb_win_in_max_ply);
            EXPECT_TRUE(std::ranges::any_of(legal, [&](Move candidate) {
                return candidate == move;
            }));
        } else {
            EXPECT_EQ(status, syzygy::Status::Error);
            EXPECT_EQ(score, 0);
        }
        EXPECT_TRUE(filter_complete);
        EXPECT_EQ(filtered.size(), 1u);
    }

    {
        Board b{"8/8/8/8/1k6/2n1KP2/p7/4NR2 b - - 0 77"};
        ASSERT_EQ(count_bits(b.color_bb[white] | b.color_bb[black]), 7);

        syzygy::Status status = syzygy::Status::Error;
        const auto [score, move] = syzygy::DTZ_probe(b, status);
        const auto legal = generate_legal_moves<black>(b);
        bool filter_complete = true;
        const auto filtered = syzygy::root_WDL_filter(b, legal, &filter_complete);

        EXPECT_EQ(syzygy::WDL_probe(b), syzygy::Status::Error);
        EXPECT_EQ(status, syzygy::Status::Error);
        EXPECT_EQ(score, 0);
        EXPECT_FALSE(move);
        EXPECT_FALSE(filter_complete);
        EXPECT_EQ(filtered.size(), legal.size());
    }
}

TEST(syzygy_root, resolve_path_ignores_hidden_staging_dirs) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = fs::temp_directory_path() / fmt::format("enyo_tb_root_{}", stamp);
    const auto hidden = root / ".6-dtz";
    const auto visible = root / "6-wdl";
    fs::create_directories(hidden);
    fs::create_directories(visible);

    {
        std::ofstream file(hidden / "KQvK.rtbw", std::ios::binary);
        file << std::string(16, '\0');
    }
    {
        std::ofstream file(visible / "KRvK.rtbw", std::ios::binary);
        file << std::string(16, '\0');
    }

    const auto resolved = syzygy::resolve_path(root.string());
    EXPECT_NE(resolved.find(visible.string()), std::string::npos) << resolved;
    EXPECT_EQ(resolved.find(hidden.string()), std::string::npos) << resolved;

    fs::remove_all(root);
}

TEST(syzygy_root, init_rejects_explicit_incomplete_tablebase_dir) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = fs::temp_directory_path() / fmt::format("enyo_incomplete_tb_{}", stamp);
    fs::create_directories(root);
    {
        std::ofstream file(root / "KQvK.rtbw", std::ios::binary);
        file.put('\0');
    }

    EXPECT_FALSE(syzygy::init(root.string()));

    fs::remove_all(root);
}

TEST(syzygy_root, six_piece_root_moves_immediately) {
    if (!init_test_syzygy(6))
        GTEST_SKIP() << "6-man Syzygy tablebases not available";

    Board b{"8/1k6/1PnK4/2B5/P7/8/8/8 w - - 5 89"};
    ASSERT_EQ(count_bits(b.color_bb[white] | b.color_bb[black]), 6);

    syzygy::Status status = syzygy::Status::Error;
    const auto [score, dtz_move] = syzygy::DTZ_probe(b, status);
    const auto legal = generate_legal_moves<white>(b);
    bool filter_complete = false;
    const auto filtered = syzygy::root_WDL_filter(b, legal, &filter_complete);

    if (dtz_move) {
        EXPECT_EQ(status, syzygy::Status::Win);
        EXPECT_EQ(score, Value::tb_win_in_max_ply);
    } else {
        EXPECT_EQ(status, syzygy::Status::Error);
        EXPECT_EQ(score, 0);
    }
    ASSERT_EQ(syzygy::WDL_probe(b), syzygy::Status::Win);
    ASSERT_TRUE(filter_complete);
    ASSERT_FALSE(filtered.empty());

    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = true;

    SearchInfo si{b, 8};
    si.board = b;
    si.nnue.refresh(si.board);
    si.starttime = std::chrono::high_resolution_clock::now();
    si.stoptime = si.starttime + std::chrono::hours(1);
    si.soft_stoptime = si.stoptime;

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    thread::pool.init_threads(si, 1);
    thread::pool.wait();
    const auto out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("string tbhit win"), std::string::npos) << out;
    EXPECT_NE(out.find("bestmove "), std::string::npos) << out;
    EXPECT_EQ(out.find("info depth 2 score"), std::string::npos) << out;
    const auto score_cp = first_uci_cp_score(out);
    EXPECT_GT(score_cp, 19000) << out;
    EXPECT_LE(score_cp, 20000) << out;
    EXPECT_NE(score_cp, Value::tb_win_in_max_ply) << out;
    EXPECT_EQ(out.find("score cp 29872"), std::string::npos) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(syzygy_root, available_six_piece_dtz_returns_root_move) {
    if (!init_test_syzygy(6))
        GTEST_SKIP() << "6-man Syzygy tablebases not available";

    Board b{"7k/8/5KQ1/8/2BB4/8/P7/8 w - - 0 1"};
    ASSERT_EQ(count_bits(b.color_bb[white] | b.color_bb[black]), 6);

    syzygy::Status status = syzygy::Status::Error;
    const auto [score, move] = syzygy::DTZ_probe(b, status);
    if (!move)
        GTEST_SKIP() << "selected 6-man DTZ material is not installed";

    EXPECT_EQ(status, syzygy::Status::Win);
    EXPECT_EQ(score, Value::tb_win_in_max_ply);
    const auto legal = generate_legal_moves<white>(b);
    EXPECT_TRUE(std::ranges::any_of(legal, [&](Move candidate) {
        return candidate == move;
    }));
}

TEST(syzygy_root, immediate_mate_reports_mate_before_tablebase_cp) {
    if (!init_test_syzygy(3))
        GTEST_SKIP() << "Syzygy tablebases not available";

    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = true;

    Board b{"7k/8/5KQ1/8/8/8/8/8 w - - 0 1"};
    SearchInfo si{b, 8};
    si.board = b;
    si.nnue.refresh(si.board);
    si.starttime = std::chrono::high_resolution_clock::now();
    si.stoptime = si.starttime + std::chrono::hours(1);
    si.soft_stoptime = si.stoptime;

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    thread::pool.init_threads(si, 1);
    thread::pool.wait();
    const auto out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("score mate 1"), std::string::npos) << out;
    EXPECT_NE(out.find("bestmove g6g7"), std::string::npos) << out;
    EXPECT_EQ(out.find("string tbhit win"), std::string::npos) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(uci_root, lost_tablebase_root_does_not_search_after_tbhit) {
    if (!init_test_syzygy(3))
        GTEST_SKIP() << "Syzygy tablebases not available";

    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = true;

    Board b;
    Uci uci{b};
    uci("position fen 6k1/2Q5/5K2/8/8/8/8/8 b - - 10 96");

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    uci("go wtime 82790 btime 68419 winc 5000 binc 5000");
    const auto out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("string tbhit loss"), std::string::npos) << out;
    EXPECT_NE(out.find("bestmove "), std::string::npos) << out;
    EXPECT_EQ(out.find("info depth 2 score"), std::string::npos) << out;
    const auto score_cp = first_uci_cp_score(out);
    EXPECT_GE(score_cp, -20000) << out;
    EXPECT_LT(score_cp, -19000) << out;
    EXPECT_NE(score_cp, Value::tb_loss_in_max_ply) << out;
    EXPECT_EQ(out.find("score cp -29872"), std::string::npos) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(search_root, reports_mate_distance_when_search_finds_mate) {
    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = false;

    Board b{"8/8/8/8/3Q4/k7/8/1K6 w - - 0 1"};
    SearchInfo si{b, 5};
    si.board = b;
    si.nnue.refresh(si.board);
    si.starttime = std::chrono::high_resolution_clock::now();
    si.stoptime = si.starttime + std::chrono::hours(1);
    si.soft_stoptime = si.stoptime;

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    thread::pool.init_threads(si, 1);
    thread::pool.wait();
    const auto out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("score mate 2"), std::string::npos) << out;
    EXPECT_NE(out.find("bestmove "), std::string::npos) << out;
    EXPECT_EQ(out.find("string tbhit"), std::string::npos) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(syzygy_root, game_tb_positions_filter_to_best_wdl) {
    if (!init_test_syzygy(6))
        GTEST_SKIP() << "6-man Syzygy tablebases not available";

    const std::vector<std::string> fens = {
        "8/8/8/8/1k6/2n1KP2/p7/5R2 b - - 0 77",
        "8/8/8/8/1k6/2n1KP2/8/R7 b - - 0 78",
        "8/8/8/3n4/1k6/5P2/4K3/R7 b - - 2 79",
        "8/8/8/8/1k3n2/5P2/8/R4K2 b - - 4 80",
        "8/8/8/8/5n2/1k3P2/5K2/R7 b - - 6 81",
        "8/8/8/8/1k3n2/5P2/5K2/5R2 b - - 8 82",
        "8/8/8/8/5n2/1k2KP2/8/5R2 b - - 10 83",
        "8/8/8/8/5n2/4KP2/1k6/2R5 b - - 12 84",
        "8/8/8/8/5K2/5P2/8/2k5 b - - 0 85",
        "8/8/8/8/8/4KP2/8/1k6 b - - 2 86",
        "8/8/8/8/5P2/4K3/k7/8 b - - 0 87",
        "8/8/8/5P2/8/k3K3/8/8 b - - 0 88",
        "8/8/5P2/8/1k6/4K3/8/8 b - - 0 89",
        "8/5P2/8/8/8/2k1K3/8/8 b - - 0 90",
        "5Q2/8/8/8/2k5/4K3/8/8 b - - 0 91",
        "3Q4/8/8/3k4/8/4K3/8/8 b - - 2 92",
        "3Q4/8/4k3/8/4K3/8/8/8 b - - 4 93",
        "3Q4/5k2/8/5K2/8/8/8/8 b - - 6 94",
        "8/2Q3k1/8/5K2/8/8/8/8 b - - 8 95",
        "6k1/2Q5/5K2/8/8/8/8/8 b - - 10 96",
    };

    const auto root_rank = [](syzygy::Status status) {
        switch (status) {
            case syzygy::Status::Win:  return 2;
            case syzygy::Status::Draw: return 1;
            case syzygy::Status::Loss: return 0;
            default:                   return -1;
        }
    };
    const auto child_to_root_rank = [&](syzygy::Status child_status) {
        return root_rank(
            child_status == syzygy::Status::Win  ? syzygy::Status::Loss
          : child_status == syzygy::Status::Loss ? syzygy::Status::Win
                                                 : syzygy::Status::Draw);
    };
    const auto tablebase_status = [](Board & board) {
        syzygy::Status status = syzygy::Status::Error;
        syzygy::DTZ_probe(board, status);
        if (status == syzygy::Status::Error)
            status = syzygy::WDL_probe(board);
        const int rule50 = board.half_moves + static_cast<int>(board.gamestate.half_moves);
        if (status == syzygy::Status::Error && rule50 <= 20) {
            Board zero_rule50 = board;
            zero_rule50.half_moves = 0;
            zero_rule50.gamestate.half_moves = 0;
            status = syzygy::WDL_probe(zero_rule50);
        }
        return status;
    };
    const auto apply_for_side = [](Board & board, Move move) {
        if (board.side == white)
            apply_move<white, true, false>(board, move);
        else
            apply_move<black, true, false>(board, move);
    };

    for (const auto & fen : fens) {
        Board b{fen};
        const auto original = b.fen();
        const auto legal = b.side == white
            ? generate_legal_moves<white>(b)
            : generate_legal_moves<black>(b);
        const auto filtered = syzygy::root_WDL_filter(b, legal);

        const auto root_status = tablebase_status(b);
        ASSERT_NE(root_status, syzygy::Status::Error) << fen;
        const int root_best_rank = root_rank(root_status);
        ASSERT_FALSE(filtered.empty()) << fen;

        for (auto move : filtered) {
            Board child = b;
            apply_for_side(child, move);
            const auto child_status = tablebase_status(child);
            ASSERT_NE(child_status, syzygy::Status::Error) << fen << " move " << mv2str(move);
            EXPECT_EQ(child_to_root_rank(child_status), root_best_rank)
                << fen << " move " << mv2str(move);
        }
        EXPECT_EQ(b.fen(), original);
    }
}

// --- AccumulatorCache pieces_hash regression ---------------------------------
// The original pieces_hash used `pt << (sq + color*6)`, which collides on
// pawn(=1)@N and knight(=2)@(N-1) — each contributes 2^N, their XORs cancel
// to zero. A board carrying such a pawn+knight pair therefore hashed
// identically to a board with those squares empty. With identical king
// squares (the cache's primary key) the stale-accumulator path was live.
//
// `compute_pieces_hash` is file-static, so we test the invariant that
// refresh_with_cache must uphold: two positions with DIFFERENT piece
// placements must produce DIFFERENT evals after refresh_with_cache, even
// when their old-hash contributions happen to cancel. If the cache returns
// a stale accumulator from the first position, the second position's eval
// will match the first.
TEST(nnue_cache, pieces_hash_pawn_knight_collision_does_not_stale_acc) {
    // Position A: kings on e8/e1 plus white pawn on f2 (sq=10) and
    //             white knight on g2 (sq=9). In H1=0 indexing:
    //             (pawn=1)<<10 ^ (knight=2)<<9 = 0x400 ^ 0x400 = 0.
    // Position B: same kings, pawn and knight absent. Old-hash contribution
    //             from those two pieces is likewise 0, so the stale hash
    //             functions compute the same value for both positions.
    Board a{"4k3/8/8/8/8/8/5PN1/4K3 w - - 0 1"};
    Board b{"4k3/8/8/8/8/8/8/4K3 w - - 0 1"};

    NNUE::Net net;
    // Prime cache with position A, then ask for position B. Under the old
    // hash this was a false cache hit and B's eval == A's eval. Under the
    // Zobrist-based hash the cache misses and B refreshes correctly.
    net.refresh_with_cache(a);
    const auto eval_a = net.Evaluate(a.side);

    net.refresh_with_cache(b);
    const auto eval_b = net.Evaluate(b.side);

    EXPECT_NE(eval_a, eval_b)
        << "AccumulatorCache returned a stale accumulator across positions "
           "whose old XOR-based pieces_hash collide. The fix in 2672d35 "
           "uses Board::zbrs.psq_ randoms and must keep these distinct.\n"
           "Position A: " << a.fen() << " eval=" << eval_a << "\n"
           "Position B: " << b.fen() << " eval=" << eval_b;

    // Belt-and-suspenders: compare B's eval from the cached path against
    // a forced full refresh of B. If the cache is behaving, they match.
    NNUE::Net ref;
    ref.refresh(b);
    EXPECT_EQ(eval_b, ref.Evaluate(b.side))
        << "refresh_with_cache(B) after priming with A produced a different "
           "eval than a fresh refresh(B). Cache validator is unsound.";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    NNUE::Init("");
    return RUN_ALL_TESTS();
}
