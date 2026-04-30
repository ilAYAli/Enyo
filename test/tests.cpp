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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    NNUE::Init("");
    return RUN_ALL_TESTS();
}
