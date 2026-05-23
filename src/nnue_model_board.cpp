// Engine integration for Network.
// Links network's forward-pass primitives to enyo::Board. Kept in a
// separate TU so tests can compile nnue_model.cpp without
// pulling in board.hpp / fmt / etc.

#include "nnue_model.hpp"
#include "board.hpp"
#include "magic/magic.hpp"
#include "precalc/bishop_attacks.hpp"
#include "precalc/king_attacks.hpp"
#include "precalc/knight_attacks.hpp"
#include "precalc/pawn_attacks.hpp"
#include "precalc/rook_attacks.hpp"
#include "util.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace Network {

// enumerate_pieces — walk `board.pt_bb` for both colors and emit
// (packed-piece-code, enyo-sq) entries. Order doesn't matter since
// ApplyDelta's add path is commutative.
size_t enumerate_pieces(const enyo::Board& b, PieceEntry* out) {
    size_t n = 0;
    for (int color = 0; color < 2; ++color) {
        for (int pt = static_cast<int>(enyo::pawn); pt <= static_cast<int>(enyo::king); ++pt) {
            enyo::bitboard_t bb = b.pt_bb[color][pt];
            while (bb) {
                const auto sq = static_cast<enyo::square_t>(enyo::pop_lsb(bb));
                const int piece_code = ((pt - 1) << 1) | color;
                out[n++] = {piece_code, sq};
            }
        }
    }
    return n;
}

void feature_indices(const enyo::Board& b,
                     enyo::Color view,
                     std::vector<int>& out)
{
    out.clear();
    const enyo::square_t view_ksq =
        view == enyo::white
            ? static_cast<enyo::square_t>(enyo::lsb(b.pt_bb[enyo::white][enyo::king]))
            : static_cast<enyo::square_t>(enyo::lsb(b.pt_bb[enyo::black][enyo::king]));

    enyo::bitboard_t pieces = b.color_bb[enyo::white] | b.color_bb[enyo::black];
    while (pieces) {
        const auto sq = static_cast<enyo::square_t>(enyo::pop_lsb(pieces));
        const enyo::PieceType pt = b.pt_mb[sq];
        const enyo::Color pc = (b.color_bb[enyo::white] & (1ULL << sq))
            ? enyo::white
            : enyo::black;
        out.push_back(FeatureIdx(pt, pc, sq, view_ksq, view));
    }
}

int ScaleEval(const enyo::Board& b, int score) {
    const auto minors =
        b.pt_bb[enyo::white][enyo::knight]
      | b.pt_bb[enyo::black][enyo::knight]
      | b.pt_bb[enyo::white][enyo::bishop]
      | b.pt_bb[enyo::black][enyo::bishop];
    const auto rooks =
        b.pt_bb[enyo::white][enyo::rook]
      | b.pt_bb[enyo::black][enyo::rook];
    const auto queens =
        b.pt_bb[enyo::white][enyo::queen]
      | b.pt_bb[enyo::black][enyo::queen];
    const int phase = 3 * enyo::count_bits(minors)
        + 5 * enyo::count_bits(rooks)
        + 10 * enyo::count_bits(queens);

    score = (128 + phase) * score / 128;
    return std::clamp(score, -2045, 2045);
}

#if !ENYO_ENABLE_KING_PRESSURE_BUCKETS
int MaterialBucket(const enyo::Board& b) {
    const int pieces = enyo::count_bits(
        b.color_bb[enyo::white] | b.color_bb[enyo::black]);
    const int bucket = (pieces - 2) / 4;
    return std::clamp(bucket, 0, N_OUTPUT_BUCKETS - 1);
}
#endif

#if ENYO_ENABLE_THREAT_NNUE || ENYO_ENABLE_KING_PRESSURE_BUCKETS
namespace {

enyo::bitboard_t attacks_from(enyo::PieceType pt,
                              enyo::Color color,
                              enyo::square_t sq,
                              enyo::bitboard_t occ)
{
    switch (pt) {
    case enyo::pawn:
        return enyo::pawn_attack_table[color][sq];
    case enyo::knight:
        return enyo::knight_attack_table[sq];
    case enyo::bishop:
        return enyo::get_bishop_attacks(sq, occ);
    case enyo::rook:
        return enyo::get_rook_attacks(sq, occ);
    case enyo::queen:
        return enyo::get_bishop_attacks(sq, occ) | enyo::get_rook_attacks(sq, occ);
    case enyo::king:
        return king_attack_table[sq];
    default:
        return 0;
    }
}

} // namespace
#endif

#if ENYO_ENABLE_KING_PRESSURE_BUCKETS
int MaterialBucket(const enyo::Board& b) {
    const auto us = b.side;
    const auto them = ~us;
    const auto king_square = static_cast<enyo::square_t>(
        enyo::lsb(b.pt_bb[us][enyo::king]));
    const enyo::bitboard_t occ = b.color_bb[enyo::white] | b.color_bb[enyo::black];
    const enyo::bitboard_t king_zone =
        king_attack_table[king_square] | (1ULL << king_square);

    int pressure = 0;
    for (int pt_int = static_cast<int>(enyo::pawn);
         pt_int <= static_cast<int>(enyo::queen);
         ++pt_int) {
        const auto pt = static_cast<enyo::PieceType>(pt_int);
        enyo::bitboard_t pieces = b.pt_bb[them][pt];
        while (pieces) {
            const auto from = static_cast<enyo::square_t>(enyo::pop_lsb(pieces));
            if ((attacks_from(pt, them, from, occ) & king_zone) == 0)
                continue;

            pressure += pt == enyo::pawn ? 1
                : pt == enyo::knight || pt == enyo::bishop ? 2
                : pt == enyo::rook ? 3
                : 4;
        }
    }

    return std::clamp(pressure, 0, N_OUTPUT_BUCKETS - 1);
}
#endif

#if ENYO_ENABLE_THREAT_NNUE
namespace {

int relative_piece(enyo::PieceType pt, enyo::Color color, enyo::Color view)
{
    const int piece = ((static_cast<int>(pt) - 1) << 1) | static_cast<int>(color);
    return 6 * ((piece ^ static_cast<int>(view)) & 0x1) + (piece >> 1);
}

int threat_from_square(enyo::square_t from,
                       enyo::square_t king_square,
                       enyo::Color view)
{
    const int sq = to_net_sq(from);
    const int kingsq = to_net_sq(king_square);
    return (7 * !(kingsq & 4)) ^ (56 * static_cast<int>(view)) ^ sq;
}

int ThreatFeatureIdx(enyo::PieceType attacker_pt,
                     enyo::Color attacker_color,
                     enyo::square_t from,
                     enyo::PieceType attacked_pt,
                     enyo::Color attacked_color,
                     enyo::square_t king_square,
                     enyo::Color view)
{
    const int attacker = relative_piece(attacker_pt, attacker_color, view);
    const int attacked = relative_piece(attacked_pt, attacked_color, view);
    const int sq = threat_from_square(from, king_square, view);
    return ((attacker * N_PIECE_TYPES) + attacked) * N_SQUARES + sq;
}

void add_threat_feature(acc_t* values, int feature)
{
    if (THREAT_FEATURE_ACTIVE[feature] == 0)
        return;

    const int8_t* weights = &THREAT_WEIGHTS[static_cast<size_t>(feature) * N_HIDDEN];
    for (size_t i = 0; i < N_HIDDEN; ++i)
        values[i] = static_cast<acc_t>(values[i] + static_cast<int>(weights[i]));
}

} // namespace

void AddThreatsToAccumulator(Accumulator* dest,
                             const enyo::Board& b,
                             enyo::Color view)
{
    if (THREAT_WEIGHTS == nullptr)
        return;

    const enyo::square_t king_square = static_cast<enyo::square_t>(
        enyo::lsb(b.pt_bb[view][enyo::king]));
    const enyo::bitboard_t occ = b.color_bb[enyo::white] | b.color_bb[enyo::black];
    const enyo::bitboard_t king_zone =
        king_attack_table[king_square] | (1ULL << king_square);
    acc_t* values = dest->values[view];

    for (auto color : {enyo::white, enyo::black}) {
        for (int pt_int = static_cast<int>(enyo::pawn);
             pt_int <= static_cast<int>(enyo::king);
             ++pt_int) {
            const auto pt = static_cast<enyo::PieceType>(pt_int);
            enyo::bitboard_t pieces = b.pt_bb[color][pt];
            while (pieces) {
                const auto from = static_cast<enyo::square_t>(enyo::pop_lsb(pieces));
                enyo::bitboard_t targets =
                    attacks_from(pt, color, from, occ) & occ & king_zone;
                while (targets) {
                    const auto to = static_cast<enyo::square_t>(enyo::pop_lsb(targets));
                    const auto attacked_pt = b.pt_mb[to];
                    if (attacked_pt == enyo::no_piece_type)
                        continue;
                    const auto attacked_color =
                        (b.color_bb[enyo::white] & (1ULL << to))
                            ? enyo::white
                            : enyo::black;
                    const int feature = ThreatFeatureIdx(
                        pt, color, from, attacked_pt, attacked_color,
                        king_square, view);
                    add_threat_feature(values, feature);
                }
            }
        }
    }

    dest->eval_correct[enyo::white] = 0;
    dest->eval_correct[enyo::black] = 0;
}
#endif

// Phase-4-v1 correctness path: fresh accumulator on every call. A
// future phase will plug into the engine's Net accumulator stack for
// search-time incremental updates.
int EvaluateFromScratch(const enyo::Board& b) {
    PieceEntry pieces[32];
    const size_t n = enumerate_pieces(b, pieces);

    enyo::square_t wk_sq = 0, bk_sq = 0;
    for (size_t i = 0; i < n; ++i) {
        if (pieces[i].piece_code == 10) wk_sq = pieces[i].sq;
        if (pieces[i].piece_code == 11) bk_sq = pieces[i].sq;
    }

    Accumulator acc;
    std::memset(&acc, 0, sizeof(acc));
    ResetAccumulator(&acc, enyo::white, wk_sq, pieces, n);
    ResetAccumulator(&acc, enyo::black, bk_sq, pieces, n);
#if ENYO_ENABLE_THREAT_NNUE
    AddThreatsToAccumulator(&acc, b, enyo::white);
    AddThreatsToAccumulator(&acc, b, enyo::black);
#endif

    const int bucket = OUTPUT_BUCKETS == 1 ? 0 : MaterialBucket(b);
    return Propagate(&acc, static_cast<int>(b.side), bucket);
}

} // namespace Network
