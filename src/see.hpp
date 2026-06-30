#pragma once
#include "precalc/pawn_attacks.hpp"
#include "precalc/knight_attacks.hpp"
#include "precalc/king_attacks.hpp"
#include "magic/magic.hpp"
#include "types.hpp"
#include "board.hpp"
#include "util.hpp"

namespace enyo {

inline bitboard_t attackers_to(const Board & b, square_t sq, bitboard_t occ)
{
    bitboard_t a = 0;
    a |= pawn_attack_table[white][sq] & b.pt_bb[white][pawn];
    a |= pawn_attack_table[black][sq] & b.pt_bb[black][pawn];
    a |= knight_attack_table[sq]       & (b.pt_bb[white][knight] | b.pt_bb[black][knight]);
    a |= king_attack_table[sq]         & (b.pt_bb[white][king]   | b.pt_bb[black][king]);
    const bitboard_t bishops_queens = b.pt_bb[white][bishop] | b.pt_bb[black][bishop]
                                    | b.pt_bb[white][queen]  | b.pt_bb[black][queen];
    const bitboard_t rooks_queens   = b.pt_bb[white][rook]   | b.pt_bb[black][rook]
                                    | b.pt_bb[white][queen]  | b.pt_bb[black][queen];
    a |= get_bishop_attacks(sq, occ) & bishops_queens;
    a |= get_rook_attacks(sq, occ)   & rooks_queens;
    return a & occ;
}

template <Color Us>
inline int see(const Board & b, Move move, int threshold = 0)
{
    constexpr auto Them = ~Us;
    const auto dst_piece = move.dst_piece();
    if (!dst_piece)
        return 0;

    const auto src_sq = move.src_sq();
    const auto dst_sq = move.dst_sq();
    const auto src_piece = move.src_piece();

    int gain[32];
    int d = 0;
    gain[d] = piece_value(dst_piece);

    bitboard_t occ = b.color_bb[white] | b.color_bb[black];
    bitboard_t attackers = attackers_to(b, dst_sq, occ);

    // Remove the initial attacker from the occupancy and attackers set.
    occ &= ~(1ULL << src_sq);
    attackers &= ~(1ULL << src_sq);

    // X-ray sliders: any slider behind the just-removed piece may now attack.
    const bitboard_t bishops_queens = b.pt_bb[white][bishop] | b.pt_bb[black][bishop]
                                    | b.pt_bb[white][queen]  | b.pt_bb[black][queen];
    const bitboard_t rooks_queens   = b.pt_bb[white][rook]   | b.pt_bb[black][rook]
                                    | b.pt_bb[white][queen]  | b.pt_bb[black][queen];
    attackers |= get_bishop_attacks(dst_sq, occ) & bishops_queens & occ;
    attackers |= get_rook_attacks(dst_sq, occ)   & rooks_queens   & occ;

    PieceType attacker_pt = src_piece;
    Color stm = Them;

    while (true) {
        // Find least-valuable attacker for side to move.
        bitboard_t stm_attackers = 0;
        PieceType next_pt = no_piece_type;
        for (PieceType pt : {pawn, knight, bishop, rook, queen, king}) {
            bitboard_t bb = attackers & b.pt_bb[stm][pt];
            if (bb) {
                stm_attackers = bb;
                next_pt = pt;
                break;
            }
        }
        if (!next_pt)
            break;

        ++d;
        gain[d] = piece_value(attacker_pt) - gain[d - 1];

        // Speculative cutoff: if the side to move cannot raise its score
        // above the caller's best (-gain[d-1]), stop.
        if (std::max(-gain[d - 1], gain[d]) < 0)
            break;

        const square_t from = lsb(stm_attackers);
        occ &= ~(1ULL << from);
        attackers &= ~(1ULL << from);

        // Re-reveal X-ray sliders behind the removed attacker.
        if (next_pt == pawn || next_pt == bishop || next_pt == queen)
            attackers |= get_bishop_attacks(dst_sq, occ) & bishops_queens & occ;
        if (next_pt == rook || next_pt == queen)
            attackers |= get_rook_attacks(dst_sq, occ) & rooks_queens & occ;

        attacker_pt = next_pt;
        stm = ~stm;

        if (attacker_pt == king) {
            // King cannot recapture if the opponent still has attackers.
            bitboard_t opp_remaining = 0;
            for (PieceType pt : {pawn, knight, bishop, rook, queen})
                opp_remaining |= attackers & b.pt_bb[stm][pt];
            if (opp_remaining) {
                --d;
                break;
            }
        }
    }

    while (--d >= 0)
        gain[d] = -std::max(-gain[d], gain[d + 1]);

    (void)threshold;
    return gain[0];
}

template <Color Us>
inline bool see_ge(const Board & b, Move move, int threshold = 0)
{
    if (move.flags() != Move::Flags::generic)
        return see<Us>(b, move) >= threshold;

    const auto dst_piece = move.dst_piece();
    if (!dst_piece)
        return threshold <= 0;

    int swap = piece_value(dst_piece) - threshold;
    if (swap < 0)
        return false;

    const auto src_piece = move.src_piece();
    swap = piece_value(src_piece) - swap;
    if (swap <= 0)
        return true;

    const auto src_sq = move.src_sq();
    const auto dst_sq = move.dst_sq();
    const bitboard_t bishops_queens = b.pt_bb[white][bishop] | b.pt_bb[black][bishop]
                                    | b.pt_bb[white][queen]  | b.pt_bb[black][queen];
    const bitboard_t rooks_queens   = b.pt_bb[white][rook] | b.pt_bb[black][rook]
                                    | b.pt_bb[white][queen] | b.pt_bb[black][queen];

    bitboard_t occ = (b.color_bb[white] | b.color_bb[black])
                   ^ (1ULL << src_sq) ^ (1ULL << dst_sq);
    bitboard_t attackers = attackers_to(b, dst_sq, occ);
    Color stm = Us;
    int result = 1;

    while (true) {
        stm = ~stm;
        attackers &= occ;
        const bitboard_t stm_attackers = attackers & b.color_bb[stm];
        if (!stm_attackers)
            break;

        result ^= 1;
        bitboard_t candidates;

        if ((candidates = stm_attackers & b.pt_bb[stm][pawn])) {
            swap = piece_value(pawn) - swap;
            if (swap < result)
                break;
            occ ^= candidates & -candidates;
            attackers |= get_bishop_attacks(dst_sq, occ) & bishops_queens;
        } else if ((candidates = stm_attackers & b.pt_bb[stm][knight])) {
            swap = piece_value(knight) - swap;
            if (swap < result)
                break;
            occ ^= candidates & -candidates;
        } else if ((candidates = stm_attackers & b.pt_bb[stm][bishop])) {
            swap = piece_value(bishop) - swap;
            if (swap < result)
                break;
            occ ^= candidates & -candidates;
            attackers |= get_bishop_attacks(dst_sq, occ) & bishops_queens;
        } else if ((candidates = stm_attackers & b.pt_bb[stm][rook])) {
            swap = piece_value(rook) - swap;
            if (swap < result)
                break;
            occ ^= candidates & -candidates;
            attackers |= get_rook_attacks(dst_sq, occ) & rooks_queens;
        } else if ((candidates = stm_attackers & b.pt_bb[stm][queen])) {
            swap = piece_value(queen) - swap;
            if (swap < result)
                break;
            occ ^= candidates & -candidates;
            attackers |= get_bishop_attacks(dst_sq, occ) & bishops_queens;
            attackers |= get_rook_attacks(dst_sq, occ) & rooks_queens;
        } else {
            return static_cast<bool>((attackers & b.color_bb[~stm]) ? result ^ 1 : result);
        }
    }

    return static_cast<bool>(result);
}

} // namespace enyo
