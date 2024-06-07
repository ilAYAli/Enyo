#pragma once
#include "precalc/pawn_attacks.hpp"
#include "precalc/knight_attacks.hpp"
#include "types.hpp"
#include "board.hpp"

namespace enyo {

namespace {

template <Color Us>
inline PieceType remove_least_valuable_attacker(Board &b, square_t sq) {
    constexpr Color Them = ~Us;
    for (PieceType pt : {pawn, knight, bishop, rook, queen}) {
        const auto attackers_bb = b.pt_bb[Them][pt];
        bitboard_t attacks;
        switch (pt) {
            case pawn:   attacks = pawn_attack_table[Us][sq]; break;
            case knight: attacks = knight_attack_table[sq]; break;
            case bishop: attacks = bishop_moves<Us>(b, sq); break;
            case rook:   attacks = rook_moves<Us>(b, sq); break;
            case queen:  attacks = bishop_moves<Us>(b, sq) | rook_moves<Us>(b, sq); break;
            default:     attacks = 0;
        }

        if (attackers_bb & attacks) {
            square_t attack_sq = lsb(attackers_bb & attacks);
            b.pt_bb[Them][pt] &= ~(1ULL << attack_sq);
            return pt;
        }
    }
    return no_piece_type;
}

}

// Static Exchange Evaluation (note: board must be passed by value)
template <Color Us>
inline bool see(Board b, Move move, int threshold)
{
    constexpr auto Them = ~Us;
    const auto dst_sq = move.dst_sq();
    const auto dst_piece = move.dst_piece();
    auto attacker = move.src_piece();
    int gain[MAX_PLY] {};

    // Remove attacker:
    b.pt_bb[Us][attacker] &= ~(1ULL << move.src_sq());
    // remove capture if any:
    if (dst_piece != no_piece_type) {
        b.pt_bb[Them][dst_piece] &= ~(1ULL << dst_sq);
    }

    gain[0] = piece_value(dst_piece);
    if (gain[0] < threshold)
        return false;

    int d = 0;
    while (true) {
        // Opponent attack:
        attacker = remove_least_valuable_attacker<Them>(b, dst_sq);
        if (attacker == no_piece_type) break;
        gain[++d] = piece_value(attacker) - gain[d - 1];
        if (std::max(-gain[d - 1], gain[d]) < 0) break;

        // Our recapture:
        attacker = remove_least_valuable_attacker<Us>(b, dst_sq);
        if (attacker == no_piece_type) break;
        gain[++d] = piece_value(attacker) - gain[d - 1];
        if (std::max(-gain[d - 1], gain[d]) < 0) break;
    }

    if (!d)
        return false;
    while (--d) {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    }

    return gain[0] >= threshold;
}

} // enyo ns
