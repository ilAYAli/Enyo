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
    const auto pawns_bb = b.pt_bb[Them][pawn];
    if (pawns_bb & pawn_attack_table[Us][sq]) {
        square_t attack_sq = 1ULL << lsb(pawns_bb & pawn_attack_table[Us][sq]);
        b.pt_bb[Them][pawn] &= ~attack_sq;
        return pawn;
    } else if (b.pt_bb[Them][knight] & knight_attack_table[sq]) {
        square_t attack_sq = 1ULL << lsb(b.pt_bb[Them][knight] & knight_attack_table[sq]);
        b.pt_bb[Them][knight] &= ~attack_sq;
        return knight;
    } else if (b.pt_bb[Them][bishop] & bishop_moves<Us>(b, sq)) {
        square_t attack_sq = 1ULL << lsb(b.pt_bb[Them][bishop] & bishop_moves<Us>(b, sq));
        b.pt_bb[Them][bishop] &= ~attack_sq;
        return bishop;
    } else if (b.pt_bb[Them][rook] & rook_moves<Us>(b, sq)) {
        square_t attack_sq = 1ULL << lsb(b.pt_bb[Them][rook] & rook_moves<Us>(b, sq));
        b.pt_bb[Them][rook] &= ~attack_sq;
        return rook;
    } else if (b.pt_bb[Them][queen] & (bishop_moves<Us>(b, sq) | rook_moves<Us>(b, sq))) {
        square_t attack_sq = 1ULL << lsb(b.pt_bb[Them][queen] & (bishop_moves<Us>(b, sq) | rook_moves<Us>(b, sq)));
        b.pt_bb[Them][queen] &= ~attack_sq;
        return queen;
    }
    return no_piece_type;
}

}

// Static Exchange Evaluation
template <Color Us>
inline bool see(Board const & b, Move move, int threshold)
{
    constexpr auto Them = ~Us;
    const auto dst_sq = move.dst_sq();
    const auto dst_piece = move.dst_piece();
    auto attacker = move.src_piece();
    int gain[MAX_PLY] {};

    Board tmpBoard = b;
    tmpBoard.pt_bb[Us][attacker] &= ~(1ULL << move.src_sq());
    if (dst_piece != no_piece_type) {
        tmpBoard.pt_bb[Them][dst_piece] &= ~(1ULL << dst_sq);
    }

    gain[0] = piece_value(dst_piece);

    int d = 0;
    while (true) {
        attacker = remove_least_valuable_attacker<Them>(tmpBoard, dst_sq);
        if (attacker == no_piece_type)
            break;
        gain[++d] = piece_value(attacker) - gain[d - 1];
        if (std::max(-gain[d - 1], gain[d]) < 0)
            break;

        attacker = remove_least_valuable_attacker<Us>(tmpBoard, dst_sq);
        if (attacker == no_piece_type)
            break;
        gain[++d] = piece_value(attacker) - gain[d - 1];
        if (std::max(-gain[d - 1], gain[d]) < 0)
            break;
    }
    while (--d)
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    return gain[0] >= threshold;
}

} // enyo ns