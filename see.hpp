#pragma once
#include "precalc/pawn_attacks.hpp"
#include "precalc/knight_attacks.hpp"
#include "types.hpp"
#include "board.hpp"
#include "util.hpp"
#include "movegen.hpp"

namespace enyo {

template <Color Us>
inline std::pair<PieceType, bitboard_t> get_lva_bb(Board &b, enyo::square_t sq)
{
    constexpr auto Them = ~Us;
    for (PieceType pt : {pawn, knight, bishop, rook, queen, king}) {
        bitboard_t pt_attacks_bb = 0;
        switch (pt) {
            case pawn:   pt_attacks_bb = pawn_attack_table[Us][sq]; break;
            case knight: pt_attacks_bb = knight_attack_table[sq]; break;
            case bishop: pt_attacks_bb = bishop_moves<Them>(b, sq); break;
            case rook:   pt_attacks_bb = rook_moves<Them>(b, sq); break;
            case queen:  pt_attacks_bb = bishop_moves<Them>(b, sq) | rook_moves<Them>(b, sq); break;
            case king:   pt_attacks_bb = king_attack_table[sq]; break;
            default:     pt_attacks_bb = 0;
        }

        auto pt_attacks_to_bb = pt_attacks_bb & b.pt_bb[Us][pt];
        if (pt_attacks_to_bb) {
            //fmt::println("LVA to {}, {} {}", pt, sq2str(sq), visualize(pt_attacks_to_bb));
            return std::make_pair(pt, pt_attacks_to_bb);
        }
    }
    return {};
}

template <Color Us>
inline int see(Board b, Move move, int threshold = 0)
{
    constexpr auto Them = ~Us;
    const auto dst_piece = move.dst_piece();
    if (!dst_piece) // Non-capture move:
        return 0;

    const auto src_sq = move.src_sq();
    const auto dst_sq = move.dst_sq();
    auto src_piece = move.src_piece();
    int gain[32] = {};

    int idx = 0;
    gain[idx++] = piece_value(dst_piece);
    gain[idx++] = piece_value(src_piece);
    // Remove initial attacker and capture:
    b.pt_bb[Them][dst_piece] &= ~(1ULL << dst_sq);
    b.pt_bb[Us][src_piece] &= ~(1ULL << src_sq);

    Color side_to_move = Them; // Start with the opponent's move

    while (true) {
        std::pair<PieceType, bitboard_t> attacker;
        if (side_to_move == Us) {
            attacker = get_lva_bb<Us>(b, dst_sq);
        } else {
            attacker = get_lva_bb<Them>(b, dst_sq);
        }

        auto [pt, bb] = attacker;
        if (!bb) break;

        gain[idx] = piece_value(pt);
        if (gain[idx] - gain[idx - 1] < threshold)
            break;

        square_t attacker_sq = lsb(bb);

        // Remove the least valuable attacker:
        b.pt_bb[side_to_move][pt] &= ~(1ULL << attacker_sq);

        idx++;
        side_to_move = ~side_to_move; // Switch sides
    }

    if (idx <= 1)
        return gain[0];

    while (--idx) {
        gain[idx - 1] = std::max(-gain[idx], gain[idx - 1]);
    }

    return gain[0];
}

}
