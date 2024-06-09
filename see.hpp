#pragma once
#include "precalc/pawn_attacks.hpp"
#include "precalc/knight_attacks.hpp"
#include "types.hpp"
#include "board.hpp"
#include "util.hpp"
#include "movegen.hpp"

namespace enyo {

namespace {

template <Color Us>
inline std::pair<PieceType, bitboard_t> get_lva_bb(Board &b, enyo::square_t sq)
{
    constexpr auto Them = ~Us;
    for (PieceType pt : {pawn, knight, bishop, rook, queen}) {
        bitboard_t pt_attacks_bb = 0;
        switch (pt) {
            case pawn:   pt_attacks_bb = pawn_attack_table[Us][sq]; break;
            case knight: pt_attacks_bb = knight_attack_table[sq]; break;
            case bishop: pt_attacks_bb = bishop_moves<Them>(b, sq); break;
            case rook:   pt_attacks_bb = rook_moves<Them>(b, sq); break;
            case queen:  pt_attacks_bb = bishop_moves<Them>(b, sq) | rook_moves<Them>(b, sq); break;
            default:     pt_attacks_bb = 0;
        }

        // fmt::print("<{}>{} attacks to: {}, {}\n", Us, pt, sq2str(sq), visualize(pt_attacks_bb));
        auto pt_attacks_to_bb = pt_attacks_bb & b.pt_bb[Us][pt];
        if (pt_attacks_to_bb) {
            //fmt::println("LVA to {}, {} {}", pt, sq2str(sq), visualize(pt_attacks_to_bb));
            return std::make_pair(pt, pt_attacks_to_bb);
        }
    }
    return {};
}

}

// Static Exchange Evaluation (note: board must be passed by value)
template <Color Us>
inline int see(Board b, Move move, int threshold = 0)
{
    constexpr auto Them = ~Us;
    const auto dst_piece = move.dst_piece();
    if (!dst_piece) return 0;

    const auto src_sq = move.src_sq();
    const auto dst_sq = move.dst_sq();
    const auto src_piece = move.dst_piece();
    auto attacker = move.src_piece();
    int gain[16]{};

    int idx = 0;
    gain[idx++] = piece_value(dst_piece);
    gain[idx++] = piece_value(src_piece);
    // remove attacker and capture:
    b.pt_bb[Us][attacker] &= ~(1ULL << src_sq);
    b.pt_bb[Them][dst_piece] &= ~(1ULL << dst_sq);

    auto Side = ~Us;
    while (true) {
        { // attack:
            constexpr auto Side = ~Us;
            auto [pt, bb] = get_lva_bb<Side>(b, dst_sq);
            if (!bb) break;
            gain[idx++] = piece_value(pt);
            b.pt_bb[Side][pt] &= ~(1ULL << lsb(bb));
        }

        { // recapture:
            constexpr auto Side = Us;
            auto [pt, bb] = get_lva_bb<Side>(b, dst_sq);
            if (!bb) break;
            gain[idx++] = piece_value(pt);
            b.pt_bb[Side][pt] &= ~(1ULL << lsb(bb));
        }
    }

    if (!idx) return 0;
    int delta = 0;
    for (int i = 0; i < idx -1; i++) {
        if (!gain[i]) break;
        if (!(i & 1))
            delta += gain[i];
        else
            delta -= gain[i];
    }

    return delta;
}

} // enyo ns
