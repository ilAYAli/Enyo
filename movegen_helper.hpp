#pragma once
// ONLY USED FOR TESTING

#include "board.hpp"
#include "types.hpp"
namespace enyo {

template <Color Us>
inline Move resolve_move(Board & b, PieceType src_piece, square_t src, uint8_t dst);

template <>
inline Move resolve_move<white>(Board &b, PieceType src_piece, square_t src, uint8_t dst)
{
    auto const dst_piece = get_piece_type<black>(b, dst);
    Move mv{ src, white, src_piece, dst, dst_piece };
    //if (dst_piece == king) BREAKPOINT("dst piece can't be king");

    uint32_t move_type = 0;
    switch (src_piece) {
        case king: {
            constexpr square_t king_home = e1;
            constexpr bitboard_t castle_mask = 0x2200000000000022;
            if (src == king_home && ((1ULL << dst) & castle_mask)) {
                move_type |= Move::Flags::Castle;
            }
            break;
        }
        case pawn: {
            auto const attack = frtab[src][0] != frtab[dst][0];
                if (frtab[dst][1] == rank_8) {
                    move_type |= Move::Flags::Promote;
                } else if ((dst == b.gamestate.enpassant_square) && attack) {
                    move_type |= Move::Flags::Enpassant;
                }
            break;
        }
        case rook:
        case bishop:
        case knight:
        case queen:
        case no_piece_type:
        case piece_type_nb:
            break;
    }

    mv.set_flags(static_cast<Move::Flags>(move_type));

    return mv;
}

template <>
inline Move resolve_move<black>(Board & b, PieceType src_piece, square_t src, uint8_t dst)
{
    auto const dst_piece = get_piece_type<white>(b, dst);
    Move mv{ src, black, src_piece, dst, dst_piece };
    //if (dst_piece == king) BREAKPOINT("dst piece can't be king");

    uint32_t move_type = 0;
    switch (src_piece) {
        case king: {
            constexpr square_t king_home = e8;
            constexpr bitboard_t castle_mask = 0x2200000000000022;
            if (src == king_home && ((1ULL << dst) & castle_mask)) {
                move_type |= static_cast<uint32_t>(Move::Flags::Castle);
            }
            break;
        }
        case pawn: {
            auto const attack = frtab[src][0] != frtab[dst][0];
            if (frtab[dst][1] == rank_1) {
                move_type |= Move::Flags::Promote;
            } else if ((dst == b.gamestate.enpassant_square) && attack) {
                move_type |= Move::Flags::Enpassant;
            }
            break;
        }
        case rook:
        case bishop:
        case knight:
        case queen:
        case no_piece_type:
        case piece_type_nb:
            break;
    }

    mv.set_flags(static_cast<Move::Flags>(move_type));

    return mv;
}

} // enyo
