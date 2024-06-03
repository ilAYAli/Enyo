#include "board.hpp"
#include "zobrist.hpp"
#include "config.hpp"
using namespace enyo;

namespace zobrist {

uint64_t generate_hash(Board const & b)
{
    uint64_t zkey = 0;
    for (unsigned c = 0; c < color_nb; c++) {
        for (int square = 0; square < 64; square++) {
            // TODO: use pt_mb
            if ((b.pt_bb[c][knight] >> square) & 1)  zkey ^= b.zbrs.psq_[c][0][square];
            if ((b.pt_bb[c][bishop] >> square) & 1)  zkey ^= b.zbrs.psq_[c][1][square];
            if ((b.pt_bb[c][rook]   >> square) & 1)  zkey ^= b.zbrs.psq_[c][2][square];
            if ((b.pt_bb[c][queen]  >> square) & 1)  zkey ^= b.zbrs.psq_[c][3][square];
            if ((b.pt_bb[c][pawn]   >> square) & 1)  zkey ^= b.zbrs.psq_[c][4][square];
            if ((b.pt_bb[c][king]   >> square) & 1)  zkey ^= b.zbrs.psq_[c][5][square];
        }
    }

    if (b.gamestate.enpassant_square) {
        auto const file = b.gamestate.enpassant_square % 8;
        zkey ^= b.zbrs.enpassant_[file];
    }

    if (b.gamestate.can_castle(CastlingRights::white_oo))  zkey ^= b.zbrs.castling_[0];
    if (b.gamestate.can_castle(CastlingRights::white_ooo)) zkey ^= b.zbrs.castling_[1];
    if (b.gamestate.can_castle(CastlingRights::black_oo))  zkey ^= b.zbrs.castling_[2];
    if (b.gamestate.can_castle(CastlingRights::black_ooo)) zkey ^= b.zbrs.castling_[3];

    zkey ^= b.gamestate.white_to_move; // TODO: is this correct?
    return zkey;
}

} // zobrist
