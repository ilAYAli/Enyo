#include "board.hpp"
#include "zobrist.hpp"
#include "util.hpp"
using namespace enyo;

namespace zobrist {

uint64_t generate_hash(Board const & b)
{
    uint64_t zkey = 0;
    for (int c = white; c < color_nb; c++) {
        for (int pt = static_cast<int>(pawn); pt <= static_cast<int>(king); pt++) {
            auto bb = b.pt_bb[c][pt];
            while (bb) {
                const auto sq = pop_lsb(bb);
                zkey ^= b.zbrs.psq_[c][pt][sq];
            }
        }
    }

    if (b.gamestate.enpassant_square) {
        auto const file = b.gamestate.enpassant_square % 8;
        zkey ^= b.zbrs.enpassant_[file];
    }

    if (b.gamestate.can_castle(CastlingRights::white_ooo)) zkey ^= b.zbrs.castling_[0];
    if (b.gamestate.can_castle(CastlingRights::white_oo))  zkey ^= b.zbrs.castling_[1];
    if (b.gamestate.can_castle(CastlingRights::black_ooo)) zkey ^= b.zbrs.castling_[2];
    if (b.gamestate.can_castle(CastlingRights::black_oo))  zkey ^= b.zbrs.castling_[3];

    if (b.side == black)
        zkey ^= b.zbrs.side_;

    return zkey;
}

uint64_t generate_pawn_hash(Board const & b)
{
    uint64_t zkey = 0;
    for (int c = white; c < color_nb; c++) {
        auto bb = b.pt_bb[c][pawn];
        while (bb) {
            const auto sq = pop_lsb(bb);
            zkey ^= b.zbrs.psq_[c][pawn][sq];
        }
    }
    return zkey;
}

} // zobrist
