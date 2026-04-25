#include "board.hpp"
#include "zobrist.hpp"
#include "config.hpp"
using namespace enyo;

namespace zobrist {

constexpr int ep_file(square_t sq)
{
    return 7 - (sq % 8);
}

bool has_legal_ep(const Board & b)
{
    const auto ep = b.gamestate.enpassant_square;
    if (!ep)
        return false;

    const auto side = b.side;
    const auto pawns = b.pt_bb[side][pawn];
    const auto ep_mask = 1ULL << ep;
    const auto attackers = side == white
        ? ((ep_mask >> 7) & ~file_h) | ((ep_mask >> 9) & ~file_a)
        : ((ep_mask << 7) & ~file_a) | ((ep_mask << 9) & ~file_h);

    return static_cast<bool>(pawns & attackers);
}

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

uint64_t generate_stockfish_hash(Board const & b)
{
    uint64_t key = 0;

    for (int square = 0; square < 64; square++) {
        const auto pt = b.pt_mb[square];
        if (pt == no_piece_type)
            continue;

        const auto sq_mask = 1ULL << square;
        const auto color = (b.color_bb[white] & sq_mask) ? white : black;
        const int sf_piece = color == white
            ? (pt == pawn ? 1 : pt == knight ? 2 : pt == bishop ? 3 : pt == rook ? 4 : pt == queen ? 5 : 6)
            : (pt == pawn ? 9 : pt == knight ? 10 : pt == bishop ? 11 : pt == rook ? 12 : pt == queen ? 13 : 14);
        const int sf_square = 7 - (square % 8) + 8 * (square / 8);
        key ^= b.zbrs.stockfish_psq_[sf_piece][sf_square];
    }

    if (has_legal_ep(b)) {
        const auto file = ep_file(b.gamestate.enpassant_square);
        key ^= b.zbrs.enpassant_[file];
    }

    if (!b.gamestate.white_to_move)
        key ^= b.zbrs.side_;

    key ^= b.zbrs.castling_[b.gamestate.castling_rights];
    return key;
}

} // zobrist
