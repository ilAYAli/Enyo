#include "probe.hpp"
#include "3rdparty/Fathom/src/tbprobe.h"

#include "board.hpp"
#include "config.hpp"
#include "movegen.hpp"

using namespace enyo;

namespace {
bool initialized = false;
}

namespace syzygy {

bool init(const std::string & tb_path)
{
    initialized = tb_init(tb_path.c_str());
    return initialized;
}

pos board2pos(Board & b)
{
    assert(b.color_bb[white] & b.pt_bb[white][king]);
    assert(b.color_bb[black] & b.pt_bb[black][king]);
    return pos {
        .white = bbconv(b.color_bb[white]),
        .black = bbconv(b.color_bb[black]),
        .kings = bbconv(b.pt_bb[white][king] | b.pt_bb[black][king]),
        .queens = bbconv(b.pt_bb[white][queen] | b.pt_bb[black][queen]),
        .rooks = bbconv(b.pt_bb[white][rook] | b.pt_bb[black][rook]),
        .bishops = bbconv(b.pt_bb[white][bishop] | b.pt_bb[black][bishop]),
        .knights = bbconv(b.pt_bb[white][knight] | b.pt_bb[black][knight]),
        .pawns = bbconv(b.pt_bb[white][pawn] | b.pt_bb[black][pawn]),
        .castling = uint8_t(b.gamestate.castling_rights),
        .rule50 = 0,
        .ep = sqconv(b.gamestate.enpassant_square),
        .turn = b.side == white,
        .move = static_cast<uint16_t>(b.histply + 1),
    };
}

Status WDL_probe(Board &board)
{
    if (!initialized)
        return Status::Error;

    auto pos = board2pos(board);
    auto ret =
        tb_probe_wdl(
            pos.white, pos.black,
            pos.kings, pos.queens, pos.rooks, pos.bishops, pos.knights, pos.pawns,
            pos.rule50,
            pos.castling,
            pos.ep,
            pos.turn
    );
    switch (ret) {
        case TB_WIN:            return Status::Win;
        case TB_CURSED_WIN:     return Status::Draw;
        case TB_DRAW:           return Status::Draw;
        case TB_BLESSED_LOSS:   return Status::Draw;
        case TB_LOSS:           return Status::Loss;
        default:
            return Status::Error;
    }
}

std::pair<int, Move> probe_DTZ(Board & board)
{
    if (!initialized) {
        return {TB_RESULT_FAILED, Move{}};
    }

    auto pos = board2pos(board);

    unsigned TBresult =
        tb_probe_root(
            pos.white, pos.black,
            pos.kings, pos.queens, pos.rooks, pos.bishops, pos.knights, pos.pawns,
            pos.rule50,
            pos.castling,
            pos.ep,
            pos.turn,
            nullptr // results
        );

    if (TBresult == TB_RESULT_FAILED || TBresult == TB_RESULT_CHECKMATE ||
        TBresult == TB_RESULT_STALEMATE)
        return {TB_RESULT_FAILED, Move{}};

    const int wdl = TB_GET_WDL(TBresult);

    int score = 0;
    if (wdl == TB_LOSS) {
        score = Value::TB_LOSS_IN_MAX_PLY;
    }
    if (wdl == TB_WIN) {
        score = Value::TB_WIN_IN_MAX_PLY;
    }
    if (wdl == TB_BLESSED_LOSS || wdl == TB_DRAW || wdl == TB_CURSED_WIN) {
        score = 0;
    }

    const int promo = TB_GET_PROMOTES(TBresult);
    const PieceType promoTranslation[] = {no_piece_type, queen, rook, bishop, knight };

    const auto sqFrom = square_t(TB_GET_FROM(TBresult));
    const auto sqTo = square_t(TB_GET_TO(TBresult));

    Movelist legalmoves;
    if (board.side == white)
        legalmoves = generate_legal_moves<white>(board);
    else
        legalmoves = generate_legal_moves<black>(board);

    // no piece
    for (auto move : legalmoves) {
        if (move.get_src() == sqFrom && move.get_dst() == sqTo) {
            if (promoTranslation[promo] == no_piece_type) {
               if (move.get_flags() != Move::Flags::Promote) {
                    return {score, move};
                }
            } else {
                if (promo < 5) {
                    if (move.get_promo_piece() == promoTranslation[promo] && move.get_flags() == Move::Flags::Promote) {
                        return {score, move};
                    }
                }
            }
        }
    }
    return {TB_RESULT_FAILED, Move{}};
}

}  // namespace syzygy
