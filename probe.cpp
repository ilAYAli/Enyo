#include "probe.hpp"
#if ENYO_USE_SYZYGY
#include "3rdparty/Fathom/src/tbprobe.h"
#endif

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
#if ENYO_USE_SYZYGY
    // tb_init returns true even when no tables were found; the real signal
    // is TB_LARGEST (max piece count of any loaded table).
    initialized = tb_init(tb_path.c_str()) && TB_LARGEST > 0;
    return initialized;
#else
    (void)tb_path;
    initialized = false;
    return false;
#endif
}

unsigned largest()
{
#if ENYO_USE_SYZYGY
    return TB_LARGEST;
#else
    return 0;
#endif
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
#if !ENYO_USE_SYZYGY
    (void)board;
    return Status::Error;
#else
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
#endif
}

std::pair<int, Move> DTZ_probe(Board & board, Status & status)
{
    status = Status::Error;
#if !ENYO_USE_SYZYGY
    (void)board;
    return {0, Move{}};
#else
    if (!initialized)
        return {0, Move{}};

    auto pos = board2pos(board);
    pos.rule50 = static_cast<uint8_t>(board.gamestate.half_moves);

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

    if (TBresult == TB_RESULT_FAILED
     || TBresult == TB_RESULT_CHECKMATE
     || TBresult == TB_RESULT_STALEMATE)
        return {0, Move{}};

    const int wdl = TB_GET_WDL(TBresult);
    int score = 0;
    switch (wdl) {
        case TB_WIN:          score = Value::tb_win_in_max_ply;  status = Status::Win;  break;
        case TB_LOSS:         score = Value::tb_loss_in_max_ply; status = Status::Loss; break;
        case TB_CURSED_WIN:
        case TB_DRAW:
        case TB_BLESSED_LOSS: score = 0;                         status = Status::Draw; break;
        default:              return {0, Move{}};
    }

    const int promo = TB_GET_PROMOTES(TBresult);
    const PieceType promoTranslation[] = {no_piece_type, queen, rook, bishop, knight};

    // Fathom hands back A1=0 squares; compare against legal moves in Enyo's
    // native H1=0 layout by translating the legal move into A1=0.
    const auto fathom_from = unsigned(TB_GET_FROM(TBresult));
    const auto fathom_to   = unsigned(TB_GET_TO(TBresult));

    Movelist legalmoves;
    if (board.side == white)
        legalmoves = generate_legal_moves<white>(board);
    else
        legalmoves = generate_legal_moves<black>(board);

    for (auto move : legalmoves) {
        if (sqconv(move.src_sq()) != fathom_from) continue;
        if (sqconv(move.dst_sq()) != fathom_to)   continue;
        if (promoTranslation[promo] == no_piece_type) {
            if (move.flags() != Move::Flags::promote)
                return {score, move};
        } else if (promo < 5) {
            if (move.flags() == Move::Flags::promote
             && move.promo_piece() == promoTranslation[promo])
                return {score, move};
        }
    }
    return {0, Move{}};
#endif
}

}  // namespace syzygy
