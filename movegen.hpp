#pragma once
#include <cstdint>
#include "board.hpp"
#include "fen.hpp"
#include "pgn.hpp"
#include "eventlog.hpp"

#include "uci.hpp"
#include "magic/magic.hpp"
#if 1
#include "precalc/pawn_attacks.hpp"
#include "precalc/knight_attacks.hpp"
#include "precalc/bishop_attacks.hpp"
#include "precalc/rook_attacks.hpp"
#include "precalc/king_attacks.hpp"
#endif

#include "config.hpp"
#include "types.hpp"
#include "movelist.hpp"
#include "util.hpp"
#include "fmt/core.h"

using namespace eventlog;
using namespace enyo;

namespace {

template <Color Us>
static constexpr bitboard_t knight_moves(Board const & b, square_t square) {
    auto attacks = knight_attack_table[square] & ~b.color_bb[Us];
    return attacks;
}

template <Color Us>
static constexpr bitboard_t rook_moves(Board const & b, square_t square) {
    bitboard_t const p1 = b.color_bb[Us];
    bitboard_t const p2 = b.color_bb[~Us];
    const auto attacks = get_rook_attacks(square, p1|p2) & ~p1;
    return attacks;
}

template <Color Us>
static constexpr bitboard_t bishop_moves(Board const & b, square_t square) {
    bitboard_t const p1 = b.color_bb[Us];
    bitboard_t const p2 = b.color_bb[~Us];
    const auto attacks = get_bishop_attacks(square, p1|p2) & ~p1;
    return attacks;
}

template <Color Us>
static constexpr bitboard_t queen_moves(Board const & b, square_t square) {
    bitboard_t const p1 = b.color_bb[Us];
    bitboard_t const p2 = b.color_bb[~Us];
    const auto attacks =
        (get_bishop_attacks(square, p1|p2)
       | get_rook_attacks(square, p1|p2)) & ~p1;
    return attacks;
}

template <Color Us, bool CheckCastle>
static constexpr bitboard_t king_moves(Board const & b, square_t square) {
    bitboard_t valid_moves = king_attack_table[square] & ~b.color_bb[Us];

    constexpr square_t king_home = Us == white ? e1 : e8;
    if (square != king_home)
        return valid_moves;

    // TODO: consider less checks as it will be validated in legal_moves
    if constexpr (CheckCastle) {
        constexpr bitboard_t kingside_busy_mask  = 0x0600000000000006;
        constexpr bitboard_t queenside_busy_mask = 0x7000000000000070;
        const auto all_pieces = b.color_bb[white] | b.color_bb[black];
        if constexpr (Us == white) {
            if (b.gamestate.can_castle(CastlingRights::white_oo))
                if (!(all_pieces & (kingside_busy_mask & rank_1)))
                    if (b.pt_bb[white][rook] & (1ULL << h1))
                        valid_moves |= (1ULL << (square - 2));

            if (b.gamestate.can_castle(CastlingRights::white_ooo))
                if (!(all_pieces & (queenside_busy_mask & rank_1)))
                    if (b.pt_bb[white][rook] & (1ULL << a1))
                        valid_moves |= (1ULL << (square + 2));
        } else {
            if (b.gamestate.can_castle(CastlingRights::black_oo))
                if (!(all_pieces & (kingside_busy_mask & rank_8)))
                    if (b.pt_bb[black][rook] & (1ULL << h8))
                        valid_moves |= (1ULL << (square - 2));

            if (b.gamestate.can_castle(CastlingRights::black_ooo))
                if (!(all_pieces & (queenside_busy_mask & rank_8)))
                    if (b.pt_bb[black][rook] & (1ULL << a8))
                        valid_moves |= (1ULL << (square + 2));
        }
    }
    return valid_moves;
}

template <Color Us, bool UpdateZobrist, bool UpdateNNUE>
static inline constexpr void apply_promotion(Board & b, enyo::Move move, NNUE::Net * nnue)
{
    if constexpr (false)
        log(Log::info, "[+] apply promotion: {}\n", move);

    constexpr auto Them = ~Us;
    const auto src_bit = move.get_src();
    const auto dst_piece = move.get_dst_piece();
    const auto dst_bit = move.get_dst();

    const square_t w_ksq = lsb(b.pt_bb[white][king]);
    const square_t b_ksq = lsb(b.pt_bb[black][king]);
    clr_piece<Us, UpdateZobrist, UpdateNNUE>(b, pawn, src_bit, nnue, w_ksq, b_ksq);
    if (dst_piece != no_piece_type)
        clr_piece<Them, UpdateZobrist, UpdateNNUE>(b, dst_piece, dst_bit, nnue, w_ksq, b_ksq);
    set_piece<Us, UpdateZobrist, UpdateNNUE>(b, move.get_promo_piece(), dst_bit, nnue, w_ksq, b_ksq);
}


template <Color Us, bool UpdateZobrist, bool UpdateNNUE>
static inline constexpr void revert_promotion(Board & b, Undo const& undo, NNUE::Net * nnue)
{
    if constexpr(false)
        log(Log::info, "[-] revert promotion: {}\n", undo.move);

    constexpr auto Them = ~Us;
    const auto src_bit = undo.move.get_src();
    const auto dst_piece = undo.move.get_dst_piece();
    const auto dst_bit = undo.move.get_dst();

    const square_t w_ksq = lsb(b.pt_bb[white][king]);
    const square_t b_ksq = lsb(b.pt_bb[black][king]);
    clr_piece<Us, UpdateZobrist, UpdateNNUE>(b, undo.move.get_promo_piece(), dst_bit, nnue, w_ksq, b_ksq);
    if (dst_piece != no_piece_type)
        set_piece<Them, UpdateZobrist, UpdateNNUE>(b, dst_piece, dst_bit, nnue, w_ksq, b_ksq);
    set_piece<Us, UpdateZobrist, UpdateNNUE>(b, pawn, src_bit, nnue, w_ksq, b_ksq);
}

template <Color Us, bool UpdateZobrist, bool UpdateNNUE>
static constexpr bool apply_enpassant(Board & b, enyo::Move move, unsigned enpassant_square, NNUE::Net * nnue)
{
    constexpr auto Them = ~Us;
    const auto src = move.get_src();
    const auto dst = move.get_dst();

    square_t const target = static_cast<square_t>(enpassant_square + (Us == black ? 8 : -8U));
    if constexpr (false)
        fmt::print("[+] apply enpassant: {}{}, target: {}\n", sq2str(src), sq2str(dst), sq2str(target));

    const square_t w_ksq = lsb(b.pt_bb[white][king]);
    const square_t b_ksq = lsb(b.pt_bb[black][king]);
    clr_piece<Us, UpdateZobrist, UpdateNNUE>(b, pawn, src, nnue, w_ksq, b_ksq);
    clr_piece<Them, UpdateZobrist, UpdateNNUE>(b, pawn, target, nnue, w_ksq, b_ksq);
    set_piece<Us, UpdateZobrist, UpdateNNUE>(b, pawn, dst, nnue, w_ksq, b_ksq);

    b.hash ^= b.zbrs.enpassant_[enpassant_square % 8];
    if constexpr (zobrist::debug)
        fmt::print("<{}> zobrist: apply_enpassant hash: {:016X}\n",
            __func__, b.hash);

    return true;
}

template <Color Us, bool UpdateZobrist, bool UpdateNNUE>
static constexpr bool revert_enpassant(Board & b, Undo const & undo, NNUE::Net * nnue)
{
    constexpr auto Them = ~Us;
    const auto move = undo.move;
    const auto src = move.get_src();
    const auto dst = move.get_dst();

    square_t const target = static_cast<square_t>(undo.gamestate.enpassant_square + (Us == black ? 8 : -8U));
    if constexpr (false)
        fmt::print("[-] revert enpassant: {}, target: {}\n",
            move, sq2str(target));

    const square_t w_ksq = lsb(b.pt_bb[white][king]);
    const square_t b_ksq = lsb(b.pt_bb[black][king]);
    clr_piece<Us, UpdateZobrist, UpdateNNUE>(b, pawn, dst, nnue, w_ksq, b_ksq);
    set_piece<Them, UpdateZobrist, UpdateNNUE>(b, pawn, target, nnue, w_ksq, b_ksq);
    set_piece<Us, UpdateZobrist, UpdateNNUE>(b, pawn, src, nnue, w_ksq, b_ksq);

    if constexpr (zobrist::debug)
        fmt::print("<{}> zobrist: apply_enpassant for square: {}\n",
        __func__, undo.gamestate.enpassant_square);
    b.hash ^= b.zbrs.enpassant_[undo.gamestate.enpassant_square % 8];

    return true;
}

// hotspot: is square attacked by Them
template<Color Us>
static inline constexpr bool is_square_attacked(const Board & b, square_t sq)
{
    constexpr Color Them = ~Us;
    if (b.pt_bb[Them][pawn] & pawn_attack_table[Them][sq])
        return true;
    if (b.pt_bb[Them][knight] & knight_attack_table[sq])
        return true;
    if (b.pt_bb[Them][king] & king_attack_table[sq])
        return true;
    if (bishop_moves<Us>(b, sq) & (b.pt_bb[Them][bishop] | b.pt_bb[Them][queen]))
        return true;
    if (rook_moves<Us>(b, sq) & (b.pt_bb[Them][rook] | b.pt_bb[Them][queen]))
        return true;
    return false;
}

enum class CastleSide { Kingside, Queenside };

template <Color Us, CastleSide Side, bool UpdateZobrist, bool UpdateNNUE>
inline void revert_castle(Board & b, NNUE::Net * nnue)
{
    const square_t w_ksq = lsb(b.pt_bb[white][king]);
    const square_t b_ksq = lsb(b.pt_bb[black][king]);
    if constexpr (Us == white) {
        // ----[ white kingside ]-----------------------------------------------
        if constexpr (Side == CastleSide::Kingside) {
            b.gamestate.set_castle(CastlingRights::white_oo, true);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[1];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }

            b.gamestate.set_castle(CastlingRights::white_ooo, true);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[0];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }

            clr_piece<white, UpdateZobrist, UpdateNNUE>(b, rook, f1, nnue, w_ksq, b_ksq);
            clr_piece<white, UpdateZobrist, UpdateNNUE>(b, king, g1, nnue, w_ksq, b_ksq);
            set_piece<white, UpdateZobrist, UpdateNNUE>(b, rook, h1, nnue, w_ksq, b_ksq);
            set_piece<white, UpdateZobrist, UpdateNNUE>(b, king, e1, nnue, w_ksq, b_ksq);
        } else { //----[ white queenside ]--------------------------------------
            b.gamestate.set_castle(CastlingRights::white_ooo, true);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[0];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }

            b.gamestate.set_castle(CastlingRights::white_oo, true);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[1];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }

            clr_piece<white, UpdateZobrist, UpdateNNUE>(b, rook, d1, nnue, w_ksq, b_ksq);
            clr_piece<white, UpdateZobrist, UpdateNNUE>(b, king, c1, nnue, w_ksq, b_ksq);
            set_piece<white, UpdateZobrist, UpdateNNUE>(b, rook, a1, nnue, w_ksq, b_ksq);
            set_piece<white, UpdateZobrist, UpdateNNUE>(b, king, e1, nnue, w_ksq, b_ksq);
        }
    } else {
        //----[ black kingside ]------------------------------------------------
        if constexpr (Side == CastleSide::Kingside) {
            b.gamestate.set_castle(CastlingRights::black_oo, true);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[3];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }

            if constexpr (UpdateZobrist) {
                b.gamestate.set_castle(CastlingRights::black_ooo, true);
                b.hash ^= b.zbrs.castling_[2];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }

            clr_piece<black, UpdateZobrist, UpdateNNUE>(b, rook, f8, nnue, w_ksq, b_ksq);
            clr_piece<black, UpdateZobrist, UpdateNNUE>(b, king, g8, nnue, w_ksq, b_ksq);
            set_piece<black, UpdateZobrist, UpdateNNUE>(b, rook, h8, nnue, w_ksq, b_ksq);
            set_piece<black, UpdateZobrist, UpdateNNUE>(b, king, e8, nnue, w_ksq, b_ksq);
        } else { //----[ black queenside ]--------------------------------------
            b.gamestate.set_castle(CastlingRights::black_oo, true);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[3];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }

            b.gamestate.set_castle(CastlingRights::black_ooo, true);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[2];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }

            clr_piece<black, UpdateZobrist, UpdateNNUE>(b, rook, d8, nnue, w_ksq, b_ksq);
            clr_piece<black, UpdateZobrist, UpdateNNUE>(b, king, c8, nnue, w_ksq, b_ksq);
            set_piece<black, UpdateZobrist, UpdateNNUE>(b, rook, a8, nnue, w_ksq, b_ksq);
            set_piece<black, UpdateZobrist, UpdateNNUE>(b, king, e8, nnue, w_ksq, b_ksq);
        }
    }
}


template <Color Us, CastleSide Side, bool UpdateZobrist, bool UpdateNNUE>
inline void apply_castle(Board & b, NNUE::Net * nnue)
{
    const square_t w_ksq = lsb(b.pt_bb[white][king]);
    const square_t b_ksq = lsb(b.pt_bb[black][king]);
    if constexpr (Us == white) {
       if constexpr (Side == CastleSide::Kingside) { // white, kingside:
            clr_piece<white, UpdateZobrist, UpdateNNUE>(b, king, e1, nnue, w_ksq, b_ksq);
            clr_piece<white, UpdateZobrist, UpdateNNUE>(b, rook, h1, nnue, w_ksq, b_ksq);
            set_piece<white, UpdateZobrist, UpdateNNUE>(b, king, g1, nnue, w_ksq, b_ksq);
            set_piece<white, UpdateZobrist, UpdateNNUE>(b, rook, f1, nnue, w_ksq, b_ksq);
            b.gamestate.set_castle(CastlingRights::white_oo, false);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[1];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }
            b.gamestate.set_castle(CastlingRights::white_ooo, true);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[0];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }
        } else { // white, queenside:
            clr_piece<white, UpdateZobrist, UpdateNNUE>(b, king, e1, nnue, w_ksq, b_ksq);
            clr_piece<white, UpdateZobrist, UpdateNNUE>(b, rook, a1, nnue, w_ksq, b_ksq);
            set_piece<white, UpdateZobrist, UpdateNNUE>(b, king, c1, nnue, w_ksq, b_ksq);
            set_piece<white, UpdateZobrist, UpdateNNUE>(b, rook, d1, nnue, w_ksq, b_ksq);

            b.gamestate.set_castle(CastlingRights::white_ooo, false);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[0];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }

            b.gamestate.set_castle(CastlingRights::white_oo, false);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[1];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }
        }
    } else {
       if constexpr (Side == CastleSide::Kingside) { // black, kingside:
            clr_piece<black, UpdateZobrist, UpdateNNUE>(b, king, e8, nnue, w_ksq, b_ksq);
            clr_piece<black, UpdateZobrist, UpdateNNUE>(b, rook, h8, nnue, w_ksq, b_ksq);
            set_piece<black, UpdateZobrist, UpdateNNUE>(b, king, g8, nnue, w_ksq, b_ksq);
            set_piece<black, UpdateZobrist, UpdateNNUE>(b, rook, f8, nnue, w_ksq, b_ksq);

            b.gamestate.set_castle(CastlingRights::black_oo, false);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[3];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }

            b.gamestate.set_castle(CastlingRights::black_ooo, false);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[2];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }
        } else { // black, queenside:
            clr_piece<black, UpdateZobrist, UpdateNNUE>(b, king, e8, nnue, w_ksq, b_ksq);
            clr_piece<black, UpdateZobrist, UpdateNNUE>(b, rook, a8, nnue, w_ksq, b_ksq);
            set_piece<black, UpdateZobrist, UpdateNNUE>(b, king, c8, nnue, w_ksq, b_ksq);
            set_piece<black, UpdateZobrist, UpdateNNUE>(b, rook, d8, nnue, w_ksq, b_ksq);

            b.gamestate.set_castle(CastlingRights::black_oo, false);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[2];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }

            b.gamestate.set_castle(CastlingRights::black_ooo, false);
            if constexpr (UpdateZobrist) {
                b.hash ^= b.zbrs.castling_[3];
                if constexpr (zobrist::debug)
                    fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                        __func__, b.hash, b.fen());
            }
        }
    }

    if constexpr (UpdateNNUE) {
        assert(nnue && "apply_castle: nnue is null");
        nnue->refresh(b);
    }
}

template <Color Us>
static inline void init_pinner_blocker_bb(Board & b)
{
    //return;
    bool constexpr debug = false;
    constexpr auto Them = ~Us;

    square_t const ksq = lsb(b.pt_bb[Us][king]);

    b.blockers_bb[Us] = 0;
    b.pinners_bb[Them] = 0;

    bitboard_t orth_attackers = b.pt_bb[Them][queen] | b.pt_bb[Them][rook];
    bitboard_t diag_attackers = b.pt_bb[Them][queen] | b.pt_bb[Them][bishop];
    bitboard_t snipers = (get_rook_attacks(ksq, 0) & orth_attackers)
                       | (get_bishop_attacks(ksq, 0) & diag_attackers);
    if (!snipers) {
        if constexpr (debug)
            fmt::print("<{}> {} king at {} not sniped, fen: {}\n",
                __func__, Us, sq2str(ksq), b.fen());
        return;
    }

    bitboard_t pieces = b.color_bb[Us] | b.color_bb[~Us];
    bitboard_t occupancy = (pieces ^ snipers);

    while (snipers) {
        square_t sniper_sq = pop_lsb(snipers);
        if constexpr (debug)
            fmt::print("sniper: {}\n", sq2str(sniper_sq));
        bitboard_t bb = b.between_bb[sniper_sq][ksq] & occupancy;
        //fmt::print("bb:{}\n", visualize(bb));
        bool const more_than_one = bb & (bb - 1);
        // if constexpr (debug) fmt::print("more_than_one:{}\n", more_than_one);
        if (bb && !more_than_one) {
            // only add blocker if ours:
            b.blockers_bb[Us] |= bb & b.color_bb[Us];
            // add pinner:
            if (bb & b.color_bb[Us])
                b.pinners_bb[Them] |= 1ULL << sniper_sq;
            if constexpr (debug) {
                fmt::print("<{}> {} pinners_bb:{}\n", __func__, Them, visualize(b.pinners_bb[Them]));
                fmt::print("<{}> {} blockers_bb:{}\n", __func__, Us, visualize(b.blockers_bb[Us]));
            }
        }
    }
}

template <Color c>
inline void init_attacks_bb(Board & b)
{
    std::memset(b.attacks_bb[c], 0, sizeof(b.attacks_bb[c]));

    bitboard_t const pieces = (b.color_bb[~c] | b.color_bb[c]) & ~b.pt_bb[~c][king];
    //bitboard_t const pieces = b.color_bb[~c] | b.color_bb[c];
    b.all_attacks_bb[c] = 0;

    bool constexpr debug = false;
    if constexpr (debug)
        fmt::print("<{}> update {} attacks, fen: {}\n", __func__, c, b.fen());
    { // pawns:
        bitboard_t pawn_attacks = 0;
        auto opponent_pawns = b.pt_bb[c][pawn];
        while (opponent_pawns) {
            const auto sq = pop_lsb(opponent_pawns);
            pawn_attacks |= pawn_attack_table[~c][sq];
        }
        if constexpr (debug)
            fmt::print("pawn_attacks:{}\n", visualize(pawn_attacks));
        b.all_attacks_bb[c] |= pawn_attacks;
    }

    { // knights:
        bitboard_t knight_attackers = 0;
        auto opponent_knights = b.pt_bb[c][knight];
        while (opponent_knights) {
            const auto sq = pop_lsb(opponent_knights);
            knight_attackers |= knight_attack_table[sq];
        }
        if constexpr (debug)
            fmt::print("knight_attackers:{}\n", visualize(knight_attackers));
        b.all_attacks_bb[c] |= knight_attackers;
    }

    { // bishops/queens:
        bitboard_t diag_attackers = b.pt_bb[c][queen] | b.pt_bb[c][bishop];
        bitboard_t diag_attacks = 0;
        while (diag_attackers) {
            auto sq = pop_lsb(diag_attackers);
            diag_attacks |= get_bishop_attacks(sq, pieces);

        }
        if constexpr (debug)
            fmt::print("diag_attacks:{}\n", visualize(diag_attacks));
        b.all_attacks_bb[c] |= diag_attacks;
    }

    { // rooks/queens:
        bitboard_t orth_attackers = b.pt_bb[c][queen] | b.pt_bb[c][rook];
        bitboard_t orth_attacks = 0;
        while (orth_attackers) {
            auto sq = pop_lsb(orth_attackers);
            orth_attacks |= get_rook_attacks(sq, pieces);
        }
        if constexpr (debug)
            fmt::print("orth_attacks: {}\n", visualize(orth_attacks));
        b.all_attacks_bb[c] |= orth_attacks;
    }


    { // king:
        bitboard_t king_attacks = king_attack_table[lsb(b.pt_bb[c][king])];
        b.all_attacks_bb[c] |= king_attacks;
        if constexpr (debug)
            fmt::print("king_attacks: {}\n", visualize(king_attacks));
    }
}

// checkers to "our" king
template <Color Us>
static inline void init_checkers_bb(Board & b) {
    constexpr auto Them = ~Us;
    b.checkers_bb[Them] = 0;

    bool constexpr debug = false;
    if constexpr (debug)
        fmt::print("<{}> update {} checkers, fen: {}\n", __func__, Them, b.fen());

    //fmt::print("<{}> {} blockers_bb:{}\n", __func__, Us, visualize(b.blockers_bb[Us]));
    //fmt::print("<{}> {} pinners_bb:{}\n", __func__, Us, visualize(b.pinners_bb[Them]));

    square_t const ksq = lsb(b.pt_bb[Us][king]);
    bitboard_t const pieces = b.color_bb[Us] | b.color_bb[Them];
    for (square_t sq = 0; sq < square_nb; ++sq) {
        if (b.attacks_bb[Them][sq] & (1ULL << ksq)) {
            bitboard_t bb = b.between_bb[sq][ksq];
            if constexpr (debug)
                fmt::print("attacks from {} :{}\n", sq2str(sq), visualize(b.attacks_bb[Them][sq]));
            //if constexpr (debug) fmt::print("bb:{}\n", visualize(bb));
            if (bb & pieces) {
                if constexpr (debug)
                    fmt::print("piece at {} is blocking sniper at {}\n", sq2str(lsb(pieces & bb)), sq2str(sq));
                if constexpr (debug)
                    fmt::print("blockers_bb:{}\n", visualize(b.blockers_bb[Us]));
                continue;
            }
            if constexpr (debug)
                fmt::print("adding checker: {}, ksq: {}, fen: {}\n", sq2str(sq), sq2str(ksq), b.fen());
            b.checkers_bb[Them] |= (1ULL << sq);
        }
    }
    if constexpr (debug)
        fmt::print("checkers:{}\n", visualize(b.checkers_bb[Them]));
}


template <Color Us, bool UpdateZobrist, bool UpdateNNUE = false>
inline bool apply_move_generic(Board & b, Move mv, NNUE::Net * nnue)
{
    const auto src = mv.get_src();
    const auto dst = mv.get_dst();
    const auto src_piece = mv.get_src_piece();
    const auto dst_piece = mv.get_dst_piece();
    constexpr auto Them = ~Us;

#if 0
    if (dst_piece == king) {
        fmt::print("error: king can't be captured, move: {}, fen: {}\n", mv, b.fen());
        print_pgn(b);
        fflush(stdout);
        exit(1);
    }
    assert(dst_piece != king && "king can't be captured"); // assert triggers on debug build
#endif

    if constexpr (Us == white) {
        switch (src_piece) {
            case rook: {
                if (src == a1) {
                    if (b.gamestate.can_castle(CastlingRights::white_ooo)) {
                        b.gamestate.set_castle(CastlingRights::white_ooo, false);
                        b.hash ^= b.zbrs.castling_[0];
                        if constexpr (zobrist::debug)
                            fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                                __func__, b.hash, b.fen());
                    }
                } else if (src == h1) {
                    if (b.gamestate.can_castle(CastlingRights::white_oo)) {
                        b.gamestate.set_castle(CastlingRights::white_oo, false);
                        b.hash ^= b.zbrs.castling_[1];
                        if constexpr (zobrist::debug)
                            fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                                __func__, b.hash, b.fen());
                    }
                }
                break;
            }
            case king: {
                if (b.gamestate.can_castle(CastlingRights::white_ooo)) {
                    b.gamestate.set_castle(CastlingRights::white_ooo, false);
                    b.hash ^= b.zbrs.castling_[0];
                    if constexpr (zobrist::debug)
                        fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                            __func__, b.hash, b.fen());
                }
                if (b.gamestate.can_castle(CastlingRights::white_oo)) {
                    b.gamestate.set_castle(CastlingRights::white_oo, false);
                    b.hash ^= b.zbrs.castling_[1];
                    if constexpr (zobrist::debug)
                        fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                            __func__, b.hash, b.fen());
                }
                break;
            }
            case pawn: {
                const auto enpassant_attack_mask = (1ULL << (dst +1)) | (1ULL << (dst -1));
                if ((frtab[src][1] == rank_2) && (frtab[dst][1] == rank_4)) {
                    if (b.pt_bb[Them][pawn] & enpassant_attack_mask) {
                        b.gamestate.enpassant_square = dst - 8;
                    }
                }
                break;
            }
            default: break;
        }
    } else { // black:
        switch (src_piece) {
            case rook: {
                if (src == a8) {
                    if (b.gamestate.can_castle(CastlingRights::black_ooo)) {
                        b.gamestate.set_castle(CastlingRights::black_ooo, false);
                        b.hash ^= b.zbrs.castling_[2];
                        if constexpr (zobrist::debug)
                            fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                                __func__, b.hash, b.fen());
                    }
                } else if (src == h8) {
                    if (b.gamestate.can_castle(CastlingRights::black_oo)) {
                        b.gamestate.set_castle(CastlingRights::black_oo, false);
                        b.hash ^= b.zbrs.castling_[3];
                        if constexpr (zobrist::debug)
                            fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                                __func__, b.hash, b.fen());
                    }
                }
                break;
            }
            case king: {
                if (b.gamestate.can_castle(CastlingRights::black_ooo)) {
                    b.gamestate.set_castle(CastlingRights::black_ooo, false);
                    b.hash ^= b.zbrs.castling_[2]; // Queenside black
                    if constexpr (zobrist::debug)
                        fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                            __func__, b.hash, b.fen());
                }
                if (b.gamestate.can_castle(CastlingRights::black_oo)) {
                    b.gamestate.set_castle(CastlingRights::black_oo, false);
                    b.hash ^= b.zbrs.castling_[3]; // Kingside black
                    if constexpr (zobrist::debug)
                        fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                            __func__, b.hash, b.fen());
                }
                break;
            }
            case pawn: {
                const auto enpassant_attack_mask = (1ULL << (dst +1)) | (1ULL << (dst -1));
                if ((frtab[src][1] == rank_7) && (frtab[dst][1] == rank_5)) {
                    if (b.pt_bb[Them][pawn] & enpassant_attack_mask) {
                        b.gamestate.enpassant_square = dst + 8;
                    }
                }
                break;
            }
            default: break;
        }

    } // black

    const square_t w_ksq = lsb(b.pt_bb[white][king]);
    const square_t b_ksq = lsb(b.pt_bb[black][king]);
    clr_piece<Us, UpdateZobrist, UpdateNNUE>(b, src_piece, src, nnue, w_ksq, b_ksq);
    if (dst_piece != no_piece_type)
        clr_piece<Them, UpdateZobrist, UpdateNNUE>(b, dst_piece, dst, nnue, w_ksq, b_ksq);
    set_piece<Us, UpdateZobrist, UpdateNNUE>(b, src_piece, dst, nnue, w_ksq, b_ksq);

    if constexpr (UpdateNNUE) {
        assert(nnue && "apply_move_generic: nnue is null");
        if (true) {
        //if (src_piece == king
        //    && (NNUE::KING_BUCKET[src ^ (b.side == white ? 0 : 56)]
        //     != NNUE::KING_BUCKET[dst ^ (b.side == white ? 0 : 56)]
        //     || square_file(src) + square_file(dst) == 7)) {
             nnue->refresh(b);
        } else {
            nnue->updateAccumulator(
                src_piece, // PWA: presumably
                Us,
                src,
                dst,
                lsb(b.pt_bb[white][king]),
                lsb(b.pt_bb[black][king])
            );

        }
    }


    return true;
}


template<Color Us>
inline void restore_castling_rights(Board & b, Undo const & undo)
{
    if constexpr (Us == white) {
        if (b.gamestate.can_castle(CastlingRights::white_oo) != undo.gamestate.can_castle(CastlingRights::white_oo)) {
            b.gamestate.set_castle(CastlingRights::white_oo, undo.gamestate.can_castle(CastlingRights::white_oo));
            b.hash ^= b.zbrs.castling_[1];
            if constexpr (zobrist::debug)
                fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                    __func__, b.hash, b.fen());
        }
        if (b.gamestate.can_castle(CastlingRights::white_ooo) != undo.gamestate.can_castle(CastlingRights::white_ooo)) {
            b.gamestate.set_castle(CastlingRights::white_ooo, undo.gamestate.can_castle(CastlingRights::white_ooo));
            b.hash ^= b.zbrs.castling_[0];
            if constexpr (zobrist::debug)
                fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                    __func__, b.hash, b.fen());
        }
    } else {
        if (b.gamestate.can_castle(CastlingRights::black_oo) != undo.gamestate.can_castle(CastlingRights::black_oo)) {
            b.gamestate.set_castle(CastlingRights::black_oo, undo.gamestate.can_castle(CastlingRights::black_oo));
            b.hash ^= b.zbrs.castling_[3];
            if constexpr (zobrist::debug)
                fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                    __func__, b.hash, b.fen());
        }
        if (b.gamestate.can_castle(CastlingRights::black_ooo) != undo.gamestate.can_castle(CastlingRights::black_ooo)) {
            b.gamestate.set_castle(CastlingRights::black_ooo, undo.gamestate.can_castle(CastlingRights::black_ooo));
            b.hash ^= b.zbrs.castling_[2];
            if constexpr (zobrist::debug)
                fmt::print("<{}> zobrist: hash: {:016X}, fen: {}\n",
                    __func__, b.hash, b.fen());
        }
    }
}

template<Color Us, bool UpdateZobrist, bool UpdateNNUE>
inline void revert_move_generic(Board & b, Undo const & undo, NNUE::Net * nnue) {
    const auto src_piece = undo.move.get_src_piece();
    const auto dst_piece = undo.move.get_dst_piece();
    const auto src = undo.move.get_src();
    const auto dst = undo.move.get_dst();

    const square_t w_ksq = lsb(b.pt_bb[white][king]);
    const square_t b_ksq = lsb(b.pt_bb[black][king]);
    clr_piece<Us, UpdateZobrist, UpdateNNUE>(b, src_piece, dst, nnue, w_ksq, b_ksq);
    if (dst_piece != no_piece_type)
        set_piece<~Us, UpdateZobrist, UpdateNNUE>(b, dst_piece, dst, nnue, w_ksq, b_ksq);
    set_piece<Us, UpdateZobrist, UpdateNNUE>(b, src_piece, src, nnue, w_ksq, b_ksq);

    if (src_piece == king || src_piece == rook) {
        restore_castling_rights<Us>(b, undo);
    }
}

} // anonymous namespace

template<Color Us>
inline static bool is_check(const Board & b) {
    const bitboard_t km = b.pt_bb[Us][king];
    if constexpr (Constexpr::check_ksq) {
        if (!km) BREAKPOINT(fmt::format("missing {} king", Us));
    }
    const square_t ksq = lsb(km);
    return is_square_attacked<Us>(b, ksq);
}

template <Color Us, bool UpdateZobrist = true, bool UpdateNNUE = false>
inline void apply_move(Board & b, Move move, [[maybe_unused]] NNUE::Net * nnue = nullptr)
{
    if constexpr (Constexpr::debug_move)
        fmt::print("[{}] {}<{}>: {} fen: {} side: {}, hash: {:016X}\n",
            b.histply, __func__, Us, move, b.fen(), b.side, b.hash);

    if constexpr (UpdateNNUE) {
        assert(nnue && "apply_move: nnue is null");
        nnue->push();
    }

    const auto src_piece = move.get_src_piece();
    const auto dst_piece = move.get_dst_piece();
    assert((src_piece != PieceType::no_piece_type) && "error, source position not specified");

    auto & undo = b.history[b.histply++] = Undo{
        .move = move,
        .gamestate = b.gamestate,
        .half_moves = b.half_moves
    };
    b.gamestate.enpassant_square = 0;
    b.side = Us;

    switch (move.get_flags()) {
        case Move::Flags::Castle:
            if (move.get_dst() < move.get_src())
                apply_castle<Us, CastleSide::Kingside, UpdateZobrist, UpdateNNUE>(b, nnue);
            else
                apply_castle<Us, CastleSide::Queenside, UpdateZobrist, UpdateNNUE>(b, nnue);
            b.half_moves = 0;
            break;
        case Move::Flags::Promote:
            apply_promotion<Us, UpdateZobrist, UpdateNNUE>(b, move, nnue);
            b.half_moves = 0;
            break;
        case Move::Flags::Enpassant:
            apply_enpassant<Us, UpdateZobrist, UpdateNNUE>(b, move, undo.gamestate.enpassant_square, nnue);
            b.half_moves = 0;
            break;
        default:
            apply_move_generic<Us, UpdateZobrist, UpdateNNUE>(b, move, nnue);
            if (dst_piece != no_piece_type) {
                b.half_moves = 0;
            } else {
                if (src_piece == pawn)
                    b.half_moves = 0;
                else
                    b.half_moves++;
            }
    }

    b.hash ^= b.zbrs.side_;
    undo.hash = b.hash;

    if constexpr (zobrist::debug)
        fmt::print("<{}> zobrist: {} move, hash: {:016X}, fen: {}\n",
            __func__, Us, b.hash, b.fen());

    b.side = ~Us;
}

template <Color Us, bool UpdateZobrist = true, bool UpdateNNUE = false>
inline void revert_move(Board & b, [[maybe_unused]] NNUE::Net * nnue = nullptr)
{
    b.side = ~Us;
    auto & undo = b.history[--b.histply];

    if constexpr (UpdateNNUE) {
        assert(nnue && "revert_move: nnue is null");
        nnue->pop();
    }

    if constexpr (Constexpr::constexpr_assert) {
        if (b.histply < 0 || undo.move.get_src_piece() == no_piece_type) {
            if (b.histply < 0) {
                fmt::print("[ERROR] {}<{}> can't undo due to negative ply: {}, fen: {}\n",
                __func__, Us, b.histply, b.fen());
                print_pgn(b);
                BREAKPOINT("nagative ply");
            } else {
                if (!undo.move) {
                    fmt::print("[ERROR] {}<Us> undo[{}] == nullmove, fen: {}\n",
                    __func__, b.histply, b.fen());
                    print_pgn(b);
                } else {
                    fmt::print("[ERROR] {}<Us> source piece not specified: undo[{}] move: {}, fen: {}\n",
                    __func__, b.histply, undo.move, b.fen());
                    print_pgn(b);
                }
                BREAKPOINT("no source piece");
            }
        }
    }

    if constexpr (Constexpr::debug_move) {
        fmt::print("[{}] {}<{}>: {} fen: {} side: {}, hash: {:016X}\n",
            b.histply, __func__, Us, undo.move, b.fen(), b.side, b.hash);
    }

    b.hash ^= b.zbrs.side_;
    if constexpr (zobrist::debug)
        fmt::print("<{}> zobrist: {} move, hash: {:016X}\n",
            __func__, Us, b.hash);

    switch (undo.move.get_flags()) {
        case Move::Flags::Castle:
            if (undo.move.get_dst() < undo.move.get_src())
                revert_castle<Us, CastleSide::Kingside, UpdateZobrist, UpdateNNUE>(b, nnue);
            else
                revert_castle<Us, CastleSide::Queenside, UpdateZobrist, UpdateNNUE>(b, nnue);
            break;
        case Move::Flags::Promote:
            revert_promotion<Us, UpdateZobrist, UpdateNNUE>(b, undo, nnue);
            break;
        case Move::Flags::Enpassant:
            revert_enpassant<Us, UpdateZobrist, UpdateNNUE>(b, undo, nnue);
            break;
        default:
            revert_move_generic<Us, UpdateZobrist, UpdateNNUE>(b, undo, nnue);
    }

    // only for move (not castling, promotion, ...)
    b.gamestate = undo.gamestate; // allow castling, ...
    b.half_moves = undo.half_moves;
    b.side = Us; // WTF!

    //if constexpr (UpdateNNUE) {
    //    b.king_square[0] = lsb(b.pt_bb[black][king]);
    //    b.king_square[1] = lsb(b.pt_bb[white][king]);
    //    b.nnue_sub.clear();
    //    b.nnue_add.clear();
    //}
}

template <Color Us>
inline void apply_null_move(Board &b)
{
    b.history[b.histply++] = Undo {
        .move = Move{},
        .gamestate = b.gamestate,
        .half_moves = b.half_moves
    };

    b.hash ^= b.zbrs.side_;

    if (b.gamestate.enpassant_square) {
        b.hash ^= b.zbrs.enpassant_[b.gamestate.enpassant_square % 8];
        b.gamestate.enpassant_square = 0;
    }

    b.side = ~Us;
}

template <Color Us>
inline void revert_null_move(Board & b)
{
    auto & undo = b.history[--b.histply];

    b.hash ^= b.zbrs.side_;

    if (undo.gamestate.enpassant_square) {
        b.hash ^= b.zbrs.enpassant_[undo.gamestate.enpassant_square % 8];
    }

    b.gamestate = undo.gamestate;
    b.side = Us;
}



//----[ pawn moves ]-----------------------------------------------------------
template <Color Us>
bitboard_t generate_pawn_moves(Board & b, Movelist & moves)
{
    constexpr auto Them = ~Us;
    constexpr auto Up          = (Us == white ? north : south);
    constexpr auto UpRight     = (Us == white ? north_east : south_west);
    constexpr auto UpLeft      = (Us == white ? north_west : south_east);
    constexpr auto DownRight   = (Us == white ? south_east : north_west);
    constexpr auto DownLeft    = (Us == white ? south_west : north_east);
    constexpr auto promo_rank  = (Us == white ? rank_7 : rank_2);
    const bitboard_t back_pawns = pieces<Us, pawn>(b) & ~promo_rank;
    const bitboard_t empty_squares = ~(b.color_bb[Us] | b.color_bb[Them]);
    const bitboard_t enemies = b.color_bb[Them];

    constexpr bool debug_attack = false;
    constexpr bool debug_enpassant = false;
    constexpr bool debug_promo = false;

    bitboard_t quiet_moves = 0;
    {
        constexpr bitboard_t double_step_rank = (Us == white ? rank_3 : rank_6);

        bitboard_t b1 = shift<Up>(back_pawns) & empty_squares;
        bitboard_t b2 = shift<Up>(b1 & double_step_rank) & empty_squares;
        quiet_moves = b1 | b2;

        while (b1) {
            square_t const dst = pop_lsb(b1);
            moves.emplace(Move{static_cast<square_t>(dst - Up), Us, pawn, dst});
        }
        while (b2) {
            square_t const dst = pop_lsb(b2);
            moves.emplace(Move{static_cast<square_t>(dst - Up - Up), Us, pawn, dst});
        }
    }

    bitboard_t regular_attacks = 0;
    { // normal attacks (left):
        bitboard_t b1 = shift<UpLeft>(back_pawns);
        b1 &= enemies;
        if constexpr (debug_attack)
            fmt::print("attack l: {}\n", visualize(b1));
        while (b1) {
            const auto dst = pop_lsb(b1);
            const auto dst_piece = get_piece_type<Them>(b, dst);
            regular_attacks |= 1ULL << dst;
            Move move{static_cast<square_t>(dst - UpLeft), Us, pawn, dst, dst_piece};
            moves.emplace(move);
            if constexpr (debug_attack)
                fmt::print("attack l: {}\n", move);
        }
    }

    { // normal attacks (right):
        bitboard_t b1 = shift<UpRight>(back_pawns);
        b1 &= enemies;
        if constexpr (debug_attack)
            fmt::print("attack r: {}\n", visualize(b1));
        while (b1) {
            const auto dst = pop_lsb(b1);
            const auto dst_piece = get_piece_type<Them>(b, dst);
            regular_attacks |= 1ULL << dst;
            Move move{static_cast<square_t>(dst - UpRight), Us, pawn, dst, dst_piece};
            moves.emplace(move);
            if constexpr (debug_attack)
                fmt::print("attack r: {}\n", move);
        }
    }

    bitboard_t enpassant_attacks = 0;
    { // en passant attack:
        if (b.gamestate.enpassant_square) {
            constexpr bitboard_t ep_attack_rank = (Us == white ? rank_5 : rank_4);
            const bitboard_t ep_mask = 1ULL << b.gamestate.enpassant_square;
            const bitboard_t pawns_on_ep_attack_rank = pieces<Us, pawn>(b) & ep_attack_rank;

            bitboard_t b1 = shift<DownLeft>(ep_mask) & pawns_on_ep_attack_rank;
            while (b1) {
                square_t const src = pop_lsb(b1);
                const auto dst = square_t(src - DownLeft);
                enpassant_attacks |= 1ULL << dst;
                Move move{src, Us, pawn, dst, pawn};
                move.set_flags(static_cast<Move::Flags>(Move::Flags::Enpassant));
                moves.emplace(move);
                if constexpr (debug_enpassant)
                    fmt::print("empass l: {}\n", move);
            }

            bitboard_t b2 = shift<DownRight>(ep_mask) & pawns_on_ep_attack_rank;
            while (b2) {
                square_t const src = pop_lsb(b2);
                const auto dst = square_t(src - DownRight);
                enpassant_attacks |= 1ULL << dst;
                Move move{src, Us, pawn, dst, pawn};
                move.set_flags(static_cast<Move::Flags>(Move::Flags::Enpassant));
                moves.emplace(move);
                if constexpr (debug_enpassant)
                    fmt::print("empass r: {}\n", move);
            }

            if constexpr (debug_enpassant)
                fmt::print("pawn enpassant: {}\n", visualize(enpassant_attacks));
        }
    }

    bitboard_t promotion_moves = 0;
    { // promotion
        const bitboard_t pawns_on_trank_7 = pieces<Us, pawn>(b) & promo_rank;
        for (const auto & promotion_piece : { knight, bishop, rook, queen }) {
            bitboard_t b1 = shift<UpLeft>(pawns_on_trank_7) & enemies;
            bitboard_t b2 = shift<UpRight>(pawns_on_trank_7) & enemies;
            bitboard_t b3 = shift<Up>(pieces<Us, pawn>(b) & promo_rank) & empty_squares;
            promotion_moves |= b1 | b2 | b3;
            while (b1) { // attack promo left:
                const auto dst = pop_lsb(b1);
                const auto dst_piece = get_piece_type<Them>(b, dst);
                Move move{static_cast<square_t>(dst - UpLeft), Us, pawn, dst, dst_piece };
                move.set_flags(static_cast<Move::Flags>(Move::Flags::Promote));
                move.set_promo_piece(promotion_piece);
                moves.emplace(move);
                if constexpr (debug_promo)
                    fmt::print("promo l: {}\n", move);
            }
            while (b2) { // attack promo right:
                const auto dst = pop_lsb(b2);
                const auto dst_piece = get_piece_type<Them>(b, dst);
                Move move{static_cast<square_t>(dst - UpRight), Us, pawn, dst, dst_piece };
                move.set_flags(static_cast<Move::Flags>(Move::Flags::Promote));
                move.set_promo_piece(promotion_piece);
                moves.emplace(move);
                if constexpr (debug_promo)
                    fmt::print("promo r: {}\n", move);
            }

            // push promotions:
            while (b3) {
                const auto dst  = pop_lsb(b3);
                Move move{static_cast<square_t>(dst - Up), Us, pawn, dst };
                move.set_flags(static_cast<Move::Flags>(Move::Flags::Promote));
                move.set_promo_piece(promotion_piece);
                moves.emplace(move);
                if constexpr (debug_promo)
                    fmt::print("promo p:  {}\n", move);
           }
        }
    }

    return quiet_moves | regular_attacks | enpassant_attacks | promotion_moves;
}


template <Color Us, bool UpdateZobrist = true, bool UpdateNNUE = true>
Movelist generate_legal_moves(Board & b)
{
    constexpr Color Them = ~Us;

    // init:
    init_pinner_blocker_bb<Us>(b);
    init_attacks_bb<Them>(b);

    square_t const king_sq = lsb(b.pt_bb[Us][king]);
    const bool king_in_check = b.all_attacks_bb[Them] & (1ULL << king_sq);
    Movelist legal_moves;

    { // pawn:
        Movelist moves;
        generate_pawn_moves<Us>(b, moves);

        if (!king_in_check) {
            for (auto mv : moves) {
                bitboard_t mm = 1ULL << mv.get_dst();
                bitboard_t src_sq_mask = 1ULL << mv.get_src();
                if (!(b.blockers_bb[Us] & src_sq_mask)) {
                    if (mv.get_dst() != b.gamestate.enpassant_square) {
                        legal_moves.emplace(mv);
                    } else { // todo: enpass might result in discovered check
                        apply_move<Us>(b, mv);
                        if (!is_check<Us>(b))
                            legal_moves.emplace(mv);
                        revert_move<Us>(b);
                    }
                } else {
                    bitboard_t pinners = b.pinners_bb[Them];
                    while (pinners) {
                        square_t const pinner_sq = pop_lsb(pinners);
                        bitboard_t const pinner_sq_mask = 1ULL << pinner_sq;
                        bitboard_t const bb = b.between_bb[pinner_sq][king_sq] | pinner_sq_mask;
                        if (bb & src_sq_mask) { // this blocker is pinned by this pinner
                            if (mm & bb) {
                                legal_moves.emplace(mv);
                                break;
                            }
                        }
                    }
                }
            }
        } else { // only consider moves that get the king out of check
            for (auto mv : moves) {
                apply_move<Us>(b, mv);
                if (!is_check<Us>(b))
                    legal_moves.emplace(mv);
                revert_move<Us>(b);
            }
        }
    }

    { // knights:
        bitboard_t src_mask = b.pt_bb[Us][knight];
        while (src_mask) {
            square_t const src_sq = pop_lsb(src_mask);
            if (b.blockers_bb[Us] & (1ULL << src_sq)) {
                continue; // can't move in pin direction
            }
            bitboard_t dst_mask = knight_moves<Us>(b, src_sq);
            while (dst_mask) {
                const auto dst_sq = pop_lsb(dst_mask);
                const auto dst_piece = get_piece_type<~Us>(b, dst_sq);
                Move mv{ src_sq, Us, knight, dst_sq, dst_piece };
                if (!king_in_check) {
                    legal_moves.emplace(mv);
                } else { // only consider moves that get the king out of check
                    apply_move<Us>(b, mv);
                    if (!is_check<Us>(b))
                        legal_moves.emplace(mv);
                    revert_move<Us>(b);
                }
            }
        }
    }

    { // bishops:
        if (!king_in_check) {
            bool constexpr debug = false;
            bitboard_t src_mask = b.pt_bb[Us][bishop];
            while (src_mask) {
                square_t const src_sq = pop_lsb(src_mask);
                bitboard_t const src_sq_mask = 1ULL << src_sq;
                bitboard_t mm = bishop_moves<Us>(b, src_sq);
                if (!(b.blockers_bb[Us] & src_sq_mask)) { // not blocker:
                    while (mm) {
                        square_t dst_sq = pop_lsb(mm);
                        const auto dst_piece = get_piece_type<~Us>(b, dst_sq);
                        Move mv{ src_sq, Us, bishop, dst_sq, dst_piece };
                        legal_moves.emplace(mv);
                    }
                } else { // allow movement in pin direction:
                    if constexpr (debug)
                        fmt::print("blocker: {}, fen: {}\n", sq2str(src_sq), b.fen());
                    bitboard_t blocker_move_mask = 0;
                    bitboard_t pinners = b.pinners_bb[Them];
                    while (pinners) {
                        square_t const pinner_sq = pop_lsb(pinners);
                        bitboard_t const pinner_sq_mask = 1ULL << pinner_sq;
                        bitboard_t const bb = b.between_bb[pinner_sq][king_sq] | pinner_sq_mask;
                        if (bb & src_sq_mask) { // this blocker is pinned by this piece
                            if (mm & pinner_sq_mask) // can move in pin direction:
                                blocker_move_mask = bb & ~src_sq_mask;
                            else // pinned in place (orthogonal pinner?)
                                blocker_move_mask = 0;
                            break;
                        }
                    }
                    while (blocker_move_mask) {
                        square_t dst_sq = pop_lsb(blocker_move_mask);
                        const auto dst_piece = get_piece_type<~Us>(b, dst_sq);
                        Move mv{ src_sq, Us, bishop, dst_sq, dst_piece };
                        legal_moves.emplace(mv);
                    }
                }
            }
        } else { // only consider moves that get the king out of check
            bitboard_t src_mask = b.pt_bb[Us][bishop];
            while (src_mask) {
                square_t const src_sq = pop_lsb(src_mask);
                bitboard_t mm = bishop_moves<Us>(b, src_sq);
                while (mm) {
                    const auto dst_sq = pop_lsb(mm);
                    const auto dst_piece = get_piece_type<~Us>(b, dst_sq);
                    Move mv{ src_sq, Us, bishop, dst_sq, dst_piece };
                    apply_move<Us>(b, mv);
                    if (!is_check<Us>(b))
                        legal_moves.emplace(mv);
                    revert_move<Us>(b);
                }
            }
        }
    }

    { // rooks:
        if (!king_in_check) {
            bool constexpr debug = false;
            bitboard_t src_mask = b.pt_bb[Us][rook];
            while (src_mask) {
                square_t const src_sq = pop_lsb(src_mask);
                bitboard_t const src_sq_mask = 1ULL << src_sq;
                bitboard_t mm = rook_moves<Us>(b, src_sq);

                if (!(b.blockers_bb[Us] & src_sq_mask)) {
                    while (mm) {
                        square_t dst_sq = pop_lsb(mm);
                        const auto dst_piece = get_piece_type<~Us>(b, dst_sq);
                        Move mv{ src_sq, Us, rook, dst_sq, dst_piece };
                        legal_moves.emplace(mv);
                    }
                } else { // blocker: allow movement in pin direction:
                    if constexpr (debug)
                        fmt::print("blocker: {}, fen: {}\n", sq2str(src_sq), b.fen());
                    bitboard_t blocker_move_mask = 0;
                    bitboard_t pinners = b.pinners_bb[Them];
                    while (pinners) {
                        square_t const pinner_sq = pop_lsb(pinners);
                        bitboard_t const pinner_sq_mask = 1ULL << pinner_sq;
                        bitboard_t const bb = b.between_bb[pinner_sq][king_sq] | pinner_sq_mask;
                        if (bb & src_sq_mask) { // this blocker is pinned by this pinner
                            if (mm & pinner_sq_mask) // can move in pin direction:
                                blocker_move_mask = bb & ~src_sq_mask;
                            else // pinned in place (diagonal pinner?)
                                blocker_move_mask = 0;
                            break;
                        }
                    }
                    while (blocker_move_mask) {
                        square_t dst_sq = pop_lsb(blocker_move_mask);
                        const auto dst_piece = get_piece_type<~Us>(b, dst_sq);
                        Move mv{ src_sq, Us, rook, dst_sq, dst_piece };
                        legal_moves.emplace(mv);
                    }
                }
            }
        } else { // only consider moves that get the king out of check
            bitboard_t src_mask = b.pt_bb[Us][rook];
            while (src_mask) {
                square_t const src_sq = pop_lsb(src_mask);
                bitboard_t mm = rook_moves<Us>(b, src_sq);
                while (mm) {
                    const auto dst_sq = pop_lsb(mm);
                    const auto dst_piece = get_piece_type<~Us>(b, dst_sq);
                    Move mv{ src_sq, Us, rook, dst_sq, dst_piece };

                    apply_move<Us>(b, mv);
                    if (!is_check<Us>(b))
                        legal_moves.emplace(mv);
                    revert_move<Us>(b);
                }
            }
        }
    }

    { // queens:
        if (!king_in_check) {
            bool constexpr debug = false;
            bitboard_t src_mask = b.pt_bb[Us][queen];
            while (src_mask) {
                square_t const src_sq = pop_lsb(src_mask);
                bitboard_t const src_sq_mask = 1ULL << src_sq;
                bitboard_t mm = queen_moves<Us>(b, src_sq);

                // blocker can't move if king is in check:
                if (!(b.blockers_bb[Us] & src_sq_mask)) {
                    while (mm) {
                        square_t dst_sq = pop_lsb(mm);
                        const auto dst_piece = get_piece_type<~Us>(b, dst_sq);
                        Move mv{ src_sq, Us, queen, dst_sq, dst_piece };
                        legal_moves.emplace(mv);
                    }
                } else { // blocker: allow movement in pin direction:
                    if constexpr (debug)
                        fmt::print("blocker: {}, fen: {}\n", sq2str(src_sq), b.fen());
                    bitboard_t blocker_move_mask = 0;
                    bitboard_t pinners = b.pinners_bb[Them];
                    while (pinners) {
                        square_t const pinner_sq = pop_lsb(pinners);
                        bitboard_t const pinner_sq_mask = 1ULL << pinner_sq;
                        bitboard_t const bb = b.between_bb[pinner_sq][king_sq] | pinner_sq_mask;
                        if (bb & src_sq_mask) { // this blocker is pinned by this pinner
                            if (mm & pinner_sq_mask) // can move in pin direction:
                                blocker_move_mask = bb & ~src_sq_mask;
                        }
                    }
                    while (blocker_move_mask) {
                        square_t dst_sq = pop_lsb(blocker_move_mask);
                        const auto dst_piece = get_piece_type<~Us>(b, dst_sq);
                        Move mv{ src_sq, Us, queen, dst_sq, dst_piece };
                        legal_moves.emplace(mv);
                    }
                }
            }
        } else { // only consider moves that get the king out of check
            bitboard_t src_mask = b.pt_bb[Us][queen];
            while (src_mask) {
                square_t const src_sq = pop_lsb(src_mask);
                bitboard_t mm = queen_moves<Us>(b, src_sq);
                while (mm) {
                    const auto dst_sq = pop_lsb(mm);
                    const auto dst_piece = get_piece_type<~Us>(b, dst_sq);
                    Move mv{ src_sq, Us, queen, dst_sq, dst_piece };
                    apply_move<Us>(b, mv);
                    if (!is_check<Us>(b))
                        legal_moves.emplace(mv);
                    revert_move<Us>(b);
                }
            }
        }
    }

    { // king:
        assert(king_sq != square_nb);

        constexpr bitboard_t kingside_castle_mask  = 0x0E0000000000000E & (Us == white ? rank_1 : rank_8);
        constexpr bitboard_t queenside_castle_mask = 0x3000000000000030 & (Us == white ? rank_1 : rank_8);
        constexpr square_t king_home = Us == white ? e1 : e8;

        // castling rights and occupancy has already been verified
        const bool can_castle_kingside  = !king_in_check && !(b.all_attacks_bb[Them] & kingside_castle_mask);
        const bool can_castle_queenside = !king_in_check && !(b.all_attacks_bb[Them] & queenside_castle_mask);

        bitboard_t king_mask = king_moves<Us, true>(b, king_sq);
        // validate castling:
        while (king_mask) {
            const auto dst_sq = pop_lsb(king_mask);
            const auto dst_mask = 1ULL << dst_sq;
            if (b.all_attacks_bb[Them] & (1ULL << dst_sq))
                continue;

            const auto dst_piece = get_piece_type<Them>(b, dst_sq);
            Move mv { king_sq, Us, king, dst_sq, dst_piece };
            if (king_sq != king_home) { // normal move
                legal_moves.emplace(mv);
            } else { // king@home
                if (dst_mask & file_g) { // kingside castle:
                    if (can_castle_kingside) {
                        mv.set_flags(Move::Flags::Castle);
                        legal_moves.emplace(mv);
                    }
                } else if (dst_mask & file_c) { // queenside
                    if (can_castle_queenside) {
                        mv.set_flags(Move::Flags::Castle);
                        legal_moves.emplace(mv);
                    }
                } else { // normal move from home square
                    legal_moves.emplace(mv);
                }
            }
        }
    }
    if constexpr (Constexpr::check_ksq) {
        if (!b.pt_bb[white][king]) BREAKPOINT("missing white king");
        if (!b.pt_bb[black][king]) BREAKPOINT("missing black king");
    }

    return legal_moves;
}
