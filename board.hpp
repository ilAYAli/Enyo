#pragma once

#include <cstdint>

#include "config.hpp"
#include "pgn.hpp"
#include "board.hpp"
#include "magic/magic.hpp"
#include "types.hpp"
#include "zobrist.hpp"
#include "fen.hpp"
#include "tt.hpp"
#include "util.hpp"
#include "nnue.hpp"

#include "fmt/core.h"


namespace enyo {

class Board {
public:
    explicit Board(std::string_view);
    Board() = default;

    Board(const Board& other) {
        copy_data(other);
    }

    Board& operator=(const Board& other) {
        if (this != &other) {
            Board temp(other);
            swap_data(temp);
        }
        return *this;
    }

    // member variables:
    PieceType pt_mb[square_nb] {};
    alignas(64) bitboard_t pt_bb[color_nb][piece_type_nb] {};
    bitboard_t color_bb[color_nb] { };
    bitboard_t checkers_bb[color_nb] { };
    bitboard_t between_bb[square_nb][square_nb] {};
    bitboard_t line_bb[square_nb][square_nb] {};
    bitboard_t blockers_bb[color_nb] {};
    bitboard_t pinners_bb[color_nb] {} ;
    bitboard_t attacks_bb[color_nb][square_nb] {} ;
    bitboard_t all_attacks_bb[color_nb]{} ;
    Move pv_table[MAX_PLY] {};
    Undo history[MAX_HISTORY] {};
    Color side { white };
    int histply {};
    uint64_t hash {};
    zobrist::zbrs zbrs{};
    int half_moves {};
    Gamestate gamestate{};

    std::string str(uint64_t attack_mask = 0, unsigned indent = 0) const;
    void set(std::string_view s = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::string fen() const;

private:
    void swap_data(Board & other);
    void copy_data(const Board & other);
    void clear_data();
};


template <Color Us, bool UpdateZobrist, bool UpdateNNUE>
static inline constexpr void set_piece(
    Board & b,
    PieceType pt,
    square_t sq,
    [[maybe_unused]] NNUE::Net * nnue,
    [[maybe_unused]] square_t w_ksq = 0,
    [[maybe_unused]] square_t b_ksq = 0)
{
    assert(pt != no_piece_type);
    assert(sq < square_nb);

    const auto sq_mask = 1ULL << sq;
    b.pt_bb[Us][pt] |= sq_mask;
    b.color_bb[Us] |= sq_mask;
    b.pt_mb[sq] = pt;

    if constexpr (UpdateZobrist) {
        const auto before = b.hash;
        b.hash ^= b.zbrs.psq_[Us][pt][sq];

        if constexpr (zobrist::debug) {
            fmt::print("<{}> zobrist: {} {}@{} before: {:#018x} after: {:#018x}, fen: {}\n",
                    __func__, Us, pt, sq2str(sq), before, b.hash, b.fen());
        }
    }

    if constexpr (UpdateNNUE) {
        nnue->updateAccumulator<true>(pt, Us, sq, w_ksq, b_ksq);
    }
}

template <Color Us, bool UpdateZobrist, bool UpdateNNUE>
static inline constexpr void clr_piece(
    Board & b,
    PieceType pt,
    square_t sq,
    [[maybe_unused]] NNUE::Net * nnue,
    [[maybe_unused]] square_t w_ksq = 0,
    [[maybe_unused]] square_t b_ksq = 0)
{
    assert(pt != no_piece_type);
    assert(sq < square_nb);

    const auto before = b.hash;
    if constexpr (UpdateZobrist) {
        b.hash ^= b.zbrs.psq_[Us][pt][sq];
    }

    const auto not_sq_mask = ~(1ULL << sq);
    b.pt_bb[Us][pt] &= not_sq_mask;
    b.color_bb[Us] &= not_sq_mask;
    b.pt_mb[sq] = no_piece_type;

    if constexpr (UpdateZobrist) {
        if constexpr (zobrist::debug) {
            fmt::print("<{}> zobrist: {} {}@{} before: {:#018x} after: {:#018x}, fen: {}\n",
                    __func__, Us, pt, sq2str(sq), before, b.hash, b.fen());
        }
    }
    if constexpr (UpdateNNUE) {
        assert(nnue && "NNUE::Net must be provided for NNUE update");
        nnue->updateAccumulator<false>(pt, Us, sq, w_ksq, b_ksq);
    }
}


template <enyo::Color c, PieceType pt>
static inline bitboard_t pieces(const Board & b)
{
    return b.pt_bb[c][pt];
}

//---[ helpers ]----------------------------------------------------------------
template <enyo::Color c>
static inline PieceType get_piece_type(Board const & b, square_t sq) {
    return b.pt_mb[sq];
}

} // enyo

static inline std::string visualize(enyo::bitboard_t const pb[enyo::color_nb][enyo::PieceType::piece_type_nb])
{
    using namespace enyo;

    std::string s = "";
    fmt::format_to(std::back_inserter(s), "\n{} ", 8);
    for (auto sq = 63; sq >= 0; sq--) {
        bitboard_t mask = 1ULL << sq;
        if      (mask & pb[white][pawn])     fmt::format_to(std::back_inserter(s), "{}", "P ");
        else if (mask & pb[white][rook])     fmt::format_to(std::back_inserter(s), "{}", "R ");
        else if (mask & pb[white][knight])   fmt::format_to(std::back_inserter(s), "{}", "N ");
        else if (mask & pb[white][bishop])   fmt::format_to(std::back_inserter(s), "{}", "B ");
        else if (mask & pb[white][queen])    fmt::format_to(std::back_inserter(s), "{}", "Q ");
        else if (mask & pb[white][king])     fmt::format_to(std::back_inserter(s), "{}", "K ");
        else if (mask & pb[black][pawn])     fmt::format_to(std::back_inserter(s), "{}", "p ");
        else if (mask & pb[black][rook])     fmt::format_to(std::back_inserter(s), "{}", "r ");
        else if (mask & pb[black][knight])   fmt::format_to(std::back_inserter(s), "{}", "n ");
        else if (mask & pb[black][bishop])   fmt::format_to(std::back_inserter(s), "{}", "b ");
        else if (mask & pb[black][queen])    fmt::format_to(std::back_inserter(s), "{}", "q ");
        else if (mask & pb[black][king])     fmt::format_to(std::back_inserter(s), "{}", "k ");
        else fmt::format_to(std::back_inserter(s), ". ");
        if (sq && (sq % 8 == 0))
            fmt::format_to(std::back_inserter(s), "\n{} ", sq / 8);
    }
    fmt::format_to(std::back_inserter(s), "\n  A B C D E F G H\n");
    return s;
}

namespace fmt {

template <>
struct formatter<enyo::Color> : formatter<const char *> {
    using formatter<const char *>::format;
    auto format(enyo::Color type, format_context & ctx) {
        return fmt::format_to(ctx.out(), "{}", type == enyo::white ? "white" : "black");
    }
};

template <>
struct formatter<enyo::Board> : formatter<const char *> {
    using formatter<const char *>::format;
    auto format(enyo::Board & b, format_context & ctx) {
        return fmt::format_to(ctx.out(), "{}", b.str());
    }
};

} // fmt
