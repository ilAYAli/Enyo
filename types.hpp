#pragma once

#include <array>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <ranges>
#include "fmt/core.h"
#include "mvlookup.hpp"

namespace enyo {

using bitboard_t = uint64_t;
using square_t   = uint8_t;

constexpr int MAX_HISTORY = 1024;   // max # moves in a game
constexpr int MAX_MOVES   = 256;    // max # moves in a given position
constexpr int MAX_PLY     = 64;     // max depth


//----[ bitboards (h1 indexed) ]-----------------------------------------------
constexpr bitboard_t rank_8 = 0xff00000000000000;
constexpr bitboard_t rank_7 = 0x00ff000000000000;
constexpr bitboard_t rank_6 = 0x0000ff0000000000;
constexpr bitboard_t rank_5 = 0x000000ff00000000;
constexpr bitboard_t rank_4 = 0x00000000ff000000;
constexpr bitboard_t rank_3 = 0x0000000000ff0000;
constexpr bitboard_t rank_2 = 0x000000000000ff00;
constexpr bitboard_t rank_1 = 0x00000000000000ff;

constexpr bitboard_t file_a = 0x8080808080808080;
constexpr bitboard_t file_b = 0x4040404040404040;
constexpr bitboard_t file_c = 0x2020202020202020;
constexpr bitboard_t file_d = 0x1010101010101010;
constexpr bitboard_t file_e = 0x0808080808080808;
constexpr bitboard_t file_f = 0x0404040404040404;
constexpr bitboard_t file_g = 0x0202020202020202;
constexpr bitboard_t file_h = 0x0101010101010101;

constexpr bitboard_t not_file_a  = 0x7f7f7f7f7f7f7f7f;
constexpr bitboard_t not_file_h  = 0xfefefefefefefefe;

// h1 indexed
enum square : square_t {
    h1, g1, f1, e1, d1, c1, b1, a1,
    h2, g2, f2, e2, d2, c2, b2, a2,
    h3, g3, f3, e3, d3, c3, b3, a3,
    h4, g4, f4, e4, d4, c4, b4, a4,
    h5, g5, f5, e5, d5, c5, b5, a5,
    h6, g6, f6, e6, d6, c6, b6, a6,
    h7, g7, f7, e7, d7, c7, b7, a7,
    h8, g8, f8, e8, d8, c8, b8, a8,
    square_nb = 64,
    nosquare = 0xff,
};

static constexpr inline bitboard_t get_file_bb(int file)
{
    return file == 0 ? file_a
         : file == 1 ? file_b
         : file == 2 ? file_c
         : file == 3 ? file_d
         : file == 4 ? file_e
         : file == 5 ? file_f
         : file == 6 ? file_g
         : file == 7 ? file_h
         : 0;
}

static constexpr inline bitboard_t get_rank_bb(int rank)
{
    return rank == 0 ? rank_1
         : rank == 1 ? rank_2
         : rank == 2 ? rank_3
         : rank == 3 ? rank_4
         : rank == 4 ? rank_5
         : rank == 5 ? rank_6
         : rank == 6 ? rank_7
         : rank == 7 ? rank_8
         : 0;
}

static constexpr bitboard_t get_adjacent_files(int file)
{
    constexpr bitboard_t adjacent_files[8] = {
        file_b,
        file_a | file_c,
        file_b | file_d,
        file_c | file_e,
        file_d | file_f,
        file_e | file_g,
        file_f | file_h,
        file_g
    };

    return adjacent_files[file];
}

static constexpr bitboard_t square_bb(int square)
{
    return 1ULL << (63 - square);
}

// get file and rank from square w/ color. Todo: consider color
static constexpr bitboard_t frtab[64][2] = { // [file][rank]
//   h1               g1               f1               e1               d1               c1               b1
    {file_h,rank_1}, {file_g,rank_1}, {file_f,rank_1}, {file_e,rank_1}, {file_d,rank_1}, {file_c,rank_1}, {file_b,rank_1}, {file_a,rank_1},
    {file_h,rank_2}, {file_g,rank_2}, {file_f,rank_2}, {file_e,rank_2}, {file_d,rank_2}, {file_c,rank_2}, {file_b,rank_2}, {file_a,rank_2},
    {file_h,rank_3}, {file_g,rank_3}, {file_f,rank_3}, {file_e,rank_3}, {file_d,rank_3}, {file_c,rank_3}, {file_b,rank_3}, {file_a,rank_3},
    {file_h,rank_4}, {file_g,rank_4}, {file_f,rank_4}, {file_e,rank_4}, {file_d,rank_4}, {file_c,rank_4}, {file_b,rank_4}, {file_a,rank_4},
    {file_h,rank_5}, {file_g,rank_5}, {file_f,rank_5}, {file_e,rank_5}, {file_d,rank_5}, {file_c,rank_5}, {file_b,rank_5}, {file_a,rank_5},
    {file_h,rank_6}, {file_g,rank_6}, {file_f,rank_6}, {file_e,rank_6}, {file_d,rank_6}, {file_c,rank_6}, {file_b,rank_6}, {file_a,rank_6},
    {file_h,rank_7}, {file_g,rank_7}, {file_f,rank_7}, {file_e,rank_7}, {file_d,rank_7}, {file_c,rank_7}, {file_b,rank_7}, {file_a,rank_7},
    {file_h,rank_8}, {file_g,rank_8}, {file_f,rank_8}, {file_e,rank_8}, {file_d,rank_8}, {file_c,rank_8}, {file_b,rank_8}, {file_a,rank_8},
};


enum Direction : int {
    north = 8,
    east  = -1,
   south = -north,
    west  = -east,

    north_east = north + east,
    south_east = south + east,
    south_west = south + west,
    north_west = north + west
};

template<Direction D>
constexpr bitboard_t shift(bitboard_t b) {
    return
        D == north         ? b << 8
      : D == south         ? b >> 8
      : D == north + north ? b << 16
      : D == south + south ? b >> 16
      : D == east          ? (b & ~file_h) >> 1
      : D == west          ? (b & ~file_a) << 1
      : D == north_east    ? (b & ~file_h) << 7
      : D == north_west    ? (b & ~file_a) << 9
      : D == south_east    ? (b & ~file_h) >> 9
      : D == south_west    ? (b & ~file_a) >> 7
      : 0;
}


//----[ value ]----------------------------------------------------------------
enum Value : int {
    DRAW                  = 0,
    MATE                  = 30000,
    MATED                 =-MATE,
    NONE                  = 32767,
    INFINITE              = 32766,
    TB_WIN_IN_MAX_PLY     = MATE - 2 * MAX_PLY,
    TB_LOSS_IN_MAX_PLY    =-TB_WIN_IN_MAX_PLY,
    MATE_IN_MAX_PLY       = MATE - MAX_PLY,
    MATED_IN_MAX_PLY      =-MATE_IN_MAX_PLY,
};

constexpr Value operator+(Value v, int val) {
    return static_cast<Value>(static_cast<int>(v) + val);
}

constexpr Value operator-(Value v, int val) {
    return static_cast<Value>(static_cast<int>(v) - val);
}

constexpr Value operator/(Value v, int val) {
    return static_cast<Value>(static_cast<int>(v) / val);
}

constexpr Value operator-(Value v) {
    return static_cast<Value>(-static_cast<int>(v));
}

constexpr Value& operator+=(Value& v, int val) {
    v = static_cast<Value>(static_cast<int>(v) + val);
    return v;
}

//----[ color ]----------------------------------------------------------------
enum Color : int {
    white,
    black,
    color_nb,
};

static inline constexpr Color operator~(Color c) {
    return Color(c ^ black);
}


//----[ castling ]-------------------------------------------------------------
enum CastlingRights {
    no_castling,

    white_oo,
    white_ooo = white_oo << 1,
    black_oo  = white_oo << 2,
    black_ooo = white_oo << 3,

    king_side      = white_oo       | black_oo,
    queen_side     = white_ooo      | black_ooo,
    white_castling = white_oo       | white_ooo,
    black_castling = black_oo       | black_ooo,
    any_castling   = white_castling | black_castling,

    castling_right_nb = 16
};

enum PieceType : uint32_t {
    no_piece_type = 0,

    pawn = 1,
    knight,
    bishop,
    rook,
    queen,
    king,

    piece_type_nb = king + 1
};

static constexpr inline int piece_value(PieceType pt) {
    switch (pt) {
        case pawn:   return 100;
        case knight: return 320;
        case bishop: return 330;
        case rook:   return 500;
        case queen:  return 900;
        case king:   return 0;
        default:     return 0;
    }
}


//----[ move ]-----------------------------------------------------------------
struct alignas(4) Move {
    static constexpr uint32_t no_move = 0x0000000000000000;

    enum Flags : int {
        Generic = 0,
        Promote = 1,
        Enpassant = 2,
        Castle = 4,
    };

    /*explicit*/ constexpr Move(uint32_t value)
        : data(value)
    { }

    constexpr Move(
        square_t src,
        Color src_color,
        PieceType src_piece,
        square_t dst,
        PieceType dst_piece = no_piece_type)
        : data(static_cast<uint32_t>(
               (src & pos_mask)
             | ((dst & pos_mask) << 6)
             | ((static_cast<uint32_t>(src_piece) & piece_mask) << 12)
             | ((static_cast<uint32_t>(dst_piece) & piece_mask) << 18)
             | ((static_cast<uint32_t>(src_color) & color_mask) << 24)
             ))
    {}

    constexpr Move()
        : data(0)
    { }

    [[nodiscard]] constexpr square_t src_sq() const {
        return static_cast<uint8_t>((data >> src_shift) & pos_mask);
    }

    [[nodiscard]] constexpr square_t dst_sq() const {
        return static_cast<uint8_t>((data >> dst_shift) & pos_mask);
    }

    [[nodiscard]] constexpr PieceType src_piece() const {
        return static_cast<PieceType>((data >> src_piece_shift) & pos_mask);
    }

    [[nodiscard]] constexpr PieceType dst_piece() const {
        return static_cast<PieceType>((data >> dst_piece_shift) & piece_mask);
    }

    [[nodiscard]] constexpr Color src_color() const {
        return static_cast<Color>((data >> src_color_shift) & color_mask);
    }

    [[nodiscard]] constexpr int flags() const {
        return static_cast<int>((data >> flags_shift) & flags_mask);
    }

    [[nodiscard]] constexpr PieceType promo_piece() const {
        uint32_t val = (data >> promo_shift) & promo_mask;
        switch (val) {
            case 0: return knight;
            case 1: return bishop;
            case 2: return rook;
            case 3: return queen;
            default:
                assert(false && "promo_piece is not knight, bishop, rook, or queen");
        }
        return no_piece_type;
    }

    constexpr void set_flags(Flags move_flags) {
        data &= ~(flags_mask << flags_shift);
        data |= (static_cast<uint32_t>(move_flags) & flags_mask) << flags_shift;
    }

    constexpr void set_promo_piece(PieceType piece) {
        uint32_t val = 0;
        switch (piece) {
            case knight: val = 0; break;
            case bishop: val = 1; break;
            case rook:   val = 2; break;
            case queen:  val = 3; break;
            default:
                assert(false && "promo_piece must be knight, bishop, rook, or queen");
        }
        data &= ~(promo_mask << promo_shift);
        data |= (val & promo_mask) << promo_shift;
    }

    constexpr auto operator<=>(const Move&) const = default;

    constexpr auto operator<=>(uint32_t value) const {
        return data <=> value;
    }

    constexpr auto & operator=(uint32_t value) {
        data = value;
        return *this;
    }

    constexpr explicit operator bool() const {
        return data != no_move;
    }

    uint32_t data = no_move;
private:
    static constexpr uint32_t piece_size = 6;
    static constexpr uint32_t color_size = 1;
    static constexpr uint32_t pos_size   = 6;
    static constexpr uint32_t flags_size = 4;
    static constexpr uint32_t promo_size = 2;

    static constexpr uint32_t piece_mask = (1 << piece_size) - 1;
    static constexpr uint32_t color_mask = (1 << color_size) - 1;
    static constexpr uint32_t pos_mask   = (1 << pos_size) - 1;
    static constexpr uint32_t flags_mask = (1 << flags_size) - 1;
    static constexpr uint32_t promo_mask = (1 << promo_size) - 1;

    static constexpr uint32_t src_shift       = 0;
    static constexpr uint32_t dst_shift       = 6;
    static constexpr uint32_t src_piece_shift = 12;
    static constexpr uint32_t dst_piece_shift = 18;
    static constexpr uint32_t src_color_shift = 24;
    static constexpr uint32_t unused_bit      = 25;
    static constexpr uint32_t flags_shift     = 26;
    static constexpr uint32_t promo_shift     = 30;
};
static_assert(sizeof(Move) == 4, "Move must be exactly 32 bits");

struct ScoredMove {
    int score {};
    enyo::Move move {};
    // sort in descending order by score:
    constexpr auto operator<=>(const ScoredMove& other) const {
        return other.score <=> score;
    }
};

struct Gamestate {
    bool white_to_move {};
    square_t enpassant_square{};
    unsigned castling_rights{};
    unsigned half_moves {};

    //bool has_castled[2] {};
    inline bool can_castle(CastlingRights cr) const {
        return castling_rights & cr;
    }
    inline unsigned set_castle(CastlingRights cr, bool val) {
        return val
            ? castling_rights |= cr
            : castling_rights &= ~cr;
    }
};

struct Undo {
    Move move {};
    Gamestate gamestate {};
    uint64_t hash {};
    int half_moves {};
};

} // enyo


//----[ formatters ]-----------------------------------------------------------
namespace fmt {

template <>
struct formatter<enyo::Value> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const enyo::Value v, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "{}", static_cast<int>(v));
    }
};


template<>
struct formatter<enyo::Move> : formatter<const char*> {
    auto format(enyo::Move type, format_context& ctx) const {
        if (type.flags() != enyo::Move::Flags::Promote) {
            return fmt::format_to(ctx.out(), "{}",
                mvlookup[type.src_sq()][type.dst_sq()]);
        }
        auto s = fmt::format("{}", mvlookup[type.src_sq()][type.dst_sq()]);
        auto const promo_piece = type.promo_piece();
        switch (promo_piece) {
            case enyo::knight: s += "n"; break;
            case enyo::bishop: s += "b"; break;
            case enyo::rook:   s += "r"; break;
            case enyo::queen:  s += "q"; break;
            default: break;
        }
        return fmt::format_to(ctx.out(), "{}", s);
    }
};

template <>
struct formatter<enyo::ScoredMove> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const enyo::ScoredMove& move, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "{}", move.move);
    }
};


template <>
struct formatter<enyo::PieceType> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(enyo::PieceType type, FormatContext& ctx) {
        auto str = [](enyo::PieceType t) {
            switch (t) {
                case enyo::PieceType::pawn:   return "pawn";
                case enyo::PieceType::rook:   return "rook";
                case enyo::PieceType::knight: return "knight";
                case enyo::PieceType::bishop: return "bishop";
                case enyo::PieceType::queen:  return "queen";
                case enyo::PieceType::king:   return "king";
                case enyo::PieceType::no_piece_type: return "None";
                default: return "??";
            }
        };
        return fmt::format_to(ctx.out(), "{}", str(type));
    }
};


constexpr inline const char * sq2str(enyo::square_t sq) {
    constexpr const char* sym[64] = {
        "h1", "g1", "f1", "e1", "d1", "c1", "b1", "a1",
        "h2", "g2", "f2", "e2", "d2", "c2", "b2", "a2",
        "h3", "g3", "f3", "e3", "d3", "c3", "b3", "a3",
        "h4", "g4", "f4", "e4", "d4", "c4", "b4", "a4",
        "h5", "g5", "f5", "e5", "d5", "c5", "b5", "a5",
        "h6", "g6", "f6", "e6", "d6", "c6", "b6", "a6",
        "h7", "g7", "f7", "e7", "d7", "c7", "b7", "a7",
        "h8", "g8", "f8", "e8", "d8", "c8", "b8", "a8"
    };
    return sym[static_cast<std::size_t>(sq) & 0x3f];
}

} // fmt

