#pragma once

#include <random>
#include <cstdint>
#include "types.hpp"
#include "mvlookup.hpp"


namespace enyo {

//---[ bit manipulation ]-------------------------------------------------------
template <typename T>
static inline __attribute__((always_inline)) uint64_t get_bit(const uint64_t b, T index) {
   return (b & (1ULL << index));
}

template <typename T>
static inline __attribute__((always_inline)) void set_bit(uint64_t & b, T index) {
    b |= (1ULL << index);
}

template <typename T>
static inline __attribute__((always_inline)) void clear_bit(uint64_t & b, T index) {
    b &= ~(1ULL << index);
}

static inline __attribute__((always_inline)) int count_bits(const uint64_t b) {
    return __builtin_popcountll(b);
}

static inline __attribute__((always_inline)) uint8_t lsb(const uint64_t b) {
    return static_cast<uint8_t>(__builtin_ctzll(b));
}

static inline __attribute__((always_inline)) uint8_t pop_lsb(uint64_t & b)
{
    uint8_t const s = static_cast<uint8_t>(__builtin_ctzll(b));
    b &= b - 1;
    return s;
}

template <typename T>
static inline __attribute__((always_inline)) void pop_bit(uint64_t & b, T index) {
    b ^= (1ULL << index);
}

//---[ random ]-----------------------------------------------------------------
static std::mt19937_64 rng(std::random_device{}());

static inline __attribute__((always_inline)) uint64_t random_uint64_t() {
    return rng();
}

static inline __attribute__((always_inline)) uint64_t random_fewbits() {
    return random_uint64_t() & random_uint64_t() & random_uint64_t();
}

constexpr inline uint8_t str2sq(char const * str) {
    return static_cast<uint8_t>((7 - (str[0] - 'a')) + ((str[1] - '1') * 8));
}

inline __attribute__((always_inline)) const char * mv2str(Move m)
{
    return mvlookup[m.get_src()][m.get_dst()];
}

//---[ misc ]-------------------------------------------------------------------
inline __attribute__((always_inline)) void prefetch(const void *addr) {
#if defined(__INTEL_COMPILER) || defined(_MSC_VER)
    _mm_prefetch((char *)addr, _MM_HINT_T0);
#else
    __builtin_prefetch(addr);
#endif
}

static inline constexpr square_t h1_to_a1[64] = {
    a1, b1, c1, d1, e1, f1, g1, h1,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a8, b8, c8, d8, e8, f8, g8, h8 /* A8 */
};

static inline constexpr square_t a1_to_h1[64] = {
    a1, b1, c1, d1, e1, f1, g1, h1,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a8, b8, c8, d8, e8, f8, g8, h8 /* A8 */
};


static inline constexpr square_t sqconv(square_t sq)
{
    return h1_to_a1[sq];
}

static inline constexpr uint64_t bbconv(uint64_t bb)
{
    uint64_t converted_bb = 0;
    while (bb) {
        const auto sq = pop_lsb(bb);
        constexpr auto & conversion = h1_to_a1;
        converted_bb |= 1ULL << conversion[sq];
    }
    return converted_bb;
}

constexpr inline int pt2p(Color c, PieceType pt) {
    if (c == white) {
        switch (pt) {
            case pawn:    return 0;
            case knight:  return 2;
            case bishop:  return 4;
            case rook:    return 6;
            case queen:   return 8;
            case king:    return 10;
            default:            return 12;
        }
    } else {
        switch (pt) { // white
            case pawn:    return 1;
            case knight:  return 3;
            case bishop:  return 5;
            case rook:    return 7;
            case queen:   return 9;
            case king:    return 11;
            default:            return 13;
        }
    }
}

//const int piece_color = !(piece_type & 1); // is 0 for white pieces
//const int piece_type = piece >> 1; // remove color
constexpr inline int get_color(int piece) {
    return !(piece & 1); // is 0 for white pieces
}

constexpr PieceType get_piece_type(int piece) {
    int color = piece & 1;
    int piece_type = piece >> 1;
    if (color == white) {
        switch (piece_type) {
            case 1:  return pawn;
            case 3:  return knight;
            case 5:  return bishop;
            case 7:  return rook;
            case 9:  return queen;
            case 11: return king;
        }
    } else {
        switch (piece_type) {
            case 0:  return pawn;
            case 2:  return knight;
            case 4:  return bishop;
            case 6:  return rook;
            case 8:  return queen;
            case 10: return king;
        }
    }
    return no_piece_type;
}

constexpr inline const char * sq2str(square_t sq) {
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

static inline std::string visualize(uint64_t bitboard)
{
    if (!bitboard) return " \"empty\"";
    std::string s = "";
    fmt::format_to(std::back_inserter(s), "\n");
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int square = (7-rank) * 8 + (7-file);
            if (!file)
                fmt::format_to(std::back_inserter(s),"  {} ", 8 - rank);
            fmt::format_to(std::back_inserter(s), " {}", get_bit(bitboard, static_cast<uint8_t>(square)) ? "1" : ".");
        }
        fmt::format_to(std::back_inserter(s), "\n");
    }
    fmt::format_to(std::back_inserter(s), "\n     A B C D E F G H\n");
    fmt::format_to(std::back_inserter(s), "     bitboard: {:#018x}\n\n", bitboard);
    return s;
}

} // enyo ns
