
#pragma once
#include <cstdint>
#include "magic_numbers.hpp"
#include "../types.hpp"

namespace enyo {

extern bitboard_t bishop_masks[64];
extern bitboard_t rook_masks[64];
extern bitboard_t bishop_attacks[64][512];
extern bitboard_t rook_attacks[64][4096];

void init_sliders_attacks(PieceType piece);

// rook relevant occupancy bits
static constexpr int rook_relevant_bits[64] = {
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12
};

// bishop relevant occupancy bits
static constexpr int bishop_rellevant_bits[64] = {
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6
};

static inline uint64_t get_bishop_attacks(int square, uint64_t occupancy)
{
    occupancy &= bishop_masks[square];
    occupancy *= bishop_magics[square];
    occupancy >>= 64 - bishop_rellevant_bits[square];
    return bishop_attacks[square][occupancy];
}

static inline uint64_t get_rook_attacks(int square, uint64_t occupancy)
{
    occupancy &= rook_masks[square];
    occupancy *= rook_magics[square];
    occupancy >>= 64 - rook_relevant_bits[square];
    return rook_attacks[square][occupancy];
}

} // enyo
