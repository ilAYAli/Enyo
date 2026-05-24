/*
 * Enyo integration config for Pyrrhic.
 *
 * Pyrrhic works in the standard A1=0 bitboard layout. Enyo converts positions
 * before calling into Pyrrhic, so the helpers here intentionally stay A1=0.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

static inline unsigned enyo_pyrrhic_poplsb(uint64_t *bb)
{
    const unsigned sq = (unsigned)__builtin_ctzll(*bb);
    *bb &= *bb - 1;
    return sq;
}

#define PYRRHIC_POPCOUNT(x) ((int)__builtin_popcountll((uint64_t)(x)))
#define PYRRHIC_LSB(x) ((unsigned)__builtin_ctzll((uint64_t)(x)))
#define PYRRHIC_POPLSB(x) enyo_pyrrhic_poplsb((x))

static inline uint64_t enyo_pyrrhic_knight_attacks(unsigned sq)
{
    const int file = (int)(sq & 7U);
    const int rank = (int)(sq >> 3U);
    uint64_t attacks = 0;
    static const int offsets[8][2] = {
        { 1,  2}, { 2,  1}, { 2, -1}, { 1, -2},
        {-1, -2}, {-2, -1}, {-2,  1}, {-1,  2},
    };

    for (unsigned i = 0; i < 8; ++i) {
        const int f = file + offsets[i][0];
        const int r = rank + offsets[i][1];
        if (0 <= f && f < 8 && 0 <= r && r < 8)
            attacks |= 1ULL << (r * 8 + f);
    }
    return attacks;
}

static inline uint64_t enyo_pyrrhic_king_attacks(unsigned sq)
{
    const int file = (int)(sq & 7U);
    const int rank = (int)(sq >> 3U);
    uint64_t attacks = 0;

    for (int dr = -1; dr <= 1; ++dr) {
        for (int df = -1; df <= 1; ++df) {
            if (df == 0 && dr == 0)
                continue;
            const int f = file + df;
            const int r = rank + dr;
            if (0 <= f && f < 8 && 0 <= r && r < 8)
                attacks |= 1ULL << (r * 8 + f);
        }
    }
    return attacks;
}

static inline uint64_t enyo_pyrrhic_pawn_attacks(unsigned sq, bool color)
{
    const uint64_t bb = 1ULL << sq;
    const uint64_t not_file_a = 0xfefefefefefefefeULL;
    const uint64_t not_file_h = 0x7f7f7f7f7f7f7f7fULL;

    if (color)
        return ((bb & not_file_a) << 7U) | ((bb & not_file_h) << 9U);
    return ((bb & not_file_h) >> 7U) | ((bb & not_file_a) >> 9U);
}

static inline uint64_t enyo_pyrrhic_slider_attacks(
    unsigned sq,
    uint64_t occ,
    const int directions[][2],
    unsigned direction_count)
{
    const int file = (int)(sq & 7U);
    const int rank = (int)(sq >> 3U);
    uint64_t attacks = 0;

    for (unsigned i = 0; i < direction_count; ++i) {
        int f = file + directions[i][0];
        int r = rank + directions[i][1];
        while (0 <= f && f < 8 && 0 <= r && r < 8) {
            const unsigned to = (unsigned)(r * 8 + f);
            const uint64_t to_bb = 1ULL << to;
            attacks |= to_bb;
            if (occ & to_bb)
                break;
            f += directions[i][0];
            r += directions[i][1];
        }
    }
    return attacks;
}

static inline uint64_t enyo_pyrrhic_bishop_attacks(unsigned sq, uint64_t occ)
{
    static const int directions[4][2] = {
        { 1,  1}, { 1, -1}, {-1, -1}, {-1,  1},
    };
    return enyo_pyrrhic_slider_attacks(sq, occ, directions, 4);
}

static inline uint64_t enyo_pyrrhic_rook_attacks(unsigned sq, uint64_t occ)
{
    static const int directions[4][2] = {
        { 1,  0}, { 0, -1}, {-1,  0}, { 0,  1},
    };
    return enyo_pyrrhic_slider_attacks(sq, occ, directions, 4);
}

#define PYRRHIC_PAWN_ATTACKS(sq, color) enyo_pyrrhic_pawn_attacks((unsigned)(sq), (color))
#define PYRRHIC_KNIGHT_ATTACKS(sq) enyo_pyrrhic_knight_attacks((unsigned)(sq))
#define PYRRHIC_BISHOP_ATTACKS(sq, occ) enyo_pyrrhic_bishop_attacks((unsigned)(sq), (occ))
#define PYRRHIC_ROOK_ATTACKS(sq, occ) enyo_pyrrhic_rook_attacks((unsigned)(sq), (occ))
#define PYRRHIC_QUEEN_ATTACKS(sq, occ) \
    (enyo_pyrrhic_bishop_attacks((unsigned)(sq), (occ)) | enyo_pyrrhic_rook_attacks((unsigned)(sq), (occ)))
#define PYRRHIC_KING_ATTACKS(sq) enyo_pyrrhic_king_attacks((unsigned)(sq))
