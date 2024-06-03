/*
generate magics:

g++ -std=c++20 -I /opt/homebrew/include/ -L /opt/homebrew/lib -lfmt magic.cpp -DFIND_MAGICS -DMAIN && ./a.out > magic_numbers.hpp

#c magic.cpp -DFIND_MAGICS && ./magic > magic_numbers.cpp

generate and test:
#c magic.cpp -DFIND_MAGICS -DMAIN && ./a.out > magic_numbers.cpp && c magic.cpp -D MAIN && ./a.out

 */
#include "fmt/format.h"

#include "magic.hpp"
#include "../util.hpp"

namespace enyo {

bitboard_t bishop_masks[64];
bitboard_t rook_masks[64];
bitboard_t bishop_attacks[64][512];
bitboard_t rook_attacks[64][4096];

uint64_t set_occupancy(int index, int bits_in_mask, uint64_t attack_mask)
{
    uint64_t occupancy = 0ULL;

    for (int count = 0; count < bits_in_mask; count++) {
        int square = lsb(attack_mask);

        pop_bit(attack_mask, square);

        if (index & (1 << count))
            occupancy |= (1ULL << square);
    }

    return occupancy;
}

// mask bishop attacks
uint64_t mask_bishop_attacks(int square)
{
    uint64_t attacks = 0ULL;

    // init target files & ranks
    int tr = square / 8;
    int tf = square % 8;

    // generate attacks
    int f = 0;
    int r = 0;
    for (r = tr + 1, f = tf + 1; r <= 6 && f <= 6; r++, f++) attacks |= (1ULL << (r * 8 + f));
    for (r = tr + 1, f = tf - 1; r <= 6 && f >= 1; r++, f--) attacks |= (1ULL << (r * 8 + f));
    for (r = tr - 1, f = tf + 1; r >= 1 && f <= 6; r--, f++) attacks |= (1ULL << (r * 8 + f));
    for (r = tr - 1, f = tf - 1; r >= 1 && f >= 1; r--, f--) attacks |= (1ULL << (r * 8 + f));

    return attacks;
}

uint64_t mask_rook_attacks(int square)
{
    uint64_t attacks = 0ULL;

    // init target files & ranks
    int tr = square / 8;
    int tf = square % 8;

    // generate attacks
    int f = 0;
    int r = 0;
    for (r = tr + 1; r <= 6; r++) attacks |= (1ULL << (r * 8 + tf));
    for (r = tr - 1; r >= 1; r--) attacks |= (1ULL << (r * 8 + tf));
    for (f = tf + 1; f <= 6; f++) attacks |= (1ULL << (tr * 8 + f));
    for (f = tf - 1; f >= 1; f--) attacks |= (1ULL << (tr * 8 + f));

    // return attack map for bishop on a given square
    return attacks;
}

// bishop attacks
uint64_t bishop_attacks_on_the_fly(int square, uint64_t block)
{
    uint64_t attacks = 0ULL;

    // init target files & ranks
    int tr = square / 8;
    int tf = square % 8;

    // generate attacks
    int f = 0;
    int r = 0;
    for (r = tr + 1, f = tf + 1; r <= 7 && f <= 7; r++, f++) {
        attacks |= (1ULL << (r * 8 + f));
        if (block & (1ULL << (r * 8 + f))) break;
    }

    for (r = tr + 1, f = tf - 1; r <= 7 && f >= 0; r++, f--) {
        attacks |= (1ULL << (r * 8 + f));
        if (block & (1ULL << (r * 8 + f))) break;
    }

    for (r = tr - 1, f = tf + 1; r >= 0 && f <= 7; r--, f++) {
        attacks |= (1ULL << (r * 8 + f));
        if (block & (1ULL << (r * 8 + f))) break;
    }

    for (r = tr - 1, f = tf - 1; r >= 0 && f >= 0; r--, f--) {
        attacks |= (1ULL << (r * 8 + f));
        if (block & (1ULL << (r * 8 + f))) break;
    }

    return attacks;
}

// rook attacks
uint64_t rook_attacks_on_the_fly(int square, uint64_t block)
{
    uint64_t attacks = 0ULL;

    int tr = square / 8;
    int tf = square % 8;

    int f = 0;
    int r = 0;
    for (r = tr + 1; r <= 7; r++) {
        attacks |= (1ULL << (r * 8 + tf));
        if (block & (1ULL << (r * 8 + tf))) break;
    }

    for (r = tr - 1; r >= 0; r--) {
        attacks |= (1ULL << (r * 8 + tf));
        if (block & (1ULL << (r * 8 + tf))) break;
    }

    for (f = tf + 1; f <= 7; f++) {
        attacks |= (1ULL << (tr * 8 + f));
        if (block & (1ULL << (tr * 8 + f))) break;
    }

    for (f = tf - 1; f >= 0; f--) {
        attacks |= (1ULL << (tr * 8 + f));
        if (block & (1ULL << (tr * 8 + f))) break;
    }

    return attacks;
}

#if 0
uint64_t get_bishop_attacks(int square, uint64_t occupancy)
{
    occupancy &= bishop_masks[square];
    occupancy *=  bishop_magics[square];
    occupancy >>= 64 - bishop_rellevant_bits[square];

    return bishop_attacks[square][occupancy];

}

uint64_t get_rook_attacks(int square, uint64_t occupancy)
{
    occupancy &= rook_masks[square];
    occupancy *=  rook_magics[square];
    occupancy >>= 64 - rook_rellevant_bits[square];

    return rook_attacks[square][occupancy];
}
#endif

// must be called prior to using the magics
void init_sliders_attacks(PieceType piece)
{
    for (int square = 0; square < 64; square++) {
        bishop_masks[square] = mask_bishop_attacks(square);
        rook_masks[square] = mask_rook_attacks(square);

        uint64_t mask = piece == bishop
            ? mask_bishop_attacks(square)
            : mask_rook_attacks(square);

        int bit_count = count_bits(mask);

        int occupancy_variations = 1 << bit_count;

        for (int count = 0; count < occupancy_variations; count++) {
            if (piece == bishop) {
                uint64_t occupancy = set_occupancy(count, bit_count, mask);
                uint64_t magic_index = occupancy * bishop_magics[square] >> (64 - bishop_rellevant_bits[square]);
                bishop_attacks[square][magic_index] = bishop_attacks_on_the_fly(square, occupancy);
            } else { // rook
                uint64_t occupancy = set_occupancy(count, bit_count, mask);
                uint64_t magic_index = occupancy * rook_magics[square] >> (64 - rook_relevant_bits[square]);
                rook_attacks[square][magic_index] = rook_attacks_on_the_fly(square, occupancy);
            }
        }
    }
}


// only used when initially creating the magic tables
uint64_t find_magic(int square, int relevant_bits, int bishop) {
    uint64_t occupancies[4096];
    uint64_t attacks[4096];
    uint64_t used_attacks[4096];
    uint64_t mask_attack = bishop ? mask_bishop_attacks(square) : mask_rook_attacks(square);

    int occupancy_variations = 1 << relevant_bits;

    for (int count = 0; count < occupancy_variations; count++) {
        occupancies[count] = set_occupancy(count, relevant_bits, mask_attack);

        attacks[count] = bishop
            ? bishop_attacks_on_the_fly(square, occupancies[count])
            : rook_attacks_on_the_fly(square, occupancies[count]);
    }

    for (int random_count = 0; random_count < 100000000; random_count++) {
        uint64_t magic = random_fewbits();

        if (count_bits((mask_attack * magic) & 0xFF00000000000000ULL) < 6)
            continue;

        memset(used_attacks, 0ULL, sizeof(used_attacks));

        int fail = 0;
        for (int count = 0; !fail && count < occupancy_variations; count++) {
            int magic_index = (int)((occupancies[count] * magic) >> (64 - relevant_bits));

            if (used_attacks[magic_index] == 0ULL)
                used_attacks[magic_index] = attacks[count];
            else if(used_attacks[magic_index] != attacks[count])
                fail = 1;
        }

        if (!fail)
            return magic;
    }

    fmt::print("***Failed***\n");
    return 0ULL;
}

// only used when initially creating the magic tables
void init_magics()
{
    fmt::print("#pragma once\n\n");
    fmt::print("#include <cstdint>\n\n");
    fmt::print("static constexpr uint64_t rook_magics[64] = {{\n    ");
    for (int square = 0; square < 64; square++) {
        if (square && (square % 4 == 0))
            fmt::print("\n    ");
        fmt::print(" {:#018x},", find_magic(square, rook_relevant_bits[square], 0));
    }

    fmt::print("\n}};\n\n");
    fmt::print("static constexpr uint64_t bishop_magics[64] = {{\n    ");

    for (int square = 0; square < 64; square++) {
        if (square && (square % 4 == 0))
            fmt::print("\n    ");
        fmt::print(" {:#018x},", find_magic(square, bishop_rellevant_bits[square], 1));
    }

    fmt::print("\n}};\n\n");
}

}


#include "../board.hpp"
int test() {
    using namespace enyo;
    init_sliders_attacks(bishop);
    init_sliders_attacks(rook);

    bitboard_t line_bb[64][64] {};
    bitboard_t between_bb[64][64] {};
    (void)line_bb;
    (void)between_bb;
    for (square_t s1 = 0; s1 < 64; ++s1) {
        for (square_t s2 = 0; s2 < 64; ++s2) {
            if (get_bishop_attacks(s1, 0) & (1ULL << s2)) {
                //fmt::print("[{}][{}] have diagonal: {} -> {}\n", s1, s2, sq2str(s1), sq2str(s2));
                bitboard_t l1 = get_bishop_attacks(s1, 0) | 1ULL << s1;
                bitboard_t l2 = get_bishop_attacks(s2, 0) | 1ULL << s2;
                bitboard_t line = l1 & l2;
                line_bb[s1][s2] = line;
                //fmt::print("{}\n", visualize(line));

                bitboard_t b1 = get_bishop_attacks(s1, 1ULL << s2) | 1ULL << s1;
                bitboard_t b2 = get_bishop_attacks(s2, 1ULL << s1) | 1ULL << s2;
                bitboard_t between = b1 & b2;
                between_bb[s1][s2] = between;
                fmt::print("{}\n", visualize(between));
            }
            if (get_rook_attacks(s1, 0) & (1ULL << s2)) {
                //fmt::print("[{}][{}] have orthogonal: {} -> {}\n", s1, s2, sq2str(s1), sq2str(s2));
                bitboard_t l1 = get_rook_attacks(s1, 0) | 1ULL << s1;
                bitboard_t l2 = get_rook_attacks(s2, 0) | 1ULL << s2;
                bitboard_t line = l1 & l2;
                line_bb[s1][s2] = line;
                //fmt::print("{}\n", visualize(line));

                bitboard_t b1 = get_rook_attacks(s1, 1ULL << s2) | 1ULL << s1;
                bitboard_t b2 = get_rook_attacks(s2, 1ULL << s1) | 1ULL << s2;
                bitboard_t between = b1 & b2;
                between_bb[s1][s2] = between;
                fmt::print("{}\n", visualize(between));
            }
        }
    }
    return 0;
}

void test_magic()
{
    using namespace enyo;

    init_sliders_attacks(bishop);
    init_sliders_attacks(rook);

    // create bishop occupancy bitboard
    uint64_t bishop_occupancy = 0ULL;
    set_bit(bishop_occupancy, g7);
    set_bit(bishop_occupancy, f6);
    set_bit(bishop_occupancy, c5);
    set_bit(bishop_occupancy, b2);
    set_bit(bishop_occupancy, g1);

    fmt::print("     Bishop occupancy\n");
    fmt::print("{}\n", visualize(bishop_occupancy));

    fmt::print("     Bishop attacks\n");
    fmt::print("{}\n", visualize(get_bishop_attacks(d4, bishop_occupancy)));

    // create rook occupancy
    uint64_t rook_occupancy = 0ULL;
    set_bit(rook_occupancy, d7);
    set_bit(rook_occupancy, d6);
    set_bit(rook_occupancy, d3);
    set_bit(rook_occupancy, a4);
    set_bit(rook_occupancy, f4);

    fmt::print("     Rook occupancy\n");
    fmt::print("{}\n", visualize(rook_occupancy));

    // get rook attacks
    fmt::print("     Rook attacks\n");
    fmt::print("{}\n", visualize(get_rook_attacks(d4, rook_occupancy)));
}

#ifdef MAIN
int main()
{
    using namespace enyo;

    if constexpr (false)
        test_magic();
    else
        init_magics();
    return 0;
}
#endif

