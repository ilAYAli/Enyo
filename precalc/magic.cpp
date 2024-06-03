/*
generate magics:

c magic.cpp -DFIND_MAGICS && ./magic > magic_numbers.cpp

generate and test:
c magic.cpp -DFIND_MAGICS -DMAIN && ./a.out > magic_numbers.cpp && c magic.cpp -D MAIN && ./a.out

 */
#include "fmt/format.h"

#include "magic_numbers.cpp"
#include "../board.hpp"

namespace enyo {

// rook rellevant occupancy bits
int rook_rellevant_bits[64] = {
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12
};

// bishop rellevant occupancy bits
int bishop_rellevant_bits[64] = {
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6
};

uint64_t bishop_masks[64];
uint64_t rook_masks[64];

uint64_t bishop_attacks[64][512];
uint64_t rook_attacks[64][4096];

uint64_t set_occupancy(int index, int bits_in_mask, uint64_t attack_mask)
{
    uint64_t occupancy = 0ULL;

    for (int count = 0; count < bits_in_mask; count++) {
        int square = get_ls1b_index(attack_mask);

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
                uint64_t magic_index = occupancy * rook_magics[square] >> (64 - rook_rellevant_bits[square]);
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
    fmt::print("static constexpr uint64_t rook_magics[64] = {{\n");

    for (int square = 0; square < 64; square++) {
        fmt::print("    {:#018x},\n", find_magic(square, rook_rellevant_bits[square], 0));
        if (square && (square % 4 == 0))
            fmt::print("\n");
    }

    fmt::print("}};\n\nstatic constexpr uint64_t bishop_magics[64] = {{\n");

    for (int square = 0; square < 64; square++) {
        fmt::print("    {:#018x},\n", find_magic(square, bishop_rellevant_bits[square], 1));
        if (square && (square % 4 == 0))
            fmt::print("\n");
    }

    fmt::print("}};\n\n");
}


}

#ifdef MAIN
// main driver
int main()
{
    using namespace enyo;
#ifdef FIND_MAGICS
    init_magics();
    return 0;
#endif

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

    return 0;
}

#endif
