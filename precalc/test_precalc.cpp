#include <iostream>
#include <getopt.h>
#include <fmt/core.h>

//#include "../board.hpp"
#include "rook_attacks.hpp"
#include "bishop_attacks.hpp"
#include "../magic/magic.hpp"

#include "util.hpp"

using namespace enyo;

size_t count_set_bits(uint64_t n) {
    return static_cast<size_t>(__builtin_popcountll(n));
}


std::vector<uint64_t> create_blocker_bitboard(uint64_t movement_mask) {
    auto const bitboard = movement_mask;
    std::vector<int> moveSquareIndices;
    for (int i = 0; i < 64; i++) {
        if (((movement_mask >> i) & 1) == 1) {
            moveSquareIndices.push_back(i);
        }
    }

    size_t const num_patterns = 1 << moveSquareIndices.size();
    std::vector<uint64_t> blocker_bitboards(num_patterns, 0);

    // hvor mange bits det er i move
    for (size_t pattern_idx = 0; pattern_idx < num_patterns; pattern_idx++) {
        for (size_t bit_idx = 0; bit_idx < moveSquareIndices.size(); bit_idx++) {
            size_t bit = (pattern_idx >> bit_idx) & 1;
            blocker_bitboards[pattern_idx] |= (uint64_t(bit) << moveSquareIndices[bit_idx]);
        }
    }
    fmt::print("mask:\n{:064b}\n", bitboard);
    fmt::print("blocker_bitboards.size: {}\n", blocker_bitboards.size());
    for (auto const elt: blocker_bitboards) {
        //fmt::print("{:#018x}\n", elt);
        fmt::print("{:064b}\n", elt);
    }
    return blocker_bitboards;
}

void test_magic(int square) {
    uint64_t occupancy = 0;
    set_bit(occupancy, b7);
    set_bit(occupancy, g7);
    set_bit(occupancy, c5);
    set_bit(occupancy, b2);
    set_bit(occupancy, c3);
    set_bit(occupancy, e3);
    set_bit(occupancy, g1);

    uint64_t queen_attacks = get_bishop_attacks(square, occupancy) | get_rook_attacks(square, occupancy);
    fmt::print("queen attacks: {}\n", visualize(queen_attacks));
}


int main(int argc, char **argv) {
    int option;
    square_t sq = square_nb;
    bool o_specified = false;
    bool d_specified = false;

    init_sliders_attacks(bishop);
    init_sliders_attacks(rook);

    while ((option = getopt(argc, argv, "ods:")) != -1) {
        switch (option) {
            case 'o':
                o_specified = true;
                break;

            case 'd':
                d_specified = true;
                break;

            case 's':
                sq = str2sq(optarg);
                break;

            default:
                std::cerr << "Usage: " << argv[0] << " [-o] [-d] -s value\n";
                return 1;
        }
    }

    if (sq == square_nb) {
        fmt::print("Error: -s square option must be specified.\n");
        return 1;
    }

    if (o_specified)
        fmt::print("{}: {}\n", sq2str(sq), visualize(rook_attack_table[sq]));
    if (d_specified)
        fmt::print("{}: {}\n", sq2str(sq), visualize(bishop_attack_table[sq]));

    test_magic(sq);
    return 0;
}
