#include <iostream>
#include <iostream>
#include <getopt.h>
#include <fmt/core.h>

#include "../util.hpp"
#include "../board.hpp"
#include "pawn_attacks.hpp"
#include "king_attacks.hpp"

using namespace enyo;


namespace {

void init_rook_attacks()
{
#if 0
    uint64_t idx = uint64_t(atoi(argv[1]));
    uint64_t rank = idx / 8;
    uint64_t file = idx % 8;
    fmt::print("rank: {}, file: {}\n", rank, file);
    fmt::print("rank: {:#018x}\n", 0xffULL << (rank * 8));
    fmt::print("file: {:#018x}\n", 0x0101010101010101ULL << file);
#else
    fmt::print("#pragma once\n\n");
    fmt::print("#include <cstdint>\n\n");
    fmt::print("constexpr uint64_t rook_attack_table[64] = {{\n    ");
    for (uint64_t square = 0; square < 64; square++) {
        uint64_t rank = square / 8;
        uint64_t file = square % 8;
        uint64_t rank_mask = 0xffULL << (rank * 8);
        uint64_t file_mask = 0x0101010101010101ULL << file;
        fmt::print("{:#018x}, ", (rank_mask | file_mask) & ~(1ULL << square));
        if (square == 63)
            fmt::print("\n}};\n");
        if (square && (square % 8) == 7)
            fmt::print("\n    ");
    }
#endif

}

void init_bishop_attacks()
{
    auto bishop_attacks = [](uint8_t srcBit) {
        uint64_t validMoves = 0;
        const int offsets[] = { 7, 9, -7, -9 };
        for (auto offset : offsets) {
            auto dstBit = (srcBit + offset);
            while (dstBit >= 0 && dstBit < 64) {
                auto const srcRow = 7 - (srcBit % 8);
                auto const srcCol = 7 - (srcBit / 8);
                auto const dstRow = 7 - (dstBit % 8);
                auto const dstCol = 7 - (dstBit / 8);
                // Check if the destination is on the same diagonal:
                if (std::abs(dstRow - srcRow) == std::abs(dstCol - srcCol)) {
                    validMoves |= (1ULL << dstBit);
                }
                dstBit += offset;
            }
        }
        return validMoves;
    };

    fmt::print("#pragma once\n\n");
    fmt::print("#include <cstdint>\n\n");
    fmt::print("constexpr uint64_t bishop_attack_table[64] = {{\n    ");
    for (square_t i = 0; i < 64; i++) {
        fmt::print("{:#018x},", bishop_attacks(i));
        //fmt::print("{:#018x} ", antidiag_mask(i));

        if (i == 63) fmt::print("\n");
        else if (i && (i % 8) == 7) fmt::print("\n    ");
    }
    fmt::print("}};\n");
}


/*
 8  . . . . . . . .
 7  . . . . . . . .
 6  . . . . . . . .
 5  . . . . . . . .
 4  . . . . . . . p
 3  . . . . . . x .
 2  . . . . . . . .
 1  . . . . . . . .
*/
uint64_t pawn_attack_mask(Color c, square_t sq)
{
    uint64_t attacks = 0;
    uint64_t bitboard = 0;

    set_bit(bitboard, sq);
    if (c == black) {
        if (bitboard & not_file_a) attacks |= bitboard << 9;
        if (bitboard & not_file_h) attacks |= bitboard << 7;
    } else {
        if (bitboard & not_file_a) attacks |= bitboard >> 7;
        if (bitboard & not_file_h) attacks |= bitboard >> 9;
        //if ((bitboard >> 15) & not_file_h) attacks |= bitboard >> 15;
        //if ((bitboard >> 17) & not_file_a) attacks |= bitboard >> 17;
    }

    return attacks;
}

[[maybe_unused]] uint64_t pawn_movement_mask(bool color, size_t square)
{
    uint64_t moves = 0;
    uint64_t bitboard = 0;

    set_bit(bitboard, square);
    if (color == white) {
        if (square < 8)
            return 0;
        if (square < 16) moves |= (bitboard << 16);
        moves |= (bitboard << 8);
    } else {
        if (square > 55)
            return 0;
        if ((square >= 48)) moves |= (bitboard >> 16);
        moves |= bitboard >> 8;
    }

    return moves;
}


[[maybe_unused]] void generate_enpassant_tables()
{
    fmt::print("constexpr uint64_t pawn_enpassant_attack[2][64] = {{\n    {{\n    ");
    if (1) { // white
        for (size_t square = 0; square < 64; square++) {
            auto const mask = 1ULL << square;

            if (!(mask & rank_5))
                fmt::print("{:#018x}, ", 0);
            else if (mask & file_a)
                fmt::print("{:#018x}, ", (1ULL << (square -1)));
            else if (mask & file_h)
                fmt::print("{:#018x}, ", (1ULL << (square +1)));
            else
                fmt::print("{:#018x}, ", (1ULL << (square +1)) | (1ULL << (square -1)) );

            if (square == 63)
                fmt::print("\n");
            else if (square % 8 == 7)
                fmt::print("\n    ");
        }
    }
    if (1) {
        fmt::print("    }}, {{\n    ");
        for (size_t square = 0; square < 64; square++) {
            auto const mask = 1ULL << square;

            if (!(mask & rank_4))
                fmt::print("{:#018x}, ", 0);
            else if (mask & file_a)
                fmt::print("{:#018x}, ", (1ULL << (square -1)));
            else if (mask & file_h)
                fmt::print("{:#018x}, ", (1ULL << (square +1)));
            else
                fmt::print("{:#018x}, ", (1ULL << (square +1)) | (1ULL << (square -1)) );

            if (square == 63)
                fmt::print("\n    }}\n");
            else if (square % 8 == 7)
                fmt::print("\n    ");
        }
    }
    fmt::print("}};\n");
}

void generate_pawn_tables()
{
    fmt::print("constexpr uint64_t pawn_attack_table[2][64] = {{\n    ");

    fmt::print("{{\n        ");
    for (square_t sq = 0; sq < 64; sq++) {
        fmt::print("{:#018x}, ", pawn_attack_mask(white, sq));
        if (sq == 63)
            fmt::print("\n");
        else if (sq % 8 == 7)
            fmt::print("\n        ");
    }
    fmt::print("    }}, {{\n        ");
    for (square_t sq = 0; sq < 64; sq++) {
        fmt::print("{:#018x}, ", pawn_attack_mask(black, sq));
        if (sq == 63)
            fmt::print("\n");
        else if (sq % 8 == 7)
            fmt::print("\n        ");
    }
    fmt::print("    }}\n}};\n");
}

/*
     0   1   2   3   4   5   6   7
   +---+---+---+---+---+---+---+---+
 8 |   |   |   |   |   |   |   |   |  56   st: [cp:1,en:1,ca:0,pr:0,ck:0,cm:0]
   +---+---+---+---+---+---+---+---+
 7 |   |   |   |   |   |   |   |   |  48   gs: [wk:0,wq:0,bk:0,bq:0]
   +---+---+---+---+---+---+---+---+
 6 |   |   | 9 |  8|  7|   |   |   |  40   ep: b6
   +---+---+---+---+---+---+---+---+
 5 |   |   | 1 | k | -1|   |   |   |  32
   +---+---+---+---+---+---+---+---+
 4 |   |   |-7 |-8 |-9 |   |   |   |  24
   +---+---+---+---+---+---+---+---+
 3 |   |   |   |   |   |   |   |   |  16
   +---+---+---+---+---+---+---+---+
 2 |   |   |   |   |   |   |   |   |  8
   +---+---+---+---+---+---+---+---+
 1 |   |   |   |   |   |   |   |   |  0
   +---+---+---+---+---+---+---+---+
     A   B   C   D   E   F   G   H
*/

[[maybe_unused]] void generate_king_moves() {
    fmt::print("constexpr uint64_t king_movement[64] = {{\n    ");
    for (size_t square = 0; square < 64; square++) {
        auto const mask = 1ULL << square;

        uint64_t bitboard = 0;

        if (!(mask & rank_8)) {
            if (!(mask & file_a))
                bitboard |= 1ULL << (square +9);
            bitboard |= 1ULL << (square +8);
            if (!(mask & file_h))
                bitboard |= 1ULL << (square +7);
        }

        if (!(mask & file_a))
            bitboard |= 1ULL << (square +1);

        if (!(mask & file_h))
            bitboard |= 1ULL << (square -1);

        if (!(mask & rank_1)) {
            if (!(mask & file_a))
                bitboard |= 1ULL << (square -7);
            bitboard |= 1ULL << (square -8);
            if (!(mask & file_h))
                bitboard |= 1ULL << (square -9);
        }

        fmt::print("{:#018x}, ", bitboard);
        //fmt::print("{} = {}\n", sq2str(square), visualize(bitboard));

        if (square == 63)
            fmt::print("\n");
        else if (square % 8 == 7)
            fmt::print("\n    ");
    }
    fmt::print("}};\n");
}


/*
     0   1   2   3   4   5   6   7
   +---+---+---+---+---+---+---+---+
 8 |   |   |   |   |   |   |   |   |  56   st: [cp:1,en:1,ca:0,pr:0,ck:0,cm:0]
   +---+---+---+---+---+---+---+---+
 7 |   |   | 17|   | 15|   |   |   |  48   gs: [wk:0,wq:0,bk:0,bq:0]
   +---+---+---+---+---+---+---+---+
 6 |   | 10|   |   |   | 6 |   |   |  40   ep: b6
   +---+---+---+---+---+---+---+---+
 5 |   |   |   | n |   |   |   |   |  32
   +---+---+---+---+---+---+---+---+
 4 |   | -6|   |   |   |-10|   |   |  24
   +---+---+---+---+---+---+---+---+
 3 |   |   |-15|   |-17|   |   |   |  16
   +---+---+---+---+---+---+---+---+
 2 |   |   |   |   |   |   |   |   |  8
   +---+---+---+---+---+---+---+---+
 1 |   |   |   |   |   |   |   |   |  0
   +---+---+---+---+---+---+---+---+
     A   B   C   D   E   F   G   H
*/
[[maybe_unused]] void generate_knight_moves() {
    fmt::print("constexpr uint64_t knight_attack_table[64] = {{\n    ");
    for (size_t square = 0; square < 64; square++) {
        auto const mask = 1ULL << square;

        uint64_t bitboard = 0;

        // up
        if (!(mask & (rank_7|rank_8))) {
            if (!(mask & file_a))
                bitboard |= 1ULL << (square +17);
            if (!(mask & file_h))
                bitboard |= 1ULL << (square +15);
        }

        if (!(mask & (rank_8))) {
            if (!(mask & (file_a|file_b)))
                bitboard |= 1ULL << (square +10);
            if (!(mask & (file_g|file_h)))
                bitboard |= 1ULL << (square +6);
        }
        // down
        if (!(mask & (rank_1|rank_2))) {
            if (!(mask & file_a))
                bitboard |= 1ULL << (square -15);
            if (!(mask & file_h))
                bitboard |= 1ULL << (square -17);
        }

        if (!(mask & (rank_1))) {
            if (!(mask & (file_a|file_b)))
                bitboard |= 1ULL << (square -6);
            if (!(mask & (file_g|file_h)))
                bitboard |= 1ULL << (square -10);
        }


        fmt::print("{:#018x}, ", bitboard);
        //fmt::print("{} = {}\n", sq2str(square), visualize(bitboard));

        if (square == 63)
            fmt::print("\n");
        else if (square % 8 == 7)
            fmt::print("\n    ");
    }
    fmt::print("}};\n");
}

}

int main(int argc, char **argv)
{
    int option;
    while ((option = getopt(argc, argv, "prb")) != -1) {
        switch (option) {
            case 'p':
                generate_pawn_tables();
                break;
            case 'r':
                init_rook_attacks();
                break;

            case 'b':
                init_bishop_attacks();
                break;

            default:
                // Handle unknown or error cases
                std::cerr << "Usage: " << argv[0] << " [-p,-r,-d value]\n";
                return 1;
        }
    }

    return 0;
}
