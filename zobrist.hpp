#pragma once

#include <cstdint>
#include "types.hpp"


namespace enyo {
    class Board;
}

namespace zobrist {

constexpr bool debug = false;

class PRNG {
public:
    explicit PRNG(uint64_t seed)
    : s(seed) {
        assert(seed);
    }

    template<typename T>
    T rand() {
        return T(rand64());
    }

    template<typename T>
    T sparse_rand() {
        return T(rand64() & rand64() & rand64());
    }

private:
    // courtsey Stockfish:
    uint64_t rand64() {
        s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
        return s * 2685821657736338717LL;
    }
    uint64_t s;
};

struct zbrs {
    zbrs() {
        PRNG rng(1070372);

        for (size_t color = 0; color < size_t(enyo::color_nb); color++) {
            for (int piece = 0; piece < int(enyo::piece_type_nb); piece++) {
                for (int square = 0; square < 64; square++) {
                    psq_[color][piece][square] = rng.rand<uint64_t>();
                }
            }
        }

        for (int piece = 0; piece < 16; piece++) {
            for (int square = 0; square < 64; square++) {
                stockfish_psq_[piece][square] = 0;
            }
        }

        auto map_piece = [](size_t color, int pt) {
            if (pt == enyo::no_piece_type)
                return 0;

            if (color == enyo::white) {
                switch (pt) {
                    case enyo::pawn: return 1;
                    case enyo::knight: return 2;
                    case enyo::bishop: return 3;
                    case enyo::rook: return 4;
                    case enyo::queen: return 5;
                    case enyo::king: return 6;
                    default: return 0;
                }
            }

            switch (pt) {
                case enyo::pawn: return 9;
                case enyo::knight: return 10;
                case enyo::bishop: return 11;
                case enyo::rook: return 12;
                case enyo::queen: return 13;
                case enyo::king: return 14;
                default: return 0;
            }
        };

        for (size_t color = 0; color < size_t(enyo::color_nb); color++) {
            for (int pt = enyo::pawn; pt <= enyo::king; pt++) {
                const int sf_piece = map_piece(color, pt);
                for (int square = 0; square < 64; square++) {
                    const int sf_square = 7 - (square % 8) + 8 * (square / 8);
                    stockfish_psq_[sf_piece][sf_square] = psq_[color][pt][square];
                }
            }
        }

        for (auto column = 0; column < 8; column++) {
            enpassant_[column] = rng.rand<uint64_t>();
        }

        for (auto i = 0; i < enyo::CastlingRights::castling_right_nb; i++) {
            castling_[i] = rng.rand<uint64_t>();
        }

        side_ = rng.rand<uint64_t>();
    }

    uint64_t psq_[enyo::color_nb][enyo::piece_type_nb][64] {};
    uint64_t stockfish_psq_[16][64] {};
    uint64_t castling_[enyo::CastlingRights::castling_right_nb] {};
    uint64_t enpassant_[8] {};
    uint64_t side_ {};
};

extern uint64_t generate_hash(enyo::Board const & b);
extern uint64_t generate_stockfish_hash(enyo::Board const & b);

}
