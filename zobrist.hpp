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

        for (auto column = 0; column < 8; column++) {
            enpassant_[column] = rng.rand<uint64_t>();
        }

        for (auto i = 0; i < 4; i++) {
            castling_[i] = rng.rand<uint64_t>();
        }

        side_ = rng.rand<uint64_t>();
    }

    uint64_t psq_[enyo::color_nb][enyo::piece_type_nb][64] {};
    uint64_t castling_[4] {};
    uint64_t enpassant_[8] {};
    uint64_t side_ {};
};

extern uint64_t generate_hash(enyo::Board const & b);

}

