#pragma once
//#include "board.hpp"
#include "simd.h"
#include "types.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>
#include <iostream>


#define BUCKETS (16)
#define INPUT_SIZE (64 * 12 * BUCKETS)
#define HIDDEN_SIZE (512)
#define HIDDEN_DSIZE (HIDDEN_SIZE * 2)
#define OUTPUT_SIZE (1)

#define INPUT_QUANTIZATION (32)
#define HIDDEN_QUANTIZATON (128)

extern std::array<int16_t, INPUT_SIZE * HIDDEN_SIZE> inputWeights;
extern std::array<int16_t, HIDDEN_SIZE> inputBias;
extern std::array<int16_t, HIDDEN_SIZE * 2> hiddenWeights;
extern std::array<int, OUTPUT_SIZE> hiddenBias;

// PWA: remove this:
#pragma GCC diagnostic ignored "-Wsign-conversion"

namespace enyo {
    class Board;
}

namespace NNUE {
inline constexpr std::array<int, 64> KING_BUCKET {
    0,  1,  2,  3,  3,  2,  1,  0,
    4,  5,  6,  7,  7,  6,  5,  4,
    8,  9,  10, 11, 11, 10, 9,  8,
    8,  9,  10, 11, 11, 10, 9,  8,
    12, 12, 13, 13, 13, 13, 12, 12,
    12, 12, 13, 13, 13, 13, 12, 12,
    14, 14, 15, 15, 15, 15, 14, 14,
    14, 14, 15, 15, 15, 15, 14, 14,
};
}

namespace {

static inline int kingSquareIndex(enyo::square_t kingSquare, enyo::Color kingColor)
{
    kingSquare = enyo::square_t((56 * kingColor) ^ kingSquare);
    return NNUE::KING_BUCKET[kingSquare];
}

// !! A1 indexed
static inline int index(
    enyo::PieceType pieceType,
    enyo::Color pieceColor,
    enyo::square_t square,
    enyo::Color view,
    enyo::square_t kingSquare)
{
    const int piece_type =
        pieceType == enyo::no_piece_type
        ? 6 // NONETYPE
        : static_cast<int>(pieceType) -1;

    const int ksIndex = kingSquareIndex(kingSquare, view);
    square = enyo::square_t(square ^ (56 * view));
    square = enyo::square_t(square ^ (7 * !!(kingSquare & 0x4)));

    return square
           + piece_type * 64
           + !(pieceColor ^ view) * 64 * 6 + ksIndex * 64 * 6 * 2;
}

static inline int16_t relu(int16_t input)
{
    return std::max(static_cast<int16_t>(0), input);
}

} // anon ns


namespace NNUE {

struct Accumulator {
#if defined(USE_SIMD)
    alignas(ALIGNMENT) std::array<int16_t, HIDDEN_SIZE> white_acc;
    alignas(ALIGNMENT) std::array<int16_t, HIDDEN_SIZE> black_acc;
#else
    std::array<int16_t, HIDDEN_SIZE> white_acc;
    std::array<int16_t, HIDDEN_SIZE> black_acc;
#endif

    std::array<int16_t, HIDDEN_SIZE> &operator[](enyo::Color side)
    {
        return side == enyo::white
            ? white_acc
            : black_acc;
    }
    std::array<int16_t, HIDDEN_SIZE> &operator[](bool side)
    {
        return side
            ? black_acc
            : white_acc;
    }

    inline void copy(NNUE::Accumulator &acc) {
        std::copy(std::begin(acc.white_acc), std::end(acc.white_acc), std::begin(white_acc));
        std::copy(std::begin(acc.black_acc), std::end(acc.black_acc), std::begin(black_acc));
    }

    inline void clear() {
        std::copy(std::begin(inputBias), std::end(inputBias), std::begin(white_acc));
        std::copy(std::begin(inputBias), std::end(inputBias), std::begin(black_acc));
    }
};

struct Net {
    size_t currentAccumulator = 0;

    std::array<Accumulator, 512> accumulator_stack;

    Net();

    inline void push() {
        accumulator_stack[currentAccumulator + 1].copy(accumulator_stack[currentAccumulator]);
        currentAccumulator++;
        assert(currentAccumulator < accumulator_stack.size()
            && "currentAccumulator >= accumulator_stack.size()");
    }
    inline void pop() {
        currentAccumulator--;
        assert(currentAccumulator >= 0 && "currentAccumulator < 0");
    }
    inline void reset_accumulators() {
        currentAccumulator = 0;
    }

    void refresh(enyo::Board &board);

    template <bool add>
    void updateAccumulator(
        enyo::PieceType pieceType,
        enyo::Color pieceColor,
        enyo::square_t square,
        enyo::square_t kingSquare_White,
        enyo::square_t kingSquare_Black
    );

    void updateAccumulator(
        enyo::PieceType pieceType,
        enyo::Color pieceColor,
        enyo::square_t from_square,
        enyo::square_t to_square,
        enyo::square_t kingSquare_White,
        enyo::square_t kingSquare_Black
    );

    int Evaluate(enyo::Color side);

    void Benchmark();

    void print_n_accumulator_inputs(const Accumulator &accumulator, size_t N) {
        for (size_t i = 0; i < N; i++) {
            std::cout << accumulator.white_acc[i] << ", ";
        }

        std::cout << std::endl;

        for (size_t i = 0; i < N; i++) {
            std::cout << accumulator.black_acc[i] << ", ";
        }

        std::cout << std::endl;
    }

    void print_indexes(
        const enyo::Board &board,
        const enyo::PieceType pt,
        const enyo::square_t sq,
        enyo::square_t kingSquare);
};

void Init(const std::string &file_name);

} // namespace NNUE
