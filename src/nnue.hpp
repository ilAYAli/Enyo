#pragma once
//#include "board.hpp"
#include "simd.h"
#include "types.hpp"
#include "nnue_model.hpp"

#include <array>
#include <cstddef>
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
inline constexpr size_t LEGACY_NETWORK_SIZE =
      INPUT_SIZE * HIDDEN_SIZE * sizeof(int16_t)
    + HIDDEN_SIZE * sizeof(int16_t)
    + HIDDEN_DSIZE * OUTPUT_SIZE * sizeof(int16_t)
    + OUTPUT_SIZE * sizeof(int32_t);

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

// Simplified Finny tables cache - stores accumulator state for each king square
// This cache enables efficient NNUE updates when the king moves by reusing
// previously computed accumulators for the new king position.
// The cache uses a simple hash of piece positions to validate entries.
struct AccumulatorCache {
    struct Entry {
        Accumulator acc;
        uint64_t pieces_hash;  // Hash of piece positions to detect cache validity
        bool valid;
        
        Entry() : pieces_hash(0), valid(false) {}
    };
    
    // Cache entry for each king square combination
    std::array<std::array<Entry, 64>, 2> entries;  // [color][king_square]
    
    void invalidate() {
        for (auto& color_entries : entries) {
            for (auto& entry : color_entries) {
                entry.valid = false;
            }
        }
    }
};

struct Net {
    struct NetworkEvalCacheEntry {
        uint64_t hash {};
        int32_t eval {};
        uint8_t valid {};
    };

    // Direct-mapped hash-indexed Evaluate2 cache. Earlier instrumentation
    // showed 19.8% hit rate at 1<<17 entries against ~3.2M computed evals
    // per 2M nodes — i.e. the table thrashes from collisions. Bumping
    // size is identity-safe: it only changes whether a Propagate runs,
    // not its result.
    static constexpr size_t network_eval_cache_size = 1 << 20;

    size_t currentAccumulator = 0;

    std::array<Accumulator, 512> accumulator_stack;
    std::vector<Network::Accumulator> network_accumulator_stack;
    std::array<Network::AccumulatorKingState, 2 * 2 * Network::N_KING_BUCKETS> network_refresh_table;
    std::vector<NetworkEvalCacheEntry> network_eval_cache;
    uint64_t network_refresh_generation = 0;
    uint64_t network_eval_cache_generation = 0;
    AccumulatorCache cache;

    Net();

    inline void push(bool copy_active_network = true) {
        if (Network::enabled && Network::INPUT_WEIGHTS != nullptr) {
            if (copy_active_network) {
                std::memcpy(&network_accumulator_stack[currentAccumulator + 1],
                            &network_accumulator_stack[currentAccumulator],
                            sizeof(Network::Accumulator));
            }
        } else {
            accumulator_stack[currentAccumulator + 1].copy(accumulator_stack[currentAccumulator]);
            if (Network::INPUT_WEIGHTS != nullptr) {
                std::memcpy(&network_accumulator_stack[currentAccumulator + 1],
                            &network_accumulator_stack[currentAccumulator],
                            sizeof(Network::Accumulator));
            }
        }
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
    void refresh_network(enyo::Board &board);
    void refresh_network(enyo::Board &board, enyo::Color side);
    void refresh_with_cache(enyo::Board &board);
    void update_cache(enyo::Board &board, enyo::square_t w_ksq, enyo::square_t b_ksq);

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
    // Fused single-pass accumulator update for a generic move (optionally
    // with a capture on to_square). Equivalent to the separate clr/clr/set
    // per-feature updates — int16 wrapping adds commute — but touches the
    // accumulator once instead of two or three times.
    void applyMoveDelta(
        enyo::PieceType pieceType,
        enyo::Color pieceColor,
        enyo::square_t from_square,
        enyo::square_t to_square,
        enyo::PieceType capturedPieceType,
        enyo::square_t kingSquare_White,
        enyo::square_t kingSquare_Black
    );
    void updateAccumulatorFromPrevious(
        enyo::PieceType pieceType,
        enyo::Color pieceColor,
        enyo::square_t from_square,
        enyo::square_t to_square,
        enyo::PieceType capturedPieceType,
        enyo::Color capturedPieceColor,
        enyo::square_t kingSquare_White,
        enyo::square_t kingSquare_Black
    );
    bool try_network_eager_move(enyo::Move move,
                              enyo::PieceType capturedPieceType,
                              enyo::square_t capturedSquare,
                              enyo::square_t kingSquare_White,
                              enyo::square_t kingSquare_Black);
    void mark_network_lazy_move(enyo::Move move,
                              enyo::square_t captured_square,
                              enyo::square_t kingSquare_White,
                              enyo::square_t kingSquare_Black);
    void ensure_network(enyo::Board &board);

    int Evaluate(enyo::Color side);
    int Evaluate2(enyo::Board &board, enyo::Color side);

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

// Enumerate active feature indices for a given perspective (view).
// Indices are in the same space the accumulator uses — each one is
// in [0, INPUT_SIZE) and corresponds to a row of inputWeights. Order
// matches Board's piece iteration: pop_lsb over color_bb[white] |
// color_bb[black], which is H1-indexed and LSB-first.
//
// Used by the `eval dump` UCI extension and by the Python parity
// test to verify that C++ and Python compute the same feature set
// (this is where king-bucket / mirror bugs manifest).
void feature_indices(const enyo::Board & board,
                     enyo::Color view,
                     std::vector<int> & out);

} // namespace NNUE
