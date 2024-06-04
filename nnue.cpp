#include "nnue.hpp"
#include "movelist.hpp"
#include "movegen.hpp"
//#include "bench.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#define INCBIN_STYLE INCBIN_STYLE_CAMEL
#include "incbin/incbin.h"
INCBIN(EVAL, EVAL_FILE);

#if defined(__AVX__) || defined(__AVX2__)
alignas(ALIGNMENT) std::array<int16_t, INPUT_SIZE * HIDDEN_SIZE> inputWeights;
alignas(ALIGNMENT) std::array<int16_t, HIDDEN_SIZE> inputBias;
alignas(ALIGNMENT) std::array<int16_t, HIDDEN_SIZE * 2> hiddenWeights;
alignas(ALIGNMENT) std::array<int32_t, OUTPUT_SIZE> hiddenBias;
#else
std::array<int16_t, INPUT_SIZE * HIDDEN_SIZE> inputWeights;
std::array<int16_t, HIDDEN_SIZE> inputBias;
std::array<int16_t, HIDDEN_SIZE * 2> hiddenWeights;
std::array<int32_t, OUTPUT_SIZE> hiddenBias;
#endif

// PWA: remove this:
#pragma GCC diagnostic ignored "-Wsign-conversion"

namespace NNUE {

Net::Net() { std::fill(accumulator_stack.begin(), accumulator_stack.end(), Accumulator()); }

template void Net::updateAccumulator<true>(
    enyo::PieceType,
    enyo::Color,
    enyo::square_t,
    enyo::square_t,
    enyo::square_t
);

template void Net::updateAccumulator<false>(
    enyo::PieceType,
    enyo::Color,
    enyo::square_t,
    enyo::square_t,
   enyo::square_t
);

template <bool add>
void Net::updateAccumulator(
    enyo::PieceType pieceType,
    enyo::Color pieceColor,
    enyo::square_t square,
    enyo::square_t kingSquare_White,
    enyo::square_t kingSquare_Black)
{
    Accumulator &accumulator = accumulator_stack[currentAccumulator];

    for (auto side : {enyo::white, enyo::black}) {
        const int inputs =
            index(
                pieceType,
                pieceColor,
                square,
                side,
                side == enyo::white
                ? kingSquare_White
                : kingSquare_Black
            );
        const auto weights = inputWeights.data() + inputs * HIDDEN_SIZE;

        if constexpr (add) {
            for (int i = 0; i < HIDDEN_SIZE / 4; ++i) {
                accumulator[side][i * 4 + 0] += weights[i * 4 + 0];
                accumulator[side][i * 4 + 1] += weights[i * 4 + 1];
                accumulator[side][i * 4 + 2] += weights[i * 4 + 2];
                accumulator[side][i * 4 + 3] += weights[i * 4 + 3];
            }
        } else {
            for (int i = 0; i < HIDDEN_SIZE / 4; ++i) {
                accumulator[side][i * 4 + 0] -= weights[i * 4 + 0];
                accumulator[side][i * 4 + 1] -= weights[i * 4 + 1];
                accumulator[side][i * 4 + 2] -= weights[i * 4 + 2];
                accumulator[side][i * 4 + 3] -= weights[i * 4 + 3];
            }
        }
    }
}

void Net::updateAccumulator(
    enyo::PieceType pieceType,
    enyo::Color pieceColor,
    enyo::square_t from_square,
    enyo::square_t to_square,
    enyo::square_t kingSquare_White,
    enyo::square_t kingSquare_Black)
{
    Accumulator &accumulator = accumulator_stack[currentAccumulator];
#if defined(USE_SIMD)
    for (auto side : {white, black}) {
        const int inputClear =
            index(
                pieceType,
                pieceColor,
                from_square,
                side,
                side == enyo::white
                ? kingSquare_White
                : kingSquare_Black
            );
        const int inputAdd =
            index(
                pieceType,
                pieceColor,
                to_square,
                side, side == enyo::white
                ? kingSquare_White
                : kingSquare_Black
            );

        const auto weightSub = reinterpret_cast<register_type16 *>(inputWeights.data() + inputClear * HIDDEN_SIZE);
        const auto weightAdd = reinterpret_cast<register_type16 *>(inputWeights.data() + inputAdd * HIDDEN_SIZE);
        const auto input = reinterpret_cast<register_type16 *>(accumulator[side].data());
        const auto output = reinterpret_cast<register_type16 *>(accumulator[side].data());

        for (int i = 0; i < HIDDEN_SIZE / STRIDE_16_BIT / 4; ++i) {
            output[i * 4 + 0] =
                register_add_epi16(register_sub_epi16(input[i * 4 + 0], weightSub[i * 4 + 0]), weightAdd[i * 4 + 0]);
            output[i * 4 + 1] =
                register_add_epi16(register_sub_epi16(input[i * 4 + 1], weightSub[i * 4 + 1]), weightAdd[i * 4 + 1]);
            output[i * 4 + 2] =
                register_add_epi16(register_sub_epi16(input[i * 4 + 2], weightSub[i * 4 + 2]), weightAdd[i * 4 + 2]);
            output[i * 4 + 3] =
                register_add_epi16(register_sub_epi16(input[i * 4 + 3], weightSub[i * 4 + 3]), weightAdd[i * 4 + 3]);
        }
    }
#else
    for (auto side : {enyo::white, enyo::black}) {
        const int inputClear =
            index(
                pieceType,
                pieceColor,
                from_square,
                side,
                side == enyo::white
                ? sqconv(kingSquare_White)
                : sqconv(kingSquare_Black)
            );
        const int inputAdd =
            index(
                pieceType,
                pieceColor,
                to_square,
                side,
                side == enyo::white
                ? sqconv(kingSquare_White)
                : sqconv(kingSquare_Black)
            );

        const auto weightSub = inputWeights.data() + inputClear * HIDDEN_SIZE;
        const auto weightAdd = inputWeights.data() + inputAdd * HIDDEN_SIZE;

        for (int i = 0; i < HIDDEN_SIZE / 4; ++i) {
            accumulator[side][i * 4 + 0] += static_cast<int16_t>(-weightSub[i * 4 + 0] + weightAdd[i * 4 + 0]);
            accumulator[side][i * 4 + 1] += static_cast<int16_t>(-weightSub[i * 4 + 1] + weightAdd[i * 4 + 1]);
            accumulator[side][i * 4 + 2] += static_cast<int16_t>(-weightSub[i * 4 + 2] + weightAdd[i * 4 + 2]);
            accumulator[side][i * 4 + 3] += static_cast<uint16_t>(-weightSub[i * 4 + 3] + weightAdd[i * 4 + 3]);
        }
    }
#endif
}

void Net::refresh(enyo::Board &board) {

    Accumulator &accumulator = accumulator_stack[currentAccumulator];
    accumulator.clear();

    uint64_t pieces = board.color_bb[enyo::white] | board.color_bb[enyo::black];
    while (pieces) {
        enyo::square_t sq = pop_lsb(pieces);
        enyo::PieceType pt = board.pt_mb[sq];
        auto color = board.color_bb[enyo::white] & (1ULL << sq) ? enyo::white : enyo::black;

        updateAccumulator<true>(
            pt,
            color,
            sq,
            lsb(board.pt_bb[enyo::white][enyo::king]),
            lsb(board.pt_bb[enyo::black][enyo::king])
        );
    }
}

int32_t Net::Evaluate(enyo::Color side) {
    Accumulator &accumulator = accumulator_stack[currentAccumulator];

#if defined(USE_SIMD)

    register_type16 reluBias{};
    register_type32 res{};

    const auto accumulator_us = reinterpret_cast<register_type16 *>(accumulator[side].data());
    const auto accumulator_them = reinterpret_cast<register_type16 *>(accumulator[!side].data());
    const auto weights = reinterpret_cast<register_type16 *>(hiddenWeights.data());

    for (int i = 0; i < HIDDEN_SIZE / STRIDE_16_BIT; i++) {
        res = register_add_epi32(res, register_madd_epi16(register_max_epi16(accumulator_us[i], reluBias), weights[i]));
    }

    for (int i = 0; i < HIDDEN_SIZE / STRIDE_16_BIT; i++) {
        res = register_add_epi32(res, register_madd_epi16(register_max_epi16(accumulator_them[i], reluBias), weights[i + HIDDEN_SIZE / STRIDE_16_BIT]));
    }

    const auto output = sumRegisterEpi32(res) + hiddenBias[0];
    return output / (INPUT_QUANTIZATION * HIDDEN_QUANTIZATON);

#else
    int32_t output = hiddenBias[0];

    for (int chunks = 0; chunks < HIDDEN_SIZE / 256; chunks++) {
        const int offset = chunks * 256;
        for (int i = 0; i < 256; i++) {
            output += relu(accumulator[side][i + offset]) * hiddenWeights[i + offset];
        }
    }

    for (int chunks = 0; chunks < HIDDEN_SIZE / 256; chunks++) {
        const int offset = chunks * 256;
        for (int i = 0; i < 256; i++) {
            output += relu(accumulator[!side][i + offset]) * hiddenWeights[HIDDEN_SIZE + i + offset];
        }
    }

    return output / INPUT_QUANTIZATION / HIDDEN_QUANTIZATON;

#endif
}

void Net::print_indexes(
    const enyo::Board &board,
    const enyo::PieceType pt,
    const enyo::square_t sq,
    enyo::square_t kingSquare)
{

    auto color = board.color_bb[enyo::white] & sq ? enyo::white : enyo::black;
    std::cout << "W [" << int(pt) << ", " << int(color) << ", " << int(sq)
              << "  ]: " << index(pt, color, sq, enyo::white, sqconv(kingSquare)) << "\n";
    std::cout << "B [" << int(pt) << ", " << int(color) << ", " << int(sq)
              << "  ]: " << index(pt, color, sq, enyo::black, sqconv(kingSquare)) << "\n";
}

#if 1
void Net::Benchmark()
{
    enyo::Board board{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"};

    std::cout << "Testing Evaluation speed...\n" << std::endl;

    float eval_time_sum = 0;
    float refresh_time_sum = 0;
    float update_time_sum = 0;

    std::vector<std::string> bench_fens = {
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    };
    for (auto &pos : bench_fens) {
        board.set(pos);

        long long time_sum = 0;
        int eval = 0;

        for (int i = 0; i < 1e7; i++) {
            auto start = std::chrono::steady_clock::now();
            eval = Evaluate(board.side);
            auto end = std::chrono::steady_clock::now();

            time_sum += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }

        float eval_time = static_cast<float>(time_sum) / 1e7f;
        eval_time_sum += eval_time;

        printf("%7.3f ns | eval: %4d | %-90s\n", eval_time, eval, pos.c_str());
    }

    std::cout << "\nTesting Refresh speed...\n" << std::endl;

    for (auto &pos : bench_fens) {
        board.set(pos);

        long long time_sum = 0;

        for (int i = 0; i < 1e6; i++) {
            auto start = std::chrono::steady_clock::now();
            refresh(board);
            auto end = std::chrono::steady_clock::now();

            time_sum += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }

        float refresh_time = static_cast<float>(time_sum) / 1e6f;
        refresh_time_sum += refresh_time;

        printf("%10.3f ns | %-90s\n", refresh_time, pos.c_str());
    }

    std::cout << "\nTesting Update speed...\n" << std::endl;

    for (auto &pos : bench_fens) {
        board.set(pos);

        auto const moves = generate_legal_moves<enyo::white>(board);

        long long time_sum = 0;

        // PWA: white only
        for (int iteration = 0; iteration < 1e5; iteration++) {
            for (auto move : moves) {
                auto start = std::chrono::steady_clock::now();

                apply_move<white>(board, move);
                revert_move<white>(board);

                auto end = std::chrono::steady_clock::now();

                time_sum += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            }
        }

        float update_time = static_cast<float>(time_sum) / static_cast<float>(moves.size()) / 1e5f;
        update_time_sum += update_time;

        printf("%10.3f ns | %-90s\n", update_time, pos.c_str());
    }

    std::cout << "\n";
    std::cout << "Average evaluation time: " << eval_time_sum / static_cast<float>(bench_fens.size()) << " ns" << std::endl;
    std::cout << "Average refresh time: " << refresh_time_sum / static_cast<float>(bench_fens.size()) << " ns" << std::endl;
    std::cout << "Average update time: " << update_time_sum / static_cast<float>(bench_fens.size()) << " ns" << std::endl;
}
#endif

void ReadBin() {
    uint64_t memoryIndex = 0;

    std::memcpy(inputWeights.data(), &gEVALData[memoryIndex], INPUT_SIZE * HIDDEN_SIZE * sizeof(int16_t));
    memoryIndex += INPUT_SIZE * HIDDEN_SIZE * sizeof(int16_t);

    std::memcpy(inputBias.data(), &gEVALData[memoryIndex], HIDDEN_SIZE * sizeof(int16_t));
    memoryIndex += HIDDEN_SIZE * sizeof(int16_t);

    std::memcpy(hiddenWeights.data(), &gEVALData[memoryIndex], HIDDEN_DSIZE * OUTPUT_SIZE * sizeof(int16_t));
    memoryIndex += HIDDEN_DSIZE * OUTPUT_SIZE * sizeof(int16_t);

    std::memcpy(hiddenBias.data(), &gEVALData[memoryIndex], OUTPUT_SIZE * sizeof(int32_t));
    memoryIndex += OUTPUT_SIZE * sizeof(int32_t);

#ifdef DEBUG
    std::cout << "Memory index: " << memoryIndex << std::endl;
    std::cout << "Size: " << gEVALSize << std::endl;
    std::cout << "Bias: " << hiddenBias[0] / INPUT_QUANTIZATION / HIDDEN_QUANTIZATON << std::endl;

    std::cout << std::endl;
#endif
}

void Init([[maybe_unused]] const std::string & file_name)
{
    ReadBin();
}

} // namespace NNUE
