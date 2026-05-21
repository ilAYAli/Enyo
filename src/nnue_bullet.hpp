#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace enyo { class Board; }

namespace BulletNetwork {

inline constexpr int N_INPUT_BUCKETS = 10;
inline constexpr int N_OUTPUT_BUCKETS = 8;
inline constexpr int N_INPUTS = 768 * N_INPUT_BUCKETS;
inline constexpr int N_HIDDEN = 1024;
inline constexpr int N_L2 = 16;
inline constexpr int N_L3 = 32;

inline constexpr size_t PADDING_SIZE = 32;

inline constexpr size_t PayloadSizeForHidden(int hidden)
{
    const size_t h = static_cast<size_t>(hidden);
    return sizeof(int16_t) * static_cast<size_t>(N_INPUTS) * h
        + sizeof(int16_t) * h
        + sizeof(int8_t) * static_cast<size_t>(N_OUTPUT_BUCKETS) * N_L2 * h
        + sizeof(float) * N_OUTPUT_BUCKETS * N_L2
        + sizeof(float) * static_cast<size_t>(N_OUTPUT_BUCKETS) * N_L3 * N_L2
        + sizeof(float) * N_OUTPUT_BUCKETS * N_L3
        + sizeof(float) * N_OUTPUT_BUCKETS * N_L3
        + sizeof(float) * N_OUTPUT_BUCKETS;
}

inline constexpr size_t NetworkSizeForHidden(int hidden)
{
    return PayloadSizeForHidden(hidden) + PADDING_SIZE;
}

inline constexpr bool IsNetworkSize(size_t size)
{
    return size == NetworkSizeForHidden(128)
        || size == NetworkSizeForHidden(256)
        || size == NetworkSizeForHidden(384)
        || size == NetworkSizeForHidden(512)
        || size == NetworkSizeForHidden(768)
        || size == NetworkSizeForHidden(1024);
}

extern bool enabled;
extern uint64_t NETWORK_GENERATION;

struct alignas(64) Accumulator {
    int16_t values[2][N_HIDDEN];
};

bool IsLoaded();
bool LoadNetwork(const char* path);
void CopyAccumulator(Accumulator* dest, const Accumulator* src);
int FeatureIdx(enyo::PieceType pt,
               enyo::Color pc,
               enyo::square_t sq,
               enyo::square_t king_sq,
               enyo::Color view);
void ResetAccumulator(Accumulator* acc, const enyo::Board& board, enyo::Color view);
void UpdateFeature(Accumulator* acc,
                   enyo::PieceType pt,
                   enyo::Color pc,
                   enyo::square_t sq,
                   enyo::square_t king_sq,
                   enyo::Color view,
                   bool add);
void MoveFeature(Accumulator* acc,
                 enyo::PieceType pt,
                 enyo::Color pc,
                 enyo::square_t from,
                 enyo::square_t to,
                 enyo::square_t king_sq,
                 enyo::Color view);
int Propagate(const Accumulator* acc, const enyo::Board& board);
int EvaluateFromScratch(const enyo::Board& board);
std::string Description();

} // namespace BulletNetwork
