#pragma once

#include "types.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace NNUE::Stockfish {

using FeatureIndex = uint32_t;

constexpr int square_count = 64;
constexpr int piece_count = 16;
constexpr int output_buckets = 8;
constexpr int hidden_size = 1024;

constexpr int ToStockfishSquare(enyo::square_t square) {
    return static_cast<int>(square) ^ 7;
}

constexpr uint8_t ToStockfishPiece(enyo::Color color, enyo::PieceType piece) {
    return static_cast<uint8_t>(
        static_cast<int>(piece) + (static_cast<int>(color) << 3));
}

constexpr int PieceType(uint8_t piece) {
    return piece & 7;
}

constexpr int PieceColor(uint8_t piece) {
    return piece >> 3;
}

template<size_t Capacity>
struct FeatureList {
    std::array<FeatureIndex, Capacity> values{};
    size_t size = 0;

    void push(FeatureIndex index) {
        assert(size < values.max_size());
        values[size++] = index;
    }

    void sort() {
        std::sort(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(size));
    }
};

} // namespace NNUE::Stockfish
