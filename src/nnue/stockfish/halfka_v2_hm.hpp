#pragma once

#include "stockfish_types.hpp"

namespace enyo {
class Board;
}

namespace NNUE::Stockfish::HalfKAv2Hm {

inline constexpr uint32_t hash_value = 0x7f234cb8u;
inline constexpr FeatureIndex dimensions = 22528;
inline constexpr size_t max_active_features = 32;
using ActiveFeatures = FeatureList<max_active_features>;

FeatureIndex MakeIndex(
    enyo::Color perspective,
    enyo::square_t square,
    enyo::PieceType piece,
    enyo::Color piece_color,
    enyo::square_t king_square);

ActiveFeatures GetActiveFeatures(const enyo::Board & board, enyo::Color perspective);

} // namespace NNUE::Stockfish::HalfKAv2Hm
