#pragma once

#include "stockfish_types.hpp"

namespace enyo {
class Board;
}

namespace NNUE::Stockfish::FullThreats {

inline constexpr uint32_t hash_value = 0x8f234cb8u;
inline constexpr FeatureIndex dimensions = 60720;
inline constexpr size_t max_active_features = 128;
using ActiveFeatures = FeatureList<max_active_features>;

FeatureIndex MakeIndex(
    enyo::Color perspective,
    uint8_t attacker,
    int from,
    int to,
    uint8_t attacked,
    int king_square);

ActiveFeatures GetActiveFeatures(const enyo::Board & board, enyo::Color perspective);

} // namespace NNUE::Stockfish::FullThreats
