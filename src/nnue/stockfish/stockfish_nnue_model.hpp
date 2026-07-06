#pragma once

#include "full_threats.hpp"
#include "halfka_v2_hm.hpp"
#include "nnue/load_result.hpp"
#include "stockfish_types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace NNUE::Stockfish {

struct PerspectiveAccumulator {
    alignas(64) std::array<int16_t, hidden_size> values{};
    std::array<int32_t, output_buckets> psqt{};
};

struct NetworkOutput {
    int psqt = 0;
    int positional = 0;
};

struct LayerStack {
    std::array<int32_t, 32> fc0_biases{};
    std::array<int8_t, 32 * 1024> fc0_weights{};
    alignas(64) std::array<int8_t, 32 * 1024> fc0_weights_sparse{};
    std::array<int32_t, 32> fc1_biases{};
    std::array<int8_t, 32 * 64> fc1_weights{};
    int32_t fc2_bias = 0;
    std::array<int8_t, 128> fc2_weights{};
};

enum class ArchitectureVariant {
    sfnnv14,
    sfnnv15,
};

class Model {
public:
    LoadResult Load(const char * path);

    void Reset(PerspectiveAccumulator & accumulator) const;
    void UpdateHalfKa(
        PerspectiveAccumulator & accumulator, FeatureIndex index, int sign) const;
    void UpdateThreat(
        PerspectiveAccumulator & accumulator, FeatureIndex index, int sign) const;
    NetworkOutput Evaluate(
        const std::array<PerspectiveAccumulator, 2> & accumulators,
        enyo::Color side,
        int pieces) const;

    const std::string & description() const { return description_; }

private:
    std::array<int16_t, hidden_size> biases_{};
    std::vector<int16_t> halfka_weights_;
    std::vector<int8_t> threat_weights_;
    std::vector<int32_t> halfka_psqt_weights_;
    std::vector<int32_t> threat_psqt_weights_;
    std::array<LayerStack, output_buckets> layers_{};
    ArchitectureVariant architecture_ = ArchitectureVariant::sfnnv15;
    std::string description_;
};

inline constexpr uint32_t format_version = 0x6A448AFAu;
uint32_t FeatureTransformerHash();
uint32_t ArchitectureHash();
uint32_t NetworkHash();

} // namespace NNUE::Stockfish
