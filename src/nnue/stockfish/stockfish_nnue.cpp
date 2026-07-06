#include "stockfish_nnue.hpp"

#include "board.hpp"
#include "full_threats.hpp"
#include "halfka_v2_hm.hpp"
#include "stockfish_nnue_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace NNUE::Stockfish {
namespace {

std::unique_ptr<Model> active_model;
uint64_t model_generation = 0;

struct PerspectiveFeatures {
    HalfKAv2Hm::ActiveFeatures halfka;
    FullThreats::ActiveFeatures threats;
};

struct PlyState {
    std::array<PerspectiveAccumulator, 2> accumulators;
    std::array<PerspectiveFeatures, 2> features;
    uint64_t hash = 0;
    uint64_t generation = 0;
    int eval = 0;
    bool valid = false;
    bool eval_valid = false;
};

template<size_t Capacity, typename Update>
void ApplyFeatureDifference(
    const FeatureList<Capacity> & previous,
    const FeatureList<Capacity> & current,
    Update update)
{
    size_t old_index = 0;
    size_t new_index = 0;
    while (old_index < previous.size || new_index < current.size) {
        if (new_index == current.size
            || (old_index < previous.size
                && previous.values[old_index] < current.values[new_index])) {
            update(previous.values[old_index++], -1);
        } else if (old_index == previous.size
            || current.values[new_index] < previous.values[old_index]) {
            update(current.values[new_index++], 1);
        } else {
            ++old_index;
            ++new_index;
        }
    }
}

std::array<PerspectiveFeatures, 2> GetFeatures(const enyo::Board & board) {
    std::array<PerspectiveFeatures, 2> features;
    for (int perspective = enyo::white; perspective <= enyo::black; ++perspective) {
        const auto color = static_cast<enyo::Color>(perspective);
        features[color].halfka = HalfKAv2Hm::GetActiveFeatures(board, color);
        features[color].threats = FullThreats::GetActiveFeatures(board, color);
    }
    return features;
}

void Refresh(
    PlyState & state,
    const std::array<PerspectiveFeatures, 2> & features)
{
    for (int perspective = enyo::white; perspective <= enyo::black; ++perspective) {
        auto & accumulator = state.accumulators[perspective];
        active_model->Reset(accumulator);
        for (size_t i = 0; i < features[perspective].halfka.size; ++i)
            active_model->UpdateHalfKa(
                accumulator, features[perspective].halfka.values[i], 1);
        for (size_t i = 0; i < features[perspective].threats.size; ++i)
            active_model->UpdateThreat(
                accumulator, features[perspective].threats.values[i], 1);
    }
}

void UpdateFromParent(
    PlyState & state,
    const PlyState & parent,
    const std::array<PerspectiveFeatures, 2> & features)
{
    state.accumulators = parent.accumulators;
    for (int perspective = enyo::white; perspective <= enyo::black; ++perspective) {
        auto & accumulator = state.accumulators[perspective];
        ApplyFeatureDifference(
            parent.features[perspective].halfka,
            features[perspective].halfka,
            [&](FeatureIndex index, int sign) {
                active_model->UpdateHalfKa(accumulator, index, sign);
            });
        ApplyFeatureDifference(
            parent.features[perspective].threats,
            features[perspective].threats,
            [&](FeatureIndex index, int sign) {
                active_model->UpdateThreat(accumulator, index, sign);
            });
    }
}

uint64_t ExpectedParentHash(const enyo::Board & board, const PlyState & root) {
    if (board.histply >= 2)
        return board.history[board.histply - 2].hash;
    return root.hash;
}

int NormalizeForEnyo(NetworkOutput output) {
    constexpr double stockfish_pawn_value = 208.0;
    return static_cast<int>(std::lround(
        (output.psqt + output.positional) * 100.0 / stockfish_pawn_value));
}

} // namespace

struct State::Impl {
    std::vector<PlyState> plies = std::vector<PlyState>(512);
};

LoadResult LoadNetwork(const char * path) {
    auto candidate = std::make_unique<Model>();
    const LoadResult result = candidate->Load(path);
    if (result.status != LoadStatus::loaded)
        return result;

    active_model = std::move(candidate);
    ++model_generation;
    return result;
}

void Disable() {
    active_model.reset();
    ++model_generation;
}

bool Enabled() {
    return active_model != nullptr;
}

std::string_view Description() {
    return active_model ? active_model->description() : std::string_view{};
}

State::State() = default;
State::~State() = default;

State::State(const State & other)
    : impl_(other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr)
{}

State & State::operator=(const State & other) {
    if (this != &other)
        impl_ = other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr;
    return *this;
}

State::State(State && other) noexcept = default;
State & State::operator=(State && other) noexcept = default;

void State::Prepare(const enyo::Board & board, size_t ply) {
    if (!active_model)
        return;
    if (!impl_)
        impl_ = std::make_unique<Impl>();
    if (ply >= impl_->plies.size())
        impl_->plies.resize(ply + 1);

    PlyState & state = impl_->plies[ply];
    if (state.valid && state.generation == model_generation && state.hash == board.hash)
        return;

    const auto features = GetFeatures(board);
    bool updated = false;
    if (ply > 0) {
        const PlyState & parent = impl_->plies[ply - 1];
        const PlyState & root = impl_->plies[0];
        if (parent.valid
            && parent.generation == model_generation
            && parent.hash == ExpectedParentHash(board, root)) {
            UpdateFromParent(state, parent, features);
            updated = true;
        }
    }
    if (!updated)
        Refresh(state, features);

    state.features = features;
    state.hash = board.hash;
    state.generation = model_generation;
    state.valid = true;
    state.eval_valid = false;
}

int State::Evaluate(const enyo::Board & board, size_t ply) {
    if (!active_model)
        return 0;
    Prepare(board, ply);

    PlyState & state = impl_->plies[ply];
    if (state.eval_valid)
        return state.eval;

    const int pieces = enyo::count_bits(board.color_bb[enyo::white] | board.color_bb[enyo::black]);
    state.eval = NormalizeForEnyo(active_model->Evaluate(state.accumulators, board.side, pieces));
    state.eval_valid = true;
    return state.eval;
}

void State::Clear() {
    impl_.reset();
}

} // namespace NNUE::Stockfish
