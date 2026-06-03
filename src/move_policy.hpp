#pragma once

#include "board.hpp"
#include "types.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace enyo::move_policy {

enum class FeatureSet {
    compact,
    board,
};

struct Layer {
    std::string activation;
    std::vector<std::vector<double>> weight;
    std::vector<double> bias;
};

class Model {
public:
    bool load(std::string_view path, std::string * error = nullptr);
    [[nodiscard]] bool loaded() const { return loaded_; }
    [[nodiscard]] double threshold() const { return threshold_; }
    [[nodiscard]] int input_dim() const { return input_dim_; }
    [[nodiscard]] FeatureSet feature_set() const { return feature_set_; }

    double score(Board const & board, Move move) const;
    double margin(Board const & board, Move best, Move played) const;

private:
    bool loaded_ = false;
    FeatureSet feature_set_ = FeatureSet::compact;
    int input_dim_ = 0;
    double threshold_ = 0.0;
    std::vector<double> mean_;
    std::vector<double> std_;
    std::vector<Layer> layers_;
};

std::vector<double> features(Board const & board, Move move, FeatureSet feature_set);
const char * feature_set_name(FeatureSet feature_set);

} // namespace enyo::move_policy
