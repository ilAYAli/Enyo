#pragma once

#include <string>

namespace NNUE {

enum class LoadStatus {
    not_recognized,
    loaded,
    invalid,
};

struct LoadResult {
    LoadStatus status = LoadStatus::not_recognized;
    std::string error;
};

} // namespace NNUE
