#pragma once

#include "nnue/load_result.hpp"

#include <cstddef>
#include <memory>
#include <string_view>

namespace enyo {
class Board;
}

namespace NNUE::Stockfish {

LoadResult LoadNetwork(const char * path);
void Disable();
bool Enabled();
std::string_view Description();

class State {
public:
    State();
    ~State();
    State(const State & other);
    State & operator=(const State & other);
    State(State && other) noexcept;
    State & operator=(State && other) noexcept;

    void Prepare(const enyo::Board & board, size_t ply);
    int Evaluate(const enyo::Board & board, size_t ply);
    void Clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace NNUE::Stockfish
