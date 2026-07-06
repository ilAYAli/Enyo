#pragma once

#include "nnue/load_result.hpp"

#include <cstddef>

namespace Network {

NNUE::LoadResult LoadEnyoNetwork(const char * path);
NNUE::LoadResult LoadEnyoNetwork(const unsigned char * data, size_t size);

} // namespace Network
