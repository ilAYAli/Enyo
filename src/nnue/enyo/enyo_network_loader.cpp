#include "berserk_nn_loader.hpp"
#include "enyo_halfka_model.hpp"
#include "enyo_nn_loader.hpp"

#include <cstdio>

namespace Network {

bool LoadNetwork(const char * path) {
    const auto enyo = LoadEnyoNetwork(path);
    if (enyo.status == NNUE::LoadStatus::loaded)
        return true;
    if (enyo.status == NNUE::LoadStatus::invalid) {
        std::fprintf(stderr, "network: %s: %s\n", path, enyo.error.c_str());
        return false;
    }

    const auto berserk = LoadBerserkNetwork(path);
    if (berserk.status == NNUE::LoadStatus::loaded)
        return true;
    if (berserk.status == NNUE::LoadStatus::invalid) {
        std::fprintf(stderr, "network: %s: %s\n", path, berserk.error.c_str());
        return false;
    }

    std::fprintf(stderr, "network: %s: unrecognized Enyo/Berserk network format\n", path);
    return false;
}

} // namespace Network
