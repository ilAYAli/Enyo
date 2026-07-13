#include "berserk_nn_loader.hpp"

#include "enyo_halfka_model.hpp"
#include "enyo_halfka_storage.hpp"
#include "nnue/binary_reader.hpp"

namespace Network {

NNUE::LoadResult LoadBerserkNetwork(const char * path) {
    NNUE::BinaryReader reader(path);
    if (!reader)
        return {NNUE::LoadStatus::invalid, reader.error()};
    if (!IsSupportedQuantizedNetworkSize(reader.size()))
        return {NNUE::LoadStatus::not_recognized, {}};

    EnyoStorage::Clear();
    if (!reader.read(
            EnyoStorage::input_weights,
            sizeof(acc_t) * static_cast<size_t>(FeatureCount(16, LEGACY_FEATURE_CHANNELS)) * N_HIDDEN,
            "INPUT_WEIGHTS")
        || !reader.read(EnyoStorage::input_biases, sizeof(EnyoStorage::input_biases), "INPUT_BIASES")
        || !reader.read(
            EnyoStorage::l1_weights,
            sizeof(int8_t) * N_L1 * N_L2,
            "L1_WEIGHTS")
        || !reader.read(
            EnyoStorage::l1_biases,
            sizeof(int32_t) * N_L2,
            "L1_BIASES")
        || !reader.read(
            EnyoStorage::quantized_l2_weights,
            sizeof(EnyoStorage::quantized_l2_weights),
            "L2_WEIGHTS")
        || !reader.read(
            EnyoStorage::quantized_l2_biases,
            sizeof(EnyoStorage::quantized_l2_biases),
            "L2_BIASES")
        || !reader.read(
            EnyoStorage::quantized_output_weights,
            sizeof(EnyoStorage::quantized_output_weights),
            "OUTPUT_WEIGHTS")
        || !reader.read(
            &EnyoStorage::quantized_output_bias,
            sizeof(EnyoStorage::quantized_output_bias),
            "OUTPUT_BIAS"))
        return {NNUE::LoadStatus::invalid, reader.error()};

    if (!reader.at_end())
        return {NNUE::LoadStatus::invalid, "trailing bytes after Berserk network payload"};

    const NetworkLayout layout{16, LEGACY_FEATURE_CHANNELS, 1, 0, N_HIDDEN, 0};
    EnyoStorage::Activate(layout, DenseLayerFormat::Quantized);
    return {NNUE::LoadStatus::loaded, {}};
}

} // namespace Network
