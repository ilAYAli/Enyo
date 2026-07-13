#include "enyo_nn_loader.hpp"

#include "enyo_halfka_model.hpp"
#include "enyo_halfka_storage.hpp"
#include "nnue/binary_reader.hpp"

#include <array>
#include <cstdint>
#include <cstring>

namespace Network {
namespace {

bool IsSupportedTrainedHidden(int hidden) {
    for (const int supported : SUPPORTED_TRAINED_HIDDEN)
        if (hidden == supported)
            return true;
    return false;
}

bool IsSupportedOutputBuckets(int buckets) {
    return buckets == 1 || buckets == 2 || buckets == 4 || buckets == 8;
}

bool IsSupportedOutputHeadFeatures(int features) {
    return features == 0 || features == N_HEAD_FEATURES
        || features == N_EXTENDED_HEAD_FEATURES;
}

NNUE::LoadResult ReadLayout(NNUE::BinaryReader & reader, NetworkLayout & layout) {
    std::array<uint8_t, NETWORK_V2_HEADER_MAGIC.size()> magic{};
    if (!reader.read(magic.data(), magic.size(), "network magic"))
        return {NNUE::LoadStatus::not_recognized, {}};

    const bool is_v2_magic = std::memcmp(
        magic.data(), NETWORK_V2_HEADER_MAGIC.data(), magic.size()) == 0;
    const bool is_v3_magic = std::memcmp(
        magic.data(), NETWORK_V3_HEADER_MAGIC.data(), magic.size()) == 0;
    const bool is_v4_magic = std::memcmp(
        magic.data(), NETWORK_V4_HEADER_MAGIC.data(), magic.size()) == 0;
    const bool is_v5_magic = std::memcmp(
        magic.data(), NETWORK_V5_HEADER_MAGIC.data(), magic.size()) == 0;
    if (!is_v2_magic && !is_v3_magic && !is_v4_magic && !is_v5_magic) {
        layout = DetectNetworkLayout(reader.size());
        if (layout.input_buckets == 0)
            return {NNUE::LoadStatus::not_recognized, {}};
        reader.seek(0);
        return {NNUE::LoadStatus::loaded, {}};
    }

    uint32_t version;
    uint32_t header_size;
    uint32_t input_buckets;
    uint32_t feature_channels;
    uint32_t trained_hidden;
    uint32_t runtime_hidden;
    uint32_t l2_size;
    uint32_t l3_size;
    uint32_t output_buckets;
    uint32_t output_head_features;
    uint32_t flags;
    uint32_t payload_size;
    if (!reader.read_little_endian(version, "format version")
        || !reader.read_little_endian(header_size, "header size")
        || !reader.read_little_endian(input_buckets, "input buckets")
        || !reader.read_little_endian(feature_channels, "feature channels")
        || !reader.read_little_endian(trained_hidden, "trained hidden")
        || !reader.read_little_endian(runtime_hidden, "runtime hidden")
        || !reader.read_little_endian(l2_size, "L2 size")
        || !reader.read_little_endian(l3_size, "L3 size")
        || !reader.read_little_endian(output_buckets, "output buckets")
        || !reader.read_little_endian(output_head_features, "output head features")
        || !reader.read_little_endian(flags, "flags")
        || !reader.read_little_endian(payload_size, "payload size"))
        return {NNUE::LoadStatus::invalid, reader.error()};

    if ((is_v2_magic && version != NETWORK_FORMAT_VERSION)
        || (is_v3_magic && version != NETWORK_V3_FORMAT_VERSION)
        || (is_v4_magic && version != NETWORK_V4_FORMAT_VERSION)
        || (is_v5_magic && version != NETWORK_V5_FORMAT_VERSION)
        || header_size != NETWORK_HEADER_SIZE
        || runtime_hidden != N_HIDDEN
        || l2_size != N_L2
        || l3_size != N_L3
        || (flags & ~(NETWORK_FLAG_FULL_THREATS | NETWORK_FLAG_FULL_HEADS
                      | NETWORK_FLAG_MIXED_ACTIVATION | NETWORK_FLAG_PSQT_RESIDUAL)) != 0
        || !IsSupportedTrainedHidden(static_cast<int>(trained_hidden))
        || !IsSupportedFeatureLayout(
            static_cast<int>(input_buckets), static_cast<int>(feature_channels))
        || !IsSupportedOutputBuckets(static_cast<int>(output_buckets))
        || !IsSupportedOutputHeadFeatures(static_cast<int>(output_head_features)))
        return {NNUE::LoadStatus::invalid, "unsupported Enyo network header"};

    const bool full_threats = (flags & NETWORK_FLAG_FULL_THREATS) != 0;
    const bool full_heads = (flags & NETWORK_FLAG_FULL_HEADS) != 0;
    const bool mixed_activation = (flags & NETWORK_FLAG_MIXED_ACTIVATION) != 0;
    const bool psqt_residual = (flags & NETWORK_FLAG_PSQT_RESIDUAL) != 0;
    if (full_heads != is_v3_magic
        || (full_heads && (output_buckets <= 1 || full_threats || output_head_features != 0)))
        return {NNUE::LoadStatus::invalid, "unsupported Enyo full-head network header"};
    if (mixed_activation != is_v4_magic
        || (mixed_activation && (full_heads || full_threats || output_buckets != 8
                                 || output_head_features != 0)))
        return {NNUE::LoadStatus::invalid, "unsupported Enyo mixed-activation header"};
    if (psqt_residual != is_v5_magic
        || (psqt_residual && (full_heads || full_threats || mixed_activation
                             || output_buckets != 8 || output_head_features != 0)))
        return {NNUE::LoadStatus::invalid, "unsupported Enyo PSQT-residual header"};
    const size_t expected_payload = NetworkSize(
        static_cast<int>(input_buckets),
        static_cast<int>(output_buckets),
        static_cast<int>(output_head_features),
        static_cast<int>(feature_channels),
        full_threats,
        full_heads,
        mixed_activation,
        psqt_residual);
    if (payload_size != expected_payload
        || reader.size() != NETWORK_HEADER_SIZE + expected_payload)
        return {NNUE::LoadStatus::invalid, "Enyo network payload size does not match its header"};

    layout = {
        static_cast<int>(input_buckets),
        static_cast<int>(feature_channels),
        static_cast<int>(output_buckets),
        static_cast<int>(output_head_features),
        static_cast<int>(trained_hidden),
        NETWORK_HEADER_SIZE,
        full_threats,
        full_heads,
        mixed_activation,
        psqt_residual,
    };
    if (!reader.seek(NETWORK_HEADER_SIZE))
        return {NNUE::LoadStatus::invalid, reader.error()};
    return {NNUE::LoadStatus::loaded, {}};
}

} // namespace

static NNUE::LoadResult LoadEnyoNetwork(NNUE::BinaryReader & reader) {
    if (!reader)
        return {NNUE::LoadStatus::invalid, reader.error()};

    NetworkLayout layout{};
    const auto layout_result = ReadLayout(reader, layout);
    if (layout_result.status != NNUE::LoadStatus::loaded)
        return layout_result;

    EnyoStorage::Clear();
    const size_t input_weight_bytes = sizeof(acc_t)
        * static_cast<size_t>(InputFeatureCount(
            layout.input_buckets, layout.feature_channels, layout.full_threats))
        * N_HIDDEN;
    const int output_width = N_L3 + layout.output_head_features;
    const size_t head_count = layout.full_heads
        ? static_cast<size_t>(layout.output_buckets)
        : 1u;
    const size_t l1_weight_bytes = head_count * N_L1 * N_L2 * sizeof(int8_t);
    const size_t l1_bias_bytes = head_count * N_L2 * sizeof(int32_t);
    const size_t l2_weight_bytes = head_count * N_L2 * N_L3 * sizeof(float);
    const size_t l2_bias_bytes = head_count * N_L3 * sizeof(float);
    const size_t l2_squared_weight_bytes = layout.mixed_activation
        ? N_L2 * N_L3 * sizeof(float) : 0;
    const size_t l2_squared_bias_bytes = layout.mixed_activation
        ? N_L3 * sizeof(float) : 0;
    const size_t output_weight_bytes = sizeof(float)
        * static_cast<size_t>(layout.output_buckets)
        * static_cast<size_t>(output_width) * N_OUTPUT;
    const size_t output_bias_bytes = sizeof(float)
        * static_cast<size_t>(layout.output_buckets) * N_OUTPUT;
    const size_t psqt_weight_bytes = layout.psqt_residual
        ? sizeof(float) * static_cast<size_t>(InputFeatureCount(
            layout.input_buckets, layout.feature_channels, layout.full_threats))
            * static_cast<size_t>(layout.output_buckets)
        : 0;
    const size_t psqt_bias_bytes = layout.psqt_residual
        ? sizeof(float) * static_cast<size_t>(layout.output_buckets) : 0;

    if (!reader.read(EnyoStorage::input_weights, input_weight_bytes, "INPUT_WEIGHTS")
        || !reader.read(EnyoStorage::input_biases, sizeof(EnyoStorage::input_biases), "INPUT_BIASES")
        || !reader.read(EnyoStorage::l1_weights, l1_weight_bytes, "L1_WEIGHTS")
        || !reader.read(EnyoStorage::l1_biases, l1_bias_bytes, "L1_BIASES")
        || !reader.read(EnyoStorage::l2_weights, l2_weight_bytes, "L2_WEIGHTS")
        || !reader.read(EnyoStorage::l2_biases, l2_bias_bytes, "L2_BIASES")
        || !reader.read(
            EnyoStorage::l2_squared_weights,
            l2_squared_weight_bytes,
            "L2_SQUARED_WEIGHTS")
        || !reader.read(
            EnyoStorage::l2_squared_biases,
            l2_squared_bias_bytes,
            "L2_SQUARED_BIASES")
        || !reader.read(EnyoStorage::output_weights, output_weight_bytes, "OUTPUT_WEIGHTS")
        || !reader.read(EnyoStorage::output_biases, output_bias_bytes, "OUTPUT_BIASES")
        || !reader.read(EnyoStorage::psqt_weights, psqt_weight_bytes, "PSQT_WEIGHTS")
        || !reader.read(EnyoStorage::psqt_biases, psqt_bias_bytes, "PSQT_BIASES"))
        return {NNUE::LoadStatus::invalid, reader.error()};

    if (!reader.at_end())
        return {NNUE::LoadStatus::invalid, "trailing bytes after Enyo network payload"};

    EnyoStorage::Activate(layout, DenseLayerFormat::Float);
    return {NNUE::LoadStatus::loaded, {}};
}

NNUE::LoadResult LoadEnyoNetwork(const char * path) {
    NNUE::BinaryReader reader(path);
    return LoadEnyoNetwork(reader);
}

NNUE::LoadResult LoadEnyoNetwork(const unsigned char * data, size_t size) {
    NNUE::BinaryReader reader(data, size);
    return LoadEnyoNetwork(reader);
}

} // namespace Network
