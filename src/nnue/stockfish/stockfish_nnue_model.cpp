// Current Stockfish NNUE serialization and inference, adapted from e4a63548 (GPLv3).

#include "stockfish_nnue_model.hpp"

#include "nnue/binary_reader.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__AVX512BW__) || defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace NNUE::Stockfish {
namespace {

constexpr char leb128_magic[] = "COMPRESSED_LEB128";
constexpr size_t leb128_magic_size = sizeof(leb128_magic) - 1;

constexpr uint32_t RotateLeft(uint32_t value) {
    return (value << 1) | (value >> 31);
}

constexpr uint32_t AffineHash(uint32_t previous, uint32_t outputs) {
    uint32_t hash = 0xCC03DAE4u + outputs;
    hash ^= previous >> 1;
    hash ^= previous << 31;
    return hash;
}

constexpr uint32_t ActivationHash(uint32_t previous) {
    return 0x538D24C7u + previous;
}

template<typename T>
bool ReadLeb128(BinaryReader & reader, std::span<T> output, std::string & error) {
    static_assert(std::is_signed_v<T> && sizeof(T) <= 4);

    std::array<char, leb128_magic_size> magic{};
    if (!reader.read(magic.data(), magic.size(), "LEB128 magic")) {
        error = reader.error();
        return false;
    }
    if (!std::equal(magic.begin(), magic.end(), leb128_magic)) {
        error = "invalid LEB128 magic";
        return false;
    }

    uint32_t byte_count;
    if (!reader.read_little_endian(byte_count, "LEB128 byte count")) {
        error = reader.error();
        return false;
    }
    std::vector<uint8_t> encoded(byte_count);
    if (!reader.read(encoded.data(), encoded.size(), "LEB128 payload")) {
        error = reader.error();
        return false;
    }

    size_t cursor = 0;
    for (T & value : output) {
        uint64_t decoded = 0;
        unsigned shift = 0;
        uint8_t byte = 0;
        do {
            if (cursor == encoded.size() || shift >= 35) {
                error = "invalid LEB128 payload";
                return false;
            }
            byte = encoded[cursor++];
            decoded |= static_cast<uint64_t>(byte & 0x7f) << shift;
            shift += 7;
        } while (byte & 0x80);

        if ((byte & 0x40) && shift < 64)
            decoded |= (~uint64_t{0}) << shift;
        const int64_t signed_value = std::bit_cast<int64_t>(decoded);
        if (signed_value < std::numeric_limits<T>::min()
            || signed_value > std::numeric_limits<T>::max()) {
            error = "LEB128 value is out of range";
            return false;
        }
        value = static_cast<T>(signed_value);
    }

    if (cursor != encoded.size()) {
        error = "unused bytes in LEB128 payload";
        return false;
    }
    return true;
}

bool ReadSectionHash(BinaryReader & reader, uint32_t expected, const char * label, std::string & error) {
    const size_t offset = reader.position();
    uint32_t actual;
    if (!reader.read_little_endian(actual, label)) {
        error = reader.error();
        return false;
    }
    if (actual != expected) {
        char message[128];
        std::snprintf(
            message,
            sizeof(message),
            "%s mismatch at %zu: expected %08x, got %08x",
            label,
            offset,
            expected,
            actual);
        error = message;
        return false;
    }
    return true;
}

bool ReadLayer(
    BinaryReader & reader,
    LayerStack & layer,
    ArchitectureVariant architecture,
    std::string & error)
{
    for (int32_t & bias : layer.fc0_biases)
        if (!reader.read_little_endian(bias, "fc0 bias")) {
            error = reader.error();
            return false;
        }
    if (!reader.read(layer.fc0_weights.data(), layer.fc0_weights.size(), "fc0 weights")) {
        error = reader.error();
        return false;
    }
    for (size_t output = 0; output < 32; ++output)
        for (size_t input = 0; input < 1024; ++input) {
            const size_t source = output * 1024 + input;
            const size_t destination = (input / 4) * 32 * 4 + output * 4 + input % 4;
            layer.fc0_weights_sparse[destination] = layer.fc0_weights[source];
        }

    for (int32_t & bias : layer.fc1_biases)
        if (!reader.read_little_endian(bias, "fc1 bias")) {
            error = reader.error();
            return false;
        }
    if (!reader.read(layer.fc1_weights.data(), layer.fc1_weights.size(), "fc1 weights")) {
        error = reader.error();
        return false;
    }

    const size_t fc2_weights = architecture == ArchitectureVariant::sfnnv14 ? 32 : 128;
    if (!reader.read_little_endian(layer.fc2_bias, "fc2 bias")
        || !reader.read(layer.fc2_weights.data(), fc2_weights, "fc2 weights")) {
        error = reader.error();
        return false;
    }
    return true;
}

template<size_t Inputs, size_t Outputs>
std::array<int32_t, Outputs> Affine(
    const std::array<uint8_t, Inputs> & input,
    const std::array<int8_t, Inputs * Outputs> & weights,
    const std::array<int32_t, Outputs> & biases)
{
    std::array<int32_t, Outputs> output = biases;
    for (size_t out = 0; out < Outputs; ++out)
        for (size_t in = 0; in < Inputs; ++in)
            output[out] += static_cast<int32_t>(weights[out * Inputs + in]) * input[in];
    return output;
}

std::array<int32_t, 32> AffineFc0(
    const std::array<uint8_t, hidden_size> & input,
    const LayerStack & layer)
{
    std::array<int32_t, 32> output = layer.fc0_biases;

#if defined(__AVX512BW__)
    __m512i accumulators[2] = {
        _mm512_loadu_si512(layer.fc0_biases.data()),
        _mm512_loadu_si512(layer.fc0_biases.data() + 16),
    };
    const __m512i ones = _mm512_set1_epi16(1);
    for (size_t chunk = 0; chunk < hidden_size / 4; ++chunk) {
        uint32_t packed;
        std::memcpy(&packed, input.data() + chunk * 4, sizeof(packed));
        if (packed == 0)
            continue;
        const __m512i inputs = _mm512_set1_epi32(static_cast<int32_t>(packed));
        const int8_t * weights = layer.fc0_weights_sparse.data() + chunk * 128;
        for (size_t block = 0; block < 2; ++block) {
            const __m512i row = _mm512_loadu_si512(weights + block * 64);
#if defined(__AVX512VNNI__)
            accumulators[block] = _mm512_dpbusd_epi32(accumulators[block], inputs, row);
#else
            const __m512i products = _mm512_maddubs_epi16(inputs, row);
            accumulators[block] = _mm512_add_epi32(
                accumulators[block], _mm512_madd_epi16(products, ones));
#endif
        }
    }
    _mm512_storeu_si512(output.data(), accumulators[0]);
    _mm512_storeu_si512(output.data() + 16, accumulators[1]);
#elif defined(__AVX2__)
    __m256i accumulators[4] = {
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(layer.fc0_biases.data())),
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(layer.fc0_biases.data() + 8)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(layer.fc0_biases.data() + 16)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(layer.fc0_biases.data() + 24)),
    };
#if !defined(__AVXVNNI__)
    const __m256i ones = _mm256_set1_epi16(1);
#endif
    for (size_t chunk = 0; chunk < hidden_size / 4; ++chunk) {
        uint32_t packed;
        std::memcpy(&packed, input.data() + chunk * 4, sizeof(packed));
        if (packed == 0)
            continue;
        const __m256i inputs = _mm256_set1_epi32(static_cast<int32_t>(packed));
        const int8_t * weights = layer.fc0_weights_sparse.data() + chunk * 128;
        for (size_t block = 0; block < 4; ++block) {
            const __m256i row = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(weights + block * 32));
#if defined(__AVXVNNI__)
            accumulators[block] = _mm256_dpbusd_avx_epi32(
                accumulators[block], inputs, row);
#else
            const __m256i products = _mm256_maddubs_epi16(inputs, row);
            accumulators[block] = _mm256_add_epi32(
                accumulators[block], _mm256_madd_epi16(products, ones));
#endif
        }
    }
    for (size_t block = 0; block < 4; ++block)
        _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(output.data() + block * 8), accumulators[block]);
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t accumulators[8];
    for (size_t block = 0; block < 8; ++block)
        accumulators[block] = vld1q_s32(layer.fc0_biases.data() + block * 4);
    for (size_t chunk = 0; chunk < hidden_size / 4; ++chunk) {
        uint32_t packed;
        std::memcpy(&packed, input.data() + chunk * 4, sizeof(packed));
        if (packed == 0)
            continue;
        const int8x16_t inputs = vreinterpretq_s8_u32(vdupq_n_u32(packed));
        const int8_t * weights = layer.fc0_weights_sparse.data() + chunk * 128;
        for (size_t block = 0; block < 8; ++block)
            accumulators[block] = vdotq_s32(
                accumulators[block], vld1q_s8(weights + block * 16), inputs);
    }
    for (size_t block = 0; block < 8; ++block)
        vst1q_s32(output.data() + block * 4, accumulators[block]);
#else
    for (size_t chunk = 0; chunk < hidden_size / 4; ++chunk) {
        const uint8_t * values = input.data() + chunk * 4;
        uint32_t packed;
        std::memcpy(&packed, values, sizeof(packed));
        if (packed == 0)
            continue;
        const int8_t * weights = layer.fc0_weights_sparse.data() + chunk * 128;
        for (size_t out = 0; out < output.size(); ++out) {
            output[out] += static_cast<int32_t>(values[0]) * weights[out * 4]
                + static_cast<int32_t>(values[1]) * weights[out * 4 + 1]
                + static_cast<int32_t>(values[2]) * weights[out * 4 + 2]
                + static_cast<int32_t>(values[3]) * weights[out * 4 + 3];
        }
    }
#endif
    return output;
}

uint8_t ClippedRelu(int32_t value, int shift) {
    return static_cast<uint8_t>(std::clamp(value >> shift, 0, 127));
}

uint8_t SquaredClippedRelu(int32_t value, int shift) {
    const int64_t square = static_cast<int64_t>(value) * value;
    return static_cast<uint8_t>(std::min<int64_t>(127, square >> (2 * shift + 7)));
}

int32_t PropagateSfnnv14(
    const LayerStack & layer,
    const std::array<uint8_t, hidden_size> & input)
{
    const auto fc0 = AffineFc0(input, layer);
    std::array<uint8_t, 64> fc1_input{};
    for (size_t i = 0; i < 31; ++i) {
        fc1_input[i] = SquaredClippedRelu(fc0[i], 7);
        fc1_input[31 + i] = ClippedRelu(fc0[i], 7);
    }

    const auto fc1 = Affine<64, 32>(fc1_input, layer.fc1_weights, layer.fc1_biases);
    int32_t output = layer.fc2_bias;
    for (size_t i = 0; i < fc1.size(); ++i)
        output += static_cast<int32_t>(layer.fc2_weights[i]) * ClippedRelu(fc1[i], 6);
    output += fc0[31];
    return static_cast<int32_t>((static_cast<int64_t>(output) * 9600) / 16384);
}

int32_t PropagateSfnnv15(
    const LayerStack & layer,
    const std::array<uint8_t, hidden_size> & input)
{
    const auto fc0 = AffineFc0(input, layer);
    std::array<uint8_t, 64> fc1_input{};
    for (size_t i = 0; i < 32; ++i) {
        fc1_input[i] = SquaredClippedRelu(fc0[i], 7);
        fc1_input[32 + i] = ClippedRelu(fc0[i], 7);
    }

    const auto fc1 = Affine<64, 32>(fc1_input, layer.fc1_weights, layer.fc1_biases);
    std::array<uint8_t, 128> fc2_input{};
    std::copy(fc1_input.begin(), fc1_input.end(), fc2_input.begin());
    for (size_t i = 0; i < 32; ++i) {
        fc2_input[64 + i] = SquaredClippedRelu(fc1[i], 6);
        fc2_input[96 + i] = ClippedRelu(fc1[i], 6);
    }

    int32_t output = layer.fc2_bias;
    for (size_t i = 0; i < fc2_input.size(); ++i)
        output += static_cast<int32_t>(layer.fc2_weights[i]) * fc2_input[i];
    output += fc0[30] - fc0[31];
    return static_cast<int32_t>((static_cast<int64_t>(output) * 9600) / 16384);
}

int32_t Propagate(
    const LayerStack & layer,
    ArchitectureVariant architecture,
    const std::array<uint8_t, hidden_size> & input)
{
    return architecture == ArchitectureVariant::sfnnv14
        ? PropagateSfnnv14(layer, input)
        : PropagateSfnnv15(layer, input);
}

void AddWrapped(int16_t & destination, int value) {
    uint16_t bits = std::bit_cast<uint16_t>(destination);
    bits = static_cast<uint16_t>(bits + value);
    destination = std::bit_cast<int16_t>(bits);
}

} // namespace

uint32_t FeatureTransformerHash() {
    uint32_t hash = RotateLeft(FullThreats::hash_value) ^ HalfKAv2Hm::hash_value;
    return hash ^ (hidden_size * 2);
}

uint32_t ArchitectureHash() {
    uint32_t hash = 0xEC42E90Du ^ (hidden_size * 2);
    hash = AffineHash(hash, 32);
    hash = ActivationHash(hash);
    hash = AffineHash(hash, 32);
    hash = ActivationHash(hash);
    return AffineHash(hash, 1);
}

uint32_t NetworkHash() {
    return FeatureTransformerHash() ^ ArchitectureHash();
}

LoadResult Model::Load(const char * path) {
    BinaryReader reader(path);
    if (!reader)
        return {LoadStatus::invalid, reader.error()};

    uint32_t version;
    uint32_t network_hash;
    uint32_t description_size;
    if (!reader.read_little_endian(version, "Stockfish version"))
        return {LoadStatus::not_recognized, reader.error()};
    if (version != format_version)
        return {LoadStatus::not_recognized, {}};
    if (!reader.read_little_endian(network_hash, "Stockfish network hash")
        || !reader.read_little_endian(description_size, "Stockfish description size"))
        return {LoadStatus::invalid, reader.error()};
    if (network_hash != NetworkHash())
        return {LoadStatus::invalid, "unsupported Stockfish NNUE architecture hash"};
    if (description_size > 1 << 20)
        return {LoadStatus::invalid, "Stockfish NNUE description is too large"};

    description_.resize(description_size);
    if (!reader.read(description_.data(), description_.size(), "Stockfish description"))
        return {LoadStatus::invalid, reader.error()};

    std::string error;
    if (!ReadSectionHash(reader, FeatureTransformerHash(), "feature transformer hash", error))
        return {LoadStatus::invalid, error};

    halfka_weights_.resize(static_cast<size_t>(HalfKAv2Hm::dimensions) * hidden_size);
    threat_weights_.resize(static_cast<size_t>(FullThreats::dimensions) * hidden_size);
    halfka_psqt_weights_.resize(static_cast<size_t>(HalfKAv2Hm::dimensions) * output_buckets);
    threat_psqt_weights_.resize(static_cast<size_t>(FullThreats::dimensions) * output_buckets);

    if (!ReadLeb128<int16_t>(reader, biases_, error)
        || !reader.read(threat_weights_.data(), threat_weights_.size(), "threat weights")
        || !ReadLeb128<int32_t>(reader, threat_psqt_weights_, error)
        || !ReadLeb128<int16_t>(reader, halfka_weights_, error)
        || !ReadLeb128<int32_t>(reader, halfka_psqt_weights_, error))
        return {LoadStatus::invalid, error.empty() ? reader.error() : error};

    constexpr size_t sfnnv14_layer_bytes = 35112;
    constexpr size_t sfnnv15_layer_bytes = 35208;
    const size_t layer_bytes = reader.size() - reader.position();
    if (layer_bytes == sfnnv14_layer_bytes * output_buckets)
        architecture_ = ArchitectureVariant::sfnnv14;
    else if (layer_bytes == sfnnv15_layer_bytes * output_buckets)
        architecture_ = ArchitectureVariant::sfnnv15;
    else
        return {LoadStatus::invalid, "unsupported Stockfish NNUE layer-stack layout"};

    for (LayerStack & layer : layers_) {
        if (!ReadSectionHash(reader, ArchitectureHash(), "network architecture hash", error)
            || !ReadLayer(reader, layer, architecture_, error))
            return {LoadStatus::invalid, error};
    }
    if (!reader.at_end())
        return {LoadStatus::invalid, "trailing bytes after Stockfish NNUE payload"};
    return {LoadStatus::loaded, {}};
}

void Model::Reset(PerspectiveAccumulator & accumulator) const {
    accumulator.values = biases_;
    accumulator.psqt.fill(0);
}

void Model::UpdateHalfKa(
    PerspectiveAccumulator & accumulator, FeatureIndex index, int sign) const
{
    const size_t base = static_cast<size_t>(index) * hidden_size;
    size_t i = 0;
#if defined(__AVX512BW__)
    for (; i + 32 <= accumulator.values.size(); i += 32) {
        const __m512i current = _mm512_loadu_si512(accumulator.values.data() + i);
        const __m512i weights = _mm512_loadu_si512(halfka_weights_.data() + base + i);
        _mm512_storeu_si512(
            accumulator.values.data() + i,
            sign > 0
                ? _mm512_add_epi16(current, weights)
                : _mm512_sub_epi16(current, weights));
    }
#elif defined(__AVX2__)
    for (; i + 16 <= accumulator.values.size(); i += 16) {
        const __m256i current = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(accumulator.values.data() + i));
        const __m256i weights = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(halfka_weights_.data() + base + i));
        _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(accumulator.values.data() + i),
            sign > 0
                ? _mm256_add_epi16(current, weights)
                : _mm256_sub_epi16(current, weights));
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; i + 8 <= accumulator.values.size(); i += 8) {
        const int16x8_t current = vld1q_s16(accumulator.values.data() + i);
        const int16x8_t weights = vld1q_s16(halfka_weights_.data() + base + i);
        vst1q_s16(
            accumulator.values.data() + i,
            sign > 0 ? vaddq_s16(current, weights) : vsubq_s16(current, weights));
    }
#endif
    for (; i < accumulator.values.size(); ++i)
        AddWrapped(accumulator.values[i], sign * halfka_weights_[base + i]);
    const size_t psqt_base = static_cast<size_t>(index) * output_buckets;
    for (size_t bucket = 0; bucket < accumulator.psqt.size(); ++bucket)
        accumulator.psqt[bucket] += sign * halfka_psqt_weights_[psqt_base + bucket];
}

void Model::UpdateThreat(
    PerspectiveAccumulator & accumulator, FeatureIndex index, int sign) const
{
    const size_t base = static_cast<size_t>(index) * hidden_size;
    size_t i = 0;
#if defined(__AVX512BW__)
    for (; i + 32 <= accumulator.values.size(); i += 32) {
        const __m512i current = _mm512_loadu_si512(accumulator.values.data() + i);
        const __m256i bytes = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(threat_weights_.data() + base + i));
        const __m512i weights = _mm512_cvtepi8_epi16(bytes);
        _mm512_storeu_si512(
            accumulator.values.data() + i,
            sign > 0
                ? _mm512_add_epi16(current, weights)
                : _mm512_sub_epi16(current, weights));
    }
#elif defined(__AVX2__)
    for (; i + 16 <= accumulator.values.size(); i += 16) {
        const __m256i current = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(accumulator.values.data() + i));
        const __m128i bytes = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(threat_weights_.data() + base + i));
        const __m256i weights = _mm256_cvtepi8_epi16(bytes);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(accumulator.values.data() + i),
            sign > 0
                ? _mm256_add_epi16(current, weights)
                : _mm256_sub_epi16(current, weights));
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; i + 8 <= accumulator.values.size(); i += 8) {
        const int16x8_t current = vld1q_s16(accumulator.values.data() + i);
        const int16x8_t weights = vmovl_s8(vld1_s8(threat_weights_.data() + base + i));
        vst1q_s16(
            accumulator.values.data() + i,
            sign > 0 ? vaddq_s16(current, weights) : vsubq_s16(current, weights));
    }
#endif
    for (; i < accumulator.values.size(); ++i)
        AddWrapped(accumulator.values[i], sign * threat_weights_[base + i]);
    const size_t psqt_base = static_cast<size_t>(index) * output_buckets;
    for (size_t bucket = 0; bucket < accumulator.psqt.size(); ++bucket)
        accumulator.psqt[bucket] += sign * threat_psqt_weights_[psqt_base + bucket];
}

NetworkOutput Model::Evaluate(
    const std::array<PerspectiveAccumulator, 2> & accumulators,
    enyo::Color side,
    int pieces) const
{
    std::array<uint8_t, hidden_size> transformed{};
    const enyo::Color perspectives[2] = {side, ~side};
    for (size_t perspective = 0; perspective < 2; ++perspective) {
        const size_t color = static_cast<size_t>(perspectives[perspective]);
        const auto & values = accumulators[color].values;
        const size_t offset = perspective * hidden_size / 2;
        for (size_t i = 0; i < hidden_size / 2; ++i) {
            const int first = std::clamp<int>(values[i], 0, 255);
            const int second = std::clamp<int>(values[i + hidden_size / 2], 0, 255);
            transformed[offset + i] = static_cast<uint8_t>((first * second) / 512);
        }
    }

    const size_t bucket = static_cast<size_t>(
        std::clamp((pieces - 1) / 4, 0, output_buckets - 1));
    const size_t us = static_cast<size_t>(side);
    const size_t them = static_cast<size_t>(~side);
    const int psqt = (
        accumulators[us].psqt[bucket] - accumulators[them].psqt[bucket]) / 2;
    return {psqt / 16, Propagate(layers_[bucket], architecture_, transformed) / 16};
}

} // namespace NNUE::Stockfish
