// Network weight storage and network loading.
// Defines the extern storage for the Enyo HalfKA model.
// The format-specific loaders are split out after the model move is verified.
//
// This TU is intentionally independent of the engine's Board type so
// Phase 2/3 parity tests can compile it directly. The Board-aware bits
// live in enyo_halfka_board.cpp.

#include "enyo_halfka_model.hpp"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>

namespace Network {

// ---------------------------------------------------------------------
// Internal weight storage. Populated by LoadNetwork() from the .nn blob;
// the extern pointers below are repointed to reference these arrays.
// ---------------------------------------------------------------------
namespace {
alignas(64) int16_t s_input_weights [N_FEATURES * N_HIDDEN];
alignas(64) int16_t s_input_biases  [N_HIDDEN];
alignas(64) int8_t  s_l1_weights    [N_L1 * N_L2];
alignas(64) int8_t  s_l1_weights_t  [N_L1 * N_L2];
alignas(64) int8_t  s_l1_weights_sparse[N_L1 * N_L2];
alignas(64) int32_t s_l1_biases     [N_L2];
alignas(64) float   s_l2_weights    [N_L2 * N_L3];
alignas(64) float   s_l2_biases     [N_L3];
alignas(64) float   s_output_weights[MAX_OUTPUT_BUCKETS * MAX_OUTPUT_WIDTH * N_OUTPUT];
alignas(64) float   s_output_biases [MAX_OUTPUT_BUCKETS * N_OUTPUT];
alignas(64) int16_t s_quantized_l2_weights[N_L2 * N_L3];
alignas(64) int32_t s_quantized_l2_biases[N_L3];
alignas(64) int16_t s_quantized_output_weights[N_L3 * N_OUTPUT];
alignas(64) int32_t s_quantized_output_bias;

uint32_t ReadU32LE(const uint8_t* bytes) {
    return static_cast<uint32_t>(bytes[0])
        | (static_cast<uint32_t>(bytes[1]) << 8)
        | (static_cast<uint32_t>(bytes[2]) << 16)
        | (static_cast<uint32_t>(bytes[3]) << 24);
}

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

bool ReadNetworkLayout(std::FILE* fh, size_t size, NetworkLayout& layout) {
    std::array<uint8_t, NETWORK_HEADER_SIZE> header{};
    std::rewind(fh);
    if (std::fread(header.data(), 1, NETWORK_HEADER_MAGIC.size(), fh)
        != NETWORK_HEADER_MAGIC.size())
        return false;

    if (std::memcmp(
            header.data(), NETWORK_HEADER_MAGIC.data(), NETWORK_HEADER_MAGIC.size()) != 0) {
        layout = DetectNetworkLayout(size);
        std::rewind(fh);
        return layout.input_buckets != 0;
    }

    if (std::fread(
            header.data() + NETWORK_HEADER_MAGIC.size(),
            1,
            NETWORK_HEADER_SIZE - NETWORK_HEADER_MAGIC.size(),
            fh) != NETWORK_HEADER_SIZE - NETWORK_HEADER_MAGIC.size())
        return false;

    const uint32_t version = ReadU32LE(&header[8]);
    const uint32_t header_size = ReadU32LE(&header[12]);
    const int input_buckets = static_cast<int>(ReadU32LE(&header[16]));
    const int feature_channels = static_cast<int>(ReadU32LE(&header[20]));
    const int trained_hidden = static_cast<int>(ReadU32LE(&header[24]));
    const int runtime_hidden = static_cast<int>(ReadU32LE(&header[28]));
    const int l2_size = static_cast<int>(ReadU32LE(&header[32]));
    const int l3_size = static_cast<int>(ReadU32LE(&header[36]));
    const int output_buckets = static_cast<int>(ReadU32LE(&header[40]));
    const int output_head_features = static_cast<int>(ReadU32LE(&header[44]));
    const uint32_t flags = ReadU32LE(&header[48]);
    const size_t payload_size = ReadU32LE(&header[52]);

    if (version != NETWORK_FORMAT_VERSION
        || header_size != NETWORK_HEADER_SIZE
        || runtime_hidden != N_HIDDEN
        || l2_size != N_L2
        || l3_size != N_L3
        || flags != 0
        || !IsSupportedTrainedHidden(trained_hidden)
        || !IsSupportedFeatureLayout(input_buckets, feature_channels)
        || !IsSupportedOutputBuckets(output_buckets)
        || !IsSupportedOutputHeadFeatures(output_head_features))
        return false;

    const size_t expected_payload = NetworkSize(
        input_buckets,
        output_buckets,
        output_head_features,
        feature_channels);
    if (payload_size != expected_payload || size != NETWORK_HEADER_SIZE + expected_payload)
        return false;

    layout = {
        input_buckets,
        feature_channels,
        output_buckets,
        output_head_features,
        trained_hidden,
        NETWORK_HEADER_SIZE,
    };
    return true;
}
}

// Weight-pointer definitions referenced from network.hpp. Start as nullptr;
// either SetWeights (tests) or LoadNetwork (runtime) repoints them.
const acc_t*  INPUT_WEIGHTS  = nullptr;
const acc_t*  INPUT_BIASES   = nullptr;
const int8_t* L1_WEIGHTS     = nullptr;
const int8_t* L1_WEIGHTS_T   = nullptr;
const int8_t* L1_WEIGHTS_SPARSE = nullptr;
alignas(16) uint16_t LOOKUP_INDICES[256][8];
const int32_t* L1_BIASES     = nullptr;
const float*  L2_WEIGHTS     = nullptr;
const float*  L2_BIASES      = nullptr;
const float*  OUTPUT_WEIGHTS = nullptr;
const float*  OUTPUT_BIASES  = nullptr;
float         OUTPUT_BIAS    = 0.0f;
const int16_t* QUANTIZED_L2_WEIGHTS = nullptr;
const int32_t* QUANTIZED_L2_BIASES = nullptr;
const int16_t* QUANTIZED_OUTPUT_WEIGHTS = nullptr;
int32_t        QUANTIZED_OUTPUT_BIAS = 0;
DenseLayerFormat DENSE_LAYER_FORMAT = DenseLayerFormat::Float;
bool          INPUT_LAYOUT_SCRAMBLED = false;
uint64_t      NETWORK_GENERATION = 0;
int           INPUT_BUCKETS = DEFAULT_INPUT_BUCKETS;
int           FEATURE_CHANNELS = DEFAULT_FEATURE_CHANNELS;
int           TRAINED_HIDDEN = N_HIDDEN;
int           OUTPUT_BUCKETS = DEFAULT_OUTPUT_BUCKETS;
int           OUTPUT_WIDTH = N_L3;
int           OUTPUT_HEAD_FEATURES = DEFAULT_OUTPUT_HEAD_FEATURES;

// Phase-6 runtime switch. Engine `evaluate()` checks this + INPUT_WEIGHTS
// and re-routes to EvaluateFromScratch when both are set.
bool enabled = false;

void InitLookupIndices() {
    std::memset(LOOKUP_INDICES, 0, sizeof(LOOKUP_INDICES));
    for (size_t i = 0; i < 256; ++i) {
        uint32_t bits = static_cast<uint32_t>(i);
        size_t k = 0;
        while (bits) {
            LOOKUP_INDICES[i][k++] = static_cast<uint16_t>(__builtin_ctz(bits));
            bits &= bits - 1;
        }
    }
}

void ShuffleInputLayout(int input_buckets, int feature_channels) {
#if !defined(__AVX512BW__) && !defined(__AVX2__)
    (void)input_buckets;
    (void)feature_channels;
#endif
#if defined(__AVX512BW__)
    constexpr size_t width = sizeof(__m512i) / sizeof(int16_t);
    const size_t weight_chunks =
        (static_cast<size_t>(FeatureCount(input_buckets, feature_channels)) * N_HIDDEN) / width;
    constexpr size_t bias_chunks = N_HIDDEN / width;

    auto* weights = reinterpret_cast<__m512i*>(s_input_weights);
    auto* biases = reinterpret_cast<__m512i*>(s_input_biases);

    for (size_t i = 0; i < weight_chunks; i += 2) {
        auto w0 = _mm512_loadu_si512(&weights[i]);
        auto w1 = _mm512_loadu_si512(&weights[i + 1]);
        const auto a1 = _mm512_extracti32x4_epi32(w0, 1);
        const auto a2 = _mm512_extracti32x4_epi32(w0, 2);
        const auto a3 = _mm512_extracti32x4_epi32(w0, 3);
        const auto b0 = _mm512_extracti32x4_epi32(w1, 0);
        const auto b1 = _mm512_extracti32x4_epi32(w1, 1);
        const auto b2 = _mm512_extracti32x4_epi32(w1, 2);

        w0 = _mm512_inserti32x4(w0, a2, 1);
        w0 = _mm512_inserti32x4(w0, b0, 2);
        w0 = _mm512_inserti32x4(w0, b2, 3);
        w1 = _mm512_inserti32x4(w1, a1, 0);
        w1 = _mm512_inserti32x4(w1, a3, 1);
        w1 = _mm512_inserti32x4(w1, b1, 2);
        _mm512_storeu_si512(&weights[i], w0);
        _mm512_storeu_si512(&weights[i + 1], w1);
    }

    for (size_t i = 0; i < bias_chunks; i += 2) {
        auto b0v = _mm512_loadu_si512(&biases[i]);
        auto b1v = _mm512_loadu_si512(&biases[i + 1]);
        const auto a1 = _mm512_extracti32x4_epi32(b0v, 1);
        const auto a2 = _mm512_extracti32x4_epi32(b0v, 2);
        const auto a3 = _mm512_extracti32x4_epi32(b0v, 3);
        const auto b0 = _mm512_extracti32x4_epi32(b1v, 0);
        const auto b1 = _mm512_extracti32x4_epi32(b1v, 1);
        const auto b2 = _mm512_extracti32x4_epi32(b1v, 2);

        b0v = _mm512_inserti32x4(b0v, a2, 1);
        b0v = _mm512_inserti32x4(b0v, b0, 2);
        b0v = _mm512_inserti32x4(b0v, b2, 3);
        b1v = _mm512_inserti32x4(b1v, a1, 0);
        b1v = _mm512_inserti32x4(b1v, a3, 1);
        b1v = _mm512_inserti32x4(b1v, b1, 2);
        _mm512_storeu_si512(&biases[i], b0v);
        _mm512_storeu_si512(&biases[i + 1], b1v);
    }
    INPUT_LAYOUT_SCRAMBLED = true;
#elif defined(__AVX2__)
    constexpr size_t width = sizeof(__m256i) / sizeof(int16_t);
    const size_t weight_chunks =
        (static_cast<size_t>(FeatureCount(input_buckets, feature_channels)) * N_HIDDEN) / width;
    constexpr size_t bias_chunks = N_HIDDEN / width;

    auto* weights = reinterpret_cast<__m256i*>(s_input_weights);
    auto* biases = reinterpret_cast<__m256i*>(s_input_biases);

    for (size_t i = 0; i < weight_chunks; i += 2) {
        auto w0 = _mm256_loadu_si256(&weights[i]);
        auto w1 = _mm256_loadu_si256(&weights[i + 1]);
        const auto a1 = _mm256_extracti128_si256(w0, 1);
        const auto b0 = _mm256_extracti128_si256(w1, 0);
        w0 = _mm256_inserti128_si256(w0, b0, 1);
        w1 = _mm256_inserti128_si256(w1, a1, 0);
        _mm256_storeu_si256(&weights[i], w0);
        _mm256_storeu_si256(&weights[i + 1], w1);
    }

    for (size_t i = 0; i < bias_chunks; i += 2) {
        auto b0v = _mm256_loadu_si256(&biases[i]);
        auto b1v = _mm256_loadu_si256(&biases[i + 1]);
        const auto a1 = _mm256_extracti128_si256(b0v, 1);
        const auto b0 = _mm256_extracti128_si256(b1v, 0);
        b0v = _mm256_inserti128_si256(b0v, b0, 1);
        b1v = _mm256_inserti128_si256(b1v, a1, 0);
        _mm256_storeu_si256(&biases[i], b0v);
        _mm256_storeu_si256(&biases[i + 1], b1v);
    }
    INPUT_LAYOUT_SCRAMBLED = true;
#else
    INPUT_LAYOUT_SCRAMBLED = false;
#endif
}

void SetWeights(const acc_t* weights, const acc_t* biases) {
    INPUT_BUCKETS = DEFAULT_INPUT_BUCKETS;
    FEATURE_CHANNELS = DEFAULT_FEATURE_CHANNELS;
    TRAINED_HIDDEN = N_HIDDEN;
    OUTPUT_BUCKETS = DEFAULT_OUTPUT_BUCKETS;
    OUTPUT_WIDTH = N_L3;
    OUTPUT_HEAD_FEATURES = DEFAULT_OUTPUT_HEAD_FEATURES;
    INPUT_WEIGHTS = weights;
    INPUT_BIASES  = biases;
    L1_WEIGHTS_T  = nullptr;
    L1_WEIGHTS_SPARSE = nullptr;
    OUTPUT_BIASES = nullptr;
    OUTPUT_BIAS = 0.0f;
    QUANTIZED_L2_WEIGHTS = nullptr;
    QUANTIZED_L2_BIASES = nullptr;
    QUANTIZED_OUTPUT_WEIGHTS = nullptr;
    QUANTIZED_OUTPUT_BIAS = 0;
    DENSE_LAYER_FORMAT = DenseLayerFormat::Float;
    INPUT_LAYOUT_SCRAMBLED = false;
    ++NETWORK_GENERATION;
}

// ---------------------------------------------------------------------
// LoadNetwork — parse a supported .nn file into the internal weight
// arrays and repoint the externs.
//
// File layout (tightly packed, little-endian; total NETWORK_SIZE bytes):
//   [N_FEATURES * N_HIDDEN] int16  INPUT_WEIGHTS  (row-major, feature-major)
//   [N_HIDDEN]              int16  INPUT_BIASES
//   [N_L1 * N_L2]           int8   L1_WEIGHTS     (scalar layout: w[j*N_L1+i])
//   [N_L2]                  int32  L1_BIASES
//   [N_L2 * N_L3]           float  L2_WEIGHTS     (row-major, output-major)
//   [N_L3]                  float  L2_BIASES
//   [OUTPUT_BUCKETS * (N_L3 + HEAD_FEATURES)] float  OUTPUT_WEIGHTS
//   [OUTPUT_BUCKETS]        float  OUTPUT_BIASES
//
// Any size mismatch is a hard failure (we refuse to silently load a
// differently-sized net — that would produce garbage evaluations).
// ---------------------------------------------------------------------
bool LoadNetwork(const char* path) {
    InitLookupIndices();

    std::FILE* fh = std::fopen(path, "rb");
    if (!fh) {
        std::fprintf(stderr, "network: can't open '%s': %s\n",
                     path, std::strerror(errno));
        return false;
    }

    // Size-check first.
    if (std::fseek(fh, 0, SEEK_END) != 0) {
        std::fclose(fh);
        return false;
    }
    const long sz = std::ftell(fh);
    if (sz < 0) { std::fclose(fh); return false; }
    const bool quantized = IsSupportedQuantizedNetworkSize(static_cast<size_t>(sz));
    NetworkLayout layout = quantized
        ? NetworkLayout{16, LEGACY_FEATURE_CHANNELS, 1, 0, N_HIDDEN, 0}
        : NetworkLayout{};
    if (!quantized && !ReadNetworkLayout(fh, static_cast<size_t>(sz), layout)) {
        std::fprintf(stderr,
            "network: '%s' is %ld bytes, expected a supported Enyo NNUE size\n"
            "  input buckets: 1, 2, 4, 8, 10, 16, or 32\n"
            "  feature channels: 12, or 11 with 10, 16, or 32 input buckets\n"
            "  trained hidden width: 512, 768, or 1024 in enyo-native-v2\n"
            "  output buckets: 1, 2, 4, or 8\n"
            "  output head features: 0, %d, or %d\n",
            path, sz, N_HEAD_FEATURES, N_EXTENDED_HEAD_FEATURES);
        std::fclose(fh);
        return false;
    }
    const int input_buckets = layout.input_buckets;
    const int feature_channels = layout.feature_channels;
    const int trained_hidden = layout.trained_hidden;
    const int output_buckets = layout.output_buckets;
    const int output_head_features = layout.output_head_features;
    const int output_width = N_L3 + output_head_features;
    if (std::fseek(fh, static_cast<long>(layout.header_size), SEEK_SET) != 0) {
        std::fclose(fh);
        return false;
    }

    auto read = [&](void* dst, size_t n, const char* label) -> bool {
        if (std::fread(dst, 1, n, fh) != n) {
            std::fprintf(stderr, "network: short read at %s\n", label);
            return false;
        }
        return true;
    };

    std::memset(s_input_weights, 0, sizeof(s_input_weights));
    std::memset(s_output_weights, 0, sizeof(s_output_weights));
    std::memset(s_output_biases, 0, sizeof(s_output_biases));
    std::memset(s_quantized_l2_weights, 0, sizeof(s_quantized_l2_weights));
    std::memset(s_quantized_l2_biases, 0, sizeof(s_quantized_l2_biases));
    std::memset(s_quantized_output_weights, 0, sizeof(s_quantized_output_weights));
    s_quantized_output_bias = 0;
    const size_t input_weight_bytes =
        sizeof(acc_t) * static_cast<size_t>(FeatureCount(input_buckets, feature_channels)) * N_HIDDEN;
    const size_t output_weight_bytes =
        sizeof(float) * static_cast<size_t>(output_buckets)
            * static_cast<size_t>(output_width) * N_OUTPUT;
    const size_t output_bias_bytes =
        sizeof(float) * static_cast<size_t>(output_buckets) * N_OUTPUT;
    if (!read(s_input_weights,  input_weight_bytes,       "INPUT_WEIGHTS")) { std::fclose(fh); return false; }
    if (!read(s_input_biases,   sizeof(s_input_biases),   "INPUT_BIASES"))  { std::fclose(fh); return false; }
    if (!read(s_l1_weights,     sizeof(s_l1_weights),     "L1_WEIGHTS"))    { std::fclose(fh); return false; }
    if (!read(s_l1_biases,      sizeof(s_l1_biases),      "L1_BIASES"))     { std::fclose(fh); return false; }
    if (quantized) {
        if (!read(s_quantized_l2_weights, sizeof(s_quantized_l2_weights), "L2_WEIGHTS")) {
            std::fclose(fh);
            return false;
        }
        if (!read(s_quantized_l2_biases, sizeof(s_quantized_l2_biases), "L2_BIASES")) {
            std::fclose(fh);
            return false;
        }
        if (!read(
                s_quantized_output_weights,
                sizeof(s_quantized_output_weights),
                "OUTPUT_WEIGHTS")) {
            std::fclose(fh);
            return false;
        }
        if (!read(
                &s_quantized_output_bias,
                sizeof(s_quantized_output_bias),
                "OUTPUT_BIAS")) {
            std::fclose(fh);
            return false;
        }
    } else {
        if (!read(s_l2_weights, sizeof(s_l2_weights), "L2_WEIGHTS")) {
            std::fclose(fh);
            return false;
        }
        if (!read(s_l2_biases, sizeof(s_l2_biases), "L2_BIASES")) {
            std::fclose(fh);
            return false;
        }
        if (!read(s_output_weights, output_weight_bytes, "OUTPUT_WEIGHTS")) {
            std::fclose(fh);
            return false;
        }
        if (!read(s_output_biases, output_bias_bytes, "OUTPUT_BIASES")) {
            std::fclose(fh);
            return false;
        }
    }

    // Sanity: trailing-byte check (should be exactly EOF now).
    unsigned char probe;
    if (std::fread(&probe, 1, 1, fh) != 0) {
        std::fprintf(stderr, "network: trailing bytes after expected EOF\n");
        std::fclose(fh);
        return false;
    }
    std::fclose(fh);

    // Repoint externs.
    for (size_t i = 0; i < N_L1; ++i)
        for (size_t j = 0; j < N_L2; ++j)
            s_l1_weights_t[i * N_L2 + j] = s_l1_weights[j * N_L1 + i];
    for (size_t i = 0; i < N_L1 * N_L2; ++i)
        s_l1_weights_sparse[WeightIdxScrambled(i)] = s_l1_weights[i];
    INPUT_BUCKETS = input_buckets;
    FEATURE_CHANNELS = feature_channels;
    TRAINED_HIDDEN = trained_hidden;
    OUTPUT_BUCKETS = output_buckets;
    OUTPUT_WIDTH = output_width;
    OUTPUT_HEAD_FEATURES = output_head_features;
    ShuffleInputLayout(input_buckets, feature_channels);

    INPUT_WEIGHTS  = s_input_weights;
    INPUT_BIASES   = s_input_biases;
    L1_WEIGHTS     = s_l1_weights;
    L1_WEIGHTS_T   = s_l1_weights_t;
    L1_WEIGHTS_SPARSE = s_l1_weights_sparse;
    L1_BIASES      = s_l1_biases;
    DENSE_LAYER_FORMAT = quantized
        ? DenseLayerFormat::Quantized
        : DenseLayerFormat::Float;
    L2_WEIGHTS = quantized ? nullptr : s_l2_weights;
    L2_BIASES = quantized ? nullptr : s_l2_biases;
    OUTPUT_WEIGHTS = quantized ? nullptr : s_output_weights;
    OUTPUT_BIASES = quantized ? nullptr : s_output_biases;
    OUTPUT_BIAS = quantized ? 0.0f : s_output_biases[0];
    QUANTIZED_L2_WEIGHTS = quantized ? s_quantized_l2_weights : nullptr;
    QUANTIZED_L2_BIASES = quantized ? s_quantized_l2_biases : nullptr;
    QUANTIZED_OUTPUT_WEIGHTS = quantized ? s_quantized_output_weights : nullptr;
    QUANTIZED_OUTPUT_BIAS = quantized ? s_quantized_output_bias : 0;
    ++NETWORK_GENERATION;

    return true;
}

} // namespace Network
