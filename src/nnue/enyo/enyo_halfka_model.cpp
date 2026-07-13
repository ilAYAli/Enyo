// Enyo HalfKA weight storage and runtime activation.
// Defines the extern storage for the Enyo HalfKA model.
//
// This TU is intentionally independent of the engine's Board type so
// Phase 2/3 parity tests can compile it directly. The Board-aware bits
// live in enyo_halfka_board.cpp.

#include "enyo_halfka_model.hpp"
#include "enyo_halfka_storage.hpp"

#include <cstdint>
#include <cstring>

namespace Network {

namespace EnyoStorage {

alignas(64) int16_t input_weights[N_FEATURES * N_HIDDEN];
alignas(64) int16_t input_biases[N_HIDDEN];
alignas(64) int8_t l1_weights[MAX_OUTPUT_BUCKETS * N_L1 * N_L2];
alignas(64) int8_t l1_weights_transposed[MAX_OUTPUT_BUCKETS * N_L1 * N_L2];
alignas(64) int8_t l1_weights_sparse[MAX_OUTPUT_BUCKETS * N_L1 * N_L2];
alignas(64) int32_t l1_biases[MAX_OUTPUT_BUCKETS * N_L2];
alignas(64) float l2_weights[MAX_OUTPUT_BUCKETS * N_L2 * N_L3];
alignas(64) float l2_biases[MAX_OUTPUT_BUCKETS * N_L3];
alignas(64) float l2_squared_weights[N_L2 * N_L3];
alignas(64) float l2_squared_biases[N_L3];
alignas(64) float output_weights[MAX_OUTPUT_BUCKETS * MAX_OUTPUT_WIDTH * N_OUTPUT];
alignas(64) float output_biases[MAX_OUTPUT_BUCKETS * N_OUTPUT];
alignas(64) int16_t quantized_l2_weights[N_L2 * N_L3];
alignas(64) int32_t quantized_l2_biases[N_L3];
alignas(64) int16_t quantized_output_weights[N_L3 * N_OUTPUT];
alignas(64) int32_t quantized_output_bias;

} // namespace EnyoStorage

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
const float*  L2_SQUARED_WEIGHTS = nullptr;
const float*  L2_SQUARED_BIASES = nullptr;
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
bool          FULL_THREATS_ENABLED = false;
bool          FULL_HEADS_ENABLED = false;
bool          MIXED_ACTIVATION_ENABLED = false;

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

void ShuffleInputLayout(int input_buckets, int feature_channels, bool full_threats) {
#if !defined(__AVX512BW__) && !defined(__AVX2__)
    (void)input_buckets;
    (void)feature_channels;
    (void)full_threats;
#endif
#if defined(__AVX512BW__)
    constexpr size_t width = sizeof(__m512i) / sizeof(int16_t);
    const size_t weight_chunks =
        (static_cast<size_t>(InputFeatureCount(input_buckets, feature_channels, full_threats))
            * N_HIDDEN) / width;
    constexpr size_t bias_chunks = N_HIDDEN / width;

    auto* weights = reinterpret_cast<__m512i*>(EnyoStorage::input_weights);
    auto* biases = reinterpret_cast<__m512i*>(EnyoStorage::input_biases);

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
        (static_cast<size_t>(InputFeatureCount(input_buckets, feature_channels, full_threats))
            * N_HIDDEN) / width;
    constexpr size_t bias_chunks = N_HIDDEN / width;

    auto* weights = reinterpret_cast<__m256i*>(EnyoStorage::input_weights);
    auto* biases = reinterpret_cast<__m256i*>(EnyoStorage::input_biases);

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
    FULL_THREATS_ENABLED = false;
    FULL_HEADS_ENABLED = false;
    MIXED_ACTIVATION_ENABLED = false;
    INPUT_WEIGHTS = weights;
    INPUT_BIASES  = biases;
    L1_WEIGHTS_T  = nullptr;
    L1_WEIGHTS_SPARSE = nullptr;
    L2_SQUARED_WEIGHTS = nullptr;
    L2_SQUARED_BIASES = nullptr;
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

void EnyoStorage::Clear() {
    std::memset(input_weights, 0, sizeof(input_weights));
    std::memset(input_biases, 0, sizeof(input_biases));
    std::memset(l1_weights, 0, sizeof(l1_weights));
    std::memset(l1_weights_transposed, 0, sizeof(l1_weights_transposed));
    std::memset(l1_weights_sparse, 0, sizeof(l1_weights_sparse));
    std::memset(l1_biases, 0, sizeof(l1_biases));
    std::memset(l2_weights, 0, sizeof(l2_weights));
    std::memset(l2_biases, 0, sizeof(l2_biases));
    std::memset(l2_squared_weights, 0, sizeof(l2_squared_weights));
    std::memset(l2_squared_biases, 0, sizeof(l2_squared_biases));
    std::memset(output_weights, 0, sizeof(output_weights));
    std::memset(output_biases, 0, sizeof(output_biases));
    std::memset(quantized_l2_weights, 0, sizeof(quantized_l2_weights));
    std::memset(quantized_l2_biases, 0, sizeof(quantized_l2_biases));
    std::memset(quantized_output_weights, 0, sizeof(quantized_output_weights));
    quantized_output_bias = 0;
}

void EnyoStorage::Activate(const NetworkLayout & layout, DenseLayerFormat format) {
    InitLookupIndices();

    const size_t head_count = layout.full_heads
        ? static_cast<size_t>(layout.output_buckets)
        : 1u;
    for (size_t head = 0; head < head_count; ++head) {
        const size_t base = head * N_L1 * N_L2;
        for (size_t i = 0; i < N_L1; ++i)
            for (size_t j = 0; j < N_L2; ++j)
                l1_weights_transposed[base + i * N_L2 + j] =
                    l1_weights[base + j * N_L1 + i];
        for (size_t i = 0; i < N_L1 * N_L2; ++i)
            l1_weights_sparse[base + WeightIdxScrambled(i)] = l1_weights[base + i];
    }

    INPUT_BUCKETS = layout.input_buckets;
    FEATURE_CHANNELS = layout.feature_channels;
    TRAINED_HIDDEN = layout.trained_hidden;
    OUTPUT_BUCKETS = layout.output_buckets;
    OUTPUT_HEAD_FEATURES = layout.output_head_features;
    FULL_THREATS_ENABLED = layout.full_threats;
    FULL_HEADS_ENABLED = layout.full_heads;
    MIXED_ACTIVATION_ENABLED = layout.mixed_activation;
    OUTPUT_WIDTH = N_L3 + OUTPUT_HEAD_FEATURES;
    ShuffleInputLayout(INPUT_BUCKETS, FEATURE_CHANNELS, FULL_THREATS_ENABLED);

    INPUT_WEIGHTS = input_weights;
    INPUT_BIASES = input_biases;
    L1_WEIGHTS = l1_weights;
    L1_WEIGHTS_T = l1_weights_transposed;
    L1_WEIGHTS_SPARSE = l1_weights_sparse;
    L1_BIASES = l1_biases;
    DENSE_LAYER_FORMAT = format;

    const bool quantized = format == DenseLayerFormat::Quantized;
    L2_WEIGHTS = quantized ? nullptr : l2_weights;
    L2_BIASES = quantized ? nullptr : l2_biases;
    L2_SQUARED_WEIGHTS = !quantized && layout.mixed_activation
        ? l2_squared_weights : nullptr;
    L2_SQUARED_BIASES = !quantized && layout.mixed_activation
        ? l2_squared_biases : nullptr;
    OUTPUT_WEIGHTS = quantized ? nullptr : output_weights;
    OUTPUT_BIASES = quantized ? nullptr : output_biases;
    OUTPUT_BIAS = quantized ? 0.0f : output_biases[0];
    QUANTIZED_L2_WEIGHTS = quantized ? quantized_l2_weights : nullptr;
    QUANTIZED_L2_BIASES = quantized ? quantized_l2_biases : nullptr;
    QUANTIZED_OUTPUT_WEIGHTS = quantized ? quantized_output_weights : nullptr;
    QUANTIZED_OUTPUT_BIAS = quantized ? quantized_output_bias : 0;
    ++NETWORK_GENERATION;
}

} // namespace Network
