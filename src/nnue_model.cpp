// Network weight storage and network loading.
// Defines the extern storage for weight pointers declared in network.hpp
// and implements LoadNetwork for the v13-compatible .nn binary layout.
//
// This TU is intentionally independent of the engine's Board type so
// Phase 2/3 parity tests can compile it directly. The Board-aware bits
// live in nnue_model_board.cpp.

#include "nnue_model.hpp"

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
bool          INPUT_LAYOUT_SCRAMBLED = false;
uint64_t      NETWORK_GENERATION = 0;
int           INPUT_BUCKETS = DEFAULT_INPUT_BUCKETS;
int           FEATURE_CHANNELS = DEFAULT_FEATURE_CHANNELS;
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
    OUTPUT_BUCKETS = DEFAULT_OUTPUT_BUCKETS;
    OUTPUT_WIDTH = N_L3;
    OUTPUT_HEAD_FEATURES = DEFAULT_OUTPUT_HEAD_FEATURES;
    INPUT_WEIGHTS = weights;
    INPUT_BIASES  = biases;
    L1_WEIGHTS_T  = nullptr;
    L1_WEIGHTS_SPARSE = nullptr;
    OUTPUT_BIASES = nullptr;
    OUTPUT_BIAS = 0.0f;
    INPUT_LAYOUT_SCRAMBLED = false;
    ++NETWORK_GENERATION;
}

// ---------------------------------------------------------------------
// LoadNetwork — parse a v13-compatible .nn file into the internal
// weight arrays and repoint the externs.
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
    const NetworkLayout layout = DetectNetworkLayout(static_cast<size_t>(sz));
    if (layout.input_buckets == 0) {
        std::fprintf(stderr,
            "network: '%s' is %ld bytes, expected a supported Enyo NNUE size\n"
            "  12-channel legacy: %zu, %zu, %zu, %zu, %zu, %zu, %zu, or %zu\n"
            "  12-channel material/phase head: %zu, %zu, %zu, %zu, %zu, %zu, %zu, or %zu\n"
            "  11-channel HalfKAv2-like: %zu, %zu, %zu, or %zu\n"
            "  11-channel HalfKAv2-like material/phase head: %zu, %zu, %zu, or %zu\n",
            path, sz,
            NetworkSize(16, 1), NetworkSize(16, 2),
            NetworkSize(16, 4), NetworkSize(16, 8),
            NetworkSize(32, 1), NetworkSize(32, 2),
            NetworkSize(32, 4), NetworkSize(32, 8),
            NetworkSize(16, 1, N_HEAD_FEATURES),
            NetworkSize(16, 2, N_HEAD_FEATURES),
            NetworkSize(16, 4, N_HEAD_FEATURES),
            NetworkSize(16, 8, N_HEAD_FEATURES),
            NetworkSize(32, 1, N_HEAD_FEATURES),
            NetworkSize(32, 2, N_HEAD_FEATURES),
            NetworkSize(32, 4, N_HEAD_FEATURES),
            NetworkSize(32, 8, N_HEAD_FEATURES),
            NetworkSize(32, 1, DEFAULT_OUTPUT_HEAD_FEATURES, HALFKA_V2_FEATURE_CHANNELS),
            NetworkSize(32, 2, DEFAULT_OUTPUT_HEAD_FEATURES, HALFKA_V2_FEATURE_CHANNELS),
            NetworkSize(32, 4, DEFAULT_OUTPUT_HEAD_FEATURES, HALFKA_V2_FEATURE_CHANNELS),
            NetworkSize(32, 8, DEFAULT_OUTPUT_HEAD_FEATURES, HALFKA_V2_FEATURE_CHANNELS),
            NetworkSize(32, 1, N_HEAD_FEATURES, HALFKA_V2_FEATURE_CHANNELS),
            NetworkSize(32, 2, N_HEAD_FEATURES, HALFKA_V2_FEATURE_CHANNELS),
            NetworkSize(32, 4, N_HEAD_FEATURES, HALFKA_V2_FEATURE_CHANNELS),
            NetworkSize(32, 8, N_HEAD_FEATURES, HALFKA_V2_FEATURE_CHANNELS));
        std::fclose(fh);
        return false;
    }
    const int input_buckets = layout.input_buckets;
    const int feature_channels = layout.feature_channels;
    const int output_buckets = layout.output_buckets;
    const int output_head_features = layout.output_head_features;
    const int output_width = N_L3 + output_head_features;
    std::rewind(fh);

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
    if (!read(s_l2_weights,     sizeof(s_l2_weights),     "L2_WEIGHTS"))    { std::fclose(fh); return false; }
    if (!read(s_l2_biases,      sizeof(s_l2_biases),      "L2_BIASES"))     { std::fclose(fh); return false; }
    if (!read(s_output_weights, output_weight_bytes,      "OUTPUT_WEIGHTS")){ std::fclose(fh); return false; }
    if (!read(s_output_biases,  output_bias_bytes,        "OUTPUT_BIASES")) { std::fclose(fh); return false; }

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
    L2_WEIGHTS     = s_l2_weights;
    L2_BIASES      = s_l2_biases;
    OUTPUT_WEIGHTS = s_output_weights;
    OUTPUT_BIASES  = s_output_biases;
    OUTPUT_BIAS    = s_output_biases[0];
    ++NETWORK_GENERATION;

    return true;
}

} // namespace Network
