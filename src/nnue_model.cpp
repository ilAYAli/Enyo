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
#include <vector>

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
alignas(64) float   s_l2_weights    [N_OUTPUT_BUCKETS * N_L2 * N_L3];
alignas(64) float   s_l2_biases     [N_OUTPUT_BUCKETS * N_L3];
alignas(64) float   s_output_weights[N_OUTPUT_BUCKETS * N_L3 * N_OUTPUT];
alignas(64) float   s_output_biases [N_OUTPUT_BUCKETS];
#if ENYO_ENABLE_THREAT_NNUE
alignas(64) int8_t  s_threat_weights[N_THREAT_FEATURES * N_HIDDEN];
alignas(64) uint8_t s_threat_feature_active[N_THREAT_FEATURES];
#endif
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
#if ENYO_ENABLE_THREAT_NNUE
const int8_t* THREAT_WEIGHTS = nullptr;
const uint8_t* THREAT_FEATURE_ACTIVE = nullptr;
#endif
int           OUTPUT_BUCKETS = 1;
bool          INPUT_LAYOUT_SCRAMBLED = false;
uint64_t      NETWORK_GENERATION = 0;

// Phase-6 runtime switch. Engine `evaluate()` checks this + INPUT_WEIGHTS
// and re-routes to EvaluateFromScratch when both are set.
bool enabled = false;

void ExpandLegacyInputWeights(const int16_t* legacy_weights) {
    constexpr size_t bucket_stride = N_PIECE_TYPES * N_SQUARES;
    for (size_t bucket = 0; bucket < N_KING_BUCKETS; ++bucket) {
        const size_t legacy_bucket = LEGACY_BUCKET_FOR_BUCKET[bucket];
        for (size_t offset = 0; offset < bucket_stride; ++offset) {
            const size_t dst_feature = bucket * bucket_stride + offset;
            const size_t src_feature = legacy_bucket * bucket_stride + offset;
            std::memcpy(
                &s_input_weights[dst_feature * N_HIDDEN],
                &legacy_weights[src_feature * N_HIDDEN],
                sizeof(int16_t) * N_HIDDEN);
        }
    }
}

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

void ShuffleInputLayout() {
#if defined(__AVX512BW__)
    constexpr size_t width = sizeof(__m512i) / sizeof(int16_t);
    constexpr size_t weight_chunks = (N_FEATURES * N_HIDDEN) / width;
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
    constexpr size_t weight_chunks = (N_FEATURES * N_HIDDEN) / width;
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
    INPUT_WEIGHTS = weights;
    INPUT_BIASES  = biases;
    L1_WEIGHTS_T  = nullptr;
    L1_WEIGHTS_SPARSE = nullptr;
    OUTPUT_BIASES = s_output_biases;
#if ENYO_ENABLE_THREAT_NNUE
    THREAT_WEIGHTS = nullptr;
    THREAT_FEATURE_ACTIVE = nullptr;
#endif
    OUTPUT_BUCKETS = 1;
    INPUT_LAYOUT_SCRAMBLED = false;
    ++NETWORK_GENERATION;
}

// ---------------------------------------------------------------------
// LoadNetwork — parse an Enyo .nn file into the internal
// weight arrays and repoint the externs.
//
// File layout (tightly packed, little-endian; total NETWORK_SIZE bytes):
//   [N_FEATURES * N_HIDDEN] int16  INPUT_WEIGHTS  (row-major, feature-major)
//   [N_HIDDEN]              int16  INPUT_BIASES
//   [N_L1 * N_L2]           int8   L1_WEIGHTS     (scalar layout: w[j*N_L1+i])
//   [N_L2]                  int32  L1_BIASES
//   [head_buckets * N_L2 * N_L3]     float  L2_WEIGHTS
//   [head_buckets * N_L3]            float  L2_BIASES
//   [head_buckets * N_L3 * N_OUTPUT] float  OUTPUT_WEIGHTS
//   [head_buckets]                   float  OUTPUT_BIASES
//
// Legacy 16-bucket files are accepted by expanding each old bucket into its
// new 32-bucket children. Other size mismatches are hard failures.
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
    const auto file_size = static_cast<size_t>(sz);
    const bool legacy_layout =
        file_size == LEGACY_NETWORK_SIZE ||
        file_size == LEGACY_BUCKETED_HEAD_NETWORK_SIZE
#if ENYO_ENABLE_THREAT_NNUE
        || file_size == LEGACY_NETWORK_SIZE + THREAT_WEIGHTS_BYTES
        || file_size == LEGACY_BUCKETED_HEAD_NETWORK_SIZE + THREAT_WEIGHTS_BYTES
#endif
        ;
    const bool bucketed_head =
        file_size == BUCKETED_HEAD_NETWORK_SIZE ||
        file_size == LEGACY_BUCKETED_HEAD_NETWORK_SIZE
#if ENYO_ENABLE_THREAT_NNUE
        || file_size == BUCKETED_HEAD_NETWORK_SIZE + THREAT_WEIGHTS_BYTES
        || file_size == LEGACY_BUCKETED_HEAD_NETWORK_SIZE + THREAT_WEIGHTS_BYTES
#endif
        ;
#if ENYO_ENABLE_THREAT_NNUE
    const bool threat_layout =
        file_size == NETWORK_SIZE + THREAT_WEIGHTS_BYTES
        || file_size == BUCKETED_HEAD_NETWORK_SIZE + THREAT_WEIGHTS_BYTES
        || file_size == LEGACY_NETWORK_SIZE + THREAT_WEIGHTS_BYTES
        || file_size == LEGACY_BUCKETED_HEAD_NETWORK_SIZE + THREAT_WEIGHTS_BYTES;
#endif
    if (file_size != NETWORK_SIZE
        && file_size != BUCKETED_HEAD_NETWORK_SIZE
        && !legacy_layout
#if ENYO_ENABLE_THREAT_NNUE
        && !threat_layout
#endif
        ) {
        std::fprintf(stderr,
            "network: '%s' is %ld bytes, expected %zu, %zu, %zu, or %zu",
            path, sz,
            LEGACY_NETWORK_SIZE,
            NETWORK_SIZE,
            LEGACY_BUCKETED_HEAD_NETWORK_SIZE,
            BUCKETED_HEAD_NETWORK_SIZE);
#if ENYO_ENABLE_THREAT_NNUE
        std::fprintf(stderr, " (+ optional %zu threat bytes)", THREAT_WEIGHTS_BYTES);
#endif
        std::fprintf(stderr, "\n");
        std::fclose(fh);
        return false;
    }
    std::rewind(fh);

    auto read = [&](void* dst, size_t n, const char* label) -> bool {
        if (std::fread(dst, 1, n, fh) != n) {
            std::fprintf(stderr, "network: short read at %s\n", label);
            return false;
        }
        return true;
    };

    std::vector<int16_t> legacy_input_weights;
    if (legacy_layout) {
        legacy_input_weights.resize(
            static_cast<size_t>(LEGACY_N_FEATURES) * N_HIDDEN);
        if (!read(legacy_input_weights.data(),
                  legacy_input_weights.size() * sizeof(int16_t),
                  "INPUT_WEIGHTS")) { std::fclose(fh); return false; }
        ExpandLegacyInputWeights(legacy_input_weights.data());
    } else if (!read(s_input_weights,  sizeof(s_input_weights),  "INPUT_WEIGHTS")) {
        std::fclose(fh);
        return false;
    }
    if (!read(s_input_biases,   sizeof(s_input_biases),   "INPUT_BIASES"))  { std::fclose(fh); return false; }
    if (!read(s_l1_weights,     sizeof(s_l1_weights),     "L1_WEIGHTS"))    { std::fclose(fh); return false; }
    if (!read(s_l1_biases,      sizeof(s_l1_biases),      "L1_BIASES"))     { std::fclose(fh); return false; }

    const size_t head_buckets = bucketed_head
        ? static_cast<size_t>(N_OUTPUT_BUCKETS)
        : 1;
    const size_t l2_weights_bytes =
        head_buckets * N_L2 * N_L3 * sizeof(float);
    const size_t l2_biases_bytes =
        head_buckets * N_L3 * sizeof(float);
    const size_t output_weights_bytes =
        head_buckets * N_L3 * N_OUTPUT * sizeof(float);
    const size_t output_biases_bytes =
        head_buckets * sizeof(float);

    if (!read(s_l2_weights,     l2_weights_bytes,      "L2_WEIGHTS"))     { std::fclose(fh); return false; }
    if (!read(s_l2_biases,      l2_biases_bytes,       "L2_BIASES"))      { std::fclose(fh); return false; }
    if (!read(s_output_weights, output_weights_bytes,  "OUTPUT_WEIGHTS")) { std::fclose(fh); return false; }
    if (!read(s_output_biases,  output_biases_bytes,   "OUTPUT_BIASES"))  { std::fclose(fh); return false; }
#if ENYO_ENABLE_THREAT_NNUE
    if (threat_layout) {
        if (!read(s_threat_weights, sizeof(s_threat_weights), "THREAT_WEIGHTS")) {
            std::fclose(fh);
            return false;
        }
        bool any_threat_weight = false;
        for (int feature = 0; feature < N_THREAT_FEATURES; ++feature) {
            const int8_t* row = &s_threat_weights[
                static_cast<size_t>(feature) * N_HIDDEN];
            bool row_active = false;
            for (int hidden = 0; hidden < N_HIDDEN; ++hidden) {
                if (row[hidden] != 0) {
                    row_active = true;
                    any_threat_weight = true;
                    break;
                }
            }
            s_threat_feature_active[feature] = static_cast<uint8_t>(row_active);
        }
        if (!any_threat_weight) {
            std::memset(s_threat_weights, 0, sizeof(s_threat_weights));
            std::memset(s_threat_feature_active, 0, sizeof(s_threat_feature_active));
        }
    }
#endif

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
    ShuffleInputLayout();

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
#if ENYO_ENABLE_THREAT_NNUE
    if (threat_layout) {
        bool any_threat_weight = false;
        for (const uint8_t active : s_threat_feature_active) {
            if (active != 0) {
                any_threat_weight = true;
                break;
            }
        }
        THREAT_WEIGHTS = any_threat_weight ? s_threat_weights : nullptr;
        THREAT_FEATURE_ACTIVE = any_threat_weight ? s_threat_feature_active : nullptr;
    } else {
        THREAT_WEIGHTS = nullptr;
        THREAT_FEATURE_ACTIVE = nullptr;
    }
#endif
    OUTPUT_BUCKETS = static_cast<int>(head_buckets);
    ++NETWORK_GENERATION;

    return true;
}

} // namespace Network
