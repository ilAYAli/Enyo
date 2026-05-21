#pragma once
// Network uses a Berserk v13-compatible network layout and feature indexing.
// KING_BUCKETS and the feature-index formula are derived from Berserk
// (GPL-3.0), adapted to Enyo's square and piece conventions.

#include "types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__AVX512BW__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace enyo { class Board; }

namespace Network {

// ---------------------------------------------------------------------
// Architecture constants.
// ---------------------------------------------------------------------
inline constexpr int N_KING_BUCKETS = 16;
inline constexpr int N_PIECE_TYPES  = 12;               // 6 types × 2 colors
inline constexpr int N_SQUARES      = 64;
inline constexpr int N_FEATURES     = N_KING_BUCKETS * N_PIECE_TYPES * N_SQUARES; // 12288
inline constexpr int N_HIDDEN       = 1024;
inline constexpr int N_L1           = 2 * N_HIDDEN;     // 2048
inline constexpr int N_L2           = 16;
inline constexpr int N_PHASE_HEAD   = 1;
inline constexpr int N_L2_HEAD      = N_L2 + N_PHASE_HEAD;
inline constexpr int N_L3           = 32;
inline constexpr int N_OUTPUT       = 1;

// ---------------------------------------------------------------------
// Network king buckets. Do not reshuffle this; feature indexing and
// bucket-crossing refresh logic are calibrated against this exact layout.
// ---------------------------------------------------------------------
inline constexpr std::array<uint16_t, 64> KING_BUCKETS = {
    15, 15, 14, 14, 14, 14, 15, 15,
    15, 15, 14, 14, 14, 14, 15, 15,
    13, 13, 12, 12, 12, 12, 13, 13,
    13, 13, 12, 12, 12, 12, 13, 13,
    11, 10,  9,  8,  8,  9, 10, 11,
    11, 10,  9,  8,  8,  9, 10, 11,
     7,  6,  5,  4,  4,  5,  6,  7,
     3,  2,  1,  0,  0,  1,  2,  3,
};

// ---------------------------------------------------------------------
// h1-indexed Enyo square -> a8-indexed network square.
//
// Enyo:    h1 = 0, g1 = 1, ..., a1 = 7, h2 = 8, ... a8 = 63
// Network: a8 = 0, b8 = 1, ..., h8 = 7, a7 = 8, ... h1 = 63
//
// The relationship is simply `net_sq = 63 - enyo_sq` -- equivalent
// to `enyo_sq ^ 63`, which flips both rank AND file bits. Equivalent
// to a 180-degree board rotation.
// ---------------------------------------------------------------------
inline constexpr int to_net_sq(int enyo_sq) {
    return enyo_sq ^ 63;
}

// ---------------------------------------------------------------------
// Network piece encoding: `Piece(pt, c) = (pt << 1) | c`, where
// pt ∈ {PAWN=0..KING=5}, c ∈ {WHITE=0, BLACK=1}.
// So WHITE_PAWN=0, BLACK_PAWN=1, WHITE_KNIGHT=2, ..., BLACK_KING=11.
//
// Enyo's PieceType has pawn=1..king=6 (no_piece_type=0). Bridge here.
// ---------------------------------------------------------------------
inline constexpr int to_net_piece(enyo::PieceType pt, enyo::Color c) {
    const int bpt = static_cast<int>(pt) - 1;
    return (bpt << 1) | static_cast<int>(c);
}

// ---------------------------------------------------------------------
// Enyo-side wrapper: takes (piece_type, piece_color, sq, king_sq, view)
// in Enyo's native representation and converts to the network layout.
//
// - `view` is the perspective (white or black); feature indices are
//   always computed from the mover's side of the board.
// - `oP` groups by piece-color-relative-to-view then by piece type,
//   so US-PAWN=0..US-KING=5, THEM-PAWN=6..THEM-KING=11 for any view.
// - The `7 * !(kingsq & 4)` term mirrors the board horizontally when
//   the king is on the king-side (files e-h of its own color's side),
//   collapsing the 8 files to 4 by symmetry.
// ---------------------------------------------------------------------
inline constexpr int FeatureIdxFormula(int pt,
                                       int piece_color,
                                       int enyo_sq,
                                       int enyo_kingsq,
                                       int view)
{
    if (pt == 0)
        return 0;

    const int piece    = ((pt - 1) << 1) | piece_color;
    const int sq       = to_net_sq(enyo_sq);
    const int kingsq   = to_net_sq(enyo_kingsq);

    const int oP  = 6 * ((piece ^ view) & 0x1) + (piece >> 1);
    const int oK  = (7 * !(kingsq & 4)) ^ (56 * view) ^ kingsq;
    const int oSq = (7 * !(kingsq & 4)) ^ (56 * view) ^ sq;

    return KING_BUCKETS[static_cast<size_t>(oK)] * 12 * 64 + oP * 64 + oSq;
}

inline constexpr size_t FEATURE_IDX_TABLE_SIZE =
    7 * 2 * N_SQUARES * N_SQUARES * 2;

inline constexpr size_t FeatureIdxTableOffset(int pt,
                                              int piece_color,
                                              int enyo_sq,
                                              int enyo_kingsq,
                                              int view)
{
    return (((static_cast<size_t>(pt) * 2 + static_cast<size_t>(piece_color))
        * N_SQUARES + static_cast<size_t>(enyo_sq))
        * N_SQUARES + static_cast<size_t>(enyo_kingsq))
        * 2 + static_cast<size_t>(view);
}

inline std::array<int, FEATURE_IDX_TABLE_SIZE> MakeFeatureIdxTable()
{
    std::array<int, FEATURE_IDX_TABLE_SIZE> table{};
    for (int pt = 0; pt <= 6; ++pt)
        for (int color = 0; color < 2; ++color)
            for (int sq = 0; sq < N_SQUARES; ++sq)
                for (int ksq = 0; ksq < N_SQUARES; ++ksq)
                    for (int view = 0; view < 2; ++view)
                        table[FeatureIdxTableOffset(pt, color, sq, ksq, view)] =
                            FeatureIdxFormula(pt, color, sq, ksq, view);
    return table;
}

inline const auto FEATURE_IDX_TABLE = MakeFeatureIdxTable();

inline int FeatureIdx(enyo::PieceType pt,
                      enyo::Color piece_color,
                      enyo::square_t enyo_sq,
                      enyo::square_t enyo_kingsq,
                      enyo::Color view)
{
    return FEATURE_IDX_TABLE[
        FeatureIdxTableOffset(
            static_cast<int>(pt),
            static_cast<int>(piece_color),
            static_cast<int>(enyo_sq),
            static_cast<int>(enyo_kingsq),
            static_cast<int>(view))];
}

// ---------------------------------------------------------------------
// A king move triggers a full accumulator refresh when either:
//   (a) the king crosses the center file (file bit 2 changes), OR
//   (b) the king moves to a different KING_BUCKETS value.
//
// Takes Enyo h1-indexed squares and transforms them into the requested
// network perspective.
// ---------------------------------------------------------------------
inline constexpr bool MoveRequiresRefresh(enyo::PieceType pt,
                                          enyo::square_t enyo_from,
                                          enyo::square_t enyo_to,
                                          enyo::Color view)
{
    if (pt != enyo::king)
        return false;

    const int from = to_net_sq(enyo_from) ^ (56 * static_cast<int>(view));
    const int to   = to_net_sq(enyo_to) ^ (56 * static_cast<int>(view));

    // Bit 2 of the a1-indexed square is the center-file crossing.
    if ((from & 4) != (to & 4))
        return true;

    return KING_BUCKETS[static_cast<size_t>(from)] !=
           KING_BUCKETS[static_cast<size_t>(to)];
}

// ---------------------------------------------------------------------
// `Accumulator` holds the live hidden-layer state for both perspectives
// plus enough metadata for lazy updates.
//
// `AccumulatorKingState` caches the accumulator for a given
// (perspective, file_half, king_bucket) tuple.
// ---------------------------------------------------------------------

using acc_t = int16_t;

struct alignas(64) Accumulator {
    uint8_t  correct[2];          // [perspective] — set when `values[p]` is live
    uint8_t  eval_correct[2];
    uint8_t  lazy_correct[2];
    uint8_t  lazy_refresh[2];
    uint8_t  lazy_rem_count[2];
    uint8_t  lazy_add_count[2];
    uint16_t captured;            // piece captured by the move that produced this acc
    uint16_t lazy_rem[2][2];
    uint16_t lazy_add[2][2];
    uint32_t move;                // move that produced this accumulator
    int32_t  eval[2];
    alignas(64) acc_t values[2][N_HIDDEN]; // [perspective][hidden_idx]
};

struct alignas(64) AccumulatorKingState {
    acc_t    values[N_HIDDEN];
    uint64_t pcs[12]; // per-piece bitboard snapshot (for diff-based refresh)
};

// ---------------------------------------------------------------------
// Weight storage. Phase 3 fills these from the `.nn` blob; Phase 2
// tests inject a mock network by calling SetWeights().
//
// Layout matches the v13-compatible .nn format. The post-L0 path uses float32
// for L2/L3/output (different from the post-v13 main-branch code,
// which is all int16):
//
//   INPUT_WEIGHTS[N_FEATURES * N_HIDDEN]  row-major, feature -> hidden
//   INPUT_BIASES [N_HIDDEN]               added during ResetAccumulator
//   L1_WEIGHTS   [N_L1 * N_L2]            int8 sparse, row-major after shuffle
//   L1_BIASES    [N_L2]                   int32
//   L2_WEIGHTS   [N_L2_HEAD * N_L3]       float32
//   L2_BIASES    [N_L3]                   float32
//   OUTPUT_WEIGHTS[N_L3 * N_OUTPUT]       float32
//   OUTPUT_BIAS                           float32
// ---------------------------------------------------------------------
extern const acc_t* INPUT_WEIGHTS;
extern const acc_t* INPUT_BIASES;

extern const int8_t*  L1_WEIGHTS;
extern const int8_t*  L1_WEIGHTS_T;
extern const int8_t*  L1_WEIGHTS_SPARSE;
extern uint16_t       LOOKUP_INDICES[256][8];
extern const int32_t* L1_BIASES;
extern const float*   L2_WEIGHTS;
extern const float*   L2_BIASES;
extern const float*   OUTPUT_WEIGHTS;
extern float          OUTPUT_BIAS;
extern bool           INPUT_LAYOUT_SCRAMBLED;
extern uint64_t       NETWORK_GENERATION;

// Phase-2 test helper: point the weight externs at externally-allocated
// arrays.  Caller is responsible for the storage outliving subsequent
// accumulator calls.
void SetWeights(const acc_t* weights, const acc_t* biases);

// Load a full `.nn` blob from disk. Returns true on success.
// The blob layout must match NETWORK_SIZE below. Weight
// pointers are updated to reference internal storage owned by nnue_model.cpp.
bool LoadNetwork(const char* path);

inline constexpr size_t LEGACY_NETWORK_SIZE =
      sizeof(int16_t) * N_FEATURES * N_HIDDEN  // input weights
    + sizeof(int16_t) * N_HIDDEN               // input biases
    + sizeof(int8_t)  * N_L1 * N_L2            // L1 weights
    + sizeof(int32_t) * N_L2                   // L1 biases
    + sizeof(float)   * N_L2 * N_L3            // L2 weights
    + sizeof(float)   * N_L3                   // L2 biases
    + sizeof(float)   * N_L3 * N_OUTPUT        // output weights
    + sizeof(float);                           // output bias

// Total byte size of Enyo material/phase-head .nn files. Legacy files
// exactly LEGACY_NETWORK_SIZE bytes are still accepted by LoadNetwork().
inline constexpr size_t NETWORK_SIZE =
      sizeof(int16_t) * N_FEATURES * N_HIDDEN  // input weights
    + sizeof(int16_t) * N_HIDDEN               // input biases
    + sizeof(int8_t)  * N_L1 * N_L2            // L1 weights
    + sizeof(int32_t) * N_L2                   // L1 biases
    + sizeof(float)   * N_L2_HEAD * N_L3       // L2 weights
    + sizeof(float)   * N_L3                   // L2 biases
    + sizeof(float)   * N_L3 * N_OUTPUT        // output weights
    + sizeof(float);                           // output bias

// ---------------------------------------------------------------------
// Delta: up to 32 features to add and 32 to remove between two
// accumulator states.
// ---------------------------------------------------------------------
struct Delta {
    uint8_t r = 0;
    uint8_t a = 0;
    int rem[32];
    int add[32];
};

// ---------------------------------------------------------------------
// Applies `-INPUT_WEIGHTS[rem[i]*N_HIDDEN ..]` and
// `+INPUT_WEIGHTS[add[i]*N_HIDDEN ..]` to a 1024-wide accumulator row.
// `dest` and `src` may alias for in-place refresh.
//
// SIMD variants process 8 (NEON) / 16 (AVX2) / 32 (AVX-512) int16 lanes
// per iteration. Scalar fallback is the literal definition.
// ---------------------------------------------------------------------
inline void ApplyDelta(acc_t* dest, const acc_t* src, const Delta* delta) {
    if (delta->r == 0 && delta->a == 0) {
        if (dest != src)
            std::memcpy(dest, src, sizeof(acc_t) * N_HIDDEN);
        return;
    }
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (size_t i = 0; i < N_HIDDEN; i += 8) {
        auto v = vld1q_s16(&src[i]);
        for (size_t r = 0; r < delta->r; ++r) {
            const acc_t* w = &INPUT_WEIGHTS[static_cast<size_t>(delta->rem[r]) * N_HIDDEN + i];
            v = vsubq_s16(v, vld1q_s16(w));
        }
        for (size_t a = 0; a < delta->a; ++a) {
            const acc_t* w = &INPUT_WEIGHTS[static_cast<size_t>(delta->add[a]) * N_HIDDEN + i];
            v = vaddq_s16(v, vld1q_s16(w));
        }
        vst1q_s16(&dest[i], v);
    }
#elif defined(__AVX512BW__)
    for (size_t i = 0; i < N_HIDDEN; i += 32) {
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&src[i]));
        for (size_t r = 0; r < delta->r; ++r) {
            const acc_t* w = &INPUT_WEIGHTS[static_cast<size_t>(delta->rem[r]) * N_HIDDEN + i];
            __m512i wv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w));
            v = _mm512_sub_epi16(v, wv);
        }
        for (size_t a = 0; a < delta->a; ++a) {
            const acc_t* w = &INPUT_WEIGHTS[static_cast<size_t>(delta->add[a]) * N_HIDDEN + i];
            __m512i wv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w));
            v = _mm512_add_epi16(v, wv);
        }
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(&dest[i]), v);
    }
#elif defined(__AVX2__)
    for (size_t i = 0; i < N_HIDDEN; i += 16) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i]));
        for (size_t r = 0; r < delta->r; ++r) {
            const acc_t* w = &INPUT_WEIGHTS[static_cast<size_t>(delta->rem[r]) * N_HIDDEN + i];
            __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
            v = _mm256_sub_epi16(v, wv);
        }
        for (size_t a = 0; a < delta->a; ++a) {
            const acc_t* w = &INPUT_WEIGHTS[static_cast<size_t>(delta->add[a]) * N_HIDDEN + i];
            __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
            v = _mm256_add_epi16(v, wv);
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dest[i]), v);
    }
#else
    for (size_t i = 0; i < N_HIDDEN; ++i) {
        int value = src[i];
        for (size_t r = 0; r < delta->r; ++r)
            value -= INPUT_WEIGHTS[static_cast<size_t>(delta->rem[r]) * N_HIDDEN + i];
        for (size_t a = 0; a < delta->a; ++a)
            value += INPUT_WEIGHTS[static_cast<size_t>(delta->add[a]) * N_HIDDEN + i];
        dest[i] = static_cast<acc_t>(value);
    }
#endif
}

// ApplySubAdd — remove feature f1, add feature f2. The hot path for
// quiet moves (piece from f1 to f2).
inline void ApplySubAdd(acc_t* dest, const acc_t* src, int f1, int f2) {
    const acc_t* w1 = &INPUT_WEIGHTS[static_cast<size_t>(f1) * N_HIDDEN];
    const acc_t* w2 = &INPUT_WEIGHTS[static_cast<size_t>(f2) * N_HIDDEN];
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (size_t i = 0; i < N_HIDDEN; i += 8) {
        auto v = vld1q_s16(&src[i]);
        v = vsubq_s16(v, vld1q_s16(&w1[i]));
        v = vaddq_s16(v, vld1q_s16(&w2[i]));
        vst1q_s16(&dest[i], v);
    }
#elif defined(__AVX512BW__)
    for (size_t i = 0; i < N_HIDDEN; i += 32) {
        auto v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&src[i]));
        v = _mm512_sub_epi16(v, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&w1[i])));
        v = _mm512_add_epi16(v, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&w2[i])));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(&dest[i]), v);
    }
#elif defined(__AVX2__)
    for (size_t i = 0; i < N_HIDDEN; i += 16) {
        auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i]));
        v = _mm256_sub_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w1[i])));
        v = _mm256_add_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w2[i])));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dest[i]), v);
    }
#else
    for (size_t i = 0; i < N_HIDDEN; ++i)
        dest[i] = static_cast<acc_t>(src[i] - w1[i] + w2[i]);
#endif
}

// ApplySubSubAdd — captures. Remove attacker-from, remove captured-to,
// add attacker-to.
inline void ApplySubSubAdd(acc_t* dest, const acc_t* src,
                           int f1, int f2, int f3) {
    const acc_t* w1 = &INPUT_WEIGHTS[static_cast<size_t>(f1) * N_HIDDEN];
    const acc_t* w2 = &INPUT_WEIGHTS[static_cast<size_t>(f2) * N_HIDDEN];
    const acc_t* w3 = &INPUT_WEIGHTS[static_cast<size_t>(f3) * N_HIDDEN];
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (size_t i = 0; i < N_HIDDEN; i += 8) {
        auto v = vld1q_s16(&src[i]);
        v = vsubq_s16(v, vld1q_s16(&w1[i]));
        v = vsubq_s16(v, vld1q_s16(&w2[i]));
        v = vaddq_s16(v, vld1q_s16(&w3[i]));
        vst1q_s16(&dest[i], v);
    }
#elif defined(__AVX512BW__)
    for (size_t i = 0; i < N_HIDDEN; i += 32) {
        auto v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&src[i]));
        v = _mm512_sub_epi16(v, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&w1[i])));
        v = _mm512_sub_epi16(v, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&w2[i])));
        v = _mm512_add_epi16(v, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&w3[i])));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(&dest[i]), v);
    }
#elif defined(__AVX2__)
    for (size_t i = 0; i < N_HIDDEN; i += 16) {
        auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i]));
        v = _mm256_sub_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w1[i])));
        v = _mm256_sub_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w2[i])));
        v = _mm256_add_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w3[i])));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dest[i]), v);
    }
#else
    for (size_t i = 0; i < N_HIDDEN; ++i)
        dest[i] = static_cast<acc_t>(src[i] - w1[i] - w2[i] + w3[i]);
#endif
}

// ApplySubSubAddAdd — castling. Remove king-from, remove rook-from,
// add king-to, add rook-to.
inline void ApplySubSubAddAdd(acc_t* dest, const acc_t* src,
                              int f1, int f2, int f3, int f4) {
    const acc_t* w1 = &INPUT_WEIGHTS[static_cast<size_t>(f1) * N_HIDDEN];
    const acc_t* w2 = &INPUT_WEIGHTS[static_cast<size_t>(f2) * N_HIDDEN];
    const acc_t* w3 = &INPUT_WEIGHTS[static_cast<size_t>(f3) * N_HIDDEN];
    const acc_t* w4 = &INPUT_WEIGHTS[static_cast<size_t>(f4) * N_HIDDEN];
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (size_t i = 0; i < N_HIDDEN; i += 8) {
        auto v = vld1q_s16(&src[i]);
        v = vsubq_s16(v, vld1q_s16(&w1[i]));
        v = vsubq_s16(v, vld1q_s16(&w2[i]));
        v = vaddq_s16(v, vld1q_s16(&w3[i]));
        v = vaddq_s16(v, vld1q_s16(&w4[i]));
        vst1q_s16(&dest[i], v);
    }
#elif defined(__AVX512BW__)
    for (size_t i = 0; i < N_HIDDEN; i += 32) {
        auto v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&src[i]));
        v = _mm512_sub_epi16(v, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&w1[i])));
        v = _mm512_sub_epi16(v, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&w2[i])));
        v = _mm512_add_epi16(v, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&w3[i])));
        v = _mm512_add_epi16(v, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&w4[i])));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(&dest[i]), v);
    }
#elif defined(__AVX2__)
    for (size_t i = 0; i < N_HIDDEN; i += 16) {
        auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i]));
        v = _mm256_sub_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w1[i])));
        v = _mm256_sub_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w2[i])));
        v = _mm256_add_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w3[i])));
        v = _mm256_add_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w4[i])));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dest[i]), v);
    }
#else
    for (size_t i = 0; i < N_HIDDEN; ++i)
        dest[i] = static_cast<acc_t>(src[i] - w1[i] - w2[i] + w3[i] + w4[i]);
#endif
}

// ---------------------------------------------------------------------
// Reset a king-state refresh table to all-biases, no pieces.
// ---------------------------------------------------------------------
inline void ResetRefreshTable(AccumulatorKingState* refreshTable) {
    for (size_t b = 0; b < 2 * 2 * N_KING_BUCKETS; ++b) {
        AccumulatorKingState* state = refreshTable + b;
        std::memcpy(state->values, INPUT_BIASES, sizeof(acc_t) * N_HIDDEN);
        std::memset(state->pcs, 0, sizeof(state->pcs));
    }
}

// ---------------------------------------------------------------------
// Board "piece list" contract: the caller supplies occupied Enyo squares
// and packed 0..11 piece codes for FeatureIdx().
// ---------------------------------------------------------------------
struct PieceEntry {
    int piece_code;
    enyo::square_t sq;   // Enyo h1-indexed square
};

// Starts from bias, adds feature weights for every piece on the board.
// `king_enyo_sq` is the side-to-move's king square (h1-indexed).
inline void ResetAccumulator(Accumulator* dest, enyo::Color view,
                             enyo::square_t king_enyo_sq,
                             const PieceEntry* pieces, size_t npieces)
{
    const int v = static_cast<int>(view);
    acc_t* values = dest->values[v];
    std::memcpy(values, INPUT_BIASES, sizeof(acc_t) * N_HIDDEN);

    Delta delta;
    delta.r = 0;
    delta.a = 0;
    for (size_t i = 0; i < npieces; ++i) {
        const int piece = pieces[i].piece_code;
        const auto pt   = static_cast<enyo::PieceType>((piece >> 1) + 1);
        const auto pc   = static_cast<enyo::Color>(piece & 1);
        delta.add[delta.a++] = FeatureIdx(pt, pc, pieces[i].sq,
                                          king_enyo_sq, view);
    }
    ApplyDelta(values, values, &delta);
    dest->correct[v] = 1;
    dest->eval_correct[enyo::white] = 0;
    dest->eval_correct[enyo::black] = 0;
}

// ---------------------------------------------------------------------
// Integration helper: walk a Board and emit the (piece, sq)
// entries that ResetAccumulator expects. Forward-declared here so the
// header stays decoupled from board.hpp; implemented in nnue_model.cpp.
//
// Writes up to 32 entries to `out`, returns the number written.
// ---------------------------------------------------------------------
size_t enumerate_pieces(const enyo::Board& b, PieceEntry* out);
float PhaseHeadInput(const enyo::Board& b);
int ScaleEval(const enyo::Board& b, int score);

// Convenience — full board evaluate in centipawns (stm-relative). Builds
// both accumulators from scratch, runs Propagate. No incremental update,
// no king-bucket refresh cache. Phase-4-v1 correctness-first path.
int EvaluateFromScratch(const enyo::Board& b);

// Phase 6: runtime toggle for the new eval. When `enabled` is true and
// a network is loaded, the engine's `evaluate()` routes through
// EvaluateFromScratch instead of the old 512-hidden NNUE.
// Slow (no incremental / SIMD) — correctness path first, speed in Phase 7.
extern bool enabled;

// ---------------------------------------------------------------------
// Forward pass.
// ---------------------------------------------------------------------
//
// Full chain, given a live Accumulator and side-to-move:
//
//   x0 = InputReLU(acc, stm)              // int8,  length 2*N_HIDDEN = N_L1
//   x1 = L1AffineReLU(x0)                 // float, length N_L2
//   x2 = L2AffineReLU(x1)                 // float, length N_L3
//   out = L3Transform(x2)                 // float
//   return out / 32                       // centipawns
//
// Quantization constant: QUANT1_BITS = 5. clipped ReLU clamps to [0,127<<5]
// then right-shifts by 5 to fit int8.
// ---------------------------------------------------------------------
inline constexpr int QUANT1_BITS = 5;
inline constexpr int SPARSE_CHUNK_SIZE = 4;

inline constexpr size_t WeightIdxScrambled(size_t idx) {
    return ((idx / SPARSE_CHUNK_SIZE) % (N_L1 / SPARSE_CHUNK_SIZE) *
            N_L2 * SPARSE_CHUNK_SIZE) +
           (idx / N_L1 * SPARSE_CHUNK_SIZE) +
           (idx % SPARSE_CHUNK_SIZE);
}

// InputReLU — concat (us, them) accumulator rows, clipped to int8.
inline void InputReLU(int8_t* outputs, const Accumulator* acc, int stm) {
    const int views[2] = {stm, !stm};
    constexpr int maxv = 127 << QUANT1_BITS;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const auto max_vec = vdupq_n_s16(maxv);
    for (int v = 0; v < 2; ++v) {
        const acc_t* in = acc->values[views[v]];
        int8_t* out = &outputs[N_HIDDEN * v];
        for (size_t i = 0; i < N_HIDDEN; i += 16) {
            auto lo = vld1q_s16(&in[i]);
            auto hi = vld1q_s16(&in[i + 8]);
            const auto lo8 = vqshrun_n_s16(vminq_s16(lo, max_vec), QUANT1_BITS);
            const auto hi8 = vqshrun_n_s16(vminq_s16(hi, max_vec), QUANT1_BITS);
            vst1q_s8(
                &out[i],
                vreinterpretq_s8_u8(vcombine_u8(lo8, hi8)));
        }
    }
#elif defined(__AVX512BW__)
    if (INPUT_LAYOUT_SCRAMBLED) {
        for (int v = 0; v < 2; ++v) {
            const acc_t* in = acc->values[views[v]];
            int8_t* out = &outputs[N_HIDDEN * v];
            for (size_t i = 0; i < N_HIDDEN / 64; i += 2) {
                const auto s0 = _mm512_srai_epi16(
                    _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&in[64 * i + 0])), QUANT1_BITS);
                const auto s1 = _mm512_srai_epi16(
                    _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&in[64 * i + 32])), QUANT1_BITS);
                const auto s2 = _mm512_srai_epi16(
                    _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&in[64 * i + 64])), QUANT1_BITS);
                const auto s3 = _mm512_srai_epi16(
                    _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&in[64 * i + 96])), QUANT1_BITS);
                const auto zero = _mm512_setzero_si512();
                _mm512_storeu_si512(
                    reinterpret_cast<__m512i*>(&out[64 * i]),
                    _mm512_max_epi8(_mm512_packs_epi16(s0, s1), zero));
                _mm512_storeu_si512(
                    reinterpret_cast<__m512i*>(&out[64 * i + 64]),
                    _mm512_max_epi8(_mm512_packs_epi16(s2, s3), zero));
            }
        }
        return;
    }
    const auto zero = _mm512_setzero_si512();
    const auto max_vec = _mm512_set1_epi16(maxv);
    for (int v = 0; v < 2; ++v) {
        const acc_t* in = acc->values[views[v]];
        int8_t* out = &outputs[N_HIDDEN * v];
        for (size_t i = 0; i < N_HIDDEN; i += 32) {
            auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&in[i]));
            x = _mm512_min_epi16(_mm512_max_epi16(x, zero), max_vec);
            x = _mm512_srli_epi16(x, QUANT1_BITS);
            const auto y = _mm512_cvtsepi16_epi8(x);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out[i]), y);
        }
    }
#elif defined(__AVX2__)
    if (INPUT_LAYOUT_SCRAMBLED) {
        for (int v = 0; v < 2; ++v) {
            const acc_t* in = acc->values[views[v]];
            int8_t* out = &outputs[N_HIDDEN * v];
            for (size_t i = 0; i < N_HIDDEN / 32; i += 2) {
                const auto s0 = _mm256_srai_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&in[32 * i + 0])), QUANT1_BITS);
                const auto s1 = _mm256_srai_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&in[32 * i + 16])), QUANT1_BITS);
                const auto s2 = _mm256_srai_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&in[32 * i + 32])), QUANT1_BITS);
                const auto s3 = _mm256_srai_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&in[32 * i + 48])), QUANT1_BITS);
                const auto zero = _mm256_setzero_si256();
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(&out[32 * i]),
                    _mm256_max_epi8(_mm256_packs_epi16(s0, s1), zero));
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(&out[32 * i + 32]),
                    _mm256_max_epi8(_mm256_packs_epi16(s2, s3), zero));
            }
        }
        return;
    }
    const auto zero = _mm256_setzero_si256();
    const auto max_vec = _mm256_set1_epi16(maxv);
    for (int v = 0; v < 2; ++v) {
        const acc_t* in = acc->values[views[v]];
        int8_t* out = &outputs[N_HIDDEN * v];
        for (size_t i = 0; i < N_HIDDEN; i += 32) {
            auto lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&in[i]));
            auto hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&in[i + 16]));
            lo = _mm256_min_epi16(_mm256_max_epi16(lo, zero), max_vec);
            hi = _mm256_min_epi16(_mm256_max_epi16(hi, zero), max_vec);
            lo = _mm256_srli_epi16(lo, QUANT1_BITS);
            hi = _mm256_srli_epi16(hi, QUANT1_BITS);
            auto packed = _mm256_packus_epi16(lo, hi);
            packed = _mm256_permute4x64_epi64(packed, 0xD8);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out[i]), packed);
        }
    }
#else
    for (int v = 0; v < 2; ++v) {
        const acc_t* in = acc->values[views[v]];
        int8_t* out     = &outputs[N_HIDDEN * v];
        for (size_t i = 0; i < N_HIDDEN; ++i) {
            int x = in[i];
            if (x < 0) x = 0;
            if (x > maxv) x = maxv;
            out[i] = static_cast<int8_t>(x >> QUANT1_BITS);
        }
    }
#endif
}

#if defined(__AVX2__)
inline uint32_t NNZ(__m256i chunk) {
    return static_cast<uint32_t>(
        _mm256_movemask_ps(
            _mm256_castsi256_ps(
                _mm256_cmpgt_epi32(chunk, _mm256_setzero_si256()))));
}

inline size_t FindNNZ(uint16_t* dest, const int32_t* inputs, size_t chunks) {
    constexpr size_t in_width = sizeof(__m256i) / sizeof(int32_t);
    constexpr size_t chunk_size = in_width;
    const size_t num_chunks = chunks / chunk_size;
    constexpr size_t in_per_chunk = chunk_size / in_width;
    constexpr size_t out_per_chunk = chunk_size / 8;

    const auto* in = reinterpret_cast<const __m256i*>(inputs);
    const __m128i increment = _mm_set1_epi16(8);
    __m128i base = _mm_setzero_si128();
    size_t count = 0;

    (void)chunks;
    for (size_t i = 0; i < num_chunks; ++i) {
        uint32_t nnz = 0;
        for (size_t j = 0; j < in_per_chunk; ++j) {
            const __m256i input_chunk = in[i * in_per_chunk + j];
            nnz |= NNZ(input_chunk) << (j * in_width);
        }

        for (size_t j = 0; j < out_per_chunk; ++j) {
            const uint16_t lookup = static_cast<uint16_t>((nnz >> (j * 8)) & 0xFF);
            const __m128i offsets = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(&LOOKUP_INDICES[lookup]));
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dest + count),
                _mm_add_epi16(base, offsets));
            count += static_cast<size_t>(__builtin_popcount(lookup));
            base = _mm_add_epi16(base, increment);
        }
    }

    return count;
}
#endif

// L1AffineReLU — sparse int8 matmul + ReLU.  Writes float output.
// dest: float[N_L2], src: int8[N_L1].
inline void L1AffineReLU(float* dest, const int8_t* src) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#if defined(__ARM_FEATURE_DOTPROD)
    if (L1_WEIGHTS_SPARSE != nullptr) {
        int32x4_t a0 = vld1q_s32(&L1_BIASES[0]);
        int32x4_t a1 = vld1q_s32(&L1_BIASES[4]);
        int32x4_t a2 = vld1q_s32(&L1_BIASES[8]);
        int32x4_t a3 = vld1q_s32(&L1_BIASES[12]);
        int32x4_t b0 = vdupq_n_s32(0);
        int32x4_t b1 = vdupq_n_s32(0);
        int32x4_t b2 = vdupq_n_s32(0);
        int32x4_t b3 = vdupq_n_s32(0);

        const int32_t* in32 = reinterpret_cast<const int32_t*>(src);
        for (size_t i = 0; i < N_L1 / SPARSE_CHUNK_SIZE; i += 2) {
            const int8x16_t input0 = vreinterpretq_s8_s32(vdupq_n_s32(in32[i]));
            const int8_t* w0 = &L1_WEIGHTS_SPARSE[i * N_L2 * SPARSE_CHUNK_SIZE];
            a0 = vdotq_s32(a0, vld1q_s8(&w0[0]), input0);
            a1 = vdotq_s32(a1, vld1q_s8(&w0[16]), input0);
            a2 = vdotq_s32(a2, vld1q_s8(&w0[32]), input0);
            a3 = vdotq_s32(a3, vld1q_s8(&w0[48]), input0);

            const int8x16_t input1 = vreinterpretq_s8_s32(vdupq_n_s32(in32[i + 1]));
            const int8_t* w1 = &L1_WEIGHTS_SPARSE[(i + 1) * N_L2 * SPARSE_CHUNK_SIZE];
            b0 = vdotq_s32(b0, vld1q_s8(&w1[0]), input1);
            b1 = vdotq_s32(b1, vld1q_s8(&w1[16]), input1);
            b2 = vdotq_s32(b2, vld1q_s8(&w1[32]), input1);
            b3 = vdotq_s32(b3, vld1q_s8(&w1[48]), input1);
        }

        a0 = vaddq_s32(a0, b0);
        a1 = vaddq_s32(a1, b1);
        a2 = vaddq_s32(a2, b2);
        a3 = vaddq_s32(a3, b3);

        a0 = vmaxq_s32(a0, vdupq_n_s32(0));
        a1 = vmaxq_s32(a1, vdupq_n_s32(0));
        a2 = vmaxq_s32(a2, vdupq_n_s32(0));
        a3 = vmaxq_s32(a3, vdupq_n_s32(0));

        vst1q_f32(&dest[0], vcvtq_f32_s32(a0));
        vst1q_f32(&dest[4], vcvtq_f32_s32(a1));
        vst1q_f32(&dest[8], vcvtq_f32_s32(a2));
        vst1q_f32(&dest[12], vcvtq_f32_s32(a3));
        return;
    }
#endif
    if (L1_WEIGHTS_SPARSE != nullptr) {
        int32_t acc[N_L2];
        for (size_t i = 0; i < N_L2; ++i)
            acc[i] = L1_BIASES[i];

        const int32_t* in32 = reinterpret_cast<const int32_t*>(src);
        for (size_t i = 0; i < N_L1 / SPARSE_CHUNK_SIZE; ++i) {
            const int32_t packed = in32[i];
            if (packed == 0)
                continue;

            const auto* s = reinterpret_cast<const int8_t*>(&packed);
            const int8_t* w = &L1_WEIGHTS_SPARSE[i * N_L2 * SPARSE_CHUNK_SIZE];
            for (size_t j = 0; j < N_L2; ++j) {
                acc[j] += static_cast<int32_t>(s[0]) * static_cast<int32_t>(w[0])
                    + static_cast<int32_t>(s[1]) * static_cast<int32_t>(w[1])
                    + static_cast<int32_t>(s[2]) * static_cast<int32_t>(w[2])
                    + static_cast<int32_t>(s[3]) * static_cast<int32_t>(w[3]);
                w += SPARSE_CHUNK_SIZE;
            }
        }

        for (size_t i = 0; i < N_L2; ++i) {
            float v = static_cast<float>(acc[i]);
            dest[i] = v < 0.0f ? 0.0f : v;
        }
        return;
    }
    if (L1_WEIGHTS_T != nullptr) {
        int32x4_t a0 = vld1q_s32(&L1_BIASES[0]);
        int32x4_t a1 = vld1q_s32(&L1_BIASES[4]);
        int32x4_t a2 = vld1q_s32(&L1_BIASES[8]);
        int32x4_t a3 = vld1q_s32(&L1_BIASES[12]);

        for (size_t i = 0; i < N_L1; ++i) {
            if (!src[i])
                continue;

            const auto s = static_cast<int16_t>(src[i]);
            const auto sv = vdupq_n_s16(s);
            const int8_t* w = &L1_WEIGHTS_T[i * N_L2];
            const int8x16_t wb = vld1q_s8(w);
            const int16x8_t w0 = vmovl_s8(vget_low_s8(wb));
            const int16x8_t w1 = vmovl_s8(vget_high_s8(wb));

            a0 = vmlal_s16(a0, vget_low_s16(w0), vget_low_s16(sv));
            a1 = vmlal_s16(a1, vget_high_s16(w0), vget_high_s16(sv));
            a2 = vmlal_s16(a2, vget_low_s16(w1), vget_low_s16(sv));
            a3 = vmlal_s16(a3, vget_high_s16(w1), vget_high_s16(sv));
        }

        alignas(16) int32_t acc[N_L2];
        vst1q_s32(&acc[0], a0);
        vst1q_s32(&acc[4], a1);
        vst1q_s32(&acc[8], a2);
        vst1q_s32(&acc[12], a3);
        for (size_t i = 0; i < N_L2; ++i) {
            float v = static_cast<float>(acc[i]);
            dest[i] = v < 0.0f ? 0.0f : v;
        }
        return;
    }
#elif defined(__AVX512BW__)
    if (L1_WEIGHTS_SPARSE != nullptr) {
        __m512i acc = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(L1_BIASES));
        const int32_t* in32 = reinterpret_cast<const int32_t*>(src);
#if !defined(__AVX512VNNI__)
        const __m512i ones = _mm512_set1_epi16(1);
#endif

        for (size_t i = 0; i < N_L1 / SPARSE_CHUNK_SIZE; ++i) {
            if (in32[i] == 0)
                continue;

            const auto input = _mm512_set1_epi32(in32[i]);
            const auto weights = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(
                    &L1_WEIGHTS_SPARSE[i * N_L2 * SPARSE_CHUNK_SIZE]));
#if defined(__AVX512VNNI__)
            acc = _mm512_dpbusd_epi32(acc, input, weights);
#else
            auto product = _mm512_maddubs_epi16(input, weights);
            product = _mm512_madd_epi16(product, ones);
            acc = _mm512_add_epi32(acc, product);
#endif
        }

        acc = _mm512_max_epi32(acc, _mm512_setzero_si512());
        _mm512_storeu_ps(dest, _mm512_cvtepi32_ps(acc));
        return;
    }
    if (L1_WEIGHTS_T != nullptr) {
        __m512i acc = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(L1_BIASES));
        for (size_t i = 0; i < N_L1; ++i) {
            if (!src[i])
                continue;

            const __m128i wb = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(&L1_WEIGHTS_T[i * N_L2]));
            const __m512i w = _mm512_cvtepi8_epi32(wb);
            const __m512i s = _mm512_set1_epi32(static_cast<int32_t>(src[i]));
            acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(w, s));
        }

        alignas(64) int32_t out[N_L2];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(out), acc);
        for (size_t i = 0; i < N_L2; ++i) {
            float v = static_cast<float>(out[i]);
            dest[i] = v < 0.0f ? 0.0f : v;
        }
        return;
    }
#elif defined(__AVX2__)
    if (L1_WEIGHTS_SPARSE != nullptr) {
        __m256i acc0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&L1_BIASES[0]));
        __m256i acc1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&L1_BIASES[8]));
        const int32_t* in32 = reinterpret_cast<const int32_t*>(src);
        constexpr size_t num_chunks = N_L1 / SPARSE_CHUNK_SIZE;
        alignas(64) uint16_t nnz[num_chunks];
        const size_t count = FindNNZ(nnz, in32, num_chunks);
#if !defined(__AVXVNNI__)
        const __m256i ones = _mm256_set1_epi16(1);
#endif

        size_t i = 0;
        for (; i + 1 < count; i += 2) {
            const uint16_t i0 = nnz[i + 0];
            const uint16_t i1 = nnz[i + 1];
            const auto input0 = _mm256_set1_epi32(in32[i0]);
            const auto input1 = _mm256_set1_epi32(in32[i1]);
            const int8_t* weights0 = &L1_WEIGHTS_SPARSE[i0 * N_L2 * SPARSE_CHUNK_SIZE];
            const int8_t* weights1 = &L1_WEIGHTS_SPARSE[i1 * N_L2 * SPARSE_CHUNK_SIZE];
            const auto w00 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&weights0[0]));
            const auto w01 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&weights0[32]));
            const auto w10 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&weights1[0]));
            const auto w11 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&weights1[32]));
#if defined(__AVXVNNI__)
            acc0 = _mm256_dpbusd_epi32(acc0, input0, w00);
            acc1 = _mm256_dpbusd_epi32(acc1, input0, w01);
            acc0 = _mm256_dpbusd_epi32(acc0, input1, w10);
            acc1 = _mm256_dpbusd_epi32(acc1, input1, w11);
#else
            auto p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(input0, w00), ones);
            auto p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(input1, w10), ones);
            acc0 = _mm256_add_epi32(_mm256_add_epi32(acc0, p0), p1);

            p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(input0, w01), ones);
            p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(input1, w11), ones);
            acc1 = _mm256_add_epi32(_mm256_add_epi32(acc1, p0), p1);
#endif
        }
        if (i < count) {
            const uint16_t idx = nnz[i];
            const auto input = _mm256_set1_epi32(in32[idx]);
            const int8_t* weights = &L1_WEIGHTS_SPARSE[idx * N_L2 * SPARSE_CHUNK_SIZE];
            const auto w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&weights[0]));
            const auto w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&weights[32]));
#if defined(__AVXVNNI__)
            acc0 = _mm256_dpbusd_epi32(acc0, input, w0);
            acc1 = _mm256_dpbusd_epi32(acc1, input, w1);
#else
            auto p0 = _mm256_maddubs_epi16(input, w0);
            auto p1 = _mm256_maddubs_epi16(input, w1);
            p0 = _mm256_madd_epi16(p0, ones);
            p1 = _mm256_madd_epi16(p1, ones);
            acc0 = _mm256_add_epi32(acc0, p0);
            acc1 = _mm256_add_epi32(acc1, p1);
#endif
        }

        acc0 = _mm256_max_epi32(acc0, _mm256_setzero_si256());
        acc1 = _mm256_max_epi32(acc1, _mm256_setzero_si256());
        _mm256_storeu_ps(&dest[0], _mm256_cvtepi32_ps(acc0));
        _mm256_storeu_ps(&dest[8], _mm256_cvtepi32_ps(acc1));
        return;
    }
    if (L1_WEIGHTS_T != nullptr) {
        __m256i acc0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&L1_BIASES[0]));
        __m256i acc1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&L1_BIASES[8]));

        for (size_t i = 0; i < N_L1; ++i) {
            if (!src[i])
                continue;

            const __m128i wb = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(&L1_WEIGHTS_T[i * N_L2]));
            const __m256i w0 = _mm256_cvtepi8_epi32(wb);
            const __m128i wh = _mm_srli_si128(wb, 8);
            const __m256i w1 = _mm256_cvtepi8_epi32(wh);
            const __m256i s = _mm256_set1_epi32(static_cast<int32_t>(src[i]));

            acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(w0, s));
            acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(w1, s));
        }

        alignas(32) int32_t out[N_L2];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out[0]), acc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out[8]), acc1);
        for (size_t i = 0; i < N_L2; ++i) {
            float v = static_cast<float>(out[i]);
            dest[i] = v < 0.0f ? 0.0f : v;
        }
        return;
    }
#endif

    int32_t acc[N_L2];
    for (size_t i = 0; i < N_L2; ++i)
        acc[i] = L1_BIASES[i];

    for (size_t i = 0; i < N_L1; ++i) {
        if (!src[i])
            continue;
        if (L1_WEIGHTS_T != nullptr) {
            const int8_t* w = &L1_WEIGHTS_T[i * N_L2];
            for (size_t j = 0; j < N_L2; ++j)
                acc[j] += static_cast<int32_t>(src[i]) *
                          static_cast<int32_t>(w[j]);
        } else {
            for (size_t j = 0; j < N_L2; ++j)
                acc[j] += static_cast<int32_t>(src[i]) *
                          static_cast<int32_t>(L1_WEIGHTS[j * N_L1 + i]);
        }
    }

    for (size_t i = 0; i < N_L2; ++i) {
        float v = static_cast<float>(acc[i]);
        dest[i] = v < 0.0f ? 0.0f : v;
    }
}

#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FEATURE_DOTPROD)
inline void L1AffineReLUFromAccumulator(float* dest, const Accumulator* acc, int stm)
{
    if (L1_WEIGHTS_SPARSE == nullptr) {
        alignas(64) int8_t x0[N_L1];
        InputReLU(x0, acc, stm);
        L1AffineReLU(dest, x0);
        return;
    }

    int32x4_t a0 = vld1q_s32(&L1_BIASES[0]);
    int32x4_t a1 = vld1q_s32(&L1_BIASES[4]);
    int32x4_t a2 = vld1q_s32(&L1_BIASES[8]);
    int32x4_t a3 = vld1q_s32(&L1_BIASES[12]);
    const int32x4_t zero = vdupq_n_s32(0);
    int32x4_t b0 = zero;
    int32x4_t b1 = zero;
    int32x4_t b2 = zero;
    int32x4_t b3 = zero;

    const int views[2] = {stm, !stm};
    constexpr int maxv = 127 << QUANT1_BITS;
    const int16x8_t max_vec = vdupq_n_s16(maxv);
    constexpr size_t chunks_per_view = N_HIDDEN / SPARSE_CHUNK_SIZE;

    for (size_t v = 0; v < 2; ++v) {
        const acc_t* in = acc->values[views[v]];
        const size_t base = v * chunks_per_view;
        for (size_t i = 0; i < chunks_per_view; i += 2) {
            const uint8x8_t relu = vqshrun_n_s16(
                vminq_s16(vld1q_s16(&in[i * SPARSE_CHUNK_SIZE]), max_vec),
                QUANT1_BITS);
            const uint32x2_t packed = vreinterpret_u32_u8(relu);

            const int8x16_t input0 = vreinterpretq_s8_u32(
                vdupq_n_u32(vget_lane_u32(packed, 0)));
            const int8_t* w0 = &L1_WEIGHTS_SPARSE[
                (base + i) * N_L2 * SPARSE_CHUNK_SIZE];
            a0 = vdotq_s32(a0, vld1q_s8(&w0[0]), input0);
            a1 = vdotq_s32(a1, vld1q_s8(&w0[16]), input0);
            a2 = vdotq_s32(a2, vld1q_s8(&w0[32]), input0);
            a3 = vdotq_s32(a3, vld1q_s8(&w0[48]), input0);

            const int8x16_t input1 = vreinterpretq_s8_u32(
                vdupq_n_u32(vget_lane_u32(packed, 1)));
            const int8_t* w1 = &L1_WEIGHTS_SPARSE[
                (base + i + 1) * N_L2 * SPARSE_CHUNK_SIZE];
            b0 = vdotq_s32(b0, vld1q_s8(&w1[0]), input1);
            b1 = vdotq_s32(b1, vld1q_s8(&w1[16]), input1);
            b2 = vdotq_s32(b2, vld1q_s8(&w1[32]), input1);
            b3 = vdotq_s32(b3, vld1q_s8(&w1[48]), input1);
        }
    }

    a0 = vmaxq_s32(vaddq_s32(a0, b0), zero);
    a1 = vmaxq_s32(vaddq_s32(a1, b1), zero);
    a2 = vmaxq_s32(vaddq_s32(a2, b2), zero);
    a3 = vmaxq_s32(vaddq_s32(a3, b3), zero);

    vst1q_f32(&dest[0], vcvtq_f32_s32(a0));
    vst1q_f32(&dest[4], vcvtq_f32_s32(a1));
    vst1q_f32(&dest[8], vcvtq_f32_s32(a2));
    vst1q_f32(&dest[12], vcvtq_f32_s32(a3));
}
#endif

#if defined(__AVX2__)
inline __m128 hadd_psx4(__m256* regs) {
    regs[0] = _mm256_hadd_ps(regs[0], regs[1]);
    regs[2] = _mm256_hadd_ps(regs[2], regs[3]);
    regs[0] = _mm256_hadd_ps(regs[0], regs[2]);

    const __m128 lo = _mm256_castps256_ps128(regs[0]);
    const __m128 hi = _mm256_extractf128_ps(regs[0], 1);
    return _mm_add_ps(lo, hi);
}
#endif

// L2AffineReLU — dense float matmul + ReLU. dest: float[N_L3],
// src: float[N_L2], plus one material/phase head input.
inline void L2AffineReLU(float* dest, const float* src, float phase_head) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const float32x4_t src0 = vld1q_f32(&src[0]);
    const float32x4_t src1 = vld1q_f32(&src[4]);
    const float32x4_t src2 = vld1q_f32(&src[8]);
    const float32x4_t src3 = vld1q_f32(&src[12]);
    for (int i = 0; i < N_L3; ++i) {
        const int offset = i * N_L2_HEAD;
        float32x4_t s0 = vmulq_f32(src0, vld1q_f32(&L2_WEIGHTS[offset + 0]));
        float32x4_t s1 = vmulq_f32(src1, vld1q_f32(&L2_WEIGHTS[offset + 4]));
        float32x4_t s2 = vmulq_f32(src2, vld1q_f32(&L2_WEIGHTS[offset + 8]));
        float32x4_t s3 = vmulq_f32(src3, vld1q_f32(&L2_WEIGHTS[offset + 12]));
        const float s = L2_BIASES[i]
            + vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)))
            + phase_head * L2_WEIGHTS[offset + N_L2];
        dest[i] = s < 0.0f ? 0.0f : s;
    }
#elif defined(__AVX2__)
    constexpr size_t in_width = sizeof(__m256) / sizeof(float);
    constexpr size_t in_chunks = N_L2 / in_width;
    constexpr size_t out_cc = 8;
    constexpr size_t out_chunks = N_L3 / out_cc;

    __m256 regs[out_cc];
    for (size_t i = 0; i < out_chunks; ++i) {
        for (size_t k = 0; k < out_cc; ++k)
            regs[k] = _mm256_setzero_ps();

        for (size_t j = 0; j < in_chunks; ++j) {
            const __m256 in = _mm256_loadu_ps(&src[j * in_width]);
            for (size_t k = 0; k < out_cc; ++k) {
                const size_t output = out_cc * i + k;
                const __m256 weights = _mm256_loadu_ps(
                    &L2_WEIGHTS[output * N_L2_HEAD + j * in_width]);
                regs[k] = _mm256_fmadd_ps(in, weights, regs[k]);
            }
        }

        const __m128 s0 = hadd_psx4(regs);
        const __m128 s1 = hadd_psx4(&regs[4]);
        __m256 sum = _mm256_insertf128_ps(_mm256_castps128_ps256(s0), s1, 1);
        const size_t output = out_cc * i;
        const __m256 phase_weights = _mm256_setr_ps(
            L2_WEIGHTS[(output + 0) * N_L2_HEAD + N_L2],
            L2_WEIGHTS[(output + 1) * N_L2_HEAD + N_L2],
            L2_WEIGHTS[(output + 2) * N_L2_HEAD + N_L2],
            L2_WEIGHTS[(output + 3) * N_L2_HEAD + N_L2],
            L2_WEIGHTS[(output + 4) * N_L2_HEAD + N_L2],
            L2_WEIGHTS[(output + 5) * N_L2_HEAD + N_L2],
            L2_WEIGHTS[(output + 6) * N_L2_HEAD + N_L2],
            L2_WEIGHTS[(output + 7) * N_L2_HEAD + N_L2]);
        sum = _mm256_fmadd_ps(
            _mm256_set1_ps(phase_head),
            phase_weights,
            sum);
        const __m256 bias = _mm256_loadu_ps(&L2_BIASES[i * out_cc]);
        sum = _mm256_max_ps(_mm256_add_ps(sum, bias), _mm256_setzero_ps());
        _mm256_storeu_ps(&dest[i * out_cc], sum);
    }
#else
    for (int i = 0; i < N_L3; ++i) {
        const int offset = i * N_L2_HEAD;
        float s = L2_BIASES[i];
        for (int j = 0; j < N_L2; ++j)
            s += src[j] * L2_WEIGHTS[offset + j];
        s += phase_head * L2_WEIGHTS[offset + N_L2];
        dest[i] = s < 0.0f ? 0.0f : s;
    }
#endif
}

// L3Transform returns post-shift units; Propagate divides by 32.
inline float L3Transform(const float* src) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < N_L3; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(&src[i]), vld1q_f32(&OUTPUT_WEIGHTS[i]));
    return vaddvq_f32(acc) + OUTPUT_BIAS;
#elif defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    acc0 = _mm512_fmadd_ps(
        _mm512_loadu_ps(&src[0]),
        _mm512_loadu_ps(&OUTPUT_WEIGHTS[0]),
        acc0);
    acc1 = _mm512_fmadd_ps(
        _mm512_loadu_ps(&src[16]),
        _mm512_loadu_ps(&OUTPUT_WEIGHTS[16]),
        acc1);
    const __m512 acc = _mm512_add_ps(acc0, acc1);
    const __m256 lo = _mm512_castps512_ps256(acc);
    const __m256 hi = _mm512_extractf32x8_ps(acc, 1);
    const __m256 sum8 = _mm256_add_ps(lo, hi);
    const __m128 sum4 = _mm_add_ps(
        _mm256_castps256_ps128(sum8),
        _mm256_extractf128_ps(sum8, 1));
    const __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    const __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 0x1));
    return _mm_cvtss_f32(sum1) + OUTPUT_BIAS;
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (size_t i = 0; i < N_L3; i += 8) {
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(&src[i]),
            _mm256_loadu_ps(&OUTPUT_WEIGHTS[i]),
            acc);
    }
    const __m128 sum4 = _mm_add_ps(
        _mm256_castps256_ps128(acc),
        _mm256_extractf128_ps(acc, 1));
    const __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    const __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 0x1));
    return _mm_cvtss_f32(sum1) + OUTPUT_BIAS;
#else
    float s = OUTPUT_BIAS;
    for (int i = 0; i < N_L3; ++i)
        s += src[i] * OUTPUT_WEIGHTS[i];
    return s;
#endif
}

// Full forward pass, scalar only. Returns eval in centipawns (stm-relative).
inline int Propagate(const Accumulator* acc, int stm, float phase_head) {
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FEATURE_DOTPROD)
    alignas(64) float x1[N_L2];
    alignas(64) float x2[N_L3];
    L1AffineReLUFromAccumulator(x1, acc, stm);
    L2AffineReLU(x2, x1, phase_head);
    return static_cast<int>(L3Transform(x2) / 32.0f);
#else
    alignas(64) int8_t x0[N_L1];
    alignas(64) float  x1[N_L2];
    alignas(64) float  x2[N_L3];
    InputReLU(x0, acc, stm);
    L1AffineReLU(x1, x0);
    L2AffineReLU(x2, x1, phase_head);
    return static_cast<int>(L3Transform(x2) / 32.0f);
#endif
}

inline int Propagate(const Accumulator* acc, int stm) {
    return Propagate(acc, stm, 0.0f);
}

} // namespace Network
