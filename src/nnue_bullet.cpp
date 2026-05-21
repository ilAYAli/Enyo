#include "nnue_bullet.hpp"

#include "board.hpp"
#include "util.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace BulletNetwork {

namespace {

alignas(64) int16_t s_l0_weights[N_INPUTS * N_HIDDEN];
alignas(64) int16_t s_l0_biases[N_HIDDEN];
alignas(64) int8_t  s_l1_weights[N_OUTPUT_BUCKETS * N_L2 * N_HIDDEN];
alignas(64) float   s_l1_weights_float[N_OUTPUT_BUCKETS * N_L2 * N_HIDDEN];
alignas(64) float   s_l1_biases[N_OUTPUT_BUCKETS * N_L2];
alignas(64) float   s_l2_weights[N_OUTPUT_BUCKETS * N_L3 * N_L2];
alignas(64) float   s_l2_biases[N_OUTPUT_BUCKETS * N_L3];
alignas(64) float   s_l3_weights[N_OUTPUT_BUCKETS * N_L3];
alignas(64) float   s_l3_biases[N_OUTPUT_BUCKETS];
alignas(64) float   s_crelu_lookup[65536];

bool s_loaded = false;
bool s_crelu_ready = false;
int s_hidden = N_HIDDEN;
int s_pairwise = N_HIDDEN / 2;

constexpr std::array<int, 32> INPUT_BUCKET_LAYOUT = {
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
};

constexpr std::array<int, 8> FILE_MIRROR = {0, 1, 2, 3, 3, 2, 1, 0};
constexpr std::array<int, 6> SUPPORTED_HIDDEN = {128, 256, 384, 512, 768, 1024};

constexpr int enyo_to_bullet_square(enyo::square_t sq)
{
    // Enyo is h1-indexed; Bullet's white-oriented board is a1-indexed.
    return static_cast<int>(sq) ^ 7;
}

int normalized_square(enyo::square_t sq, enyo::Color view)
{
    int out = enyo_to_bullet_square(sq);
    if (view == enyo::black)
        out ^= 56;
    return out;
}

int normalized_piece(enyo::PieceType pt, enyo::Color pc, enyo::Color view)
{
    int out = (static_cast<int>(pc == enyo::black) << 3)
        | (static_cast<int>(pt) - 1);
    if (view == enyo::black)
        out ^= 8;
    return out;
}

int mirrored_bucket(int normalized_king_sq)
{
    const int rank = normalized_king_sq / 8;
    const int file = normalized_king_sq % 8;
    return INPUT_BUCKET_LAYOUT[rank * 4 + FILE_MIRROR[file]];
}

int bullet_feature(int piece, int square, int king_square)
{
    const int color = (piece & 8) ? 1 : 0;
    const int pc = 64 * (piece & 7);
    const int flip = (king_square % 8 > 3) ? 7 : 0;
    const int bucket = 768 * mirrored_bucket(king_square);
    const int base = (color ? 384 : 0) + pc + square;
    return bucket + (base ^ flip);
}

int material_bucket(const enyo::Board& board)
{
    const int pieces = enyo::count_bits(
        board.color_bb[enyo::white] | board.color_bb[enyo::black]);
    return std::clamp((pieces - 2) / 4, 0, N_OUTPUT_BUCKETS - 1);
}

template <typename T>
bool read_exact(std::FILE* fh, T* dst, size_t count, const char* label)
{
    if (std::fread(dst, sizeof(T), count, fh) == count)
        return true;
    std::fprintf(stderr, "bullet network: short read at %s\n", label);
    return false;
}

bool read_padding(std::FILE* fh)
{
    std::array<unsigned char, PADDING_SIZE> padding {};
    if (std::fread(padding.data(), 1, padding.size(), fh) != padding.size()) {
        std::fprintf(stderr, "bullet network: short read at padding\n");
        return false;
    }

    constexpr std::array<unsigned char, 6> bullet = {'b', 'u', 'l', 'l', 'e', 't'};
    for (size_t i = 0; i < padding.size(); ++i) {
        if (padding[i] != bullet[i % bullet.size()]) {
            std::fprintf(stderr, "bullet network: unexpected padding byte\n");
            return false;
        }
    }
    return true;
}

float crelu_from_acc(int16_t value)
{
    return s_crelu_lookup[static_cast<int>(value) + 32768];
}

void init_crelu_lookup()
{
    if (s_crelu_ready)
        return;

    for (int value = -32768; value <= 32767; ++value) {
        const int clipped = std::clamp(value, 0, 255);
        s_crelu_lookup[value + 32768] = static_cast<float>(clipped) / 255.0f;
    }
    s_crelu_ready = true;
}

} // namespace

bool enabled = false;
uint64_t NETWORK_GENERATION = 0;

bool IsLoaded()
{
    return s_loaded;
}

std::string Description()
{
    return "Bullet/Reckless-like 10hm->" + std::to_string(s_hidden)
        + " pairwise, material-bucketed 16/32 head";
}

bool LoadNetwork(const char* path)
{
    std::FILE* fh = std::fopen(path, "rb");
    if (!fh) {
        std::fprintf(stderr, "bullet network: can't open '%s': %s\n",
                     path, std::strerror(errno));
        return false;
    }

    if (std::fseek(fh, 0, SEEK_END) != 0) {
        std::fclose(fh);
        return false;
    }
    const long sz = std::ftell(fh);
    if (sz < 0) {
        std::fclose(fh);
        return false;
    }
    int hidden = 0;
    for (const int candidate : SUPPORTED_HIDDEN) {
        if (static_cast<size_t>(sz) == NetworkSizeForHidden(candidate)) {
            hidden = candidate;
            break;
        }
    }
    if (hidden == 0) {
        std::fprintf(stderr,
                     "bullet network: '%s' is %ld bytes, expected hidden 128/256/384/512/768/1024\n",
                     path, sz);
        std::fclose(fh);
        return false;
    }
    std::rewind(fh);
    init_crelu_lookup();

    std::memset(s_l0_weights, 0, sizeof(s_l0_weights));
    std::memset(s_l0_biases, 0, sizeof(s_l0_biases));
    std::memset(s_l1_weights, 0, sizeof(s_l1_weights));
    std::memset(s_l1_weights_float, 0, sizeof(s_l1_weights_float));

    bool ok = true;
    ok = ok && read_exact(fh, s_l0_weights, static_cast<size_t>(N_INPUTS) * hidden, "l0w");
    ok = ok && read_exact(fh, s_l0_biases, hidden, "l0b");
    ok = ok && read_exact(
        fh, s_l1_weights,
        static_cast<size_t>(N_OUTPUT_BUCKETS) * N_L2 * hidden,
        "l1w");
    ok = ok && read_exact(fh, s_l1_biases, N_OUTPUT_BUCKETS * N_L2, "l1b");
    ok = ok && read_exact(
        fh, s_l2_weights,
        static_cast<size_t>(N_OUTPUT_BUCKETS) * N_L3 * N_L2,
        "l2w");
    ok = ok && read_exact(fh, s_l2_biases, N_OUTPUT_BUCKETS * N_L3, "l2b");
    ok = ok && read_exact(fh, s_l3_weights, N_OUTPUT_BUCKETS * N_L3, "l3w");
    ok = ok && read_exact(fh, s_l3_biases, N_OUTPUT_BUCKETS, "l3b");
    ok = ok && read_padding(fh);

    unsigned char probe = 0;
    if (ok && std::fread(&probe, 1, 1, fh) != 0) {
        std::fprintf(stderr, "bullet network: trailing bytes after padding\n");
        ok = false;
    }

    std::fclose(fh);
    s_loaded = ok;
    if (ok) {
        s_hidden = hidden;
        s_pairwise = hidden / 2;
        const auto n = static_cast<size_t>(N_OUTPUT_BUCKETS) * N_L2 * s_hidden;
        for (size_t i = 0; i < n; ++i)
            s_l1_weights_float[i] = static_cast<float>(s_l1_weights[i]) / 64.0f;
        ++NETWORK_GENERATION;
    }
    return ok;
}

int FeatureIdx(enyo::PieceType pt,
               enyo::Color pc,
               enyo::square_t sq,
               enyo::square_t king_sq,
               enyo::Color view)
{
    return bullet_feature(
        normalized_piece(pt, pc, view),
        normalized_square(sq, view),
        normalized_square(king_sq, view));
}

void ResetAccumulator(Accumulator* acc, const enyo::Board& board, enyo::Color view)
{
    for (int i = 0; i < s_hidden; ++i)
        acc->values[view][i] = s_l0_biases[i];

    const auto king_sq = static_cast<enyo::square_t>(
        enyo::lsb(board.pt_bb[view][enyo::king]));
    enyo::bitboard_t pieces = board.color_bb[enyo::white] | board.color_bb[enyo::black];
    while (pieces) {
        const auto sq = static_cast<enyo::square_t>(enyo::pop_lsb(pieces));
        const enyo::PieceType pt = board.pt_mb[sq];
        const enyo::Color pc = (board.color_bb[enyo::white] & (1ULL << sq))
            ? enyo::white
            : enyo::black;
        const int feature = FeatureIdx(pt, pc, sq, king_sq, view);
        const int16_t* weights = &s_l0_weights[static_cast<size_t>(feature) * s_hidden];
        for (int i = 0; i < s_hidden; ++i)
            acc->values[view][i] = static_cast<int16_t>(acc->values[view][i] + weights[i]);
    }
}

void UpdateFeature(Accumulator* acc,
                   enyo::PieceType pt,
                   enyo::Color pc,
                   enyo::square_t sq,
                   enyo::square_t king_sq,
                   enyo::Color view,
                   bool add)
{
    const int feature = FeatureIdx(pt, pc, sq, king_sq, view);
    const int16_t* weights = &s_l0_weights[static_cast<size_t>(feature) * s_hidden];
    if (add) {
        for (int i = 0; i < s_hidden; ++i)
            acc->values[view][i] = static_cast<int16_t>(acc->values[view][i] + weights[i]);
    } else {
        for (int i = 0; i < s_hidden; ++i)
            acc->values[view][i] = static_cast<int16_t>(acc->values[view][i] - weights[i]);
    }
}

void MoveFeature(Accumulator* acc,
                 enyo::PieceType pt,
                 enyo::Color pc,
                 enyo::square_t from,
                 enyo::square_t to,
                 enyo::square_t king_sq,
                 enyo::Color view)
{
    UpdateFeature(acc, pt, pc, from, king_sq, view, false);
    UpdateFeature(acc, pt, pc, to, king_sq, view, true);
}

int Propagate(const Accumulator* acc, const enyo::Board& board)
{
    float hidden[N_HIDDEN];
    const auto them = static_cast<enyo::Color>(!board.side);
    for (int i = 0; i < s_pairwise; ++i) {
        hidden[i] = crelu_from_acc(acc->values[board.side][i])
            * crelu_from_acc(acc->values[board.side][i + s_pairwise]);
        hidden[s_pairwise + i] = crelu_from_acc(acc->values[them][i])
            * crelu_from_acc(acc->values[them][i + s_pairwise]);
    }

    const int bucket = material_bucket(board);

    float l1[N_L2];
    const size_t l1_offset = static_cast<size_t>(bucket) * N_L2 * s_hidden;
    const size_t l1_bias_offset = static_cast<size_t>(bucket) * N_L2;
    for (int i = 0; i < N_L2; ++i) {
        float sum = s_l1_biases[l1_bias_offset + i];
        const float* weights = &s_l1_weights_float[l1_offset + static_cast<size_t>(i) * s_hidden];
        for (int j = 0; j < s_hidden; ++j)
            sum += hidden[j] * weights[j];
        const float clipped = std::clamp(sum, 0.0f, 1.0f);
        l1[i] = clipped * clipped;
    }

    float l2[N_L3];
    const size_t l2_offset = static_cast<size_t>(bucket) * N_L3 * N_L2;
    const size_t l2_bias_offset = static_cast<size_t>(bucket) * N_L3;
    for (int i = 0; i < N_L3; ++i) {
        float sum = s_l2_biases[l2_bias_offset + i];
        const float* weights = &s_l2_weights[l2_offset + static_cast<size_t>(i) * N_L2];
        for (int j = 0; j < N_L2; ++j)
            sum += l1[j] * weights[j];
        const float clipped = std::clamp(sum, 0.0f, 1.0f);
        l2[i] = clipped * clipped;
    }

    float out = s_l3_biases[bucket];
    const float* weights = &s_l3_weights[static_cast<size_t>(bucket) * N_L3];
    for (int i = 0; i < N_L3; ++i)
        out += l2[i] * weights[i];

    return static_cast<int>(std::lround(400.0f * out));
}

int EvaluateFromScratch(const enyo::Board& board)
{
    Accumulator acc{};
    ResetAccumulator(&acc, board, enyo::white);
    ResetAccumulator(&acc, board, enyo::black);
    return Propagate(&acc, board);
}

} // namespace BulletNetwork
