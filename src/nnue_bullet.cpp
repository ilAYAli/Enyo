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
alignas(64) int8_t  s_l1_weights[N_OUTPUT_BUCKETS * N_L2 * N_L1_INPUTS];
alignas(64) float   s_l1_biases[N_OUTPUT_BUCKETS * N_L2];
alignas(64) float   s_l2_weights[N_OUTPUT_BUCKETS * N_L3 * N_L2];
alignas(64) float   s_l2_biases[N_OUTPUT_BUCKETS * N_L3];
alignas(64) float   s_l3_weights[N_OUTPUT_BUCKETS * N_L3];
alignas(64) float   s_l3_biases[N_OUTPUT_BUCKETS];

bool s_loaded = false;

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

constexpr int enyo_to_bullet_square(enyo::square_t sq)
{
    // Enyo is h1-indexed; Bullet's white-oriented board is a1-indexed.
    return static_cast<int>(sq) ^ 7;
}

int normalized_square(enyo::square_t sq, enyo::Color stm)
{
    int out = enyo_to_bullet_square(sq);
    if (stm == enyo::black)
        out ^= 56;
    return out;
}

int normalized_piece(enyo::PieceType pt, enyo::Color pc, enyo::Color stm)
{
    int out = (static_cast<int>(pc == enyo::black) << 3)
        | (static_cast<int>(pt) - 1);
    if (stm == enyo::black)
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

float crelu_from_acc(int value)
{
    const int clipped = std::clamp(value, 0, 255);
    return static_cast<float>(clipped) / 255.0f;
}

void refresh_accumulator(float* out,
                         const enyo::Board& board,
                         enyo::Color view)
{
    int acc[N_HIDDEN];
    for (int i = 0; i < N_HIDDEN; ++i)
        acc[i] = s_l0_biases[i];

    const enyo::Color stm = board.side;
    const enyo::Color them = static_cast<enyo::Color>(!view);
    const enyo::square_t view_king = static_cast<enyo::square_t>(
        enyo::lsb(board.pt_bb[view][enyo::king]));
    const enyo::square_t other_king = static_cast<enyo::square_t>(
        enyo::lsb(board.pt_bb[them][enyo::king]));

    const int view_king_sq = view == stm
        ? normalized_square(view_king, stm)
        : (normalized_square(view_king, stm) ^ 56);
    const int other_king_sq = view == stm
        ? normalized_square(other_king, stm)
        : (normalized_square(other_king, stm) ^ 56);
    (void)other_king_sq;

    enyo::bitboard_t pieces = board.color_bb[enyo::white] | board.color_bb[enyo::black];
    while (pieces) {
        const auto sq = static_cast<enyo::square_t>(enyo::pop_lsb(pieces));
        const enyo::PieceType pt = board.pt_mb[sq];
        const enyo::Color pc = (board.color_bb[enyo::white] & (1ULL << sq))
            ? enyo::white
            : enyo::black;

        int piece = normalized_piece(pt, pc, stm);
        int square = normalized_square(sq, stm);
        if (view != stm) {
            piece ^= 8;
            square ^= 56;
        }

        const int feature = bullet_feature(piece, square, view_king_sq);
        const int16_t* weights = &s_l0_weights[static_cast<size_t>(feature) * N_HIDDEN];
        for (int i = 0; i < N_HIDDEN; ++i)
            acc[i] += weights[i];
    }

    for (int i = 0; i < N_PAIRWISE; ++i)
        out[i] = crelu_from_acc(acc[i]) * crelu_from_acc(acc[i + N_PAIRWISE]);
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
    return "Bullet/Reckless-like 768x10hm->1024 pairwise, material-bucketed 16/32 head";
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
    if (static_cast<size_t>(sz) != NETWORK_SIZE) {
        std::fprintf(stderr,
                     "bullet network: '%s' is %ld bytes, expected %zu\n",
                     path, sz, NETWORK_SIZE);
        std::fclose(fh);
        return false;
    }
    std::rewind(fh);

    bool ok = true;
    ok = ok && read_exact(fh, s_l0_weights, static_cast<size_t>(N_INPUTS) * N_HIDDEN, "l0w");
    ok = ok && read_exact(fh, s_l0_biases, N_HIDDEN, "l0b");
    ok = ok && read_exact(
        fh, s_l1_weights,
        static_cast<size_t>(N_OUTPUT_BUCKETS) * N_L2 * N_L1_INPUTS,
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
    if (ok)
        ++NETWORK_GENERATION;
    return ok;
}

int EvaluateFromScratch(const enyo::Board& board)
{
    float hidden[N_L1_INPUTS];
    refresh_accumulator(&hidden[0], board, board.side);
    refresh_accumulator(&hidden[N_PAIRWISE], board, static_cast<enyo::Color>(!board.side));

    const int bucket = material_bucket(board);

    float l1[N_L2];
    const size_t l1_offset = static_cast<size_t>(bucket) * N_L2 * N_L1_INPUTS;
    const size_t l1_bias_offset = static_cast<size_t>(bucket) * N_L2;
    for (int i = 0; i < N_L2; ++i) {
        float sum = s_l1_biases[l1_bias_offset + i];
        const int8_t* weights = &s_l1_weights[l1_offset + static_cast<size_t>(i) * N_L1_INPUTS];
        for (int j = 0; j < N_L1_INPUTS; ++j)
            sum += hidden[j] * (static_cast<float>(weights[j]) / 64.0f);
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

} // namespace BulletNetwork
