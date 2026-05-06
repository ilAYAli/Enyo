#include "nnue.hpp"
#include "movelist.hpp"
#include "movegen.hpp"
//#include "bench.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <fmt/core.h>

#define INCBIN_STYLE INCBIN_STYLE_CAMEL
#include "incbin/incbin.h"
INCBIN(EVAL, EVAL_FILE);

#if defined(__AVX__) || defined(__AVX2__)
alignas(ALIGNMENT) std::array<int16_t, INPUT_SIZE * HIDDEN_SIZE> inputWeights;
alignas(ALIGNMENT) std::array<int16_t, HIDDEN_SIZE> inputBias;
alignas(ALIGNMENT) std::array<int16_t, HIDDEN_SIZE * 2> hiddenWeights;
alignas(ALIGNMENT) std::array<int32_t, OUTPUT_SIZE> hiddenBias;
#else
std::array<int16_t, INPUT_SIZE * HIDDEN_SIZE> inputWeights;
std::array<int16_t, HIDDEN_SIZE> inputBias;
std::array<int16_t, HIDDEN_SIZE * 2> hiddenWeights;
std::array<int32_t, OUTPUT_SIZE> hiddenBias;
#endif

// PWA: remove this:
#pragma GCC diagnostic ignored "-Wsign-conversion"

namespace NNUE {

Net::Net() {
    std::fill(accumulator_stack.begin(), accumulator_stack.end(), Accumulator());
    nnue2_accumulator_stack.resize(accumulator_stack.size());
    nnue2_eval_cache.resize(nnue2_eval_cache_size);
    nnue2_eval_cache_generation = NNUE2::NETWORK_GENERATION;
    std::memset(nnue2_accumulator_stack.data(), 0,
                nnue2_accumulator_stack.size() * sizeof(NNUE2::Accumulator));
}

namespace {

bool use_nnue2() {
    return NNUE2::enabled && NNUE2::INPUT_WEIGHTS != nullptr;
}

void invalidate_nnue2_eval(NNUE2::Accumulator & accumulator)
{
    accumulator.eval_correct[enyo::white] = 0;
    accumulator.eval_correct[enyo::black] = 0;
}

size_t nnue2_refresh_table_index(enyo::square_t king_square, enyo::Color side)
{
    const int ksq = NNUE2::to_berserk_sq(king_square)
        ^ (56 * static_cast<int>(side));
    const int file_half = (ksq & 4) ? 1 : 0;
    const auto bucket = static_cast<size_t>(
        NNUE2::KING_BUCKETS[static_cast<size_t>(ksq)]);
    return static_cast<size_t>(side) * 2 * NNUE2::N_KING_BUCKETS
        + static_cast<size_t>(file_half) * NNUE2::N_KING_BUCKETS
        + bucket;
}

void reset_nnue2_refresh_table_if_needed(NNUE::Net & net)
{
    if (net.nnue2_refresh_generation == NNUE2::NETWORK_GENERATION)
        return;

    NNUE2::ResetRefreshTable(net.nnue2_refresh_table.data());
    net.nnue2_refresh_generation = NNUE2::NETWORK_GENERATION;
}

void refresh_nnue2_from_table(NNUE::Net & net,
                              NNUE2::Accumulator & accumulator,
                              const enyo::Board & board,
                              enyo::Color side)
{
    reset_nnue2_refresh_table_if_needed(net);

    const auto king_square = static_cast<enyo::square_t>(
        lsb(board.pt_bb[side][enyo::king]));
    auto & state = net.nnue2_refresh_table[
        nnue2_refresh_table_index(king_square, side)];

    NNUE2::Delta delta{};
    for (int pc = 0; pc < NNUE2::N_PIECE_TYPES; ++pc) {
        const auto pt = static_cast<enyo::PieceType>((pc >> 1) + 1);
        const auto color = static_cast<enyo::Color>(pc & 1);
        const uint64_t curr = board.pt_bb[color][pt];
        const uint64_t prev = state.pcs[pc];

        uint64_t rem = prev & ~curr;
        while (rem) {
            const auto sq = static_cast<enyo::square_t>(pop_lsb(rem));
            delta.rem[delta.r++] = NNUE2::FeatureIdx(
                pt,
                color,
                sq,
                king_square,
                side);
        }

        uint64_t add = curr & ~prev;
        while (add) {
            const auto sq = static_cast<enyo::square_t>(pop_lsb(add));
            delta.add[delta.a++] = NNUE2::FeatureIdx(
                pt,
                color,
                sq,
                king_square,
                side);
        }

        state.pcs[pc] = curr;
    }

    NNUE2::ApplyDelta(state.values, state.values, &delta);
    std::memcpy(
        accumulator.values[side],
        state.values,
        sizeof(NNUE2::acc_t) * NNUE2::N_HIDDEN);
    accumulator.correct[side] = 1;
    invalidate_nnue2_eval(accumulator);
}

void update_nnue2_feature(NNUE2::Accumulator & accumulator,
                          enyo::PieceType pieceType,
                          enyo::Color pieceColor,
                          enyo::square_t square,
                          enyo::square_t kingSquare_White,
                          enyo::square_t kingSquare_Black,
                          bool add)
{
    if (NNUE2::INPUT_WEIGHTS == nullptr)
        return;

    invalidate_nnue2_eval(accumulator);
    for (auto side : {enyo::white, enyo::black}) {
        NNUE2::Delta delta{};
        const int feature = NNUE2::FeatureIdx(
            pieceType,
            pieceColor,
            square,
            side == enyo::white ? kingSquare_White : kingSquare_Black,
            side);
        if (add)
            delta.add[delta.a++] = feature;
        else
            delta.rem[delta.r++] = feature;
        NNUE2::ApplyDelta(accumulator.values[side], accumulator.values[side], &delta);
    }
}

void move_nnue2_feature(NNUE2::Accumulator & accumulator,
                        enyo::PieceType pieceType,
                        enyo::Color pieceColor,
                        enyo::square_t from_square,
                        enyo::square_t to_square,
                        enyo::square_t kingSquare_White,
                        enyo::square_t kingSquare_Black)
{
    if (NNUE2::INPUT_WEIGHTS == nullptr)
        return;

    invalidate_nnue2_eval(accumulator);
    for (auto side : {enyo::white, enyo::black}) {
        const auto king_square = side == enyo::white ? kingSquare_White : kingSquare_Black;
        NNUE2::Delta delta{};
        delta.rem[delta.r++] = NNUE2::FeatureIdx(
            pieceType, pieceColor, from_square, king_square, side);
        delta.add[delta.a++] = NNUE2::FeatureIdx(
            pieceType, pieceColor, to_square, king_square, side);
        NNUE2::ApplyDelta(accumulator.values[side], accumulator.values[side], &delta);
    }
}

void move_nnue2_feature_from(NNUE2::Accumulator & dest,
                             const NNUE2::Accumulator & src,
                             enyo::PieceType pieceType,
                             enyo::Color pieceColor,
                             enyo::square_t from_square,
                             enyo::square_t to_square,
                             enyo::PieceType capturedPieceType,
                             enyo::Color capturedPieceColor,
                             enyo::square_t kingSquare_White,
                             enyo::square_t kingSquare_Black)
{
    if (NNUE2::INPUT_WEIGHTS == nullptr)
        return;

    invalidate_nnue2_eval(dest);
    for (auto side : {enyo::white, enyo::black}) {
        const auto king_square = side == enyo::white ? kingSquare_White : kingSquare_Black;
        NNUE2::Delta delta{};
        delta.rem[delta.r++] = NNUE2::FeatureIdx(
            pieceType, pieceColor, from_square, king_square, side);
        if (capturedPieceType != enyo::no_piece_type) {
            delta.rem[delta.r++] = NNUE2::FeatureIdx(
                capturedPieceType, capturedPieceColor, to_square, king_square, side);
        }
        delta.add[delta.a++] = NNUE2::FeatureIdx(
            pieceType, pieceColor, to_square, king_square, side);
        NNUE2::ApplyDelta(dest.values[side], src.values[side], &delta);
    }
}

uint16_t pack_nnue2_capture(enyo::PieceType pt, enyo::square_t sq)
{
    return static_cast<uint16_t>(
        static_cast<uint16_t>(pt)
      | (static_cast<uint16_t>(sq) << 8));
}

enyo::PieceType packed_nnue2_capture_piece(uint16_t packed)
{
    return static_cast<enyo::PieceType>(packed & 0xff);
}

enyo::square_t packed_nnue2_capture_square(uint16_t packed)
{
    return static_cast<enyo::square_t>(packed >> 8);
}

void move_nnue2_feature_from_for_side(NNUE2::Accumulator & dest,
                                      const NNUE2::Accumulator & src,
                                      enyo::Move move,
                                      uint16_t captured,
                                      enyo::Color side,
                                      enyo::square_t king_square)
{
    const auto mover = move.src_color();
    const bool own_king_move = mover == side && move.src_piece() == enyo::king;
    const auto add_king_square = own_king_move ? move.dst_sq() : king_square;
    const auto captured_piece = packed_nnue2_capture_piece(captured);
    const auto captured_square = packed_nnue2_capture_square(captured);
    const auto to_piece = move.flags() == enyo::Move::Flags::promote
        ? move.promo_piece()
        : move.src_piece();

    const int from_feature = NNUE2::FeatureIdx(
        move.src_piece(), mover, move.src_sq(), king_square, side);
    const int to_feature = NNUE2::FeatureIdx(
        to_piece, mover, move.dst_sq(), add_king_square, side);

    if (move.flags() == enyo::Move::Flags::castle) {
        const bool kingside = move.dst_sq() < move.src_sq();
        const enyo::square_t rook_from = mover == enyo::white
            ? (kingside ? enyo::h1 : enyo::a1)
            : (kingside ? enyo::h8 : enyo::a8);
        const enyo::square_t rook_to = mover == enyo::white
            ? (kingside ? enyo::f1 : enyo::d1)
            : (kingside ? enyo::f8 : enyo::d8);
        NNUE2::ApplySubSubAddAdd(
            dest.values[side],
            src.values[side],
            from_feature,
            NNUE2::FeatureIdx(enyo::rook, mover, rook_from, king_square, side),
            to_feature,
            NNUE2::FeatureIdx(enyo::rook, mover, rook_to, king_square, side));
        return;
    }

    if (captured_piece != enyo::no_piece_type) {
        NNUE2::ApplySubSubAdd(
            dest.values[side],
            src.values[side],
            from_feature,
            NNUE2::FeatureIdx(captured_piece, ~mover, captured_square, king_square, side),
            to_feature);
        return;
    }

    NNUE2::ApplySubAdd(dest.values[side], src.values[side], from_feature, to_feature);
}

void build_nnue2_lazy_for_side(NNUE2::Accumulator & accumulator,
                               enyo::Move move,
                               enyo::PieceType to_piece,
                               enyo::PieceType captured_piece,
                               enyo::square_t captured_square,
                               enyo::Color side,
                               enyo::square_t king_square)
{
    const auto mover = move.src_color();
    const bool own_king_move = mover == side && move.src_piece() == enyo::king;
    accumulator.lazy_correct[side] = 1;
    accumulator.lazy_refresh[side] = own_king_move
        && NNUE2::MoveRequiresRefresh(enyo::king, move.src_sq(), move.dst_sq(), side);
    accumulator.lazy_rem_count[side] = 0;
    accumulator.lazy_add_count[side] = 0;

    if (accumulator.lazy_refresh[side])
        return;

    const auto add_king_square = own_king_move ? move.dst_sq() : king_square;

    accumulator.lazy_rem[side][accumulator.lazy_rem_count[side]++] =
        static_cast<uint16_t>(NNUE2::FeatureIdx(
            move.src_piece(), mover, move.src_sq(), king_square, side));
    accumulator.lazy_add[side][accumulator.lazy_add_count[side]++] =
        static_cast<uint16_t>(NNUE2::FeatureIdx(
            to_piece, mover, move.dst_sq(), add_king_square, side));

    if (move.flags() == enyo::Move::Flags::castle) {
        const bool kingside = move.dst_sq() < move.src_sq();
        const enyo::square_t rook_from = mover == enyo::white
            ? (kingside ? enyo::h1 : enyo::a1)
            : (kingside ? enyo::h8 : enyo::a8);
        const enyo::square_t rook_to = mover == enyo::white
            ? (kingside ? enyo::f1 : enyo::d1)
            : (kingside ? enyo::f8 : enyo::d8);
        accumulator.lazy_rem[side][accumulator.lazy_rem_count[side]++] =
            static_cast<uint16_t>(NNUE2::FeatureIdx(
                enyo::rook, mover, rook_from, king_square, side));
        accumulator.lazy_add[side][accumulator.lazy_add_count[side]++] =
            static_cast<uint16_t>(NNUE2::FeatureIdx(
                enyo::rook, mover, rook_to, king_square, side));
        return;
    }

    if (captured_piece != enyo::no_piece_type) {
        accumulator.lazy_rem[side][accumulator.lazy_rem_count[side]++] =
            static_cast<uint16_t>(NNUE2::FeatureIdx(
                captured_piece, ~mover, captured_square, king_square, side));
    }
}

void apply_nnue2_lazy_from_for_side(NNUE2::Accumulator & dest,
                                    const NNUE2::Accumulator & src,
                                    enyo::Color side)
{
    if (dest.lazy_rem_count[side] == 2 && dest.lazy_add_count[side] == 2) {
        NNUE2::ApplySubSubAddAdd(
            dest.values[side],
            src.values[side],
            dest.lazy_rem[side][0],
            dest.lazy_rem[side][1],
            dest.lazy_add[side][0],
            dest.lazy_add[side][1]);
        return;
    }

    if (dest.lazy_rem_count[side] == 2 && dest.lazy_add_count[side] == 1) {
        NNUE2::ApplySubSubAdd(
            dest.values[side],
            src.values[side],
            dest.lazy_rem[side][0],
            dest.lazy_rem[side][1],
            dest.lazy_add[side][0]);
        return;
    }

    NNUE2::ApplySubAdd(
        dest.values[side],
        src.values[side],
        dest.lazy_rem[side][0],
        dest.lazy_add[side][0]);
}

} // namespace

template void Net::updateAccumulator<true>(
    enyo::PieceType,
    enyo::Color,
    enyo::square_t,
    enyo::square_t,
    enyo::square_t
);

template void Net::updateAccumulator<false>(
    enyo::PieceType,
    enyo::Color,
    enyo::square_t,
    enyo::square_t,
   enyo::square_t
);

template <bool add>
void Net::updateAccumulator(
    enyo::PieceType pieceType,
    enyo::Color pieceColor,
    enyo::square_t square,
    enyo::square_t kingSquare_White,
    enyo::square_t kingSquare_Black)
{
    if (use_nnue2()) {
        update_nnue2_feature(
            nnue2_accumulator_stack[currentAccumulator],
            pieceType,
            pieceColor,
            square,
            kingSquare_White,
            kingSquare_Black,
            add);
        return;
    }
    if (NNUE2::INPUT_WEIGHTS != nullptr) {
        update_nnue2_feature(
            nnue2_accumulator_stack[currentAccumulator],
            pieceType,
            pieceColor,
            square,
            kingSquare_White,
            kingSquare_Black,
            add);
    }

    Accumulator &accumulator = accumulator_stack[currentAccumulator];
    for (auto side : {enyo::white, enyo::black}) {
        const int inputs =
            index(
                pieceType,
                pieceColor,
                square,
                side,
                side == enyo::white
                ? kingSquare_White
                : kingSquare_Black
            );
        const auto weights = inputWeights.data() + inputs * HIDDEN_SIZE;

        if constexpr (add) {
            for (int i = 0; i < HIDDEN_SIZE / 4; ++i) {
                accumulator[side][i * 4 + 0] += weights[i * 4 + 0];
                accumulator[side][i * 4 + 1] += weights[i * 4 + 1];
                accumulator[side][i * 4 + 2] += weights[i * 4 + 2];
                accumulator[side][i * 4 + 3] += weights[i * 4 + 3];
            }
        } else {
            for (int i = 0; i < HIDDEN_SIZE / 4; ++i) {
                accumulator[side][i * 4 + 0] -= weights[i * 4 + 0];
                accumulator[side][i * 4 + 1] -= weights[i * 4 + 1];
                accumulator[side][i * 4 + 2] -= weights[i * 4 + 2];
                accumulator[side][i * 4 + 3] -= weights[i * 4 + 3];
            }
        }
    }
}

void Net::updateAccumulator(
    enyo::PieceType pieceType,
    enyo::Color pieceColor,
    enyo::square_t from_square,
    enyo::square_t to_square,
    enyo::square_t kingSquare_White,
    enyo::square_t kingSquare_Black)
{
    if (use_nnue2()) {
        move_nnue2_feature(
            nnue2_accumulator_stack[currentAccumulator],
            pieceType,
            pieceColor,
            from_square,
            to_square,
            kingSquare_White,
            kingSquare_Black);
        return;
    }
    if (NNUE2::INPUT_WEIGHTS != nullptr) {
        move_nnue2_feature(
            nnue2_accumulator_stack[currentAccumulator],
            pieceType,
            pieceColor,
            from_square,
            to_square,
            kingSquare_White,
            kingSquare_Black);
    }

    Accumulator &accumulator = accumulator_stack[currentAccumulator];
#if defined(USE_SIMD)
    for (auto side : {white, black}) {
        const int inputClear =
            index(
                pieceType,
                pieceColor,
                from_square,
                side,
                side == enyo::white
                ? kingSquare_White
                : kingSquare_Black
            );
        const int inputAdd =
            index(
                pieceType,
                pieceColor,
                to_square,
                side, side == enyo::white
                ? kingSquare_White
                : kingSquare_Black
            );

        const auto weightSub = reinterpret_cast<register_type16 *>(inputWeights.data() + inputClear * HIDDEN_SIZE);
        const auto weightAdd = reinterpret_cast<register_type16 *>(inputWeights.data() + inputAdd * HIDDEN_SIZE);
        const auto input = reinterpret_cast<register_type16 *>(accumulator[side].data());
        const auto output = reinterpret_cast<register_type16 *>(accumulator[side].data());

        for (int i = 0; i < HIDDEN_SIZE / STRIDE_16_BIT / 4; ++i) {
            output[i * 4 + 0] =
                register_add_epi16(register_sub_epi16(input[i * 4 + 0], weightSub[i * 4 + 0]), weightAdd[i * 4 + 0]);
            output[i * 4 + 1] =
                register_add_epi16(register_sub_epi16(input[i * 4 + 1], weightSub[i * 4 + 1]), weightAdd[i * 4 + 1]);
            output[i * 4 + 2] =
                register_add_epi16(register_sub_epi16(input[i * 4 + 2], weightSub[i * 4 + 2]), weightAdd[i * 4 + 2]);
            output[i * 4 + 3] =
                register_add_epi16(register_sub_epi16(input[i * 4 + 3], weightSub[i * 4 + 3]), weightAdd[i * 4 + 3]);
        }
    }
#else
    for (auto side : {enyo::white, enyo::black}) {
        const int inputClear =
            index(
                pieceType,
                pieceColor,
                from_square,
                side,
                side == enyo::white
                ? kingSquare_White
                : kingSquare_Black
            );
        const int inputAdd =
            index(
                pieceType,
                pieceColor,
                to_square,
                side,
                side == enyo::white
                ? kingSquare_White
                : kingSquare_Black
            );

        const auto weightSub = inputWeights.data() + inputClear * HIDDEN_SIZE;
        const auto weightAdd = inputWeights.data() + inputAdd * HIDDEN_SIZE;

        for (int i = 0; i < HIDDEN_SIZE / 4; ++i) {
            accumulator[side][i * 4 + 0] += static_cast<int16_t>(-weightSub[i * 4 + 0] + weightAdd[i * 4 + 0]);
            accumulator[side][i * 4 + 1] += static_cast<int16_t>(-weightSub[i * 4 + 1] + weightAdd[i * 4 + 1]);
            accumulator[side][i * 4 + 2] += static_cast<int16_t>(-weightSub[i * 4 + 2] + weightAdd[i * 4 + 2]);
            accumulator[side][i * 4 + 3] += static_cast<int16_t>(-weightSub[i * 4 + 3] + weightAdd[i * 4 + 3]);
        }
    }
#endif
}

void Net::updateAccumulatorFromPrevious(
    enyo::PieceType pieceType,
    enyo::Color pieceColor,
    enyo::square_t from_square,
    enyo::square_t to_square,
    enyo::PieceType capturedPieceType,
    enyo::Color capturedPieceColor,
    enyo::square_t kingSquare_White,
    enyo::square_t kingSquare_Black)
{
    assert(currentAccumulator > 0 && "NNUE2 previous accumulator missing");
    move_nnue2_feature_from(
        nnue2_accumulator_stack[currentAccumulator],
        nnue2_accumulator_stack[currentAccumulator - 1],
        pieceType,
        pieceColor,
        from_square,
        to_square,
        capturedPieceType,
        capturedPieceColor,
        kingSquare_White,
        kingSquare_Black);
}

bool Net::try_nnue2_eager_move(enyo::Move move,
                               enyo::PieceType capturedPieceType,
                               enyo::square_t capturedSquare,
                               enyo::square_t kingSquare_White,
                               enyo::square_t kingSquare_Black)
{
    if (!use_nnue2() || currentAccumulator == 0)
        return false;
    if (move.src_piece() == enyo::king
     && NNUE2::MoveRequiresRefresh(
            enyo::king,
            move.src_sq(),
            move.dst_sq(),
            move.src_color()))
        return false;

    const NNUE2::Accumulator & parent =
        nnue2_accumulator_stack[currentAccumulator - 1];
    if (!parent.correct[enyo::white] || !parent.correct[enyo::black])
        return false;

    NNUE2::Accumulator & child = nnue2_accumulator_stack[currentAccumulator];
    child.captured = pack_nnue2_capture(capturedPieceType, capturedSquare);
    move_nnue2_feature_from_for_side(
        child,
        parent,
        move,
        child.captured,
        enyo::white,
        kingSquare_White);
    move_nnue2_feature_from_for_side(
        child,
        parent,
        move,
        child.captured,
        enyo::black,
        kingSquare_Black);
    child.correct[enyo::white] = 1;
    child.correct[enyo::black] = 1;
    invalidate_nnue2_eval(child);
    child.move = move.data;
    return true;
}

void Net::mark_nnue2_lazy_move(enyo::Move move,
                               enyo::square_t captured_square,
                               enyo::square_t kingSquare_White,
                               enyo::square_t kingSquare_Black)
{
    if (!use_nnue2())
        return;

    assert(currentAccumulator > 0 && "NNUE2 lazy move needs a child slot");
    NNUE2::Accumulator & accumulator = nnue2_accumulator_stack[currentAccumulator];
    accumulator.correct[enyo::white] = 0;
    accumulator.correct[enyo::black] = 0;
    invalidate_nnue2_eval(accumulator);
    accumulator.move = move.data;
    accumulator.captured = pack_nnue2_capture(
        move.flags() == enyo::Move::Flags::enpassant
            ? enyo::pawn
            : move.dst_piece(),
        captured_square == enyo::nosquare ? move.dst_sq() : captured_square);
    const auto captured_piece = packed_nnue2_capture_piece(accumulator.captured);
    const auto captured_sq = packed_nnue2_capture_square(accumulator.captured);
    const auto to_piece = move.flags() == enyo::Move::Flags::promote
        ? move.promo_piece()
        : move.src_piece();
    build_nnue2_lazy_for_side(
        accumulator,
        move,
        to_piece,
        captured_piece,
        captured_sq,
        enyo::white,
        kingSquare_White);
    build_nnue2_lazy_for_side(
        accumulator,
        move,
        to_piece,
        captured_piece,
        captured_sq,
        enyo::black,
        kingSquare_Black);
}

void Net::ensure_nnue2(enyo::Board &board)
{
    if (!use_nnue2())
        return;

    NNUE2::Accumulator & current = nnue2_accumulator_stack[currentAccumulator];
    bool need[2] = {
        !current.correct[enyo::white],
        !current.correct[enyo::black],
    };
    if (!need[enyo::white] && !need[enyo::black])
        return;

    size_t base[2] = {currentAccumulator, currentAccumulator};
    for (auto side : {enyo::white, enyo::black}) {
        while (base[side] > 0 && !nnue2_accumulator_stack[base[side]].correct[side])
            --base[side];
        if (!nnue2_accumulator_stack[base[side]].correct[side]) {
            refresh_nnue2(board, side);
            need[side] = false;
        }
    }
    if (!need[enyo::white] && !need[enyo::black])
        return;

    const size_t min_base = need[enyo::white] && need[enyo::black]
        ? std::min(base[enyo::white], base[enyo::black])
        : (need[enyo::white] ? base[enyo::white] : base[enyo::black]);

    for (size_t ply = min_base + 1; ply <= currentAccumulator; ++ply) {
        NNUE2::Accumulator & dest = nnue2_accumulator_stack[ply];
        const enyo::Move move{dest.move};
        if (!move) {
            for (auto side : {enyo::white, enyo::black}) {
                if (need[side]) {
                    refresh_nnue2(board, side);
                    need[side] = false;
                }
            }
            return;
        }

        for (auto side : {enyo::white, enyo::black}) {
            if (!need[side] || ply <= base[side] || dest.correct[side])
                continue;

            if (!dest.lazy_correct[side]) {
                refresh_nnue2(board, side);
                need[side] = false;
                continue;
            }

            if (dest.lazy_refresh[side]) {
                refresh_nnue2(board, side);
                need[side] = false;
                continue;
            }

            apply_nnue2_lazy_from_for_side(
                dest, nnue2_accumulator_stack[ply - 1], side);
            dest.correct[side] = 1;
            invalidate_nnue2_eval(dest);
        }
    }
}

void Net::refresh(enyo::Board &board) {
    if (use_nnue2()) {
        refresh_nnue2(board);
        return;
    }

    Accumulator &accumulator = accumulator_stack[currentAccumulator];
    accumulator.clear();

    uint64_t pieces = board.color_bb[enyo::white] | board.color_bb[enyo::black];
    while (pieces) {
        enyo::square_t sq = pop_lsb(pieces);
        enyo::PieceType pt = board.pt_mb[sq];
        auto color = board.color_bb[enyo::white] & (1ULL << sq) ? enyo::white : enyo::black;

        updateAccumulator<true>(
            pt,
            color,
            sq,
            lsb(board.pt_bb[enyo::white][enyo::king]),
            lsb(board.pt_bb[enyo::black][enyo::king])
        );
    }
    refresh_nnue2(board);
}

void Net::refresh_nnue2(enyo::Board &board) {
    if (NNUE2::INPUT_WEIGHTS == nullptr || NNUE2::INPUT_BIASES == nullptr)
        return;

    NNUE2::Accumulator & accumulator = nnue2_accumulator_stack[currentAccumulator];
    std::memset(&accumulator, 0, sizeof(accumulator));
    refresh_nnue2_from_table(*this, accumulator, board, enyo::white);
    refresh_nnue2_from_table(*this, accumulator, board, enyo::black);
}

void Net::refresh_nnue2(enyo::Board &board, enyo::Color side) {
    if (NNUE2::INPUT_WEIGHTS == nullptr || NNUE2::INPUT_BIASES == nullptr)
        return;

    NNUE2::Accumulator & accumulator = nnue2_accumulator_stack[currentAccumulator];
    refresh_nnue2_from_table(*this, accumulator, board, side);
}

// Compute a collision-resistant hash of piece positions for
// AccumulatorCache validation.
//
// The previous implementation used `pt << (sq + color*6)` which
// XOR-collides on trivial swaps — pawn(=1) at sq=N and knight(=2)
// at sq=N-1 contribute identical values and cancel, so boards
// that differ in those two pieces hash identically. The shift
// also exceeds 63 for black pieces at high squares (UB).
//
// Cache hits on collisions returned a stale accumulator. Fixing
// just this hash (distill_clean vs default.net) moved SPRT from
// -711 Elo to -336 Elo — a ~375 Elo improvement that was
// silently costing every trained-net SPRT in the project.
//
// This implementation reuses the Board's Zobrist psq randoms (the
// same ones the transposition table trusts). Collision probability
// is ~2^-64.
static uint64_t compute_pieces_hash(const enyo::Board &board) {
    uint64_t hash = 0;
    uint64_t pieces = board.color_bb[enyo::white] | board.color_bb[enyo::black];
    while (pieces) {
        enyo::square_t sq = pop_lsb(pieces);
        enyo::PieceType pt = board.pt_mb[sq];
        auto color = board.color_bb[enyo::white] & (1ULL << sq)
            ? enyo::white : enyo::black;
        hash ^= board.zbrs.psq_[color][pt][sq];
    }
    return hash;
}

void Net::update_cache(enyo::Board &board, enyo::square_t w_ksq, enyo::square_t b_ksq) {
    // Update cache for both king positions
    auto pieces_hash = compute_pieces_hash(board);
    
    cache.entries[enyo::white][w_ksq].acc.copy(accumulator_stack[currentAccumulator]);
    cache.entries[enyo::white][w_ksq].pieces_hash = pieces_hash;
    cache.entries[enyo::white][w_ksq].valid = true;
    
    cache.entries[enyo::black][b_ksq].acc.copy(accumulator_stack[currentAccumulator]);
    cache.entries[enyo::black][b_ksq].pieces_hash = pieces_hash;
    cache.entries[enyo::black][b_ksq].valid = true;
}

void Net::refresh_with_cache(enyo::Board &board) {
    if (use_nnue2()) {
        refresh_nnue2(board);
        return;
    }

    enyo::square_t w_ksq = lsb(board.pt_bb[enyo::white][enyo::king]);
    enyo::square_t b_ksq = lsb(board.pt_bb[enyo::black][enyo::king]);
    auto pieces_hash = compute_pieces_hash(board);
    
    // Check if we have a valid cached accumulator for these king positions
    auto& w_entry = cache.entries[enyo::white][w_ksq];
    auto& b_entry = cache.entries[enyo::black][b_ksq];
    
    if (w_entry.valid && b_entry.valid && 
        w_entry.pieces_hash == pieces_hash && b_entry.pieces_hash == pieces_hash) {
        // Cache hit! Copy from cache
        accumulator_stack[currentAccumulator].copy(w_entry.acc);
        refresh_nnue2(board);
        return;
    }
    
    // Cache miss - do full refresh and update cache
    refresh(board);
    update_cache(board, w_ksq, b_ksq);
}


int32_t Net::Evaluate(enyo::Color side) {
    Accumulator &accumulator = accumulator_stack[currentAccumulator];

#if defined(USE_SIMD)

    register_type16 reluBias{};
    register_type32 res{};

    const auto accumulator_us = reinterpret_cast<register_type16 *>(accumulator[side].data());
    const auto accumulator_them = reinterpret_cast<register_type16 *>(accumulator[!side].data());
    const auto weights = reinterpret_cast<register_type16 *>(hiddenWeights.data());

    for (int i = 0; i < HIDDEN_SIZE / STRIDE_16_BIT; i++) {
        res = register_add_epi32(res, register_madd_epi16(register_max_epi16(accumulator_us[i], reluBias), weights[i]));
    }

    for (int i = 0; i < HIDDEN_SIZE / STRIDE_16_BIT; i++) {
        res = register_add_epi32(res, register_madd_epi16(register_max_epi16(accumulator_them[i], reluBias), weights[i + HIDDEN_SIZE / STRIDE_16_BIT]));
    }

    const auto output = sumRegisterEpi32(res) + hiddenBias[0];
    return output / (INPUT_QUANTIZATION * HIDDEN_QUANTIZATON);

#else
    int32_t output = hiddenBias[0];

    for (int chunks = 0; chunks < HIDDEN_SIZE / 256; chunks++) {
        const int offset = chunks * 256;
        for (int i = 0; i < 256; i++) {
            output += relu(accumulator[side][i + offset]) * hiddenWeights[i + offset];
        }
    }

    for (int chunks = 0; chunks < HIDDEN_SIZE / 256; chunks++) {
        const int offset = chunks * 256;
        for (int i = 0; i < 256; i++) {
            output += relu(accumulator[!side][i + offset]) * hiddenWeights[HIDDEN_SIZE + i + offset];
        }
    }

    return output / INPUT_QUANTIZATION / HIDDEN_QUANTIZATON;

#endif
}

int32_t Net::Evaluate2(enyo::Board &board, enyo::Color side) {
    assert(side == board.side && "NNUE2 eval cache is side-to-move keyed");
    NNUE2::Accumulator & accumulator = nnue2_accumulator_stack[currentAccumulator];
    if (accumulator.eval_correct[side])
        return accumulator.eval[side];

    if (nnue2_eval_cache_generation != NNUE2::NETWORK_GENERATION) {
        for (auto & entry : nnue2_eval_cache) {
            entry.hash = 0;
            entry.valid = 0;
        }
        nnue2_eval_cache_generation = NNUE2::NETWORK_GENERATION;
    }

    auto & entry = nnue2_eval_cache[
        board.hash & (nnue2_eval_cache_size - 1)];
    if (entry.hash == board.hash && entry.valid) {
        accumulator.eval[side] = entry.eval;
        accumulator.eval_correct[side] = 1;
        return accumulator.eval[side];
    }

    ensure_nnue2(board);

    const int32_t eval = NNUE2::ScaleEval(
        board,
        NNUE2::Propagate(&accumulator, static_cast<int>(side)));
    accumulator.eval[side] = eval;
    accumulator.eval_correct[side] = 1;
    if (entry.hash != board.hash) {
        entry.hash = board.hash;
        entry.valid = 0;
    }
    entry.eval = eval;
    entry.valid = 1;
    return eval;
}

void Net::print_indexes(
    const enyo::Board &board,
    const enyo::PieceType pt,
    const enyo::square_t sq,
    enyo::square_t kingSquare)
{

    auto color = board.color_bb[enyo::white] & sq ? enyo::white : enyo::black;
    std::cout << "W [" << int(pt) << ", " << int(color) << ", " << int(sq)
              << "  ]: " << index(pt, color, sq, enyo::white, h1a1(kingSquare)) << "\n";
    std::cout << "B [" << int(pt) << ", " << int(color) << ", " << int(sq)
              << "  ]: " << index(pt, color, sq, enyo::black, h1a1(kingSquare)) << "\n";
}

#if 1
void Net::Benchmark()
{
    enyo::Board board{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"};

    std::cout << "Testing Evaluation speed...\n" << std::endl;

    float eval_time_sum = 0;
    float refresh_time_sum = 0;
    float update_time_sum = 0;

    std::vector<std::string> bench_fens = {
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    };
    for (auto &pos : bench_fens) {
        board.set(pos);

        long long time_sum = 0;
        int eval = 0;

        for (int i = 0; i < 1e7; i++) {
            auto start = std::chrono::steady_clock::now();
            eval = Evaluate(board.side);
            auto end = std::chrono::steady_clock::now();

            time_sum += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }

        float eval_time = static_cast<float>(time_sum) / 1e7f;
        eval_time_sum += eval_time;

        printf("%7.3f ns | eval: %4d | %-90s\n", eval_time, eval, pos.c_str());
    }

    std::cout << "\nTesting Refresh speed...\n" << std::endl;

    for (auto &pos : bench_fens) {
        board.set(pos);

        long long time_sum = 0;

        for (int i = 0; i < 1e6; i++) {
            auto start = std::chrono::steady_clock::now();
            refresh(board);
            auto end = std::chrono::steady_clock::now();

            time_sum += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }

        float refresh_time = static_cast<float>(time_sum) / 1e6f;
        refresh_time_sum += refresh_time;

        printf("%10.3f ns | %-90s\n", refresh_time, pos.c_str());
    }

    std::cout << "\nTesting Update speed...\n" << std::endl;

    for (auto &pos : bench_fens) {
        board.set(pos);

        auto const moves = generate_legal_moves<enyo::white>(board);

        long long time_sum = 0;

        // PWA: white only
        for (int iteration = 0; iteration < 1e5; iteration++) {
            for (auto move : moves) {
                auto start = std::chrono::steady_clock::now();

                apply_move<white>(board, move);
                revert_move<white>(board);

                auto end = std::chrono::steady_clock::now();

                time_sum += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            }
        }

        float update_time = static_cast<float>(time_sum) / static_cast<float>(moves.size()) / 1e5f;
        update_time_sum += update_time;

        printf("%10.3f ns | %-90s\n", update_time, pos.c_str());
    }

    std::cout << "\n";
    std::cout << "Average evaluation time: " << eval_time_sum / static_cast<float>(bench_fens.size()) << " ns" << std::endl;
    std::cout << "Average refresh time: " << refresh_time_sum / static_cast<float>(bench_fens.size()) << " ns" << std::endl;
    std::cout << "Average update time: " << update_time_sum / static_cast<float>(bench_fens.size()) << " ns" << std::endl;
}
#endif

// Unpack a contiguous net blob (int16 input weights + int16 input bias
// + int16 hidden weights + int32 hidden bias) from memory. Shared
// between the embedded-blob and on-disk paths.
static void ReadBinFromMemory(const unsigned char * data, size_t size)
{
    constexpr size_t expected =
          INPUT_SIZE * HIDDEN_SIZE * sizeof(int16_t)
        + HIDDEN_SIZE * sizeof(int16_t)
        + HIDDEN_DSIZE * OUTPUT_SIZE * sizeof(int16_t)
        + OUTPUT_SIZE * sizeof(int32_t);
    if (size < expected) {
        fmt::print("nnue: net blob too small ({} < {} bytes); aborting load\n",
                   size, expected);
        std::exit(EXIT_FAILURE);
    }

    size_t memoryIndex = 0;

    std::memcpy(inputWeights.data(), &data[memoryIndex], INPUT_SIZE * HIDDEN_SIZE * sizeof(int16_t));
    memoryIndex += INPUT_SIZE * HIDDEN_SIZE * sizeof(int16_t);

    std::memcpy(inputBias.data(), &data[memoryIndex], HIDDEN_SIZE * sizeof(int16_t));
    memoryIndex += HIDDEN_SIZE * sizeof(int16_t);

    std::memcpy(hiddenWeights.data(), &data[memoryIndex], HIDDEN_DSIZE * OUTPUT_SIZE * sizeof(int16_t));
    memoryIndex += HIDDEN_DSIZE * OUTPUT_SIZE * sizeof(int16_t);

    std::memcpy(hiddenBias.data(), &data[memoryIndex], OUTPUT_SIZE * sizeof(int32_t));
    memoryIndex += OUTPUT_SIZE * sizeof(int32_t);

#ifdef DEBUG
    std::cout << "Memory index: " << memoryIndex << std::endl;
    std::cout << "Size: " << size << std::endl;
    std::cout << "Bias: " << hiddenBias[0] / INPUT_QUANTIZATION / HIDDEN_QUANTIZATON << std::endl;
    std::cout << std::endl;
#endif
}

void ReadBin() {
    ReadBinFromMemory(gEVALData, gEVALSize);
}

// Load a .net file from disk into the module-level weight arrays.
// Returns true on success, false on any filesystem/size error (caller
// falls back to the embedded blob). Safe to call repeatedly — the
// weight arrays are plain std::array so the old contents are simply
// overwritten.
static bool LoadFromDisk(const std::string & path)
{
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        // Silent: a missing file is the normal case for the sentinel
        // default ("default.net"), which is expected to fall through
        // to the embedded blob. Real errors surface on open/read.
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fmt::print("nnue: failed to open '{}' for reading\n", path);
        return false;
    }
    std::vector<unsigned char> buf(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
    if (buf.empty()) {
        fmt::print("nnue: empty file '{}'\n", path);
        return false;
    }
    ReadBinFromMemory(buf.data(), buf.size());
    return true;
}

void Init(const std::string & file_name)
{
    // Empty path or missing file → use the embedded default network
    // (produced by INCBIN at build time from nnue/default.net). This
    // is the behavior the engine has always had; runtime override via
    // setoption/config is new.
    if (!file_name.empty() && LoadFromDisk(file_name)) {
        fmt::print("nnue: loaded network from '{}'\n", file_name);
        return;
    }
    ReadBin();
}

void feature_indices(const enyo::Board & board,
                     enyo::Color view,
                     std::vector<int> & out)
{
    out.clear();
    const enyo::square_t view_ksq =
        view == enyo::white
            ? lsb(board.pt_bb[enyo::white][enyo::king])
            : lsb(board.pt_bb[enyo::black][enyo::king]);

    // Mirror refresh()'s enumeration exactly: iterate both colors'
    // occupied squares via pop_lsb so order matches what the
    // accumulator saw when it was built.
    uint64_t pieces = board.color_bb[enyo::white] | board.color_bb[enyo::black];
    while (pieces) {
        enyo::square_t sq = pop_lsb(pieces);
        enyo::PieceType pt = board.pt_mb[sq];
        enyo::Color pc = (board.color_bb[enyo::white] & (1ULL << sq))
            ? enyo::white : enyo::black;
        out.push_back(index(pt, pc, sq, view, view_ksq));
    }
}

} // namespace NNUE
