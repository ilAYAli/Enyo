// Engine integration for Network.
// Links network's forward-pass primitives to enyo::Board. Kept in a
// separate TU so tests can compile enyo_halfka_model.cpp without
// pulling in board.hpp / fmt / etc.

#include "enyo_halfka_model.hpp"
#include "board.hpp"
#include "util.hpp"

#include <algorithm>
#include <cstring>

namespace Network {

// enumerate_pieces — walk `board.pt_bb` for both colors and emit
// (packed-piece-code, enyo-sq) entries. Order doesn't matter since
// ApplyDelta's add path is commutative.
size_t enumerate_pieces(const enyo::Board& b, PieceEntry* out) {
    size_t n = 0;
    for (int color = 0; color < 2; ++color) {
        for (int pt = static_cast<int>(enyo::pawn); pt <= static_cast<int>(enyo::king); ++pt) {
            enyo::bitboard_t bb = b.pt_bb[color][pt];
            while (bb) {
                const auto sq = static_cast<enyo::square_t>(enyo::pop_lsb(bb));
                const int piece_code = ((pt - 1) << 1) | color;
                out[n++] = {piece_code, sq};
            }
        }
    }
    return n;
}

MaterialSummary SummarizeMaterial(const enyo::Board& b) {
    const auto pawns =
        b.pt_bb[enyo::white][enyo::pawn]
      | b.pt_bb[enyo::black][enyo::pawn];
    const auto minors =
        b.pt_bb[enyo::white][enyo::knight]
      | b.pt_bb[enyo::black][enyo::knight]
      | b.pt_bb[enyo::white][enyo::bishop]
      | b.pt_bb[enyo::black][enyo::bishop];
    const auto rooks =
        b.pt_bb[enyo::white][enyo::rook]
      | b.pt_bb[enyo::black][enyo::rook];
    const auto queens =
        b.pt_bb[enyo::white][enyo::queen]
      | b.pt_bb[enyo::black][enyo::queen];
    const auto kings =
        b.pt_bb[enyo::white][enyo::king]
      | b.pt_bb[enyo::black][enyo::king];
    MaterialSummary summary;
    summary.pawn_count = enyo::count_bits(pawns);
    summary.minor_count = enyo::count_bits(minors);
    summary.rook_count = enyo::count_bits(rooks);
    summary.queen_count = enyo::count_bits(queens);
    summary.phase = 3 * summary.minor_count
        + 5 * summary.rook_count
        + 10 * summary.queen_count;
    summary.piece_count = summary.pawn_count
        + summary.minor_count
        + summary.rook_count
        + summary.queen_count
        + enyo::count_bits(kings);
    return summary;
}

HeadFeatures MaterialHeadFeatures(const MaterialSummary& summary) {
    HeadFeatures features;
    features.values[HEAD_PHASE_DELTA] = static_cast<float>(summary.phase) / 128.0f;
    features.values[HEAD_PIECE_COUNT] =
        (static_cast<float>(summary.piece_count) - 16.0f) / 16.0f;
    features.values[HEAD_PAWN_COUNT] =
        (static_cast<float>(summary.pawn_count) - 16.0f) / 8.0f;
    features.values[HEAD_MINOR_COUNT] =
        (static_cast<float>(summary.minor_count) - 8.0f) / 4.0f;
    features.values[HEAD_ROOK_COUNT] =
        (static_cast<float>(summary.rook_count) - 4.0f) / 2.0f;
    features.values[HEAD_QUEEN_COUNT] =
        (static_cast<float>(summary.queen_count) - 2.0f) / 2.0f;
    features.values[HEAD_NON_PAWN_COUNT] =
        (static_cast<float>(
            summary.minor_count + summary.rook_count + summary.queen_count) - 14.0f) / 7.0f;
    features.values[HEAD_PAWN_PHASE] =
        features.values[HEAD_PHASE_DELTA] * features.values[HEAD_PAWN_COUNT];
    return features;
}

int ScaleEval(const MaterialSummary& summary, int score) {
    score = (128 + summary.phase) * score / 128;
    return std::clamp(score, -2045, 2045);
}

int MaterialCountBucket(const MaterialSummary& summary) {
    return OutputBucketForPieceCount(summary.piece_count, OUTPUT_BUCKETS);
}

// Phase-4-v1 correctness path: fresh accumulator on every call. A
// future phase will plug into the engine's Net accumulator stack for
// search-time incremental updates.
int EvaluateFromScratch(const enyo::Board& b) {
    PieceEntry pieces[32];
    const size_t n = enumerate_pieces(b, pieces);

    enyo::square_t wk_sq = 0, bk_sq = 0;
    for (size_t i = 0; i < n; ++i) {
        if (pieces[i].piece_code == 10) wk_sq = pieces[i].sq;
        if (pieces[i].piece_code == 11) bk_sq = pieces[i].sq;
    }

    Accumulator acc;
    std::memset(&acc, 0, sizeof(acc));
    ResetAccumulator(&acc, enyo::white, wk_sq, pieces, n);
    ResetAccumulator(&acc, enyo::black, bk_sq, pieces, n);
    if (FULL_THREATS_ENABLED) {
        const auto threats = NNUE::Stockfish::FullThreats::GetActiveFeatures(b);
        for (auto side : {enyo::white, enyo::black}) {
            Delta delta{};
            for (size_t i = 0; i < threats[side].size; ++i)
                delta.add[delta.a++] = ThreatFeatureIdx(threats[side].values[i]);
            ApplyDelta(acc.values[side], acc.values[side], &delta);
            ApplyPsqtDelta(acc.psqt[side], acc.psqt[side], &delta);
        }
    }

    const MaterialSummary summary = SummarizeMaterial(b);
    const HeadFeatures head_features = MaterialHeadFeatures(summary);
    const int raw = Propagate(
        &acc,
        static_cast<int>(b.side),
        MaterialCountBucket(summary),
        &head_features);
    return ScaleEval(summary, raw);
}

} // namespace Network
