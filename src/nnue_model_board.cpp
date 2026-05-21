// Engine integration for Network.
// Links network's forward-pass primitives to enyo::Board. Kept in a
// separate TU so tests can compile nnue_model.cpp without
// pulling in board.hpp / fmt / etc.

#include "nnue_model.hpp"
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

int ScaleEval(const enyo::Board& b, int score) {
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
    const int phase = 3 * enyo::count_bits(minors)
        + 5 * enyo::count_bits(rooks)
        + 10 * enyo::count_bits(queens);

    score = (128 + phase) * score / 128;
    return std::clamp(score, -2045, 2045);
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

    return Propagate(&acc, static_cast<int>(b.side));
}

} // namespace Network
