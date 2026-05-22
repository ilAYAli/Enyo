// Engine integration for Network.
// Links network's forward-pass primitives to enyo::Board. Kept in a
// separate TU so tests can compile nnue_model.cpp without
// pulling in board.hpp / fmt / etc.

#include "nnue_model.hpp"
#include "board.hpp"
#include "util.hpp"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Network {

#if ENYO_ENABLE_CHECK_BUCKET_NNUE
namespace {

int rank_of(enyo::square_t sq) {
    return static_cast<int>(sq) / 8;
}

int file_of(enyo::square_t sq) {
    return 7 - static_cast<int>(sq) % 8;
}

enyo::square_t square_at(int rank, int file) {
    return static_cast<enyo::square_t>(rank * 8 + (7 - file));
}

bool on_board(int rank, int file) {
    return rank >= 0 && rank < 8 && file >= 0 && file < 8;
}

enyo::bitboard_t step_attacks(
    enyo::square_t sq,
    std::initializer_list<std::pair<int, int>> deltas)
{
    const int rank = rank_of(sq);
    const int file = file_of(sq);
    enyo::bitboard_t attacks = 0;
    for (const auto [dr, df] : deltas) {
        const int dst_rank = rank + dr;
        const int dst_file = file + df;
        if (on_board(dst_rank, dst_file))
            attacks |= 1ULL << square_at(dst_rank, dst_file);
    }
    return attacks;
}

enyo::bitboard_t slide_attacks(
    enyo::square_t sq,
    enyo::bitboard_t occupied,
    std::initializer_list<std::pair<int, int>> deltas)
{
    const int rank = rank_of(sq);
    const int file = file_of(sq);
    enyo::bitboard_t attacks = 0;
    for (const auto [dr, df] : deltas) {
        int dst_rank = rank + dr;
        int dst_file = file + df;
        while (on_board(dst_rank, dst_file)) {
            const auto dst = square_at(dst_rank, dst_file);
            attacks |= 1ULL << dst;
            if (occupied & (1ULL << dst))
                break;
            dst_rank += dr;
            dst_file += df;
        }
    }
    return attacks;
}

enyo::bitboard_t attacks_from_piece(
    enyo::PieceType pt,
    enyo::Color color,
    enyo::square_t sq,
    enyo::bitboard_t occupied)
{
    switch (pt) {
    case enyo::pawn: {
        const int direction = color == enyo::white ? 1 : -1;
        return step_attacks(sq, {{direction, -1}, {direction, 1}});
    }
    case enyo::knight:
        return step_attacks(
            sq,
            {{2, 1}, {2, -1}, {1, 2}, {1, -2},
             {-1, 2}, {-1, -2}, {-2, 1}, {-2, -1}});
    case enyo::bishop:
        return slide_attacks(
            sq, occupied, {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}});
    case enyo::rook:
        return slide_attacks(
            sq, occupied, {{1, 0}, {-1, 0}, {0, 1}, {0, -1}});
    case enyo::queen:
        return attacks_from_piece(enyo::bishop, color, sq, occupied)
             | attacks_from_piece(enyo::rook, color, sq, occupied);
    case enyo::king:
        return step_attacks(
            sq,
            {{1, 1}, {1, 0}, {1, -1}, {0, 1},
             {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}});
    default:
        return 0;
    }
}

bool is_slider(enyo::PieceType pt) {
    return pt == enyo::bishop || pt == enyo::rook || pt == enyo::queen;
}

} // namespace
#endif

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

void feature_indices(const enyo::Board& b,
                     enyo::Color view,
                     std::vector<int>& out)
{
    out.clear();
    const enyo::square_t view_ksq =
        view == enyo::white
            ? static_cast<enyo::square_t>(enyo::lsb(b.pt_bb[enyo::white][enyo::king]))
            : static_cast<enyo::square_t>(enyo::lsb(b.pt_bb[enyo::black][enyo::king]));

    enyo::bitboard_t pieces = b.color_bb[enyo::white] | b.color_bb[enyo::black];
    while (pieces) {
        const auto sq = static_cast<enyo::square_t>(enyo::pop_lsb(pieces));
        const enyo::PieceType pt = b.pt_mb[sq];
        const enyo::Color pc = (b.color_bb[enyo::white] & (1ULL << sq))
            ? enyo::white
            : enyo::black;
        out.push_back(FeatureIdx(pt, pc, sq, view_ksq, view));
    }
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

#if ENYO_ENABLE_CHECK_BUCKET_NNUE
int CheckStateBucket(const enyo::Board& b) {
    if (OUTPUT_BUCKETS <= 1)
        return 0;

    const auto stm = b.side;
    const auto them = ~stm;
    const auto king_sq = static_cast<enyo::square_t>(
        enyo::lsb(b.pt_bb[stm][enyo::king]));
    const auto occupied = b.color_bb[enyo::white] | b.color_bb[enyo::black];
    const auto king_bit = 1ULL << king_sq;
    const auto king_zone =
        attacks_from_piece(enyo::king, stm, king_sq, occupied) | king_bit;

    int checkers = 0;
    int slider_checkers = 0;
    int pressure = 0;
    for (int pt = static_cast<int>(enyo::pawn);
         pt <= static_cast<int>(enyo::queen);
         ++pt) {
        auto pieces = b.pt_bb[them][pt];
        while (pieces) {
            const auto sq = static_cast<enyo::square_t>(enyo::pop_lsb(pieces));
            const auto piece_type = static_cast<enyo::PieceType>(pt);
            const auto attacks =
                attacks_from_piece(piece_type, them, sq, occupied);
            if (attacks & king_bit) {
                ++checkers;
                if (is_slider(piece_type))
                    ++slider_checkers;
            }
            if (attacks & king_zone)
                ++pressure;
        }
    }

    if (checkers >= 2)
        return std::min(3, OUTPUT_BUCKETS - 1);
    if (checkers == 1)
        return std::min(slider_checkers ? 2 : 1, OUTPUT_BUCKETS - 1);
    if (pressure <= 0)
        return 0;
    return std::min(3 + pressure, OUTPUT_BUCKETS - 1);
}
#endif

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

#if ENYO_ENABLE_CHECK_BUCKET_NNUE
    return Propagate(
        &acc,
        static_cast<int>(b.side),
        CheckStateBucket(b));
#else
    return Propagate(&acc, static_cast<int>(b.side));
#endif
}

} // namespace Network
