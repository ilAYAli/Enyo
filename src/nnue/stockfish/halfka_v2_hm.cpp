// Adapted from Stockfish HalfKAv2_hm at e4a63548 (GPLv3).

#include "halfka_v2_hm.hpp"

#include "board.hpp"
#include "util.hpp"

#include <array>

namespace NNUE::Stockfish::HalfKAv2Hm {
namespace {

constexpr int piece_square_index[2][16] = {
    {0, 0, 128, 256, 384, 512, 640, 0, 0, 64, 192, 320, 448, 576, 640, 0},
    {0, 64, 192, 320, 448, 576, 640, 0, 0, 0, 128, 256, 384, 512, 640, 0},
};

constexpr std::array<int, square_count> king_buckets = {
    28 * 704, 29 * 704, 30 * 704, 31 * 704, 31 * 704, 30 * 704, 29 * 704, 28 * 704,
    24 * 704, 25 * 704, 26 * 704, 27 * 704, 27 * 704, 26 * 704, 25 * 704, 24 * 704,
    20 * 704, 21 * 704, 22 * 704, 23 * 704, 23 * 704, 22 * 704, 21 * 704, 20 * 704,
    16 * 704, 17 * 704, 18 * 704, 19 * 704, 19 * 704, 18 * 704, 17 * 704, 16 * 704,
    12 * 704, 13 * 704, 14 * 704, 15 * 704, 15 * 704, 14 * 704, 13 * 704, 12 * 704,
     8 * 704,  9 * 704, 10 * 704, 11 * 704, 11 * 704, 10 * 704,  9 * 704,  8 * 704,
     4 * 704,  5 * 704,  6 * 704,  7 * 704,  7 * 704,  6 * 704,  5 * 704,  4 * 704,
     0 * 704,  1 * 704,  2 * 704,  3 * 704,  3 * 704,  2 * 704,  1 * 704,  0 * 704,
};

constexpr int orientation(int king_square) {
    return king_square % 8 < 4 ? 7 : 0;
}

} // namespace

FeatureIndex MakeIndex(
    enyo::Color perspective,
    enyo::square_t square,
    enyo::PieceType piece,
    enyo::Color piece_color,
    enyo::square_t king_square)
{
    const int stockfish_square = ToStockfishSquare(square);
    const int stockfish_king = ToStockfishSquare(king_square);
    const int flip = 56 * static_cast<int>(perspective);
    const uint8_t stockfish_piece = ToStockfishPiece(piece_color, piece);
    return static_cast<FeatureIndex>(
        (stockfish_square ^ orientation(stockfish_king) ^ flip)
        + piece_square_index[perspective][stockfish_piece]
        + king_buckets[stockfish_king ^ flip]);
}

ActiveFeatures GetActiveFeatures(const enyo::Board & board, enyo::Color perspective) {
    ActiveFeatures features;
    const auto king_square = static_cast<enyo::square_t>(
        enyo::lsb(board.pt_bb[perspective][enyo::king]));

    for (int color = enyo::white; color <= enyo::black; ++color) {
        for (int piece = enyo::pawn; piece <= static_cast<int>(enyo::king); ++piece) {
            uint64_t pieces = board.pt_bb[color][piece];
            while (pieces) {
                const auto square = static_cast<enyo::square_t>(enyo::pop_lsb(pieces));
                features.values[features.size++] = MakeIndex(
                    perspective,
                    square,
                    static_cast<enyo::PieceType>(piece),
                    static_cast<enyo::Color>(color),
                    king_square);
            }
        }
    }

    features.sort();
    return features;
}

} // namespace NNUE::Stockfish::HalfKAv2Hm
