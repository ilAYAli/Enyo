// Adapted from Stockfish FullThreats at e4a63548 (GPLv3).

#include "full_threats.hpp"

#include "board.hpp"
#include "util.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <utility>

namespace NNUE::Stockfish::FullThreats {
namespace {

constexpr std::array<int, piece_count> valid_targets = {
    0, 6, 10, 8, 8, 10, 0, 0,
    0, 6, 10, 8, 8, 10, 0, 0,
};

constexpr int target_map[6][6] = {
    { 0,  1, -1,  2, -1, -1},
    { 0,  1,  2,  3,  4, -1},
    { 0,  1,  2,  3, -1, -1},
    { 0,  1,  2,  3, -1, -1},
    { 0,  1,  2,  3,  4, -1},
    {-1, -1, -1, -1, -1, -1},
};

constexpr std::array<uint8_t, 12> all_pieces = {
    1, 2, 3, 4, 5, 6,
    9, 10, 11, 12, 13, 14,
};

constexpr bool on_board(int file, int rank) {
    return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

uint64_t LeaperAttacks(int piece_type, int square) {
    static constexpr int knight_deltas[8][2] = {
        {1, 2}, {2, 1}, {2, -1}, {1, -2},
        {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2},
    };
    static constexpr int king_deltas[8][2] = {
        {1, 1}, {1, 0}, {1, -1}, {0, -1},
        {-1, -1}, {-1, 0}, {-1, 1}, {0, 1},
    };

    const auto & deltas = piece_type == enyo::knight ? knight_deltas : king_deltas;
    uint64_t attacks = 0;
    const int from_file = square % 8;
    const int from_rank = square / 8;
    for (const auto & delta : deltas) {
        const int file = from_file + delta[0];
        const int rank = from_rank + delta[1];
        if (on_board(file, rank))
            attacks |= uint64_t{1} << (rank * 8 + file);
    }
    return attacks;
}

uint64_t SliderAttacks(int piece_type, int square, uint64_t occupied) {
    static constexpr int bishop_deltas[4][2] = {
        {1, 1}, {1, -1}, {-1, -1}, {-1, 1},
    };
    static constexpr int rook_deltas[4][2] = {
        {1, 0}, {0, -1}, {-1, 0}, {0, 1},
    };

    uint64_t attacks = 0;
    const int from_file = square % 8;
    const int from_rank = square / 8;
    const auto add_rays = [&](const auto & deltas) {
        for (const auto & delta : deltas) {
            int file = from_file + delta[0];
            int rank = from_rank + delta[1];
            while (on_board(file, rank)) {
                const int target = rank * 8 + file;
                const uint64_t target_bit = uint64_t{1} << target;
                attacks |= target_bit;
                if (occupied & target_bit)
                    break;
                file += delta[0];
                rank += delta[1];
            }
        }
    };

    if (piece_type == enyo::bishop || piece_type == enyo::queen)
        add_rays(bishop_deltas);
    if (piece_type == enyo::rook || piece_type == enyo::queen)
        add_rays(rook_deltas);
    return attacks;
}

uint64_t PawnPushOrAttacks(int color, int square) {
    const int file = square % 8;
    const int rank = square / 8;
    const int rank_delta = color == enyo::white ? 1 : -1;
    uint64_t attacks = 0;
    for (const int file_delta : {-1, 0, 1}) {
        const int target_file = file + file_delta;
        const int target_rank = rank + rank_delta;
        if (on_board(target_file, target_rank))
            attacks |= uint64_t{1} << (target_rank * 8 + target_file);
    }
    return attacks;
}

uint64_t PseudoAttacks(uint8_t piece, int square) {
    const int piece_type = PieceType(piece);
    if (piece_type == enyo::pawn)
        return PawnPushOrAttacks(PieceColor(piece), square);
    if (piece_type == enyo::knight || piece_type == enyo::king)
        return LeaperAttacks(piece_type, square);
    return SliderAttacks(piece_type, square, 0);
}

struct HelperOffset {
    int piece_span = 0;
    int global = 0;
};

struct IndexTables {
    std::array<HelperOffset, piece_count> helper{};
    std::array<std::array<int, square_count>, piece_count> offsets{};
    std::array<
        std::array<std::array<FeatureIndex, 2>, piece_count>,
        piece_count> target_offsets{};
    std::array<
        std::array<std::array<uint8_t, square_count>, square_count>,
        piece_count> attack_offsets{};

    IndexTables() {
        int global = 0;
        for (const uint8_t piece : all_pieces) {
            int piece_span = 0;
            for (int square = 0; square < square_count; ++square) {
                offsets[piece][square] = piece_span;
                if (PieceType(piece) != enyo::pawn || (square >= 8 && square < 56))
                    piece_span += std::popcount(PseudoAttacks(piece, square));
            }
            helper[piece] = {piece_span, global};
            global += valid_targets[piece] * piece_span;

            for (int from = 0; from < square_count; ++from) {
                const uint64_t attacks = PseudoAttacks(piece, from);
                for (int to = 0; to < square_count; ++to) {
                    const uint64_t below = to == 0 ? 0 : (uint64_t{1} << to) - 1;
                    attack_offsets[piece][from][to] = static_cast<uint8_t>(
                        std::popcount(attacks & below));
                }
            }
        }

        for (const uint8_t attacker : all_pieces) {
            for (const uint8_t attacked : all_pieces) {
                const int attacker_type = PieceType(attacker);
                const int attacked_type = PieceType(attacked);
                const int mapped_target = target_map[attacker_type - 1][attacked_type - 1];
                const bool excluded = mapped_target < 0;
                const bool enemy = (attacker ^ attacked) == 8;
                const bool same_type_excluded = attacker_type == attacked_type
                    && (enemy || attacker_type != enyo::pawn);
                const FeatureIndex base = excluded
                    ? dimensions
                    : static_cast<FeatureIndex>(
                        helper[attacker].global
                        + (PieceColor(attacked) * (valid_targets[attacker] / 2) + mapped_target)
                            * helper[attacker].piece_span);
                target_offsets[attacker][attacked][0] = base;
                target_offsets[attacker][attacked][1] = excluded || same_type_excluded
                    ? dimensions
                    : base;
            }
        }
    }
};

const IndexTables & Tables() {
    static const IndexTables tables;
    return tables;
}

struct PositionView {
    std::array<uint8_t, square_count> piece_at{};
    std::array<uint64_t, piece_count> pieces{};
    uint64_t occupied = 0;
};

PositionView MakePositionView(const enyo::Board & board) {
    PositionView view;
    for (int color = enyo::white; color <= enyo::black; ++color) {
        for (int piece = enyo::pawn; piece <= static_cast<int>(enyo::king); ++piece) {
            uint64_t pieces = board.pt_bb[color][piece];
            while (pieces) {
                const int enyo_square = enyo::pop_lsb(pieces);
                const int square = enyo_square ^ 7;
                const uint8_t encoded = ToStockfishPiece(
                    static_cast<enyo::Color>(color), static_cast<enyo::PieceType>(piece));
                view.piece_at[square] = encoded;
                view.pieces[encoded] |= uint64_t{1} << square;
                view.occupied |= uint64_t{1} << square;
            }
        }
    }
    return view;
}

uint64_t PiecesOfTypes(const PositionView & view, std::initializer_list<int> types) {
    uint64_t result = 0;
    for (const int type : types)
        result |= view.pieces[type] | view.pieces[type + 8];
    return result;
}

} // namespace

FeatureIndex MakeIndex(
    enyo::Color perspective,
    uint8_t attacker,
    int from,
    int to,
    uint8_t attacked,
    int king_square)
{
    const int orientation = (king_square % 8 < 4 ? 0 : 7)
        ^ (56 * static_cast<int>(perspective));
    const int oriented_from = from ^ orientation;
    const int oriented_to = to ^ orientation;
    const int color_swap = 8 * static_cast<int>(perspective);
    const uint8_t oriented_attacker = static_cast<uint8_t>(attacker ^ color_swap);
    const uint8_t oriented_attacked = static_cast<uint8_t>(attacked ^ color_swap);
    const auto & tables = Tables();
    const FeatureIndex target_offset =
        tables.target_offsets[oriented_attacker][oriented_attacked][oriented_from < oriented_to];
    if (target_offset >= dimensions)
        return dimensions;
    return static_cast<FeatureIndex>(
        target_offset
        + tables.offsets[oriented_attacker][oriented_from]
        + tables.attack_offsets[oriented_attacker][oriented_from][oriented_to]);
}

ActiveFeatures GetActiveFeatures(const enyo::Board & board, enyo::Color perspective) {
    const PositionView position = MakePositionView(board);
    const int king_square = ToStockfishSquare(static_cast<enyo::square_t>(
        enyo::lsb(board.pt_bb[perspective][enyo::king])));
    const uint64_t pawn_targets = PiecesOfTypes(
        position, {enyo::pawn, enyo::knight, enyo::rook});
    const uint64_t minor_slider_targets = PiecesOfTypes(
        position, {enyo::pawn, enyo::knight, enyo::bishop, enyo::rook});
    const uint64_t queen_targets = PiecesOfTypes(
        position, {enyo::pawn, enyo::knight, enyo::bishop, enyo::rook, enyo::queen});

    ActiveFeatures features;
    for (const int relative_color : {enyo::white, enyo::black}) {
        const int color = static_cast<int>(perspective) ^ relative_color;
        const uint8_t pawn = ToStockfishPiece(
            static_cast<enyo::Color>(color), enyo::pawn);
        uint64_t pawns = position.pieces[pawn];
        while (pawns) {
            const int from = std::countr_zero(pawns);
            pawns &= pawns - 1;
            const int file = from % 8;
            const int rank = from / 8;
            const int rank_delta = color == enyo::white ? 1 : -1;

            for (const int file_delta : {-1, 1}) {
                const int target_file = file + file_delta;
                const int target_rank = rank + rank_delta;
                if (!on_board(target_file, target_rank))
                    continue;
                const int to = target_rank * 8 + target_file;
                if ((pawn_targets & (uint64_t{1} << to)) == 0)
                    continue;
                const FeatureIndex index = MakeIndex(
                    perspective, pawn, from, to, position.piece_at[to], king_square);
                if (index < dimensions)
                    features.push(index);
            }

            const int target_rank = rank + rank_delta;
            if (on_board(file, target_rank)) {
                const int to = target_rank * 8 + file;
                if (PieceType(position.piece_at[to]) == enyo::pawn) {
                    const FeatureIndex index = MakeIndex(
                        perspective, pawn, from, to, position.piece_at[to], king_square);
                    if (index < dimensions)
                        features.push(index);
                }
            }
        }

        for (int piece_type = enyo::knight;
             piece_type < static_cast<int>(enyo::king);
             ++piece_type) {
            const uint8_t attacker = ToStockfishPiece(
                static_cast<enyo::Color>(color), static_cast<enyo::PieceType>(piece_type));
            uint64_t attackers = position.pieces[attacker];
            while (attackers) {
                const int from = std::countr_zero(attackers);
                attackers &= attackers - 1;
                const uint64_t targets = piece_type == enyo::knight || piece_type == enyo::queen
                    ? queen_targets
                    : minor_slider_targets;
                uint64_t attacks = (piece_type == enyo::knight
                    ? LeaperAttacks(piece_type, from)
                    : SliderAttacks(piece_type, from, position.occupied)) & targets;
                while (attacks) {
                    const int to = std::countr_zero(attacks);
                    attacks &= attacks - 1;
                    const FeatureIndex index = MakeIndex(
                        perspective, attacker, from, to, position.piece_at[to], king_square);
                    if (index < dimensions)
                        features.push(index);
                }
            }
        }
    }

    features.sort();
    return features;
}

} // namespace NNUE::Stockfish::FullThreats
