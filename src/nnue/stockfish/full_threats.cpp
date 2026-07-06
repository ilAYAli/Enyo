// Adapted from Stockfish FullThreats at e4a63548 (GPLv3).

#include "full_threats.hpp"

#include "board.hpp"
#include "magic/magic.hpp"
#include "precalc/knight_attacks.hpp"
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

std::array<ActiveFeatures, 2> GetActiveFeatures(const enyo::Board & board) {
    // Enumerates in enyo square space (engine attack tables) and flips to
    // the Stockfish orientation only at MakeIndex; one pass feeds both
    // perspectives since attacks are perspective-independent.
    const uint64_t occupied =
        board.color_bb[enyo::white] | board.color_bb[enyo::black];
    const auto both = [&](enyo::PieceType piece) {
        return board.pt_bb[enyo::white][piece] | board.pt_bb[enyo::black][piece];
    };
    const uint64_t pawn_targets =
        both(enyo::pawn) | both(enyo::knight) | both(enyo::rook);
    const uint64_t minor_slider_targets = pawn_targets | both(enyo::bishop);
    const uint64_t queen_targets = minor_slider_targets | both(enyo::queen);

    const int king_square[2] = {
        ToStockfishSquare(static_cast<enyo::square_t>(
            enyo::lsb(board.pt_bb[enyo::white][enyo::king]))),
        ToStockfishSquare(static_cast<enyo::square_t>(
            enyo::lsb(board.pt_bb[enyo::black][enyo::king]))),
    };

    std::array<ActiveFeatures, 2> features;
    const auto emit = [&](uint8_t attacker, int from, int to) {
        const auto attacked_color =
            (board.color_bb[enyo::black] >> to) & 1 ? enyo::black : enyo::white;
        const uint8_t attacked = ToStockfishPiece(attacked_color, board.pt_mb[to]);
        const int from_sf = from ^ 7;
        const int to_sf = to ^ 7;
        for (int perspective = enyo::white; perspective <= enyo::black; ++perspective) {
            const FeatureIndex index = MakeIndex(
                static_cast<enyo::Color>(perspective),
                attacker, from_sf, to_sf, attacked, king_square[perspective]);
            if (index < dimensions)
                features[perspective].push(index);
        }
    };

    for (int color = enyo::white; color <= enyo::black; ++color) {
        const int rank_delta = color == enyo::white ? 1 : -1;
        const uint8_t pawn = ToStockfishPiece(
            static_cast<enyo::Color>(color), enyo::pawn);
        uint64_t pawns = board.pt_bb[color][enyo::pawn];
        while (pawns) {
            const int from = enyo::pop_lsb(pawns);
            const int file = from % 8;
            const int target_rank = from / 8 + rank_delta;

            for (const int file_delta : {-1, 1}) {
                const int target_file = file + file_delta;
                if (!on_board(target_file, target_rank))
                    continue;
                const int to = target_rank * 8 + target_file;
                if (pawn_targets & (uint64_t{1} << to))
                    emit(pawn, from, to);
            }

            if (on_board(file, target_rank)) {
                const int to = target_rank * 8 + file;
                if (board.pt_mb[to] == enyo::pawn)
                    emit(pawn, from, to);
            }
        }

        for (int piece_type = enyo::knight;
             piece_type < static_cast<int>(enyo::king);
             ++piece_type) {
            const uint8_t attacker = ToStockfishPiece(
                static_cast<enyo::Color>(color), static_cast<enyo::PieceType>(piece_type));
            const uint64_t targets = piece_type == enyo::knight || piece_type == enyo::queen
                ? queen_targets
                : minor_slider_targets;
            uint64_t attackers = board.pt_bb[color][piece_type];
            while (attackers) {
                const int from = enyo::pop_lsb(attackers);
                uint64_t attacks = 0;
                switch (piece_type) {
                case enyo::knight:
                    attacks = enyo::knight_attack_table[from];
                    break;
                case enyo::bishop:
                    attacks = enyo::get_bishop_attacks(from, occupied);
                    break;
                case enyo::rook:
                    attacks = enyo::get_rook_attacks(from, occupied);
                    break;
                default:
                    attacks = enyo::get_bishop_attacks(from, occupied)
                            | enyo::get_rook_attacks(from, occupied);
                    break;
                }
                attacks &= targets;
                while (attacks)
                    emit(attacker, from, enyo::pop_lsb(attacks));
            }
        }
    }

    features[enyo::white].sort();
    features[enyo::black].sort();
    return features;
}

} // namespace NNUE::Stockfish::FullThreats
