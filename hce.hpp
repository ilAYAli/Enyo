#pragma once

#include "types.hpp"
#include "board.hpp"
#include "util.hpp"
#include "movegen.hpp"

////namespace {

namespace enyo {

// eval = ((opening * (256 - phase)) + (endgame * phase)) / 256
static constexpr int taper_value(int mg, int eg, int phase)
{
    return ((mg * (256 - phase)) + (eg * phase)) / 256;
}

template <Color Us>
[[maybe_unused]] static inline int evaluate_placement(const Board & b, int phase)
{
    enum {
        op_phase = 0,
        mg_phase = 1,
        eg_phase = 2
    };

    int total_score = 0;
    auto pieces = b.color_bb[Us];
    while (pieces) {
        auto sq = pop_lsb(pieces);
        auto pt = b.pt_mb[sq];
        const auto op_score = get_value(op_phase, Us, pt, sq); // Opening values
        const auto mg_score = get_value(mg_phase, Us, pt, sq); // Middlegame values
        const auto eg_score = get_value(eg_phase, Us, pt, sq); // Endgame values
        //const auto score = op_score + taper_value(mg_score, eg_score, phase);
        const auto score = op_score + taper_value(mg_score, eg_score, phase);
        //fmt::print("sq: {}, {}, op_score: {:3d}, mg_score: {:3d}, eg_score: {:3d}, phase: {}, score: {}\n",
        //sq2str(sq), pt, op_score, mg_score, eg_score, phase, score);
        total_score += score;
    }
    return total_score;
}

template <enyo::Color Us>
[[maybe_unused]] static inline int evaluate_material(const enyo::Board & b, int phase)
{
    constexpr int pawn_value[2]   = { 100, 200 };
    constexpr int knight_value[2] = { 320, 300 };
    constexpr int bishop_value[2] = { 325, 300 };
    constexpr int rook_value[2]   = { 500, 550 };
    constexpr int queen_value[2]  = { 975, 1000 };

    int score = 0;
    score += taper_value(pawn_value[0],   pawn_value[1],   phase) * count_bits(b.pt_bb[Us][enyo::pawn]);
    score += taper_value(knight_value[0], knight_value[1], phase) * count_bits(b.pt_bb[Us][enyo::knight]);
    score += taper_value(bishop_value[0], bishop_value[1], phase) * count_bits(b.pt_bb[Us][enyo::bishop]);
    score += taper_value(rook_value[0],   rook_value[1],   phase) * count_bits(b.pt_bb[Us][enyo::rook]);
    score += taper_value(queen_value[0],  queen_value[1],  phase) * count_bits(b.pt_bb[Us][enyo::queen]);

    return score;
}

template <enyo::Color Us>
[[maybe_unused]] int evaluate_pawn_structure(const enyo::Board & b, int phase)
{
    constexpr int DoubledPawnPenalty[2] = { -20, -40 };
    constexpr int IsolatedPawnPenalty[2] = { -10, -20 };
    constexpr int PassedPawnBonus[2] = { 20, 40 };

    int score = 0;

    enyo::bitboard_t pawns = b.pt_bb[Us][enyo::pawn];
    while (pawns) {
        const auto square = pop_lsb(pawns);
        const auto file = square % 8;
        const auto rank = square / 8;

        // Check for doubled pawns
        if (enyo::get_file_bb(file) & (b.pt_bb[Us][enyo::pawn] & ~enyo::square_bb(square))) {
            score += taper_value(DoubledPawnPenalty[0], DoubledPawnPenalty[1], phase);
        }

        // Check for isolated pawns
        if (!(enyo::get_adjacent_files(file) & b.pt_bb[Us][enyo::pawn])) {
            score += taper_value(IsolatedPawnPenalty[0], IsolatedPawnPenalty[1], phase);
        }

        // Check for passed pawns
        if ((enyo::get_rank_bb(rank + 1) & b.pt_bb[~Us][enyo::pawn]) == 0) {
            score += taper_value(PassedPawnBonus[0], PassedPawnBonus[1], phase);
        }
    }

    return score;
}

template <enyo::Color Us>
[[maybe_unused]] static inline int get_pt_mobility(const enyo::Board & b, enyo::square_t sq)
{
    const enyo::bitboard_t friendly = b.color_bb[Us];
    switch (b.pt_bb[Us][sq]) {
        case enyo::pawn:    return count_bits(enyo::pawn_attack_table[~Us][sq] & ~friendly);
        case enyo::knight:  return count_bits(enyo::knight_attack_table[~sq] & ~friendly);
        case enyo::bishop:  return count_bits(bishop_moves<Us>(b, sq) & ~friendly);
        case enyo::rook:    return count_bits(rook_moves<Us>(b, sq) & ~friendly);
        case enyo::queen:   return count_bits(bishop_moves<Us>(b, sq) & ~friendly)
                                 + count_bits(rook_moves<Us>(b, sq) & ~friendly);
        case enyo::king:    return count_bits(king_moves<Us, false>(b, sq));
        default: return 0;
    }
    return 0;
}

template <Color Us>
[[maybe_unused]] static inline int evaluate_mobility(const Board & b, int phase)
{
    constexpr int mobility_bonus[2] = { 5, 10 };

    int score = 0;
    for (size_t pt = pawn; pt <= king; pt++) {
        bitboard_t pieces = b.pt_bb[Us][static_cast<PieceType>(pt)];
        while (pieces) {
            const auto sq = pop_lsb(pieces);
            score += taper_value(mobility_bonus[0], mobility_bonus[1], phase) * get_pt_mobility<Us>(b, sq);
            //score += mobility_bonus<Us>(b, sq);
        }
    }

    return score;
}

template <Color Us>
[[maybe_unused]] static inline bool is_pinned(const Board& b, square_t sq)
{
    constexpr Color Them = ~Us;
    square_t king_sq = lsb(b.pt_bb[Them][king]);

    bitboard_t our_sliders = b.pt_bb[Us][bishop] | b.pt_bb[Us][rook] | b.pt_bb[Us][queen];
    while (our_sliders) {
        square_t slider_sq = pop_lsb(our_sliders);
        bitboard_t between = b.between_bb[slider_sq][king_sq];
        if (between & square_bb(sq))
            return true;
    }

    return false;
}

template <Color Us>
[[maybe_unused]] static inline int evaluate_pinned_pieces(const Board& b, int phase)
{
    constexpr Color Them = ~Us;
    constexpr int pinned_piece_penalty[2] = { -20, -40 }; // Adjust these values as needed

    int score = 0;
    for (size_t pt = pawn; pt <= king; ++pt) {
        bitboard_t pieces = b.pt_bb[Them][static_cast<PieceType>(pt)];
        while (pieces) {
            square_t sq = pop_lsb(pieces);
            if (is_pinned<Them>(b, sq))
                score += taper_value(pinned_piece_penalty[0], pinned_piece_penalty[1], phase);
        }
    }

    return score;
}

// Function to evaluate checking
template <Color Us>
[[maybe_unused]] static inline int evaluate_checking(const Board& b, int phase)
{
    constexpr int check_penalty[2] = { -100, -200 };
    return is_check<~Us>(b)
        ? taper_value(check_penalty[0], check_penalty[1], phase)
        : 0;
}

template <Color Us>
[[maybe_unused]] static inline int evaluate_king_safety(const Board& b, int phase)
{
    // king safety: castling is prioritized in prioritize_moves
    constexpr int king_pawn_shelter_bonus[2] = { 10, 20 };  // Adjust these values as needed
    constexpr int king_open_file_penalty[2] = { -10, -20 }; // Adjust these values as needed

    square_t king_sq = lsb(b.pt_bb[Us][king]);
    int score = 0;

    // Pawn Shelter
    bitboard_t pawn_shelter = pawn_attack_table[Us][king_sq] & b.pt_bb[Us][pawn];
    score += taper_value(king_pawn_shelter_bonus[0], king_pawn_shelter_bonus[1], phase) * count_bits(pawn_shelter);

    // Open Files
    int file = king_sq % 8;
    if ((get_file_bb(file) & b.pt_bb[~Us][rook]) || (get_file_bb(file) & b.pt_bb[~Us][queen]))
        score += taper_value(king_open_file_penalty[0], king_open_file_penalty[1], phase);

    return score;
}

template <Color Us>
[[maybe_unused]] static inline int evaluate_trapped_pieces(const Board& b, int phase)
{
    constexpr int trapped_piece_penalty[2] = { -50, -100 }; // Adjust these values as needed

    int score = 0;
    for (unsigned pt = pawn; pt <= king; ++pt) {
        bitboard_t pieces = b.pt_bb[Us][static_cast<PieceType>(pt)];
        while (pieces) {
            square_t sq = pop_lsb(pieces);
            if (get_pt_mobility<Us>(b, sq) == 0)
                score += taper_value(trapped_piece_penalty[0], trapped_piece_penalty[1], phase);
        }
    }

    return score;
}

[[maybe_unused]] static inline int calculate_game_phase(const Board & b)
{
    int total_pieces = 0;

    for (unsigned pt = pawn; pt <= king; pt++) {
        total_pieces += count_bits(b.pt_bb[white][pt]);
        total_pieces += count_bits(b.pt_bb[black][pt]);
    }

    // todo: consider adjusting these values:
    constexpr int OPENING_PIECES = 32;
    constexpr int ENDGAME_PIECES = 8;

    int phase = std::clamp(OPENING_PIECES - total_pieces, 0, OPENING_PIECES - ENDGAME_PIECES);
    phase = (phase * 256) / (OPENING_PIECES - ENDGAME_PIECES);

    return phase;
}

template <Color Us>
static inline int evaluate_position_tapered(const Board & b)
{
    const int phase = calculate_game_phase(b);
    const int material_score = evaluate_material<Us>(b, phase);
    const int placement_score = evaluate_placement<Us>(b, phase);
    const int pawn_structure_score = evaluate_pawn_structure<Us>(b, phase);
    //const int king_safety_score = evaluate_king_safety<Us>(b, phase);
    //const int trapped_pieces_score = evaluate_trapped_pieces<Us>(b, phase);
    ////const int mobility_score = evaluate_mobility<Us>(b, phase);
    const int check_score = evaluate_checking<Us>(b, phase);
    int score = 0;
    score += placement_score;
    score += material_score;
    score += pawn_structure_score;
    //score += king_safety_score;
    //score += trapped_pieces_score;
    score += check_score;
    //score += mobility_score;
    return score;
}

//} // anon ns


template <Color Us>
int HCE_evaluation(const Board & b)
{
    return evaluate_position_tapered<Us>(b) - evaluate_position_tapered<~Us>(b);
}

}
