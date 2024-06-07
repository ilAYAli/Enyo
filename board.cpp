#include <string>
#include "board.hpp"
#include "fen.hpp"
#include "magic/magic.hpp"
#include "zobrist.hpp"
#include "movegen.hpp"

extern uint64_t generate_hash(enyo::Board const & );

using namespace enyo;

// used by copy assignment operator
void Board::swap_data(Board& other) {
    using std::swap;
    swap(half_moves, other.half_moves);
    swap(histply, other.histply);
    swap(side, other.side);
    swap(gamestate, other.gamestate);
    swap(hash, other.hash);
    swap(zbrs, other.zbrs);

    std::swap(pt_mb, other.pt_mb);
    std::swap(pt_bb, other.pt_bb);
    std::swap(color_bb, other.color_bb);
    std::swap(checkers_bb, other.checkers_bb);
    std::swap(between_bb, other.between_bb);
    std::swap(line_bb, other.line_bb);
    std::swap(blockers_bb, other.blockers_bb);
    std::swap(pinners_bb, other.pinners_bb);
    std::swap(attacks_bb, other.attacks_bb);
    std::swap(all_attacks_bb, other.all_attacks_bb);
    std::swap(pv_table, other.pv_table);
    std::swap(history, other.history);
}

// used by copy constructor
void Board::copy_data(const Board& other) {
    half_moves = other.half_moves;
    histply = other.histply;
    side = other.side;
    gamestate = other.gamestate;
    hash = other.hash;
    zbrs = other.zbrs;

    std::memcpy(pt_mb, other.pt_mb, sizeof(pt_mb));
    std::memcpy(pt_bb, other.pt_bb, sizeof(pt_bb));
    std::memcpy(color_bb, other.color_bb, sizeof(color_bb));
    std::memcpy(checkers_bb, other.checkers_bb, sizeof(checkers_bb));
    std::memcpy(between_bb, other.between_bb, sizeof(between_bb));
    std::memcpy(line_bb, other.line_bb, sizeof(line_bb));
    std::memcpy(blockers_bb, other.blockers_bb, sizeof(blockers_bb));
    std::memcpy(pinners_bb, other.pinners_bb, sizeof(pinners_bb));
    std::memcpy(attacks_bb, other.attacks_bb, sizeof(attacks_bb));
    std::memcpy(all_attacks_bb, other.all_attacks_bb, sizeof(all_attacks_bb));
    std::memcpy(pv_table, other.pv_table, sizeof(pv_table));
    std::memcpy(history, other.history, sizeof(history));
}

// called on set_fen:
void Board::clear_data() {
    std::memset(pt_mb, 0, sizeof(pt_mb));
    std::memset(pt_bb, 0, sizeof(pt_bb));
    std::memset(color_bb, 0, sizeof(color_bb));
    std::memset(checkers_bb, 0, sizeof(checkers_bb));
    std::memset(between_bb, 0, sizeof(between_bb));
    std::memset(line_bb, 0, sizeof(line_bb));
    std::memset(blockers_bb, 0, sizeof(blockers_bb));
    std::memset(pinners_bb, 0, sizeof(pinners_bb));
    std::memset(attacks_bb, 0, sizeof(attacks_bb));
    std::memset(all_attacks_bb, 0, sizeof(all_attacks_bb));
    std::fill(std::begin(pv_table), std::end(pv_table), enyo::Move{});
    std::fill(std::begin(history), std::end(history), enyo::Undo{});
    side = {};
    gamestate = {};
    histply = {};
    half_moves = {};
    hash = {};
}

Board::Board(std::string_view fenstr)
{
    set(fenstr);
}

void Board::set(std::string_view fen) {
    // clear ply, captues, bitboards, ...
    clear_data();

    set_fen(*this, fen);
    hash = zobrist::generate_hash(*this);

    init_sliders_attacks(bishop);
    init_sliders_attacks(rook);

    for (square_t s1 = 0; s1 < 64; ++s1) {
        for (square_t s2 = 0; s2 < 64; ++s2) {
            if (get_bishop_attacks(s1, 0) & (1ULL << s2)) {
                bitboard_t l1 = get_bishop_attacks(s1, 0) | 1ULL << s1;
                bitboard_t l2 = get_bishop_attacks(s2, 0) | 1ULL << s2;
                bitboard_t line = l1 & l2;
                line_bb[s1][s2] = line;

                bitboard_t b1 = get_bishop_attacks(s1, 1ULL << s2); // don't include b1 (king et al).
                bitboard_t b2 = get_bishop_attacks(s2, 1ULL << s1);
                bitboard_t between = b1 & b2;
                between_bb[s1][s2] = between;
            }
            if (get_rook_attacks(s1, 0) & (1ULL << s2)) {
                bitboard_t l1 = get_rook_attacks(s1, 0) | 1ULL << s1;
                bitboard_t l2 = get_rook_attacks(s2, 0) | 1ULL << s2;
                bitboard_t line = l1 & l2;
                line_bb[s1][s2] = line;

                bitboard_t b1 = get_rook_attacks(s1, 1ULL << s2);// | 1ULL << s1;
                bitboard_t b2 = get_rook_attacks(s2, 1ULL << s1);// | 1ULL << s2;
                bitboard_t between = b1 & b2;
                between_bb[s1][s2] = between;
            }
        }
    }

    // only needed for tests that don't use movegen
    init_pinner_blocker_bb<white>(*this);
    init_pinner_blocker_bb<black>(*this);
}

std::string Board::fen() const {
    return ::get_fen(*this);
}


// Lowercase letters describe the black pieces
std::string Board::str(uint64_t attack_mask, unsigned indent) const
{
    using namespace enyo;

    auto const prefix = std::string(indent, ' ');
    fmt::memory_buffer tmp;
    auto out = fmt::appender(tmp);

    fmt::format_to(out, "\n{}     0   1   2   3   4   5   6   7\n", prefix);
    fmt::format_to(out, "{}   +---+---+---+---+---+---+---+---+\n", prefix);

    for (int bit = 63; bit >= 0; bit--) {
        auto const mask = 1ULL << bit;
        if (bit == 63) {
            fmt::format_to(out, "{}{:>2d} | ", prefix, /*rank*/8);
        } else if (bit && (bit % 8 == 7)) {
            int const rank = bit / 8 + 1;
            std::string extra = "";
            if (rank == 7) extra = fmt::format("  key: {:16X}", hash);
            if (rank == 6) extra = fmt::format("  side: {}", side);
            if (rank == 5) extra = fmt::format("  enpassant: {}",
                (gamestate.enpassant_square & (rank_4|rank_5))
                ? sq2str(gamestate.enpassant_square)
                : std::string("(none)"));
            fmt::format_to(out,
                " {} {}\n"
                " {}  +---+---+---+---+---+---+---+---+\n"
                "{}{:>2d} | ",
                /*bit*/bit + 1, extra,
                prefix,
                /*idx:*/prefix,
                /* rank */ rank);
        }

        if (attack_mask & (1ULL << bit)) {
            fmt::format_to(out, "* | ");
        } else {
            if (mask & pt_bb[black][static_cast<uint8_t>(pawn)]) {
                fmt::format_to(out, "p | ");
            } else if (mask & pt_bb[black][rook]) {
                fmt::format_to(out, "r | ");
            } else if (mask & pt_bb[black][knight]) {
                fmt::format_to(out, "n | ");
            } else if (mask & pt_bb[black][bishop]) {
                fmt::format_to(out, "b | ");
            } else if (mask & pt_bb[black][queen]) {
                fmt::format_to(out, "q | ");
            } else if (mask & pt_bb[black][king]) {
                fmt::format_to(out, "k | ");
            // white:
            } else if (mask & pt_bb[white][pawn]) {
                fmt::format_to(out, "P | ");
            } else if (mask & pt_bb[white][rook]) {
                fmt::format_to(out, "R | ");
            } else if (mask & pt_bb[white][knight]) {
                fmt::format_to(out, "N | ");
            } else if (mask & pt_bb[white][bishop]) {
                fmt::format_to(out, "B | ");
            } else if (mask & pt_bb[white][queen]) {
                fmt::format_to(out, "Q | ");
            } else if (mask & pt_bb[white][king]) {
                fmt::format_to(out, "K | ");
            } else {
                fmt::format_to(out, "  | ");
            }
        }
    }
    fmt::format_to(out, " {}\n", 0); // lower bitpos
    fmt::format_to(out, "{}   +---+---+---+---+---+---+---+---+\n", prefix);
    fmt::format_to(out, "{}     A   B   C   D   E   F   G   H  \n", prefix);
    fmt::format_to(out, "\nFen: {}", ::get_fen(*this));

    return fmt::to_string(tmp);
};
