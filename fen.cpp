// Forsyth-Edwards Notation (FEN)

#include <string>
#include <sstream>
#include <charconv>
#include "eventlog.hpp"
#include "fen.hpp"
#include "types.hpp"
#include "board.hpp"
#include "util.hpp"

#include "fmt/format.h"

using namespace enyo;
using namespace eventlog;

namespace {

constexpr inline uint8_t fencr2bit(uint8_t col, uint8_t row)
{
    return static_cast<uint8_t>((7 - col) * 8 + (7 - row));
}

std::vector<std::string> split(const std::string & str, char delim = ' ') {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream istr(str);

    while (std::getline(istr, token, delim))
        tokens.push_back(token);
    return tokens;
}

inline void set_white_to_move(Board & b, std::vector<std::string> tokens) {
    bool const white_to_move = tokens[0][0] == 'w';
    b.gamestate.white_to_move = white_to_move;
}

inline void set_castling_rights(Board &b, std::vector<std::string> tokens) {
    auto const & token = tokens[1];
    b.gamestate.set_castle(CastlingRights::white_oo,  token.find('K') != std::string::npos);
    b.gamestate.set_castle(CastlingRights::white_ooo, token.find('Q') != std::string::npos);
    b.gamestate.set_castle(CastlingRights::black_oo,  token.find('k') != std::string::npos);
    b.gamestate.set_castle(CastlingRights::black_ooo, token.find('q') != std::string::npos);
}

inline void set_enpassant(Board & b, std::vector<std::string> tokens)
{
    if (tokens.size() > 2) {
        b.gamestate.enpassant_square = str2sq(tokens[2].c_str());
    }
}

inline void set_half_move_clock(Board & b, std::vector<std::string> tokens) {
    if (tokens.size() > 3) {
        auto const str = tokens[3];
        uint16_t num = 0;
        std::from_chars(str.data(), str.data() + str.size(), num);
        b.gamestate.half_moves = num;
    }
}

inline void set_full_moves(Board & /* b */, std::vector<std::string> tokens) {
    if (tokens.size() > 4) {
        auto const str = tokens[4];
        uint16_t num = 0;
        std::from_chars(str.data(), str.data() + str.size(), num);
    }
}

} // anon ns.


namespace enyo {

std::string get_fen(Board const & b) {
    fmt::memory_buffer tmp;
    auto out = fmt::appender(tmp);
    std::string s = "";
    for (uint8_t col = 0; col < 8; ++col) {
        uint8_t row = 0;
        while (row < 8) {
            uint64_t mask = 1ULL << fencr2bit(col, row); // row = actual row
            // black:
            if (mask & b.pt_bb[black][pawn]) {
                s += "p"; ++row;
            } else if (mask & b.pt_bb[black][rook]) {
                s += "r"; ++row;
            } else if (mask & b.pt_bb[black][knight]) {
                s += "n"; ++row;
            } else if (mask & b.pt_bb[black][bishop]) {
                s += "b"; ++row;
            } else if (mask & b.pt_bb[black][queen]) {
                s += "q"; ++row;
            } else if (mask & b.pt_bb[black][king]) {
                s += "k"; ++row;
            // white:
            } else if (mask & b.pt_bb[white][pawn]) {
                s += "P"; ++row;
            } else if (mask & b.pt_bb[white][rook]) {
                s += "R"; ++row;
            } else if (mask & b.pt_bb[white][knight]) {
                s += "N"; ++row;
            } else if (mask & b.pt_bb[white][bishop]) {
                s += "B"; ++row;
            } else if (mask & b.pt_bb[white][queen]) {
                s += "Q"; ++row;
            } else if (mask & b.pt_bb[white][king]) {
                s += "K"; ++row;
            } else {
                uint8_t empty = 0;
                for (uint8_t r = row + 0; r < 8; r++) {
                    auto const m = 1ULL << fencr2bit(col, r);
                    if (m & (b.color_bb[white] | b.color_bb[black]))
                        break;
                    empty++;
                }
                s += std::to_string(empty);
                row += empty;
            }
        }
        if (col != 7)
            s += "/";
    }
    fmt::format_to(out, "{}", s);
    s = "";

    fmt::format_to(out, " ");

    //if (b.gamestate.white_to_move) // TODO: is this correct?
    if (b.side == white)
        fmt::format_to(out, "w ");
    else
        fmt::format_to(out, "b ");

    // castling rights, uppercase letters come first to indicate white's castling availability:
    std::string castling_rights = "";
    if (b.gamestate.can_castle(CastlingRights::white_oo))  castling_rights += "K";
    if (b.gamestate.can_castle(CastlingRights::white_ooo)) castling_rights += "Q";
    if (b.gamestate.can_castle(CastlingRights::black_oo))  castling_rights += "k";
    if (b.gamestate.can_castle(CastlingRights::black_ooo)) castling_rights += "q";
    fmt::format_to(out, "{}", castling_rights.empty() ? "-" : castling_rights);

    if (b.gamestate.enpassant_square > 8 && b.gamestate.enpassant_square < 56)
        fmt::format_to(out, " {} ", sq2str(b.gamestate.enpassant_square));
    else
        fmt::format_to(out, " - ");

    // "This is the number of halfmoves since the last capture or pawn advance".
    // Note: position fen fenstr can specify halfmoves != 0, so this needs to be added to the fen:
    fmt::format_to(out, "{} ", b.half_moves + b.gamestate.half_moves);

    // "The number of the full move. It starts at 1, and is incremented after black's move."
    fmt::format_to(out, "{}", (b.histply + 1) / 2);

    //fmt::print("* {} {}\n", __func__, fmt::to_string(tmp));
    return fmt::to_string(tmp);
}

bool set_fen(Board & b, std::string_view fenstr)
{
    if (fenstr == "startpos")
        fenstr = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    if constexpr(false)
        log(Log::debug, "{} {}\n", __func__, fenstr);

    auto tokens = split(std::string(fenstr), '/');
    if  (tokens.empty() || tokens.size() != 8) {
        log(Log::error, "incorrect FEN string\n");
        return false;
    }

    auto gs_v = split(tokens[7], ' ');
    tokens[7] = gs_v[0];
    gs_v.erase(gs_v.begin());

    set_white_to_move(b, gs_v);
    set_castling_rights(b, gs_v);
    set_enpassant(b, gs_v);
    set_half_move_clock(b, gs_v);
    set_full_moves(b, gs_v);

    b.side = b.gamestate.white_to_move ? white : black;

    for (uint8_t col = 0; col < 8; ++col) {
        auto token = tokens[col];
        uint8_t row = 0;
        for (uint8_t trow = 0; trow < token.length(); ++trow) {
            uint64_t mask = 1ULL << fencr2bit(col, row);
            auto const c = token[trow];
            auto const sq = ((7 - col) * 8) + (7 - row);
            switch (c) {
                case 'p': b.pt_bb[black][pawn]      |= mask; b.pt_mb[sq] = pawn;   break;
                case 'r': b.pt_bb[black][rook]      |= mask; b.pt_mb[sq] = rook;   break;
                case 'n': b.pt_bb[black][knight]    |= mask; b.pt_mb[sq] = knight; break;
                case 'b': b.pt_bb[black][bishop]    |= mask; b.pt_mb[sq] = bishop; break;
                case 'q': b.pt_bb[black][queen]     |= mask; b.pt_mb[sq] = queen;  break;
                case 'k': b.pt_bb[black][king]      |= mask; b.pt_mb[sq] = king;   break;

                case 'P': b.pt_bb[white][pawn]      |= mask; b.pt_mb[sq] = pawn;   break;
                case 'R': b.pt_bb[white][rook]      |= mask; b.pt_mb[sq] = rook;   break;
                case 'N': b.pt_bb[white][knight]    |= mask; b.pt_mb[sq] = knight; break;
                case 'B': b.pt_bb[white][bishop]    |= mask; b.pt_mb[sq] = bishop; break;
                case 'Q': b.pt_bb[white][queen]     |= mask; b.pt_mb[sq] = queen;  break;
                case 'K': b.pt_bb[white][king]      |= mask; b.pt_mb[sq] = king;   break;
                default:
                    auto const jump = static_cast<uint8_t>((c - '0')-1);
                    row += jump; // -1 due to next line incr.
                    break;
            }
            row++;
        }
    }

    b.color_bb[white] =
        b.pt_bb[white][pawn] |
        b.pt_bb[white][rook] |
        b.pt_bb[white][knight] |
        b.pt_bb[white][bishop] |
        b.pt_bb[white][queen] |
        b.pt_bb[white][king];

    b.color_bb[black] =
        b.pt_bb[black][pawn] |
        b.pt_bb[black][rook] |
        b.pt_bb[black][knight] |
        b.pt_bb[black][bishop] |
        b.pt_bb[black][queen] |
        b.pt_bb[black][king];

    return true;
}

} // enyo ns.
