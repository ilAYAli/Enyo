#include <iostream>
#include <string>
#include <regex>
#include <chrono>
#include <sstream>
#include <vector>

#include <fstream>
#include <filesystem>
#include <string>

#include "config.hpp"
#include "fmt/format.h"
#include "fmt/core.h"

#include "pgn.hpp"
#include "board.hpp"
#include "movegen.hpp"
#include "movegen_helper.hpp"
#include "types.hpp"

using namespace enyo;

namespace fs = std::filesystem;

namespace enyo { Pgn pgn; }

auto resolve_pt(char  c) -> PieceType {
    switch (c) {
        case 'P': return PieceType::pawn;
        case 'N': return PieceType::knight;
        case 'B': return PieceType::bishop;
        case 'R': return PieceType::rook;
        case 'Q': return PieceType::queen;
        case 'K': return PieceType::king;
    }
    return PieceType::no_piece_type;
}

auto resolve_file(char c) -> uint64_t {
    switch (c) {
        case 'a': return 0x8080808080808080;
        case 'b': return 0x4040404040404040;
        case 'c': return 0x2020202020202020;
        case 'd': return 0x1010101010101010;
        case 'e': return 0x0808080808080808;
        case 'f': return 0x0404040404040404;
        case 'g': return 0x0202020202020202;
        case 'h': return 0x0101010101010101;
    }
    return 0xffffffffffffffff;
}

auto resolve_rank(char c) -> uint64_t {
    switch (c) {
        case '1': return 0x00000000000000ff;
        case '2': return 0x000000000000ff00;
        case '3': return 0x0000000000ff0000;
        case '4': return 0x00000000ff000000;
        case '5': return 0x000000ff00000000;
        case '6': return 0x0000ff0000000000;
        case '7': return 0x00ff000000000000;
        case '8': return 0xff00000000000000;
    }
    return 0xffffffffffffffff;
}

auto file_to_char = [](square_t src) {
    auto f = frtab[src][0];
    if      (f & file_a) return "a";
    else if (f & file_b) return "b";
    else if (f & file_c) return "c";
    else if (f & file_d) return "d";
    else if (f & file_e) return "e";
    else if (f & file_f) return "f";
    else if (f & file_g) return "g";
    else if (f & file_h) return "h";
    return "";
};

auto rank_to_char = [](square_t src) {
    auto r = frtab[src][1];
    if      (r & rank_1) return "1";
    else if (r & rank_2) return "2";
    else if (r & rank_3) return "3";
    else if (r & rank_4) return "4";
    else if (r & rank_5) return "5";
    else if (r & rank_6) return "6";
    else if (r & rank_7) return "7";
    else if (r & rank_8) return "8";
    return "";
};

auto piece_type_to_string(enyo::PieceType type) {
    switch (type) {
        case enyo::PieceType::pawn:   return "pawn";
        case enyo::PieceType::rook:   return "rook";
        case enyo::PieceType::knight: return "knight";
        case enyo::PieceType::bishop: return "bishop";
        case enyo::PieceType::queen:  return "queen";
        case enyo::PieceType::king:   return "king";
        case enyo::PieceType::no_piece_type: return "None";
        default: return "??";
    }
}

struct PgnMove {
    std::string token;
    Color side = white;
    std::string piece;
    PieceType pt = PieceType::no_piece_type;
    uint64_t src_file {};
    uint64_t src_rank;
    bool capture = false;
    std::string target_square;
    square_t dst = 0;
    PieceType promotion;
    bool check = false;
    bool checkmate = false;
    bool castle = false;
    enyo::Move move {};
};

namespace fmt {

template <>
struct formatter<PgnMove> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }
    template <typename FormatContext>
    auto format(const PgnMove& pm, FormatContext& ctx) {
        return fmt::format_to(ctx.out(),
            "Pgn {{\n"
            "  token: \"{}\",\n"
            "  color: {},\n"
            "  piece: \"{}\",\n"
            "  source_file: {:016X},\n"
            "  source_rank: {:016X},\n"
            "  capture: {},\n"
            "  dst: {},\n"
            "  promotion: {},\n"
            "  check: {},\n"
            "  checkmate: {},\n"
            "  castle: {}\n"
            "  move: {}\n"
            "}}",
            pm.token,
            pm.side ? "white" : "black", // TODO
            piece_type_to_string(pm.pt),
            pm.src_file, pm.src_rank,
            pm.capture,
            sq2str(pm.dst),
            piece_type_to_string(pm.promotion),
            pm.check, pm.checkmate, pm.castle,
            mv2str(pm.move));
    }
};

}

PgnMove parsePgnToken(const std::string& token)
{
    PgnMove result;
    result.token = token;

    std::regex pattern(R"(^(([KQRBN])?([a-h])?([1-8])?(x)?([a-h][1-8])(=[QRBN])?([+#])?|O-O(-O)?)$)");
    std::smatch matches;

    std::string piece;
    std::string target_square;
    std::string src_file;
    std::string src_rank;
    std::string promotion;
    if (std::regex_match(token, matches, pattern)) {
        if (matches[9].matched) {
            result.castle = true;
        } else {
            piece = matches[2].matched ? matches[2].str() : "P";  // Default to pawn if piece is not specified

            src_file = matches[3].matched ? matches[3].str() : "";
            src_rank = matches[4].matched ? matches[4].str() : "";
            result.src_rank = resolve_rank(src_rank[0]);

            result.capture = matches[5].matched;
            target_square = matches[6].str();

            if (piece == "P" && src_file.empty() && target_square.length() == 2) {
                src_file = target_square[0];
            }
            result.src_file = resolve_file(src_file[0]);
            result.dst = str2sq(target_square.c_str());

            if (matches[7].matched) {
                promotion = matches[7].str().substr(1);
                result.promotion = resolve_pt(promotion[0]);
            }

            if (matches[8].matched) {
                std::string checkOrMate = matches[8].str();
                if (checkOrMate == "+") {
                    result.check = true;
                } else if (checkOrMate == "#") {
                    result.checkmate = true;
                }
            }
            result.pt = resolve_pt(piece[0]);

        }
    }

    return result;
}


template <Color Us>
bool pgntokenToMove(Board & b, PgnMove & pm, int half_move = 0)
{
    const Movelist moves = Us == white
        ? generate_legal_moves<white, false, false>(b)
        : generate_legal_moves<black, false, false>(b);
    if (moves.empty()) {
        fmt::print("error, no moves\n");
        return false;
    }

    pm.side = Us;
    if (pm.token == "0-1") return false;
    if (pm.token == "1-0") return false;
    if (pm.token == "1-1") return false;
    if (pm.token == "O-O") {
        pm.move = Us == white
            ? resolve_move<white>(b, king, e1, g1)
            : resolve_move<black>(b, king, e8, g8);
    } else if (pm.token == "O-O-O") {
        pm.move = Us == white
            ? resolve_move<white>(b, king, e1, c1)
            : resolve_move<black>(b, king, e8, c8);
    } else if (pm.promotion != enyo::no_piece_type) {
        for (auto const move : moves) {
            const auto src_mask = 1ULL << move.get_src();
            if (move.get_src_piece() == pm.pt)
                continue;
            if (!(src_mask & pm.src_file))
                continue;
            if (!(src_mask & pm.src_rank))
                continue;
            if (move.get_dst() != pm.dst)
                continue;
            pm.move = Us == white
                ? resolve_move<white>(b, pm.pt, move.get_src(), pm.dst)
                : resolve_move<black>(b, pm.pt, move.get_src(), pm.dst);
            pm.move.set_promo_piece(pm.promotion);
            break;
        }
    } else {
        for (auto const move : moves) {
            const auto src_mask = 1ULL << move.get_src();
            if (move.get_src_piece() != pm.pt)
                continue;
            if (!(src_mask & pm.src_file))
                continue;
            if (!(src_mask & pm.src_rank))
                continue;
            if (move.get_dst() != pm.dst)
                continue;
            pm.move = move;
            break;
        }
    }
    if (!pm.move) {
        fmt::print("\nError, move {} not found\n{}\n", half_move, pm);
        fmt::print("{}\n", b);
        fmt::print("\n{} moves:\n", pm.pt);
        for (auto const move : moves) {
            if (move.get_src_piece() == pm.pt) {
                fmt::print("  {}\n", move);
            }
        }
        exit(1);
    }
    return true;
}

std::string extract_string_between_quotes(const std::string& str) {
    size_t startPos = str.find_first_of('"');
    if (startPos == std::string::npos) {
        return "";
    }
    size_t endPos = str.find_first_of('"', startPos + 1);
    if (endPos == std::string::npos) {
        return "";
    }

    return str.substr(startPos + 1, endPos - startPos - 1);
}

std::vector<std::string> extract_pgn_tokens(Pgn & p, const std::string & data)
{
    std::vector<std::string> tokens;
    bool within_curly = false;
    bool within_brackets = false;
    bool within_square = false;

    for (size_t i = 0; i < data.size(); i++) {
        auto c = data[i];
        if (c == '{') {
            within_curly = true;
        } else if (c == '}') {
            within_curly = false;
        } else if (c == '(') {
            within_brackets = true;
        } else if (c == ')') {
            within_brackets = false;
        } else if (c == '[') {
            std::string key;
            size_t j = 0;
            while (i + j < data.size()) {
                const auto wc = data[i + j++];
                if ((wc == '\n') || (wc == ']'))
                    break;
                if (wc == '[')
                    continue;
                key += wc;
            }
            i += j;
            auto value = extract_string_between_quotes(key);
            if (key.find("Event ") != std::string::npos) {
                p.event = value;
            } else if (key.find("Site ") != std::string::npos) {
                p.site = value;
            } else if (key.find("Date ") != std::string::npos) {
                p.date = value;
            } else if (key.find("White ") != std::string::npos) {
                p.white_player = value;
            } else if (key.find("Black ") != std::string::npos) {
                p.black_player = value;
            } else if (key.find("Result ") != std::string::npos) {
                p.result = value;
            }
        } else if (!within_curly && !within_brackets && !within_square) {
            if (c == '\n' || c == ' ') {
                continue;
            }
            std::string word;
            size_t j = 0;
            while (i + j < data.size()) {
                const auto wc = data[i + j];
                if ((wc == ' ') || (wc == '\n'))
                    break;
                word += wc;
                j++;
            }
            i += j;

            if (word.find("...") != std::string::npos) {
                continue;
            }
            if (word.find("$") != std::string::npos) {
                continue;
            }

            auto containsNumberAndDot = [](const std::string& input) -> bool {
                std::regex pattern(R"(\d+\.)");
                return std::regex_search(input, pattern);
            };
            if (containsNumberAndDot(word)) {
                continue;
            }
            tokens.push_back(word);
        }
    }
    return tokens;
}

std::vector<Move> extract_moves(Board & b, const std::vector<std::string> & tokens)
{
    constexpr bool debug_conversion = false;

    std::vector<Move> resolved;
    size_t i = 0;
    int half_move = 1;
    while (i < tokens.size()) {
        auto & white_s = tokens[i++];
        if (i > tokens.size())
            break;
        { // white:
            auto pgnm = parsePgnToken(white_s);
            if (!pgntokenToMove<white>(b, pgnm, half_move))
                break;
            apply_move<white>(b, pgnm.move);
            //if (half_move == 33) fmt::print("[{}] {}\n{}\n", half_move, pgnm, b);
            resolved.push_back(pgnm.move);
            if constexpr (debug_conversion)
                fmt::print("{:2d}. {:>4s} -> {}", half_move, white_s, pgnm.move);
        }

        auto & black_s = tokens[i++];
        if (i > tokens.size())
            break;
        { // black:
            auto pgnm = parsePgnToken(black_s);
            if (!pgntokenToMove<black>(b, pgnm, half_move))
                break;
            apply_move<black>(b, pgnm.move);
            //if (half_move == 33) fmt::print("[{}] {}\n{}\n", half_move, pgnm, b);
            resolved.push_back(pgnm.move);
            if constexpr (debug_conversion)
                fmt::print("     {:2d}. {:>4s} -> {}\n", half_move, black_s, pgnm.move);
        }
        half_move++;
    }
    if constexpr (debug_conversion)
        fmt::print("\n");

    return resolved;
}

// from move to PGN:
std::string move2algebra(Move m, bool check = false)
{
    std::string s;
    const auto src = m.get_src();
    const auto dst = m.get_dst();
    const auto src_piece = m.get_src_piece();
    const auto dst_piece = m.get_dst_piece();
    const auto flags = m.get_flags();

    bool is_capture = dst_piece != PieceType::no_piece_type;
    //s = fmt::format("({}) ", m);

    if (flags & Move::Flags::Castle) {
        if (dst == g1) s += "O-O";
        else if (dst == c1) s += "O-O-O";
        else if (dst == g8) s += "O-O";
        else if (dst == c8) s += "O-O-O";
    } else if (flags & Move::Flags::Promote) {
        const auto promo_piece = m.get_promo_piece();
        s += sq2str(dst);
        switch (promo_piece) {
            case PieceType::queen:  s += "Q"; break;
            case PieceType::rook:   s += "R"; break;
            case PieceType::bishop: s += "B"; break;
            case PieceType::knight: s += "N"; break;
            default:                s += "?"; break;
        }
    } else { // regular move:
        // axb3 -> a2b3
        switch (src_piece) {
            case PieceType::pawn:{
                if (is_capture) { // [e]xd6
                    s += file_to_char(src);
                }
                break;
            }
            case PieceType::knight: {
                s += "N";      // [N]d5xe6
                s += file_to_char(src);         // N[d]5xe6
                s += rank_to_char(src);         // Nd[5]xe6
                break;
            }
            case PieceType::bishop: {
                s += "B";
                s += file_to_char(src);
                s += rank_to_char(src);         // Nd[5]xe6
                break;
            }
            case PieceType::rook: {
                s += "R";
                s += file_to_char(src);
                s += rank_to_char(src);         // Nd[5]xe6
                break;
            }
            case PieceType::queen: {
                s += "Q";
                s += file_to_char(src);
                s += rank_to_char(src);         // Nd[5]xe6
                break;
            }
            case PieceType::king: {
                s += "K";
                break;
            }
            default: {
                s += "?" ;
                break;
            }
        }

        if (is_capture) {
            s += "x";
        }
        s += sq2str(m.get_dst());
        s += check ? "+" : "";
    }
    return s;
}

std::vector<Move> copy_played_moves(const Board & b)
{
    std::vector<Move> moves;
    moves.reserve(b.histply);
    for (auto i = 0; i < b.histply; i++)
        moves.push_back(b.history[i].move);
    return moves;
}

Board copy_initial_board(const Board & from)
{
    // board might have started from a FEN != "startpos"
    Board cpy = from;
    auto moves = copy_played_moves(from);

    auto side = ~from.side;
    for (auto it = std::ranges::rbegin(moves); it != std::ranges::rend(moves); ++it) {
        if (side == white)
            revert_move<white, true, false>(cpy);
        else
            revert_move<black, true, false>(cpy);
        side = ~side;
    }

    cpy.histply = 0;
    return cpy;
}

namespace enyo {

void print_pgn(Board & b)
{
    Board pgn_board = copy_initial_board(b);
    auto moves = copy_played_moves(b);

    constexpr bool debug = false;

    fmt::print("\n");
    std::ostringstream oss;
    int ply = 1;
    auto side = pgn_board.side;
    for (auto move : moves) {
        bool check = false;
        if (side == white) {
            apply_move<white, true, false>(pgn_board, move);
            check = is_check<black>(pgn_board);
        } else {
            apply_move<black, true, false>(pgn_board, move);
            check = is_check<white>(pgn_board);
        }
        const std::string s = fmt::format("{}. {} ", ply++, move2algebra(move, check));
        if constexpr (debug) {
            fmt::print("applying move: {}\n", move);
            fmt::print("s: {}\n", s);
        }
        oss << s;
        if (ply % 10 == 0)
            oss << "\n";
    }

    bool white_mate = false;
    auto white_moves = generate_legal_moves<white, false, false>(pgn_board);
    if (!white_moves.size() && is_check<white>(pgn_board))
        white_mate = true;

    bool black_mate = false;
    auto black_moves = generate_legal_moves<black, false, false>(pgn_board);
    if (!black_moves.size() && is_check<black>(pgn_board))
        black_mate = true;

    auto result = [&]() {
        if (white_mate)
            return "0-1";
        else if (black_mate)
            return "1-0";
        return "0-0";
    }();

    auto current_date = [] () {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm = *std::localtime(&now_time_t);
        return fmt::format("{:04d}.{:02d}.{:02d}", now_tm.tm_year + 1900, now_tm.tm_mon + 1, now_tm.tm_mday);
    }();

    auto hostname = []() -> std::string {
        char data[256];
        if (!gethostname(data, sizeof(data)))
            return std::string(data);
        return {};
    }();

    std::string headline = fmt::format(
        "[Event \"{}\"]\n"
        "[Site \"{}\"]\n"
        "[Date \"{}\"]\n"
        "[Round \"{}\"]\n"
        "[White \"{}\"]\n"
        "[Black \"{}\"]\n"
        "[Result \"{}\"]\n\n",
            "?",
            hostname.empty() ? "?" : hostname,
            current_date.empty() ? "????.??.??" : current_date,
            "1",
            pgn.white_player.empty() ? "?" : pgn.white_player,
            pgn.black_player.empty() ? "?" : pgn.black_player,
            result
    );

    fmt::print("\n{} {}\n", headline, oss.str());
}

std::string load_pgn(Board & b, const std::string& filename)
{
    constexpr bool debug_moves = false;
    if (!fs::exists(filename)) {
        fmt::print("error, pgn file does not exist: '{}'\n", filename);
        return "";
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        fmt::print("error, failed to open pgn file: '{}'\n", filename);
        return "";
    }

    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    file.close();

    auto tokens = extract_pgn_tokens(pgn, data);
    auto moves = extract_moves(b, tokens);

    if constexpr (debug_moves) {
        fmt::print("position startpos moves ");
        for (auto move : moves) {
            fmt::print("{} ", move);
        }
        fmt::print("\n");
    }

    return data;
}

} // enyo ns
