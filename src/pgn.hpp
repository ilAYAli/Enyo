#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "types.hpp"

namespace enyo {
class Board;

struct Pgn {
    std::string event;
    std::string site;
    std::string date;
    std::string white_player;
    std::string black_player;
    std::string result;
};

extern Pgn pgn;

std::string load_pgn(enyo::Board & b, const std::string & filename);
void print_pgn(enyo::Board & b);
std::optional<Move> uci_to_move(Board const & b, std::string_view uci);
std::string move_to_algebra(Board const & b, Move move);
std::string uci_to_algebra(Board const & b, std::string_view uci);

}
