#pragma once

#include <string>

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

}
