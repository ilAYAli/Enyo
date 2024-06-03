#pragma once
#include <string_view>

namespace enyo {
class Board;

extern bool set_fen(Board & b, std::string_view);
extern std::string get_fen(Board const & b);

}
