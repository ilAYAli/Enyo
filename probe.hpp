#pragma once

#include "types.hpp"
#include "board.hpp"
#include "config.hpp"
#include "fmt/core.h"


namespace syzygy {

extern bool init(const std::string & tb_path);

enum Status {
    Error = -1,
    Win,
    Draw,
    Loss,
};


Status WDL_probe(enyo::Board & board);
std::pair<int, enyo::Move> DTZ_probe(enyo::Board&  board, Status & status);

struct pos {
    uint64_t white {};
    uint64_t black {};
    uint64_t kings {};
    uint64_t queens {};
    uint64_t rooks {};
    uint64_t bishops {};
    uint64_t knights {};
    uint64_t pawns {};
    uint8_t castling {};
    uint8_t rule50 {};
    uint8_t ep {};
    bool turn {};
    uint16_t move {};
};

}  // syzygy ns.

template <>
struct fmt::formatter<syzygy::pos> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const syzygy::pos& p, FormatContext& ctx) {
        return format_to(ctx.out(),
            "{{\n"
            "    white:    {:#018x},\n"
            "    black:    {:#018x},\n"
            "    kings:    {:#018x},\n"
            "    queens:   {:#018x},\n"
            "    rooks:    {:#018x},\n"
            "    bishops:  {:#018x},\n"
            "    knights:  {:#018x},\n"
            "    pawns:    {:#018x},\n"
            "    castling: {},\n"
            "    rule50:   {},\n"
            "    ep:       {},\n"
            "    turn:     {},\n"
            "    move:     {}\n"
            "}}",
        p.white,
        p.black,
        p.kings,
        p.queens,
        p.rooks,
        p.bishops,
        p.knights,
        p.pawns,
        p.castling,
        p.rule50,
        p.ep,
        p.turn,
        p.move);
    }
};


#if 0
template <>
struct fmt::formatter<syzygy::pv_status> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const syzygy::pv_status& status, FormatContext& ctx) {
        switch (status) {
            case syzygy::pv_status::PV_OK:
                return format_to(ctx.out(), "PV_OK");
            case syzygy::pv_status::PV_FAILED:
                return format_to(ctx.out(), "PV_FAILED");
            case syzygy::pv_status::PV_CHECKMATE:
                return format_to(ctx.out(), "PV_CHECKMATE");
            case syzygy::pv_status::PV_STALEMATE:
                return format_to(ctx.out(), "PV_STALEMATE");
        }
        return format_to(ctx.out(), "Unknown");
    }
};
#endif

