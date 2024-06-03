#pragma once

#include "types.hpp"
#include <cstdint>
#include <chrono>
#include <thread>
#include <fmt/format.h>
#include "board.hpp"
#include "movelist.hpp"

namespace enyo {

extern void init_search();

struct Stack {
    Move * pv {nullptr};
    int  ply {};
    Move move {};
    Move excluded_move {};
    Move killers[2] {};
    Value eval {};
    int move_count {};
    bool in_check {};
    bool ttPv {};
    bool tthit {};
    int doubleExtensions {};
    int cutoffCnt {};
};

struct SearchInfo {
    explicit SearchInfo(const Board & b, int d = MAX_PLY)
        : board(b)
        , depth(d)
    {
        nnue.refresh(board);
    }
    SearchInfo() = default;

    bool time_expired() {
        using namespace std::chrono;
        const auto now = high_resolution_clock::now();
        elapsed_time = duration_cast<milliseconds>(now - starttime);
        return (starttime != stoptime) && now > stoptime;
    }

    Board board{};
    NNUE::Net nnue = NNUE::Net{};

    std::chrono::high_resolution_clock::time_point starttime;
    std::chrono::high_resolution_clock::time_point stoptime;
    std::chrono::milliseconds elapsed_time{};
    int depth = MAX_PLY;
    int wtime = -1;
    int btime = -1;
    int winc = -1;
    int binc = -1;
    int movestogo {40};
    int movetime {-1};

    uint64_t tbhits = 0; //
    uint64_t nodes {};
};


enum class NodeType {
    NonPV,
    PV,
    Root
};


struct QuadraticPV {
    QuadraticPV() {
        clear();
    }

    void setmove(Move move, int ply) {
        table[ply][ply] = move;
        std::ranges::copy(
            std::views::drop(
                table[ply + 1],
                ply + 1) | std::views::take(len[ply + 1] - ply - 1),
                std::ranges::begin(table[ply]) + ply + 1
        );
        len[ply] = len[ply + 1];
    }

    void setlen(int ply) {
        len[ply] = static_cast<uint8_t>(ply);
    }

    void clear() {
        std::fill(std::begin(len), std::end(len), 0);
        std::fill(std::begin(table[0]), std::end(table[std::size(table) - 1]), Move{});
    }

    std::string str() const {
        return fmt::format("{}", fmt::join(std::views::take(table[0], len[0]), " "));
    }

    constexpr Move bestmove() const {
        return table[0][0];
    }

    uint8_t len[MAX_PLY] = {};
    Move table[MAX_PLY][MAX_PLY] = {};
};

struct Worker {
    Worker() = default;
    Worker(const SearchInfo & si, int thread_id)
        : si(si)
        , id(thread_id)
    {}

    bool time_expired();
    void start();

    SearchInfo si {};
    Stack * ss { nullptr };
    Move killers[2] { 0, 0 };
    Move bestmove{};
    QuadraticPV pvline {};
    int id { };
};


bool is_repetition(const Board & b, int draw = 1);

template <Color Us, NodeType node>
Value negamax(
    int depth,
    Worker &,
    Stack *,
    Value alpha = static_cast<Value>(-Value::INFINITE),
    Value beta = Value::INFINITE
);

template <Color Us, NodeType node>
Value qsearch(Board &, Worker &, Stack *, int depth, int alpha, int beta);

template <Color Us>
Value evaluate(Board & b, int depth = 0);

void clear_stats(Worker &);

void search_position(Worker &);
void search_thread_main(Board &, SearchInfo &);
std::thread create_search_thread(Board & b, SearchInfo &);
void join_search_thread(std::thread & search_thread);

} // enyo
