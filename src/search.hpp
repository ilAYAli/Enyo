#pragma once

#include "types.hpp"
#include <cstdint>
#include <chrono>
#include <thread>
#include <fmt/format.h>
#include <fmt/ranges.h>
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

    void update_elapsed_time() {
        using namespace std::chrono;
        elapsed_time = duration_cast<milliseconds>(high_resolution_clock::now() - starttime);
    }

    bool time_expired() {
        using namespace std::chrono;
        const auto now = high_resolution_clock::now();
        elapsed_time = duration_cast<milliseconds>(now - starttime);
        return (starttime != stoptime) && now > stoptime;
    }

    // Soft limit: checked only between root iterations. Once crossed, the
    // current iteration is the last one we'll start — we don't abort it.
    // soft_scale_pct < 100 shrinks the budget (node-based TM: when one
    // root move absorbs nearly all effort, the decision is settled and
    // further iterations rarely change it).
    bool soft_time_expired() {
        using namespace std::chrono;
        if (starttime == soft_stoptime)
            return false;
        auto deadline = soft_stoptime;
        if (soft_scale_pct != 100
            && soft_stoptime != high_resolution_clock::time_point::max())
            deadline = starttime + (soft_stoptime - starttime) * soft_scale_pct / 100;
        return high_resolution_clock::now() > deadline;
    }

    Board board{};
    NNUE::Net nnue = NNUE::Net{};

    std::chrono::high_resolution_clock::time_point starttime;
    std::chrono::high_resolution_clock::time_point stoptime;      // hard limit
    std::chrono::high_resolution_clock::time_point soft_stoptime; // optimum
    std::chrono::milliseconds elapsed_time{};
    int depth = MAX_PLY;
    int wtime = -1;
    int btime = -1;
    int winc = -1;
    int binc = -1;
    int movestogo {0};  // 0 = sudden-death (no movestogo sent by GUI)
    int movetime {-1};
    Movelist searchmoves{};
    bool has_searchmoves = false;
    bool suppress_tablebase_cutoffs = false;

    uint64_t tbhits = 0; //
    uint64_t nodes {};
    uint64_t nodes_limit {};
    int soft_scale_pct = 100;
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

    void setmove(Move move, int ply, bool copy_tail = true) {
        table[ply][ply] = move;
        if (!copy_tail) {
            len[ply] = static_cast<uint8_t>(ply + 1);
            if (ply + 1 < MAX_PLY)
                table[ply][ply + 1] = Move {};
            return;
        }

        const int child_len = len[ply + 1];
        const int tail_begin = ply + 1;
        const int tail_len = std::max(0, child_len - tail_begin);
        std::ranges::copy_n(
            std::ranges::begin(table[ply + 1]) + tail_begin,
            tail_len,
            std::ranges::begin(table[ply]) + tail_begin);
        len[ply] = static_cast<uint8_t>(tail_begin + tail_len);
    }

    void setlen(int ply) {
        len[ply] = static_cast<uint8_t>(ply);
        table[ply][ply] = Move {};
    }

    void clear() {
        std::fill(std::begin(len), std::end(len), 0);
        std::fill(std::begin(table[0]), std::end(table[std::size(table) - 1]), Move{});
    }

    std::string str() const {
        return fmt::format("{}", fmt::join(std::views::take(table[0], len[0]), " "));
    }

    constexpr Move bestmove() const {
        return len[0] > 0 ? table[0][0] : Move {};
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
    Movelist root_moves {};
    QuadraticPV pvline {};
    std::array<std::array<std::array<int16_t, square_nb>, square_nb>, color_nb> history{};
    // [color][moving piece][to-square][captured piece]
    std::array<std::array<std::array<std::array<int16_t, piece_type_nb>, square_nb>, piece_type_nb>, color_nb> capture_history{};
    std::array<std::array<std::array<Move, square_nb>, square_nb>, color_nb> countermove{};
    using CmhPieceTable = std::array<std::array<int16_t, square_nb>, piece_type_nb>;
    std::array<std::array<std::array<CmhPieceTable, square_nb>, piece_type_nb>, color_nb> cmh{};
    // Follow-up move history: like cmh but keyed by our own move two
    // plies back ((ss-2)->move) instead of the opponent's reply.
    std::array<std::array<std::array<CmhPieceTable, square_nb>, piece_type_nb>, color_nb> fmh{};
    // Pawn-structure-keyed static-eval correction, scaled by corrhist_grain.
    static constexpr int corrhist_size = 16384;
    std::array<std::array<int16_t, corrhist_size>, color_nb> corrhist{};
    // Non-pawn sibling, keyed by hash ^ pawn_hash (piece placement minus
    // pawn structure); read at half the pawn table's weight.
    std::array<std::array<int16_t, corrhist_size>, color_nb> nonpawn_corrhist{};
    // Node-based TM: nodes spent under each root move this `go`,
    // indexed [src][dst].
    std::array<std::array<uint64_t, square_nb>, square_nb> root_effort{};
    int id { };
};


bool is_repetition(const Board & b, int draw = 1);

template <Color Us, NodeType node>
Value negamax(
    int depth,
    Worker &,
    Stack *,
    Value alpha = static_cast<Value>(-Value::infinite),
    Value beta = Value::infinite
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
