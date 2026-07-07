#pragma once

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

#include "movelist.hpp"
#include "search.hpp"

namespace thread {

class Pool {
public:
    uint64_t get_nodes() const;
    uint64_t get_nps() const;
    uint64_t get_tbhits() const;

    void init_threads(const enyo::SearchInfo & si, int num_threads);

    void kill();
    void wait();
    uint64_t wait_and_get_nodes();

    // Drop move-ordering statistics carried between searches (new game).
    void reset_saved_stats();

    std::atomic_bool stop { false };

private:
    // Move-ordering statistics survive Worker teardown so consecutive
    // `go` commands within one game keep learning instead of restarting
    // from zero on every move. Indexed by thread id.
    struct WorkerStats {
        decltype(enyo::Worker::history) history;
        decltype(enyo::Worker::countermove) countermove;
        decltype(enyo::Worker::cmh) cmh;
    };

    void save_stats();

    std::vector<std::unique_ptr<enyo::Worker>> pool_;
    std::vector<std::thread> threads_;
    std::vector<std::unique_ptr<WorkerStats>> saved_stats_;
};

extern Pool pool;

}
