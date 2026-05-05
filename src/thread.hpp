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

    std::atomic_bool stop { false };

private:
    std::vector<std::unique_ptr<enyo::Worker>> pool_;
    std::vector<std::thread> threads_;
};

extern Pool pool;

}
