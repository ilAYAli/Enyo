#include <iostream>

#include "thread.hpp"

using namespace enyo;

[[nodiscard]] bool Worker::time_expired()
{
    if (thread::pool.stop.load(std::memory_order_relaxed)) {
        return true;
    }

    if (id != 0)
        return false;

    const bool out_of_time = si.time_expired();
    const bool out_of_nodes =
        si.nodes_limit != 0 && thread::pool.get_nodes() >= si.nodes_limit;
    if (out_of_time || out_of_nodes) {
        thread::pool.stop = true;
        return true;
    }

    return false;
}

void Worker::start()
{
    search_position(*this);
}

namespace thread {
Pool pool;

uint64_t Pool::get_nps() const
{
    using namespace std::chrono;
    const auto elapsed = pool_[0]->si.elapsed_time;
    const auto s = static_cast<double>(duration_cast<milliseconds>(elapsed).count()) / 1000.0;
    return s > 0
        ? static_cast<uint64_t>(static_cast<double>(get_nodes()) / s)
        : 0;
}

uint64_t Pool::get_nodes() const
{
    return std::accumulate(
        std::ranges::begin(pool_),
        std::ranges::end(pool_),
        uint64_t{0},
        [](uint64_t total, const auto & st) {
            return total + st->si.nodes;
        }
    );
}

uint64_t Pool::get_tbhits() const
{
    return std::accumulate(
        std::ranges::begin(pool_),
        std::ranges::end(pool_),
        uint64_t{0},
        [](uint64_t total, const auto & st) {
            return total + st->si.tbhits;
        }
    );
}

void Pool::wait()
{
    std::ranges::for_each(threads_, [](auto& th) {
        if (th.joinable()) {
            th.join();
        }
    });
    pool_.clear();
    threads_.clear();
}

void Pool::kill()
{
    stop = true;
    wait();
}

void Pool::init_threads(const SearchInfo & si, int num_threads)
{
    assert(threads_.empty());

    stop = false;
    pool_.clear();
    threads_.clear();

    pool_.reserve(num_threads);
    threads_.reserve(num_threads);

    // init pool:
    for (int i = 0; i < num_threads; ++i) {
        pool_.emplace_back(std::make_unique<enyo::Worker>(si, i));
    }

    // start workers:
    std::ranges::for_each(pool_, [this](const auto & wptr) {
        threads_.emplace_back([this, id = wptr->id] {
            pool_[id]->start();
        });
    });

}

} // thread ns
