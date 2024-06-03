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

    if (si.time_expired() && bestmove) {
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
