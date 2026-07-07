#include <iostream>

#include "config.hpp"
#include "thread.hpp"

using namespace enyo;

[[nodiscard]] bool Worker::time_expired()
{
    if (thread::pool.stop.load(std::memory_order_relaxed)) {
        return true;
    }

    if (id != 0)
        return false;

    // The watchdog still observes hard deadlines every millisecond.
    if ((si.nodes & 1023U) != 0)
        return false;

    const bool out_of_nodes =
        si.nodes_limit != 0 && thread::pool.get_nodes() >= si.nodes_limit;
    const bool has_time_limit =
        si.stoptime != std::chrono::high_resolution_clock::time_point::max();
    const bool out_of_time = !out_of_nodes && has_time_limit && si.time_expired();
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
    (void)wait_and_get_nodes();
}

uint64_t Pool::wait_and_get_nodes()
{
    std::ranges::for_each(threads_, [](auto& th) {
        if (th.joinable()) {
            th.join();
        }
    });
    const uint64_t nodes = pool_.empty() ? 0 : get_nodes();
    save_stats();
    pool_.clear();
    threads_.clear();
    return nodes;
}

void Pool::save_stats()
{
    for (const auto & worker : pool_) {
        const auto id = static_cast<size_t>(worker->id);
        if (saved_stats_.size() <= id)
            saved_stats_.resize(id + 1);
        if (!saved_stats_[id])
            saved_stats_[id] = std::make_unique<WorkerStats>();
        saved_stats_[id]->history = worker->history;
        saved_stats_[id]->countermove = worker->countermove;
        saved_stats_[id]->cmh = worker->cmh;
    }
}

void Pool::reset_saved_stats()
{
    saved_stats_.clear();
}

void Pool::kill()
{
    stop = true;
    wait();
}

void Pool::init_threads(const SearchInfo & si, int num_threads)
{
    assert(threads_.empty());

    if (cfgmgr.lmr_dirty)
        init_search();

    stop = false;
    pool_.clear();
    threads_.clear();

    pool_.reserve(num_threads);
    threads_.reserve(num_threads);

    // init pool:
    for (int i = 0; i < num_threads; ++i) {
        auto worker = std::make_unique<enyo::Worker>(si, i);
        if (static_cast<size_t>(i) < saved_stats_.size() && saved_stats_[static_cast<size_t>(i)]) {
            const auto & saved = *saved_stats_[static_cast<size_t>(i)];
            worker->history = saved.history;
            worker->countermove = saved.countermove;
            worker->cmh = saved.cmh;
        }
        pool_.emplace_back(std::move(worker));
    }

    // start workers:
    std::ranges::for_each(pool_, [this](const auto & wptr) {
        threads_.emplace_back([this, id = wptr->id] {
            pool_[id]->start();
        });
    });

}

} // thread ns
