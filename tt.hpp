#pragma once
#include "config.hpp"
#include "types.hpp"

#if defined(__APPLE__)
    #include <sys/sysctl.h>
#endif
#include <optional>
#include <utility>

namespace {
static constexpr inline auto flag_size =  2;
static constexpr inline auto value_size = 16;
static constexpr inline auto depth_size =  6;
static constexpr inline auto move_size  = 32;

static constexpr inline auto flag_shift = 0;
static constexpr inline auto value_shift = flag_shift + flag_size;
static constexpr inline auto depth_shift = value_shift + value_size;
static constexpr inline auto move_shift  = depth_shift + depth_size;

static constexpr inline auto flag_mask = 0x3;
static constexpr inline auto value_mask = 0xFFFF;
static constexpr inline auto depth_mask = 0x3F;
static constexpr inline auto move_mask  = 0xFFFFFFFF;
} // anon ns


namespace enyo {

class Board;

static inline constexpr bool use_aligned_alloc = true;

namespace tt {

enum type {
    NoneBound,
    LowerBound,
    UpperBound,
    ExactBound,
};

struct HashEntry {
    Move move {};
    Value value {};
    int flag {};
    int depth {};
};

struct SMPentry {
    uint8_t age {};
    uint64_t key {};
    HashEntry entry {};
    bool occupied {};
};


constexpr inline Value value_from(Value v, int plies) {
    if (v == Value::none)
        return Value::none;
    else if (v >= Value::tb_win_in_max_ply)
        return static_cast<Value>(static_cast<int>(v) - plies);
    else if (v <= Value::tb_loss_in_max_ply)
        return static_cast<Value>(static_cast<int>(v) + plies);
    return v;
}

constexpr inline Value value_to(Value v, int plies) {
    if (v >= Value::tb_win_in_max_ply)
        return static_cast<Value>(static_cast<int>(v) + plies);
    else if (v <= Value::tb_loss_in_max_ply)
        return static_cast<Value>(static_cast<int>(v) - plies);
    return v;
}

class Transposition {
public:
     static Transposition& instance() {
        static Transposition This;
        return This;
    }

    ~Transposition() {
        if constexpr (use_aligned_alloc) {
            std::free(hash_table);
        } else {
            delete[] hash_table;
        }
        hash_table = nullptr;
    }

    Transposition(const Transposition&) = delete;
    Transposition& operator=(const Transposition&) = delete;

    std::vector<ScoredMove> get_pv_line(enyo::Board& b, int maxdepth = enyo::MAX_PLY);
    enyo::ScoredMove get_best_move(enyo::Board& b);

    void store(uint64_t poskey, enyo::Move move, enyo::Value value, type flag, int depth)
    {
        if (flag == NoneBound)
            return;

        if (move == enyo::Move::no_move && flag != ExactBound)
            return;
        auto index = poskey % buckets;
        auto & entry = hash_table[index];

        if (entry.occupied
            && entry.age == current_age
            && entry.entry.depth > depth
            && flag != ExactBound) {
            return;
        }

        if (!entry.occupied)
            new_write++;
        else
            over_write++;

        entry.key = poskey;
        entry.entry = HashEntry{move, value, flag, depth};
        entry.age = current_age;
        entry.occupied = true;
    }

    std::optional<HashEntry> probe(uint64_t poskey) {
        auto index = poskey % buckets;
        auto & entry = hash_table[index];

        if (!entry.occupied || entry.key != poskey) {
            return std::nullopt;
        }

        auto he = entry.entry;
        if (he.flag == NoneBound)
            return std::nullopt;
        if (he.move == enyo::Move::no_move && he.flag != ExactBound)
            return std::nullopt;
        return he;
    }

    int get_hashfull() const {
        return static_cast<int>(ceil((new_write * 1000.0) / static_cast<double>(buckets)));
    }

    void prepare() {
        current_age++;
        over_write = 0;
        cut = 0;
        hit = 0;
    }

    void clear() {
        std::fill(hash_table, hash_table + buckets, SMPentry{});
        new_write = 0;
        over_write = 0;
        cut = 0;
    }

    SMPentry * hash_table { nullptr };

    uint8_t current_age {};
    int new_write {};
    int over_write {};
    int cut {};
    int hit {};
    size_t buckets {};

private:
    Transposition() {
        resize(enyo::cfgmgr.hash_size);
    }

    [[maybe_unused]] size_t get_page_size() {
        #if defined(__APPLE__)
            int mib[2];
            int page_size;
            size_t len = sizeof(page_size);
            mib[0] = CTL_HW;
            mib[1] = HW_PAGESIZE;
            if (sysctl(mib, 2, &page_size, &len, NULL, 0) == -1) {
                return 4096;
            }
            return static_cast<size_t>(page_size);
        #else
            return static_cast<size_t>(getpagesize());
        #endif
    }

    void resize(int megabytes) {
        const auto bytes = static_cast<size_t>(megabytes * 1024 * 1024);
        if constexpr (use_aligned_alloc) {
            buckets = bytes / sizeof(SMPentry);
            size_t alloc_size = buckets * sizeof(SMPentry);
            size_t alignment = get_page_size();
            alloc_size = ((alloc_size + alignment - 1) / alignment) * alignment;

            hash_table = static_cast<SMPentry*>(std::aligned_alloc(alignment, alloc_size));
            if (!hash_table) {
                fmt::print("Failed to allocate hash table\n");
                std::exit(EXIT_FAILURE);
            }
            std::fill(hash_table, hash_table + buckets, SMPentry{});
        } else {
            buckets = bytes / sizeof(SMPentry);
            hash_table = new SMPentry[buckets];
        }
    }

};

inline Transposition& ttable = Transposition::instance();

} // tt


}

template<>
struct fmt::formatter<enyo::tt::type> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const enyo::tt::type& entry, FormatContext& ctx) const {
        switch (entry) {
            case enyo::tt::LowerBound:
                return fmt::format_to(ctx.out(), "LowerBound");
            case enyo::tt::UpperBound:
                return fmt::format_to(ctx.out(), "UpperBound");
            case enyo::tt::ExactBound:
                return fmt::format_to(ctx.out(), "ExactBound");
            default:
                return fmt::format_to(ctx.out(), "Unknown Bound");
        }
    }
};
