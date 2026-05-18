#pragma once
#include "config.hpp"
#include "types.hpp"

#include <cstdlib>
#include <new>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <malloc.h>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
#else
    #include <unistd.h>
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
        free_table();
    }

    void free_table() {
        if (!hash_table) return;
        release_table(hash_table);
        hash_table = nullptr;
        size_megabytes = 0;
        buckets = 0;
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
        if (buckets == 0)
            return 0;
        const int hashfull = static_cast<int>(ceil((new_write * 1000.0) / static_cast<double>(buckets)));
        return hashfull > 1000 ? 1000 : hashfull;
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

    bool set_size(int megabytes) {
        const auto requested_buckets = bucket_count_for(megabytes);
        if (requested_buckets == 0)
            return false;
        if (requested_buckets == buckets)
            return true;
        return replace_table(megabytes);
    }

    int size_mb() const {
        return size_megabytes;
    }

    SMPentry * hash_table { nullptr };

    uint8_t current_age {};
    int new_write {};
    int over_write {};
    int cut {};
    int hit {};
    int size_megabytes {};
    size_t buckets {};

private:
    Transposition() {
        const int requested = enyo::cfgmgr.hash_size;
        if (replace_table(requested))
            return;

        for (const int fallback : fallback_sizes) {
            if (requested > 0 && fallback >= requested)
                continue;
            if (!replace_table(fallback))
                continue;

            enyo::cfgmgr.hash_size = fallback;
            fmt::print("info string WARNING: failed to allocate {} MB hash; using {} MB\n",
                requested,
                fallback);
            return;
        }

        fmt::print("ERROR: failed to allocate hash table\n");
        std::exit(EXIT_FAILURE);
    }

    [[maybe_unused]] static size_t get_page_size() {
        #if defined(_WIN32)
            SYSTEM_INFO system_info;
            GetSystemInfo(&system_info);
            return static_cast<size_t>(system_info.dwPageSize);
        #elif defined(__APPLE__)
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

    static constexpr size_t bytes_in_megabyte = 1024ULL * 1024ULL;
    static constexpr int fallback_sizes[] = { 1024, 512, 256, 128, 64, 16 };

    struct TableAllocation {
        SMPentry * table {};
        size_t buckets {};
    };

    static size_t byte_count_for(int megabytes) {
        if (megabytes <= 0)
            return 0;
        return static_cast<size_t>(megabytes) * bytes_in_megabyte;
    }

    static size_t bucket_count_for(int megabytes) {
        return byte_count_for(megabytes) / sizeof(SMPentry);
    }

    static void release_table(SMPentry * table) {
        if constexpr (use_aligned_alloc) {
            #if defined(_WIN32)
                _aligned_free(table);
            #else
                std::free(table);
            #endif
        } else {
            delete[] table;
        }
    }

    static TableAllocation allocate_table(int megabytes) {
        const auto bytes = byte_count_for(megabytes);
        const auto requested_buckets = bytes / sizeof(SMPentry);
        if (requested_buckets == 0)
            return {};

        if constexpr (use_aligned_alloc) {
            size_t alloc_size = requested_buckets * sizeof(SMPentry);
            size_t alignment = get_page_size();
            alloc_size = ((alloc_size + alignment - 1) / alignment) * alignment;

            #if defined(_WIN32)
                auto * table = static_cast<SMPentry*>(_aligned_malloc(alloc_size, alignment));
            #else
                auto * table = static_cast<SMPentry*>(std::aligned_alloc(alignment, alloc_size));
            #endif

            if (!table)
                return {};
            std::fill(table, table + requested_buckets, SMPentry{});
            return { table, requested_buckets };
        } else {
            auto * table = new (std::nothrow) SMPentry[requested_buckets];
            if (!table)
                return {};
            std::fill(table, table + requested_buckets, SMPentry{});
            return { table, requested_buckets };
        }
    }

    bool replace_table(int megabytes) {
        const auto allocation = allocate_table(megabytes);
        if (!allocation.table)
            return false;

        free_table();
        hash_table = allocation.table;
        buckets = allocation.buckets;
        size_megabytes = megabytes;
        reset_stats();
        return true;
    }

    void reset_stats() {
        new_write = 0;
        over_write = 0;
        cut = 0;
        hit = 0;
        current_age = 0;
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
