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
static constexpr inline auto flag_size = 3;
static constexpr inline auto value_size = 16;
static constexpr inline auto depth_size = 8;
static constexpr inline auto move_size = 30;
static constexpr inline auto age_size = 7;

static constexpr inline auto flag_shift = 0;
static constexpr inline auto value_shift = flag_shift + flag_size;
static constexpr inline auto depth_shift = value_shift + value_size;
static constexpr inline auto move_shift  = depth_shift + depth_size;
static constexpr inline auto age_shift = move_shift + move_size;

static constexpr inline uint64_t flag_mask = (1ULL << flag_size) - 1;
static constexpr inline uint64_t value_mask = (1ULL << value_size) - 1;
static constexpr inline uint64_t depth_mask = (1ULL << depth_size) - 1;
static constexpr inline uint64_t move_mask = (1ULL << move_size) - 1;
static constexpr inline uint64_t age_mask = (1ULL << age_size) - 1;

static_assert(age_shift + age_size == 64);
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

// HashEntry::flag packs the bound (bits 0-1) and an optional wasPv hint
// (bit 2). bound() / set_was_pv() / was_pv() preserve callers that compare
// flag to the bound enum directly. The wasPv bit never gates cutoffs or
// replacement — it flows store -> probe -> ss->ttPv -> LMR only.
inline constexpr int was_pv_bit = 4;
inline constexpr int bound_mask = 0x3;
inline constexpr int bound_of(int flag) { return flag & bound_mask; }
inline constexpr bool was_pv_of(int flag) { return (flag & was_pv_bit) != 0; }

struct HashEntry {
    Move move {};
    Value value {};
    int flag {};
    int depth {};
};

struct alignas(16) SMPentry {
    uint64_t key {};
    uint64_t data {};
};

static_assert(sizeof(SMPentry) == 16);

constexpr inline uint32_t pack_move(Move move)
{
    constexpr uint32_t low_fields_mask = (1U << 25) - 1;
    constexpr uint32_t flags_mask = 0x7;
    constexpr uint32_t promotion_mask = 0x3;

    assert((move.data & ((1U << 25) | (1U << 29))) == 0);
    return (move.data & low_fields_mask)
         | (((move.data >> 26) & flags_mask) << 25)
         | (((move.data >> 30) & promotion_mask) << 28);
}

constexpr inline Move unpack_move(uint32_t packed)
{
    constexpr uint32_t low_fields_mask = (1U << 25) - 1;
    constexpr uint32_t flags_mask = 0x7;
    constexpr uint32_t promotion_mask = 0x3;

    return Move {
        (packed & low_fields_mask)
      | (((packed >> 25) & flags_mask) << 26)
      | (((packed >> 28) & promotion_mask) << 30)
    };
}

constexpr inline uint64_t pack_entry(
    Move move,
    Value value,
    int flag,
    int depth,
    uint8_t age)
{
    assert(depth >= -128 && depth <= 127);
    return (static_cast<uint64_t>(flag) & flag_mask) << flag_shift
         | (static_cast<uint64_t>(static_cast<uint16_t>(value)) & value_mask) << value_shift
         | (static_cast<uint64_t>(static_cast<uint8_t>(depth)) & depth_mask) << depth_shift
         | (static_cast<uint64_t>(pack_move(move)) & move_mask) << move_shift
         | (static_cast<uint64_t>(age) & age_mask) << age_shift;
}

constexpr inline HashEntry unpack_entry(uint64_t data)
{
    return HashEntry {
        unpack_move(static_cast<uint32_t>((data >> move_shift) & move_mask)),
        static_cast<Value>(static_cast<int16_t>((data >> value_shift) & value_mask)),
        static_cast<int>((data >> flag_shift) & flag_mask),
        static_cast<int>(static_cast<int8_t>((data >> depth_shift) & depth_mask)),
    };
}

constexpr inline uint8_t unpack_age(uint64_t data)
{
    return static_cast<uint8_t>((data >> age_shift) & age_mask);
}


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

    void store(uint64_t poskey, enyo::Move move, enyo::Value value, type flag,
               int depth, bool was_pv = false)
    {
        if (flag == NoneBound)
            return;

        auto index = poskey % buckets;
        auto & entry = hash_table[index];
        const auto previous_data = entry.data;

        // age is stored in 7 bits; mask the counter so the comparison
        // stays consistent when current_age wraps past 127.
        if (previous_data != 0
            && unpack_age(previous_data) == (current_age & age_mask)
            && unpack_entry(previous_data).depth > depth
            && flag != ExactBound) {
            return;
        }

        // A move-less bound (fail-low, stand-pat, TB probe) replacing an
        // entry for the same position would erase its move-ordering hint;
        // keep the previous move alongside the new bound.
        if (move == enyo::Move::no_move
            && previous_data != 0
            && (entry.key ^ previous_data) == poskey) {
            move = unpack_entry(previous_data).move;
        }

        if (previous_data == 0)
            new_write++;
        else
            over_write++;

        // Pack wasPv into flag bit 2; the bound stays in bits 0-1.
        const int packed_flag = static_cast<int>(flag) | (was_pv ? was_pv_bit : 0);
        const auto data = pack_entry(move, value, packed_flag, depth, current_age);
        entry.data = data;
        entry.key = poskey ^ data;
    }

    std::optional<HashEntry> probe(uint64_t poskey) {
        auto index = poskey % buckets;
        auto & entry = hash_table[index];
        const auto data = entry.data;

        if (data == 0 || (entry.key ^ data) != poskey) {
            return std::nullopt;
        }

        auto he = unpack_entry(data);
        if (bound_of(he.flag) == NoneBound)
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
        const auto requested_buckets = bucket_count_for(megabytes);
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
