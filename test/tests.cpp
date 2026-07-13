// cppcheck-suppress knownConditionTrueFalse

#include <gtest/gtest.h>

#include "types.hpp"
#include "board.hpp"
#include "movegen.hpp"
#include "movegen_helper.hpp"
#include "uci.hpp"
#include "perft.hpp"
#include "zobrist.hpp"
#include "see.hpp"
#include "nnue.hpp"
#include "nnue/stockfish/stockfish_nnue_model.hpp"
#include "nnue/enyo/enyo_nn_loader.hpp"
#include "probe.hpp"
#include "search.hpp"
#include "movepicker.hpp"
#include "thread.hpp"
#include "tt.hpp"
#include "pgn.hpp"
#include "move_policy.hpp"

#include <chrono>
#include <array>
#include <bit>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <ranges>
#include <vector>
#include <sstream>
#include <nlohmann/json.hpp>

using namespace enyo;
using namespace nlohmann;

std::string get_default_config_file_path() {
    const char * home_directory = getenv("HOME");
    if (home_directory) {
        const std::string default_config_path = std::string(home_directory) + "/.config/enyo/settings json";
        if (fs::exists(default_config_path))
            return default_config_path;
    }
    return "";
}

bool load_config() {
    auto config_file_path = get_default_config_file_path();
    if (cfgmgr.load_config(config_file_path)) {
        fmt::print("Using config file: '{}'\n", config_file_path);
        return true;
    }
    return false;
}

// Deterministic legacy weights, independent of the embedded net.
void seed_legacy_eval_weights() {
    uint32_t rng = 0x9E3779B9u;
    const auto next = [&rng]() -> int16_t {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<int16_t>(static_cast<int>((rng >> 16) & 0xFF) - 128);
    };
    for (auto & w : inputWeights) w = next();
    for (auto & w : inputBias) w = next();
    for (auto & w : hiddenWeights) w = next();
    for (auto & w : hiddenBias) w = next();
}

struct LegacyEvalScope {
    bool old_enabled = Network::enabled;
    std::vector<int16_t> old_input_weights{inputWeights.begin(), inputWeights.end()};
    std::vector<int16_t> old_input_bias{inputBias.begin(), inputBias.end()};
    std::vector<int16_t> old_hidden_weights{hiddenWeights.begin(), hiddenWeights.end()};
    std::vector<int> old_hidden_bias{hiddenBias.begin(), hiddenBias.end()};
    LegacyEvalScope() {
        Network::enabled = false;
        seed_legacy_eval_weights();
    }
    ~LegacyEvalScope() {
        std::copy(old_input_weights.begin(), old_input_weights.end(), inputWeights.begin());
        std::copy(old_input_bias.begin(), old_input_bias.end(), inputBias.begin());
        std::copy(old_hidden_weights.begin(), old_hidden_weights.end(), hiddenWeights.begin());
        std::copy(old_hidden_bias.begin(), old_hidden_bias.end(), hiddenBias.begin());
        Network::enabled = old_enabled;
    }
};

TEST(tt, packed_entry_round_trips_search_fields)
{
    Move promotion {a7, white, pawn, a8};
    promotion.set_flags(Move::Flags::promote);
    promotion.set_promo_piece(queen);

    struct EntryCase {
        Move move;
        Value value;
        tt::type flag;
        int depth;
        uint8_t age;
    };

    const std::array cases {
        EntryCase {Move {}, Value::draw, tt::ExactBound, -5, 0},
        EntryCase {promotion, static_cast<Value>(-1234), tt::LowerBound, 0, 90},
        // age is 7 bits since the wasPv bit widened the flag field
        EntryCase {promotion, Value::none, tt::UpperBound, MAX_PLY, 127},
    };

    EXPECT_EQ(sizeof(tt::SMPentry), 16U);
    for (const auto & expected : cases) {
        const auto data = tt::pack_entry(
            expected.move,
            expected.value,
            expected.flag,
            expected.depth,
            expected.age);
        const auto actual = tt::unpack_entry(data);

        EXPECT_EQ(actual.move, expected.move);
        EXPECT_EQ(actual.value, expected.value);
        EXPECT_EQ(actual.flag, expected.flag);
        EXPECT_EQ(actual.depth, expected.depth);
        EXPECT_EQ(tt::unpack_age(data), expected.age);
    }
}

TEST(tt, hash_size_matches_allocated_entry_capacity)
{
    constexpr size_t bytes_per_megabyte = 1024ULL * 1024ULL;
    const auto & table = tt::ttable;

    EXPECT_EQ(
        table.buckets * sizeof(tt::SMPentry),
        static_cast<size_t>(table.size_mb()) * bytes_per_megabyte);
}

TEST(tt, packed_table_preserves_replacement_contract)
{
    auto & table = tt::ttable;
    table.clear();
    table.prepare();

    Move original {e2, white, pawn, e4};
    const uint64_t key = 0x123456789ABCDEF0ULL;
    table.store(key, original, static_cast<Value>(321), tt::LowerBound, MAX_PLY);

    auto hit = table.probe(key);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->move, original);
    EXPECT_EQ(hit->value, static_cast<Value>(321));
    EXPECT_EQ(hit->flag, tt::LowerBound);
    EXPECT_EQ(hit->depth, MAX_PLY);

    Move shallower {d2, white, pawn, d4};
    table.store(key, shallower, static_cast<Value>(123), tt::UpperBound, -5);
    hit = table.probe(key);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->move, original);
    EXPECT_EQ(hit->depth, MAX_PLY);

    table.store(key, shallower, static_cast<Value>(-456), tt::ExactBound, -5);
    hit = table.probe(key);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->move, shallower);
    EXPECT_EQ(hit->value, static_cast<Value>(-456));
    EXPECT_EQ(hit->flag, tt::ExactBound);
    EXPECT_EQ(hit->depth, -5);
}

TEST(move_picker, lazy_ordering_returns_moves_by_score) {
    PrioritizedMoves moves;
    const Move low {a2, white, pawn, a3};
    const Move high {b2, white, pawn, b3};
    const Move middle {c2, white, pawn, c3};

    moves.add(low, -10);
    moves.add(high, 200);
    moves.add(middle, 50);

    EXPECT_EQ(moves.count(), 3U);
    EXPECT_EQ(moves.next(), high);
    EXPECT_EQ(moves[0], high);
    EXPECT_EQ(moves.next(), middle);
    EXPECT_EQ(moves[0], high);
    EXPECT_EQ(moves[1], middle);
    EXPECT_EQ(moves.next(), low);
    EXPECT_EQ(moves[2], low);
    EXPECT_EQ(moves.next(), Move {});
}

TEST(move_picker, lazy_ordering_matches_eager_sort_with_ties) {
    std::array<ScoredMove, 6> expected {{
        {40, Move {a2, white, pawn, a3}},
        {200, Move {b2, white, pawn, b3}},
        {40, Move {c2, white, pawn, c3}},
        {-10, Move {d2, white, pawn, d3}},
        {40, Move {e2, white, pawn, e3}},
        {20, Move {f2, white, pawn, f3}},
    }};

    PrioritizedMoves moves;
    for (const auto & entry : expected)
        moves.add(entry.move, entry.score);

    std::ranges::sort(expected, [](const auto & lhs, const auto & rhs) {
        return lhs.score > rhs.score;
    });

    for (const auto & entry : expected)
        EXPECT_EQ(moves.next(), entry.move);
    EXPECT_EQ(moves.next(), Move {});
}

json make_zero_move_policy_model(double output_bias)
{
    constexpr int input_dim = 1708;
    json layer0;
    layer0["activation"] = "relu";
    layer0["weight"] = std::vector<std::vector<double>>(2, std::vector<double>(input_dim, 0.0));
    layer0["bias"] = std::vector<double>{0.0, 0.0};

    json layer1;
    layer1["activation"] = "relu";
    layer1["weight"] = std::vector<std::vector<double>>(2, std::vector<double>(2, 0.0));
    layer1["bias"] = std::vector<double>{0.0, 0.0};

    json layer2;
    layer2["activation"] = "linear";
    layer2["weight"] = std::vector<std::vector<double>>(1, std::vector<double>(2, 0.0));
    layer2["bias"] = std::vector<double>{output_bias};

    return json{
        {"schema", "enyo.move_policy.v1"},
        {"feature_version", "move_policy_v1"},
        {"feature_set", "board"},
        {"input_dim", input_dim},
        {"hidden", 2},
        {"threshold", 18.0},
        {"mean", std::vector<double>(input_dim, 0.0)},
        {"std", std::vector<double>(input_dim, 1.0)},
        {"layers", json::array({layer0, layer1, layer2})},
        {"config", json::object()},
    };
}

json make_weighted_move_policy_model(int feature_index, double weight)
{
    constexpr int input_dim = 1708;
    std::vector<double> weights(input_dim, 0.0);
    weights[static_cast<size_t>(feature_index)] = weight;

    json layer;
    layer["activation"] = "linear";
    layer["weight"] = std::vector<std::vector<double>>{weights};
    layer["bias"] = std::vector<double>{0.0};

    return json{
        {"schema", "enyo.move_policy.v1"},
        {"feature_version", "move_policy_v1"},
        {"feature_set", "board"},
        {"input_dim", input_dim},
        {"hidden", 0},
        {"threshold", 18.0},
        {"mean", std::vector<double>(input_dim, 0.0)},
        {"std", std::vector<double>(input_dim, 1.0)},
        {"layers", json::array({layer})},
        {"config", json::object()},
    };
}

constexpr Color get_side_to_move(std::string_view fen) {
    size_t pos = fen.find(' ');
    if (pos != std::string_view::npos) {
        char side = fen[pos + 1];
        return (side == 'w') ? Color::white : Color::black;
    }
    return white;
}

bool init_test_syzygy(int min_pieces)
{
    std::vector<std::string> candidates;
    if (const char * env_path = std::getenv("ENYO_TEST_SYZYGY_PATH"); env_path && *env_path)
        candidates.emplace_back(env_path);
    if (!cfgmgr.syzygy_path.empty())
        candidates.push_back(cfgmgr.syzygy_path);
    candidates.emplace_back("~/code/cpp/chess/assets/tablebases");
    candidates.emplace_back("../assets/tablebases");

    for (const auto & path : candidates) {
        if (syzygy::init(path) && static_cast<int>(syzygy::largest()) >= min_pieces)
            return true;
    }
    return false;
}

fs::path expand_test_path(std::string path)
{
    if (path.starts_with("~/")) {
        if (const char * home = std::getenv("HOME"); home && *home)
            return fs::path(home) / path.substr(2);
    }
    return path;
}

fs::path find_test_tablebase_file(std::string_view filename)
{
    std::vector<std::string> candidates;
    if (const char * env_path = std::getenv("ENYO_TEST_SYZYGY_PATH"); env_path && *env_path)
        candidates.emplace_back(env_path);
    if (!cfgmgr.syzygy_path.empty())
        candidates.push_back(cfgmgr.syzygy_path);
    candidates.emplace_back("~/code/cpp/chess/assets/tablebases");
    candidates.emplace_back("../assets/tablebases");

    for (const auto & path : candidates) {
        const auto root = expand_test_path(path);
        std::error_code ec;
        if (!fs::exists(root, ec) || ec)
            continue;
        for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
             !ec && it != end;
             it.increment(ec)) {
            if (it->path().filename().string() == filename)
                return it->path();
        }
    }
    return {};
}

bool link_wdl_table_files(const fs::path & source_dir, const fs::path & target_dir)
{
    std::error_code ec;
    for (fs::directory_iterator it(source_dir, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end;
         it.increment(ec)) {
        if (it->path().extension() != ".rtbw")
            continue;

        const auto target = target_dir / it->path().filename();
        if (fs::exists(target, ec)) {
            ec.clear();
            continue;
        }
        fs::create_symlink(it->path(), target, ec);
        if (ec)
            return false;
    }
    return !ec;
}

std::optional<uint64_t> uci_nodes_at_depth(std::string_view output, int depth)
{
    const auto needle = fmt::format("info depth {} score ", depth);
    const auto line_begin = output.find(needle);
    if (line_begin == std::string_view::npos)
        return std::nullopt;

    const auto line_end = output.find('\n', line_begin);
    const auto line_limit = line_end == std::string_view::npos ? output.size() : line_end;
    const auto nodes_pos = output.find(" nodes ", line_begin);
    if (nodes_pos == std::string_view::npos
        || nodes_pos >= line_limit)
        return std::nullopt;

    const auto nodes_begin = nodes_pos + std::string_view(" nodes ").size();
    auto nodes_end = output.find(' ', nodes_begin);
    if (nodes_end == std::string_view::npos || nodes_end > line_limit)
        nodes_end = line_limit;
    if (nodes_begin >= nodes_end)
        return std::nullopt;

    return static_cast<uint64_t>(std::stoull(
        std::string(output.substr(nodes_begin, nodes_end - nodes_begin))));
}

#if 1
TEST(check, see_pawn_takes_pawn_defended) {
    // Pxe5: e5 defended by {Qc7, Bf6, Re6}; attacked by {Pf4, Rf5, Nd3}.
    // LVA capture sequence yields cumulative (white POV): +100,-100,+330,-320,+500,-500 = +10.
    // Minimax via backward induction settles on +10.
    std::string fen = "8/2q2k2/4rb2/4pR2/5P2/3N4/Q7/4K3 w - - 0 1";
    Board b(fen);

    auto move = resolve_move<white>(b, pawn, f4, e5);
    ASSERT_EQ(10, see<white>(b, move, 0));
}
TEST(check, see_queen_takes_defended_pawn) {
    // Qxb6: b6 is defended by a7 pawn only (d8 bishop blocked by c7 knight;
    // c7 knight does not attack b6). Exchange: Qxp, axQ, Bxp = p - q + p.
    std::string fen = "3b3k/p1n5/1p6/8/8/1Q2B3/8/7K w - - 0 1";
    Board b(fen);

    auto move = resolve_move<white>(b, queen, b3, b6);
    constexpr auto score = piece_value(pawn) - piece_value(queen) + piece_value(pawn);
    ASSERT_EQ(score, see<white>(b, move, 0));
}
TEST(check, see_knight_takes_defended_pawn) {
    // Nxd5: d5 defended by f6 knight; attackers N, B. Score = p - n + n.
    std::string fen = "k7/8/5n2/3p4/8/2N2B2/8/K7 w - - 0 1";
    Board b(fen);

    auto move = resolve_move<white>(b, knight, c3, d5);
    constexpr auto score = piece_value(pawn) - piece_value(knight) + piece_value(knight);
    ASSERT_EQ(score, see<white>(b, move, 0));
}
TEST(check, see_rook_battery_xray) {
    // Rxd4: rank-4 battery "3pRrRr" → d4=p, e4=R, f4=r, g4=R, h4=r.
    // Exchange (with X-ray): Rxp, Rxr, Rxr, Rxr = p - r + r - r.
    std::string fen = "2K5/8/8/8/3pRrRr/8/8/2k5 w - - 0 1";
    Board b(fen);

    auto move = resolve_move<white>(b, rook, e4, d4);
    constexpr auto score = piece_value(pawn) - piece_value(rook) + piece_value(rook) - piece_value(rook);
    ASSERT_EQ(score, see<white>(b, move, 0));
}

TEST(check, see_rejects_illegal_king_recapture) {
    Board b("8/8/8/8/8/4k3/3p4/2KR4 w - - 0 1");
    const auto move = resolve_move<white>(b, rook, d1, d2);

    EXPECT_EQ(piece_value(pawn), see<white>(b, move));
    EXPECT_TRUE(see_ge<white>(b, move));
}

template <Color Us>
void expect_see_ge_matches_exact(std::string_view fen)
{
    Board b(fen);
    Movelist moves;
    generate_legal_moves<Us>(b, moves);
    constexpr std::array thresholds = {
        -1000, -900, -500, -330, -320, -100, -1, 0,
        1, 100, 320, 330, 500, 900, 1000
    };

    for (const auto move : moves) {
        if (!move.dst_piece())
            continue;
        for (const int threshold : thresholds) {
            EXPECT_EQ(see<Us>(b, move) >= threshold, see_ge<Us>(b, move, threshold))
                << "move=" << fmt::format("{}", move) << " threshold=" << threshold;
        }
    }
}

template <Color Us>
void expect_see_ge_matches_exact_tree(Board & b, int depth, std::size_t & comparisons)
{
    Movelist moves;
    generate_legal_moves<Us>(b, moves);
    constexpr std::array thresholds = {
        -1000, -900, -500, -330, -320, -100, -1, 0,
        1, 100, 320, 330, 500, 900, 1000
    };

    for (const auto move : moves) {
        if (!move.dst_piece())
            continue;
        for (const int threshold : thresholds) {
            const bool exact = see<Us>(b, move) >= threshold;
            const bool fast = see_ge<Us>(b, move, threshold);
            if (exact != fast) {
                ADD_FAILURE()
                    << "fen=" << b.fen()
                    << " move=" << fmt::format("{}", move)
                    << " threshold=" << threshold
                    << " see=" << see<Us>(b, move)
                    << " see_ge=" << fast;
            }
            ++comparisons;
        }
    }

    if (depth == 0)
        return;

    for (const auto move : moves) {
        apply_move<Us>(b, move);
        expect_see_ge_matches_exact_tree<~Us>(b, depth - 1, comparisons);
        revert_move<Us>(b);
    }
}

TEST(check, see_ge_matches_exact) {
    expect_see_ge_matches_exact<white>(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    expect_see_ge_matches_exact<white>(
        "8/2q2k2/4rb2/4pR2/5P2/3N4/Q7/4K3 w - - 0 1");
    expect_see_ge_matches_exact<white>(
        "2K5/8/8/8/3pRrRr/8/8/2k5 w - - 0 1");
    expect_see_ge_matches_exact<black>(
        "2k5/8/8/3PrRrR/8/8/8/2K5 b - - 0 1");
    expect_see_ge_matches_exact<white>(
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    expect_see_ge_matches_exact<white>(
        "4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1");
    expect_see_ge_matches_exact<white>(
        "8/8/8/8/8/4k3/3p4/2KR4 w - - 0 1");
}

TEST(check, see_ge_matches_exact_across_move_trees) {
    std::size_t comparisons = 0;
    Board start("startpos");
    expect_see_ge_matches_exact_tree<white>(start, 4, comparisons);
    Board kiwipete(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    expect_see_ge_matches_exact_tree<white>(kiwipete, 3, comparisons);
    EXPECT_GT(comparisons, 1'000'000U);
}

template <Color Us>
void expect_tactical_moves_match_filtered_legal_moves(std::string_view fen)
{
    Board all_board(fen);
    Board tactical_board(fen);
    Movelist all_moves;
    Movelist tactical_moves;
    Movelist expected;
    generate_legal_moves<Us>(all_board, all_moves);
    generate_legal_tactical_moves<Us>(tactical_board, tactical_moves);

    for (const auto move : all_moves) {
        if (move.dst_piece() || (move.flags() & Move::Flags::promote))
            expected.emplace(move);
    }

    ASSERT_EQ(expected.size(), tactical_moves.size()) << fen;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], tactical_moves[i])
            << "fen=" << fen << " index=" << i;
    }
}

TEST(movegen, tactical_moves_match_filtered_legal_moves) {
    expect_tactical_moves_match_filtered_legal_moves<white>(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    expect_tactical_moves_match_filtered_legal_moves<black>(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1");
    expect_tactical_moves_match_filtered_legal_moves<white>(
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    expect_tactical_moves_match_filtered_legal_moves<white>(
        "4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1");
    expect_tactical_moves_match_filtered_legal_moves<white>(
        "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1");
    expect_tactical_moves_match_filtered_legal_moves<white>(
        "6k1/8/8/8/8/8/4r3/4K3 w - - 0 1");
}
TEST(hash, recompute_matches_initial_position) {
    Board b{"startpos"};
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));
}

TEST(config, loads_only_uci_options)
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = fs::temp_directory_path() / fmt::format("enyo_config_{}.json", stamp);
    {
        std::ofstream out(path);
        out << R"({"uci_options":{"Threads":4,"Hash":128,"use_syzygy":false}})";
    }

    ConfigManager config;
    ASSERT_TRUE(config.load_config(path.string()));
    const auto & options = config.configured_uci_options();
    const auto value_of = [&](std::string_view name) -> std::optional<std::string> {
        const auto it = std::ranges::find_if(options, [&](const auto & option) {
            return option.first == name;
        });
        return it == options.end()
            ? std::nullopt
            : std::optional<std::string>{it->second};
    };

    EXPECT_EQ(value_of("Threads"), "4");
    EXPECT_EQ(value_of("Hash"), "128");
    EXPECT_EQ(value_of("use_syzygy"), "false");
    EXPECT_EQ(config.num_threads, 1);
    EXPECT_EQ(config.hash_size, 1024);
    fs::remove(path);
}

TEST(config, rejects_legacy_and_unknown_configuration)
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = fs::temp_directory_path() / fmt::format("enyo_legacy_config_{}.json", stamp);
    {
        std::ofstream out(path);
        out << R"({"constants":{"threads":4},"toggles":{"use_lmr":true}})";
    }

    ConfigManager config;
    testing::internal::CaptureStderr();
    EXPECT_FALSE(config.load_config(path.string()));
    const auto error = testing::internal::GetCapturedStderr();
    EXPECT_NE(error.find("must contain only an object named 'uci_options'"), std::string::npos);
    EXPECT_FALSE(config.setopt("use_tt", "false"));
    EXPECT_FALSE(config.setopt("use_lmr", "false"));
    EXPECT_FALSE(config.setopt("use_nnue", "false"));

    {
        std::ofstream out(path, std::ios::trunc);
        out << R"({"uci_options":{"not_an_option":1}})";
    }
    testing::internal::CaptureStderr();
    EXPECT_FALSE(config.load_config(path.string()));
    const auto unknown_error = testing::internal::GetCapturedStderr();
    EXPECT_NE(unknown_error.find("unknown UCI option 'not_an_option'"), std::string::npos);
    fs::remove(path);
}

TEST(move_policy, startpos_e2e4_features_match_python_layout) {
    Board b("startpos");
    const auto move = uci_to_move(b, "e2e4");
    ASSERT_TRUE(move.has_value());

    const auto features = move_policy::features(b, *move, move_policy::FeatureSet::board);
    ASSERT_EQ(features.size(), 1708U);

    EXPECT_DOUBLE_EQ(features[0], 1.0);
    EXPECT_DOUBLE_EQ(features[1], 20.0 / 80.0);
    EXPECT_DOUBLE_EQ(features[2], 1.0 / 120.0);
    EXPECT_DOUBLE_EQ(features[3], 0.0);
    EXPECT_DOUBLE_EQ(features[4], 0.0);
    EXPECT_DOUBLE_EQ(features[5], 0.0);
    EXPECT_DOUBLE_EQ(features[6], 0.0);
    EXPECT_DOUBLE_EQ(features[7], 0.0);
    EXPECT_DOUBLE_EQ(features[8], 0.0);
    EXPECT_DOUBLE_EQ(features[9], 0.0);
    EXPECT_DOUBLE_EQ(features[10], 0.0);
    EXPECT_DOUBLE_EQ(features[11], 2.0 / 7.0);
    EXPECT_DOUBLE_EQ(features[12], 20.0 / 80.0);
    EXPECT_DOUBLE_EQ(features[13], 0.0);

    constexpr int from_offset = 14;
    constexpr int to_offset = from_offset + 64;
    constexpr int moving_offset = to_offset + 64;
    constexpr int material_offset = moving_offset + 6 + 6 + 6;
    constexpr int parent_board_offset = material_offset + 12;
    constexpr int child_board_offset = parent_board_offset + 12 * 64;

    EXPECT_DOUBLE_EQ(features[from_offset + 12], 1.0); // e2 in python-chess square numbering
    EXPECT_DOUBLE_EQ(features[to_offset + 28], 1.0);   // e4 in python-chess square numbering
    EXPECT_DOUBLE_EQ(features[moving_offset + 0], 1.0); // pawn
    EXPECT_DOUBLE_EQ(features[material_offset + 0], 1.0); // 8 white pawns / 8
    EXPECT_DOUBLE_EQ(features[material_offset + 6], 1.0); // 8 black pawns / 8
    EXPECT_DOUBLE_EQ(features[parent_board_offset + 12], 1.0); // white pawn on e2
    EXPECT_DOUBLE_EQ(features[parent_board_offset + 11 * 64 + 60], 1.0); // black king on e8
    EXPECT_DOUBLE_EQ(features[child_board_offset + 12], 0.0);
    EXPECT_DOUBLE_EQ(features[child_board_offset + 28], 1.0); // white pawn moved to e4
}

TEST(move_policy, loads_json_and_scores_move) {
    const auto path = fs::temp_directory_path() / "enyo_move_policy_zero_model.json";
    std::ofstream out(path);
    out << make_zero_move_policy_model(3.5).dump();
    out.close();

    move_policy::Model model;
    std::string error;
    ASSERT_TRUE(model.load(path.string(), &error)) << error;
    EXPECT_TRUE(model.loaded());
    EXPECT_EQ(model.input_dim(), 1708);
    EXPECT_DOUBLE_EQ(model.threshold(), 18.0);

    Board b("startpos");
    const auto move = uci_to_move(b, "e2e4");
    ASSERT_TRUE(move.has_value());
    EXPECT_DOUBLE_EQ(model.score(b, *move), 3.5);

    fs::remove(path);
}

TEST(hash, apply_revert_restores_hash) {
    Board b{"startpos"};
    const auto initial_hash = b.hash;

    auto move = resolve_move<white>(b, pawn, e2, e4);
    apply_move<white>(b, move);
    const auto applied_recomputed = zobrist::generate_hash(b);
    EXPECT_EQ(b.hash, applied_recomputed) << fmt::format("after apply fen={} inc={:016X} rec={:016X}", b.fen(), b.hash, applied_recomputed);

    revert_move<white>(b);
    const auto reverted_recomputed = zobrist::generate_hash(b);
    EXPECT_EQ(b.hash, reverted_recomputed) << fmt::format("after revert fen={} inc={:016X} rec={:016X}", b.fen(), b.hash, reverted_recomputed);
    EXPECT_EQ(b.hash, initial_hash) << fmt::format("initial={:016X} reverted={:016X}", initial_hash, b.hash);
}

TEST(hash, enpassant_double_push_matches_recompute) {
    Board b{"8/8/8/8/3p1p2/8/4P3/4K2k w - - 0 1"};
    auto move = resolve_move<white>(b, pawn, e2, e4);
    apply_move<white>(b, move);

    ASSERT_NE(0, b.gamestate.enpassant_square);
    const auto recomputed = zobrist::generate_hash(b);
    EXPECT_EQ(b.hash, recomputed) << fmt::format("fen={} inc={:016X} rec={:016X} ep={}", b.fen(), b.hash, recomputed, sq2str(b.gamestate.enpassant_square));
}

TEST(hash, compare_single_push_vs_double_push_delta) {
    Board single{"8/8/8/8/3p1p2/8/4P3/4K2k w - - 0 1"};
    Board dbl{single};

    auto e2e3 = resolve_move<white>(single, pawn, e2, e3);
    auto e2e4 = resolve_move<white>(dbl, pawn, e2, e4);
    apply_move<white>(single, e2e3);
    apply_move<white>(dbl, e2e4);

    const auto single_recomputed = zobrist::generate_hash(single);
    const auto double_recomputed = zobrist::generate_hash(dbl);

    EXPECT_EQ(single.hash, single_recomputed);
    EXPECT_EQ(double_recomputed ^ single_recomputed, dbl.hash ^ single.hash)
        << fmt::format("single={:016X}/{:016X} double={:016X}/{:016X}", single.hash, single_recomputed, dbl.hash, double_recomputed);
}

TEST(hash, castling_rights_mask_hash_matches_recompute) {
    Board b{"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"};
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));

    auto move = resolve_move<white>(b, rook, h1, h2);
    apply_move<white>(b, move);
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));

    revert_move<white>(b);
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));
}

TEST(fen, round_trips_fullmove_counter) {
    Board b{"1r3rk1/p1q1pp1p/1np1b1p1/2Q5/8/1PNB1P2/P1P3PP/2KR3R b - - 0 17"};

    EXPECT_EQ(b.fen(), "1r3rk1/p1q1pp1p/1np1b1p1/2Q5/8/1PNB1P2/P1P3PP/2KR3R b - - 0 17");
}

// get_fen computes the fullmove counter as full_moves + histply/2, which is
// only correct when the starting FEN was white-to-move. After parsing a
// black-to-move FEN, black's first reply (histply=1) already completes a
// full move and must tick the counter — histply/2 is still 0 and the FEN
// emits the wrong number, which poisons any downstream consumer that
// trusts the FEN (GUI display, PGN export, external reference engines).
TEST(fen, fullmove_counter_advances_correctly_from_black_to_move_start) {
    // White-to-move baseline: counter ticks after each black move (even ply).
    {
        Board b{"4k3/8/8/8/8/8/8/4K3 w - - 0 5"};
        EXPECT_EQ(b.fen(), "4k3/8/8/8/8/8/8/4K3 w - - 0 5");
        apply_move<white>(b, resolve_move<white>(b, king, e1, e2));
        EXPECT_EQ(b.fen(), "4k3/8/8/8/8/8/4K3/8 b - - 1 5");  // white moved; still move 5
        apply_move<black>(b, resolve_move<black>(b, king, e8, e7));
        EXPECT_EQ(b.fen(), "8/4k3/8/8/8/8/4K3/8 w - - 2 6");  // black completed move 5; now 6
    }
    // Black-to-move start: counter must tick after black's first move.
    {
        Board b{"4k3/8/8/8/8/8/8/4K3 b - - 0 5"};
        EXPECT_EQ(b.fen(), "4k3/8/8/8/8/8/8/4K3 b - - 0 5");
        apply_move<black>(b, resolve_move<black>(b, king, e8, e7));
        EXPECT_EQ(b.fen(), "8/4k3/8/8/8/8/8/4K3 w - - 1 6");  // black completed move 5; now 6
        apply_move<white>(b, resolve_move<white>(b, king, e1, e2));
        EXPECT_EQ(b.fen(), "8/4k3/8/8/8/8/4K3/8 b - - 2 6");  // white moved; still 6
        apply_move<black>(b, resolve_move<black>(b, king, e7, e8));
        EXPECT_EQ(b.fen(), "4k3/8/8/8/8/8/4K3/8 w - - 3 7");  // black completed move 6; now 7
    }
}

// get_fen() emits `b.half_moves + b.gamestate.half_moves`, treating
// b.half_moves as a delta from the FEN-parsed base. When a reset event
// (capture, pawn move, en-passant, promotion) zeroed b.half_moves but
// left b.gamestate.half_moves at the parsed base, the emitted FEN still
// showed the stale base instead of 0. Syzygy rule50 (which reads the
// same sum via board2pos) then fed tablebase probing an inflated counter and its
// WDL wrapper rejected legal probes under "nonzero rule50."
//
// Fix: reset_halfmove_clock() clears both fields together; Undo captures
// gamestate before apply mutates it, so revert rolls both back.
TEST(fen, halfmove_clock_resets_to_zero_with_nonzero_fen_base) {
    struct Case {
        const char * fen;
        square_t     src;
        square_t     dst;
        PieceType    src_piece;
        PieceType    promo;       // no_piece_type for non-promotion
        const char * post_fen;
        const char * label;
    };
    const Case cases[] = {
        {"4k3/8/8/8/8/8/4P3/4K3 w - - 7 12", e2, e4, pawn, no_piece_type,
         "4k3/8/8/8/4P3/8/8/4K3 b - - 0 12", "pawn double-push"},
        {"4k3/8/8/3p4/4P3/8/8/4K3 w - - 7 12", e4, d5, pawn, no_piece_type,
         "4k3/8/8/3P4/8/8/8/4K3 b - - 0 12", "pawn capture"},
        {"4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12", e5, d6, pawn, no_piece_type,
         "4k3/8/3P4/8/8/8/8/4K3 b - - 0 12", "en-passant"},
        {"4k3/P7/8/8/8/8/8/4K3 w - - 7 12", a7, a8, pawn, queen,
         "Q3k3/8/8/8/8/8/8/4K3 b - - 0 12", "promotion without capture"},
    };
    for (const auto & c : cases) {
        Board b{c.fen};
        auto m = resolve_move<white>(b, c.src_piece, c.src, c.dst);
        if (c.promo != no_piece_type)
            m.set_promo_piece(c.promo);
        apply_move<white>(b, m);
        EXPECT_EQ(b.fen(), c.post_fen) << c.label;
    }

    // Counter-check: castle and quiet king moves must still carry the
    // FEN base forward, incrementing rather than resetting.
    {
        Board b{"r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 7 12"};
        apply_move<white>(b, resolve_move<white>(b, king, e1, g1));
        EXPECT_EQ(b.fen(), "r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R4RK1 b kq - 8 12");
    }
    {
        Board b{"4k3/8/8/8/8/5N2/8/4K3 w - - 7 12"};
        apply_move<white>(b, resolve_move<white>(b, knight, f3, e5));
        EXPECT_EQ(b.fen(), "4k3/8/8/4N3/8/8/8/4K3 b - - 8 12");
    }
}

TEST(pgn, uci_to_algebra_formats_check_without_mutating_board) {
    Board b{"4k3/8/8/8/8/8/8/3R3K w - - 0 1"};
    const auto before = b.fen();

    auto move = uci_to_move(b, "d1d8");
    ASSERT_TRUE(move.has_value());
    EXPECT_EQ("Rd8+", move_to_algebra(b, *move));
    EXPECT_EQ("Rd8+", uci_to_algebra(b, "d1d8"));
    EXPECT_EQ(before, b.fen());
}

TEST(movegen, lichess_fjmp5yck_endgame_sequence_preserves_position) {
    Board b{"1r6/2K5/R6p/1P5P/3b2k1/8/8/8 w - - 0 118"};
    const auto initial_hash = b.hash;
    const auto initial_fen = b.fen();

    apply_move<white>(b, resolve_move<white>(b, king, c7, b8));
    EXPECT_EQ(b.fen(), "1K6/8/R6p/1P5P/3b2k1/8/8/8 b - - 0 118");

    apply_move<black>(b, resolve_move<black>(b, king, g4, h5));
    EXPECT_EQ(b.fen(), "1K6/8/R6p/1P5k/3b4/8/8/8 w - - 0 119");

    apply_move<white>(b, resolve_move<white>(b, rook, a6, d6));
    EXPECT_EQ(b.fen(), "1K6/8/3R3p/1P5k/3b4/8/8/8 b - - 1 119");

    apply_move<black>(b, resolve_move<black>(b, bishop, d4, e3));
    EXPECT_EQ(b.fen(), "1K6/8/3R3p/1P5k/8/4b3/8/8 w - - 2 120");

    apply_move<white>(b, resolve_move<white>(b, rook, d6, d5));
    EXPECT_EQ(b.fen(), "1K6/8/7p/1P1R3k/8/4b3/8/8 b - - 3 120");

    apply_move<black>(b, resolve_move<black>(b, king, h5, g4));
    EXPECT_EQ(b.fen(), "1K6/8/7p/1P1R4/6k1/4b3/8/8 w - - 4 121");

    revert_move<black>(b);
    revert_move<white>(b);
    revert_move<black>(b);
    revert_move<white>(b);
    revert_move<black>(b);
    revert_move<white>(b);

    EXPECT_EQ(b.fen(), initial_fen);
    EXPECT_EQ(b.hash, initial_hash);
}

// Castling bugs caught before this was deployed:
//   1. white O-O incorrectly SET white_ooo (copy-paste of "false" -> "true")
//   2. black O-O-O XOR'd the wrong zobrist indices (2/3 swapped)
//   3. opposite-side rights XOR'd unconditionally, injecting phantom bits
//      into the hash whenever the bit was already cleared
// Each assertion compares the incrementally-maintained hash against
// zobrist::generate_hash (which rebuilds the hash from the current state),
// and verifies the castling bits are correct in gamestate. Apply/revert
// is also checked to round-trip both state and hash.
TEST(movegen, castle_updates_rights_and_hash_white_kingside) {
    Board b{"r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1"};
    const auto before_hash = b.hash;
    const auto before_fen = b.fen();

    apply_move<white>(b, resolve_move<white>(b, king, e1, g1));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::white_oo));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::white_ooo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::black_oo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::black_ooo));
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));

    revert_move<white>(b);
    EXPECT_EQ(b.hash, before_hash);
    EXPECT_EQ(b.fen(), before_fen);
}

TEST(movegen, castle_updates_rights_and_hash_white_queenside) {
    Board b{"r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1"};
    const auto before_hash = b.hash;
    const auto before_fen = b.fen();

    apply_move<white>(b, resolve_move<white>(b, king, e1, c1));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::white_oo));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::white_ooo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::black_oo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::black_ooo));
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));

    revert_move<white>(b);
    EXPECT_EQ(b.hash, before_hash);
    EXPECT_EQ(b.fen(), before_fen);
}

TEST(movegen, castle_updates_rights_and_hash_black_kingside) {
    Board b{"r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R b KQkq - 0 1"};
    const auto before_hash = b.hash;
    const auto before_fen = b.fen();

    apply_move<black>(b, resolve_move<black>(b, king, e8, g8));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::black_oo));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::black_ooo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::white_oo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::white_ooo));
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));

    revert_move<black>(b);
    EXPECT_EQ(b.hash, before_hash);
    EXPECT_EQ(b.fen(), before_fen);
}

TEST(movegen, castle_updates_rights_and_hash_black_queenside) {
    Board b{"r3k2r/pppqbppp/2np1n2/4p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R b KQkq - 0 1"};
    const auto before_hash = b.hash;
    const auto before_fen = b.fen();

    apply_move<black>(b, resolve_move<black>(b, king, e8, c8));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::black_oo));
    EXPECT_FALSE(b.gamestate.can_castle(CastlingRights::black_ooo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::white_oo));
    EXPECT_TRUE(b.gamestate.can_castle(CastlingRights::white_ooo));
    EXPECT_EQ(b.hash, zobrist::generate_hash(b));

    revert_move<black>(b);
    EXPECT_EQ(b.hash, before_hash);
    EXPECT_EQ(b.fen(), before_fen);
}

// Catches unconditional XOR of opposite-side rights: if the "other" castle bit
// is already cleared when we castle, XOR'ing its zobrist key injects a phantom
// bit into the hash. All four FENs start with only the *castling* side's
// rights, so the only correct hash delta is the one for the side that moved.
TEST(movegen, castle_does_not_touch_cleared_opposite_side_rights) {
    struct Case {
        const char * fen;
        const char * post_castle_fen;
    };
    // Each case: one-sided pre-castle rights; exercising the side that still
    // has rights must not XOR or set the already-cleared opposite bit, and
    // apply/revert must round-trip hash and FEN exactly.
    const Case cases[] = {
        {"4k3/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w K - 0 1",
         "4k3/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R4RK1 b - - 1 1"},  // W OO, W-OOO already cleared
        {"4k3/pppq1ppp/2np1n2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w Q - 0 1",
         "4k3/pppq1ppp/2np1n2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/2KR3R b - - 1 1"},  // W OOO, W-OO already cleared
        {"r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/4K3 b k - 0 1",
         "r4rk1/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/4K3 w - - 1 2"},  // B OO, B-OOO already cleared
        {"r3k2r/pppqbppp/2np1n2/4p3/2B1P3/2NP1N2/PPPQ1PPP/4K3 b q - 0 1",
         "2kr3r/pppqbppp/2np1n2/4p3/2B1P3/2NP1N2/PPPQ1PPP/4K3 w - - 1 2"},    // B OOO, B-OO already cleared
    };
    for (const auto & c : cases) {
        Board b{c.fen};
        const auto expected_before = zobrist::generate_hash(b);
        ASSERT_EQ(b.hash, expected_before) << c.fen;
        if (b.side == white) {
            const auto dst = b.gamestate.can_castle(CastlingRights::white_oo) ? g1 : c1;
            apply_move<white>(b, resolve_move<white>(b, king, e1, dst));
            EXPECT_EQ(b.fen(), c.post_castle_fen) << "apply: " << c.fen;
            EXPECT_EQ(b.hash, zobrist::generate_hash(b)) << "apply: " << c.fen;
            revert_move<white>(b);
        } else {
            const auto dst = b.gamestate.can_castle(CastlingRights::black_oo) ? g8 : c8;
            apply_move<black>(b, resolve_move<black>(b, king, e8, dst));
            EXPECT_EQ(b.fen(), c.post_castle_fen) << "apply: " << c.fen;
            EXPECT_EQ(b.hash, zobrist::generate_hash(b)) << "apply: " << c.fen;
            revert_move<black>(b);
        }
        EXPECT_EQ(b.fen(), c.fen) << "revert: " << c.fen;
        EXPECT_EQ(b.hash, expected_before) << "revert: " << c.fen;
        EXPECT_EQ(b.hash, zobrist::generate_hash(b)) << "revert: " << c.fen;
    }
}

// When a rook is captured on its home square, the corresponding side's
// castling right must be cleared in both state and hash. apply_move_generic's
// switch keys on src_piece (the capturer), so historically only a rook
// *moving* from its home cleared the right — the capture path never did.
// Same reachable position via "rook captured" vs "rook moved and returned"
// then produces different 64-bit hashes: TT fragmentation.
TEST(movegen, captured_home_rook_clears_castling_rights) {
    struct Case {
        const char * fen;
        square_t     src;
        square_t     dst;
        const char * post_fen;   // expected FEN after the capture
        Color        mover;
    };
    const Case cases[] = {
        // White knight b6 captures black rook on a8 — black queenside right clears.
        {"r3k2r/2pq1ppp/pN1p1n2/4p3/8/2N5/PPPQ1PPP/R3K2R w KQkq - 0 1", b6, a8,
         "N3k2r/2pq1ppp/p2p1n2/4p3/8/2N5/PPPQ1PPP/R3K2R b KQk - 0 1", white},
        // White knight g6 captures black rook on h8 — black kingside right clears.
        {"r3k2r/ppp2pp1/3p1nNp/4p3/8/2N5/PPPQ1PPP/R3K2R w KQkq - 0 1", g6, h8,
         "r3k2N/ppp2pp1/3p1n1p/4p3/8/2N5/PPPQ1PPP/R3K2R b KQq - 0 1", white},
        // Black queen a2 captures white rook on a1 — white queenside right clears.
        {"r3k2r/pppq1ppp/2np1n2/4p3/8/2N5/qPPQ1PPP/R3K2R b KQkq - 0 1", a2, a1,
         "r3k2r/pppq1ppp/2np1n2/4p3/8/2N5/1PPQ1PPP/q3K2R w Kkq - 0 2", black},
        // Black queen h5 captures white rook on h1 — white kingside right clears.
        {"r3k2r/pppq1pp1/2np1n2/4p2q/8/2N5/PPPQ1PP1/R3K2R b KQkq - 0 1", h5, h1,
         "r3k2r/pppq1pp1/2np1n2/4p3/8/2N5/PPPQ1PP1/R3K2q w Qkq - 0 2", black},
    };
    for (const auto & c : cases) {
        Board b{c.fen};
        const auto before_hash = b.hash;
        ASSERT_EQ(b.hash, zobrist::generate_hash(b)) << c.fen;
        if (c.mover == white) {
            const auto pt = get_piece_type<white>(b, c.src);
            apply_move<white>(b, resolve_move<white>(b, pt, c.src, c.dst));
        } else {
            const auto pt = get_piece_type<black>(b, c.src);
            apply_move<black>(b, resolve_move<black>(b, pt, c.src, c.dst));
        }
        EXPECT_EQ(b.fen(), c.post_fen) << "apply: " << c.fen;
        EXPECT_EQ(b.hash, zobrist::generate_hash(b)) << "apply: " << c.fen;
        if (c.mover == white) revert_move<white>(b); else revert_move<black>(b);
        EXPECT_EQ(b.fen(), c.fen) << "revert: " << c.fen;
        EXPECT_EQ(b.hash, before_hash) << "revert: " << c.fen;
    }
}

// Promotion-capture variant of the above. Promotion captures flow through
// apply_promotion/revert_promotion, not apply_move_generic, so they need
// their own path for clearing the captured home rook's castling right.
// Same TT-fragmentation hazard as captured_home_rook_clears_castling_rights.
TEST(movegen, promotion_capture_home_rook_clears_castling_rights) {
    struct Case {
        const char * fen;
        square_t     src;
        square_t     dst;
        const char * post_fen;   // expected FEN after the promotion capture
        Color        mover;
    };
    const Case cases[] = {
        // White pawn b7 captures black rook on a8, promoting to queen — black queenside right clears.
        {"r3k2r/1P6/8/8/8/8/8/4K3 w kq - 0 1", b7, a8,
         "Q3k2r/8/8/8/8/8/8/4K3 b k - 0 1", white},
        // White pawn g7 captures black rook on h8, promoting to queen — black kingside right clears.
        {"r3k2r/6P1/8/8/8/8/8/4K3 w kq - 0 1", g7, h8,
         "r3k2Q/8/8/8/8/8/8/4K3 b q - 0 1", white},
        // Black pawn b2 captures white rook on a1, promoting to queen — white queenside right clears.
        {"4k3/8/8/8/8/8/1p6/R3K2R b KQ - 0 1", b2, a1,
         "4k3/8/8/8/8/8/8/q3K2R w K - 0 2", black},
        // Black pawn g2 captures white rook on h1, promoting to queen — white kingside right clears.
        {"4k3/8/8/8/8/8/6p1/R3K2R b KQ - 0 1", g2, h1,
         "4k3/8/8/8/8/8/8/R3K2q w Q - 0 2", black},
    };
    for (const auto & c : cases) {
        Board b{c.fen};
        const auto before_hash = b.hash;
        ASSERT_EQ(b.hash, zobrist::generate_hash(b)) << c.fen;
        if (c.mover == white) {
            auto m = resolve_move<white>(b, pawn, c.src, c.dst);
            m.set_promo_piece(PieceType::queen);
            apply_move<white>(b, m);
        } else {
            auto m = resolve_move<black>(b, pawn, c.src, c.dst);
            m.set_promo_piece(PieceType::queen);
            apply_move<black>(b, m);
        }
        EXPECT_EQ(b.fen(), c.post_fen) << "apply: " << c.fen;
        EXPECT_EQ(b.hash, zobrist::generate_hash(b)) << "apply: " << c.fen;
        if (c.mover == white) revert_move<white>(b); else revert_move<black>(b);
        EXPECT_EQ(b.fen(), c.fen) << "revert: " << c.fen;
        EXPECT_EQ(b.hash, before_hash) << "revert: " << c.fen;
    }
}

// Pawn double-push probes the two diagonal neighbours on the landing rank
// for an enemy pawn that would make en-passant possible. The neighbour
// lookup used (1ULL << (dst+1)) | (1ULL << (dst-1)) without masking to the
// landing rank, so a-file and h-file double-pushes wrapped onto the
// adjacent rank: e.g. white a2a4 (dst=a4=bit31) set dst+1=32=h5, and a
// black pawn sitting on h5 was then read as a "b4 neighbour" that armed
// en-passant. Result: bogus ep square in the post-move state, wrong FEN,
// and a poisoned zobrist (the ep term gets XOR'd in). Legal moves generated
// from the poisoned state can include an illegal en-passant capture.
TEST(movegen, double_push_does_not_wrap_ep_to_opposite_file) {
    struct Case {
        const char * fen;
        square_t     src;
        square_t     dst;
        const char * post_fen;   // expected FEN after the double push (no ep)
        Color        mover;
    };
    const Case cases[] = {
        // White a2-a4 with a black pawn on h5: pre-fix, (a4+1)=h5 hit the
        // black pawn and spuriously set ep_square=a3.
        {"4k3/8/8/7p/8/8/P7/4K3 w - - 0 1", a2, a4,
         "4k3/8/8/7p/P7/8/8/4K3 b - - 0 1", white},
        // White h2-h4 with a black pawn on a3: pre-fix, (h4-1)=a3 hit it.
        {"4k3/8/8/8/8/p7/7P/4K3 w - - 0 1", h2, h4,
         "4k3/8/8/8/7P/p7/8/4K3 b - - 0 1", white},
        // Black a7-a5 with a white pawn on h6: pre-fix, (a5+1)=h6 hit it.
        {"4k3/p7/7P/8/8/8/8/4K3 b - - 0 1", a7, a5,
         "4k3/8/7P/p7/8/8/8/4K3 w - - 0 2", black},
        // Black h7-h5 with a white pawn on a4: pre-fix, (h5-1)=a4 hit it.
        {"4k3/7p/8/8/P7/8/8/4K3 b - - 0 1", h7, h5,
         "4k3/8/8/7p/P7/8/8/4K3 w - - 0 2", black},
    };
    for (const auto & c : cases) {
        Board b{c.fen};
        const auto before_hash = b.hash;
        ASSERT_EQ(b.hash, zobrist::generate_hash(b)) << c.fen;
        if (c.mover == white) {
            apply_move<white>(b, resolve_move<white>(b, pawn, c.src, c.dst));
        } else {
            apply_move<black>(b, resolve_move<black>(b, pawn, c.src, c.dst));
        }
        EXPECT_EQ(b.fen(), c.post_fen) << "apply: " << c.fen;
        EXPECT_EQ(b.hash, zobrist::generate_hash(b)) << "apply: " << c.fen;
        EXPECT_EQ(b.gamestate.enpassant_square, 0) << "apply: " << c.fen;
        if (c.mover == white) revert_move<white>(b); else revert_move<black>(b);
        EXPECT_EQ(b.fen(), c.fen) << "revert: " << c.fen;
        EXPECT_EQ(b.hash, before_hash) << "revert: " << c.fen;
    }
}

// Castling is neither a capture nor a pawn move, so per FIDE the 50-move
// halfmove clock must *increment* across a castle, not reset. apply_move
// historically zeroed b.half_moves in the castle branch, which (a) emits a
// wrong FEN, (b) mis-counts the 50-move rule, and (c) truncates the
// is_repetition() lookback window — which uses b.half_moves as the scan
// depth into history — allowing repetitions that straddle a castle to go
// undetected.
TEST(movegen, castle_does_not_reset_halfmove_clock) {
    struct Case {
        const char * fen;
        square_t     src;
        square_t     dst;
        Color        mover;
        const char * label;
    };
    const Case cases[] = {
        {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 5 10", e1, g1, white, "white O-O"},
        {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 5 10", e1, c1, white, "white O-O-O"},
        {"r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 5 10", e8, g8, black, "black O-O"},
        {"r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 5 10", e8, c8, black, "black O-O-O"},
    };
    for (const auto & c : cases) {
        Board b{c.fen};
        const int before_counter = b.half_moves + b.gamestate.half_moves;
        if (c.mover == white)
            apply_move<white>(b, resolve_move<white>(b, king, c.src, c.dst));
        else
            apply_move<black>(b, resolve_move<black>(b, king, c.src, c.dst));
        const int after_counter = b.half_moves + b.gamestate.half_moves;
        EXPECT_EQ(after_counter, before_counter + 1) << c.label;
    }
}

TEST(search, repetition_detects_claimable_third_occurrence) {
    Board b{"startpos"};

    apply_move<white>(b, resolve_move<white>(b, knight, g1, f3));
    apply_move<black>(b, resolve_move<black>(b, knight, g8, f6));
    apply_move<white>(b, resolve_move<white>(b, knight, f3, g1));
    apply_move<black>(b, resolve_move<black>(b, knight, f6, g8));
    apply_move<white>(b, resolve_move<white>(b, knight, g1, f3));
    apply_move<black>(b, resolve_move<black>(b, knight, g8, f6));
    apply_move<white>(b, resolve_move<white>(b, knight, f3, g1));
    apply_move<black>(b, resolve_move<black>(b, knight, f6, g8));

    EXPECT_TRUE(is_repetition(b, 1));
    EXPECT_FALSE(is_repetition(b, 2));
}

TEST(search, repetition_parity_scan_matches_full_history_scan) {
    const auto reference = [](const Board & board, int draw) {
        if (board.half_moves < 2)
            return false;

        const int begin = board.histply - board.half_moves;
        int matches = 0;
        for (int index = begin; index < board.histply; ++index)
            matches += board.history[index].hash == board.hash;
        return matches > draw;
    };

    Board b {"startpos"};
    const auto expect_matches_reference = [&] {
        for (int draw = 0; draw <= 6; ++draw)
            EXPECT_EQ(is_repetition(b, draw), reference(b, draw));
    };

    for (int cycle = 0; cycle < 8; ++cycle) {
        apply_move<white>(b, resolve_move<white>(b, knight, g1, f3));
        expect_matches_reference();

        apply_move<black>(b, resolve_move<black>(b, knight, g8, f6));
        expect_matches_reference();

        apply_move<white>(b, resolve_move<white>(b, knight, f3, g1));
        expect_matches_reference();

        apply_move<black>(b, resolve_move<black>(b, knight, f6, g8));
        expect_matches_reference();
    }
}

TEST(search, qsearch_pv_does_not_copy_stale_tail) {
    Board b{"startpos"};

    SearchInfo si{b, 1};
    si.board = b;
    si.searchmoves.emplace(resolve_move<white>(b, pawn, e2, e4));
    si.has_searchmoves = true;
    si.nnue.refresh(si.board);

    thread::pool.stop = false;
    tt::ttable.clear();
    tt::ttable.prepare();

    Worker worker{si, 0};
    worker.pvline.clear();
    worker.pvline.table[1][1] = resolve_move<black>(b, pawn, a7, a6);
    worker.pvline.len[1] = 2;

    Stack stack[MAX_PLY + 5];
    stack[0].ply = 4;
    stack[1].ply = 3;
    stack[2].ply = 2;
    stack[3].ply = 1;
    for (int i = 0; i <= MAX_PLY; i++)
        stack[i + 4].ply = i;
    Stack * ss = stack + 4;

    (void)negamax<white, NodeType::Root>(
        1, worker, ss, -Value::infinite, Value::infinite);

    EXPECT_EQ(worker.pvline.bestmove(), resolve_move<white>(b, pawn, e2, e4));
    EXPECT_EQ(worker.pvline.len[0], 1);
    EXPECT_EQ(worker.pvline.str(), "e2e4");
}

TEST(search, pv_setlen_clears_stale_bestmove) {
    Board b{"startpos"};
    QuadraticPV pv;
    const auto move = resolve_move<white>(b, pawn, e2, e4);

    pv.setmove(move, 0);
    ASSERT_EQ(pv.bestmove(), move);
    ASSERT_EQ(pv.str(), "e2e4");

    pv.setlen(0);

    EXPECT_EQ(pv.bestmove(), Move {});
    EXPECT_EQ(pv.str(), "");
}

TEST(search, pv_setmove_without_tail_does_not_copy_stale_child) {
    Board b{"startpos"};
    QuadraticPV pv;

    pv.table[1][1] = resolve_move<black>(b, pawn, a7, a6);
    pv.len[1] = 2;

    const auto move = resolve_move<white>(b, pawn, e2, e4);
    pv.setmove(move, 0, false);

    EXPECT_EQ(pv.bestmove(), move);
    EXPECT_EQ(pv.len[0], 1);
    EXPECT_EQ(pv.str(), "e2e4");
}

TEST(search, root_repetition_is_penalized_when_not_worse) {
    Board b{"startpos"};

    apply_move<white>(b, resolve_move<white>(b, knight, g1, f3));
    apply_move<black>(b, resolve_move<black>(b, knight, g8, f6));
    apply_move<white>(b, resolve_move<white>(b, knight, f3, g1));
    apply_move<black>(b, resolve_move<black>(b, knight, f6, g8));
    apply_move<white>(b, resolve_move<white>(b, knight, g1, f3));
    apply_move<black>(b, resolve_move<black>(b, knight, g8, f6));
    apply_move<white>(b, resolve_move<white>(b, knight, f3, g1));
    apply_move<black>(b, resolve_move<black>(b, knight, f6, g8));

    const int old_contempt = cfgmgr.root_repetition_contempt;
    cfgmgr.root_repetition_contempt = 24;

    SearchInfo si{b, 1};
    si.board = b;
    si.searchmoves.emplace(resolve_move<white>(b, knight, g1, f3));
    si.has_searchmoves = true;
    si.nnue.refresh(si.board);

    thread::pool.stop = false;
    tt::ttable.clear();
    tt::ttable.prepare();

    Worker worker{si, 0};
    worker.pvline.clear();
    worker.pvline.table[1][1] = resolve_move<black>(b, knight, g8, f6);
    worker.pvline.len[1] = 2;

    Stack stack[MAX_PLY + 5];
    stack[0].ply = 4;
    stack[1].ply = 3;
    stack[2].ply = 2;
    stack[3].ply = 1;
    for (int i = 0; i <= MAX_PLY; i++)
        stack[i + 4].ply = i;
    Stack * ss = stack + 4;

    const auto value = negamax<white, NodeType::Root>(
        1, worker, ss, -Value::infinite, Value::infinite);

    EXPECT_EQ(value, static_cast<Value>(-cfgmgr.root_repetition_contempt));
    EXPECT_EQ(worker.pvline.bestmove(), resolve_move<white>(b, knight, g1, f3));
    EXPECT_EQ(worker.pvline.len[0], 1);
    EXPECT_EQ(worker.pvline.str(), "g1f3");

    cfgmgr.root_repetition_contempt = old_contempt;
}

TEST(search, root_repetition_contempt_is_configurable) {
    const int old_contempt = cfgmgr.root_repetition_contempt;

    EXPECT_TRUE(cfgmgr.setopt("root_repetition_contempt", "48"));
    EXPECT_EQ(cfgmgr.root_repetition_contempt, 48);

    EXPECT_TRUE(cfgmgr.setopt("root_repetition_contempt", "-1"));
    EXPECT_EQ(cfgmgr.root_repetition_contempt, 0);

    cfgmgr.root_repetition_contempt = old_contempt;
}

TEST(search, hypersion_check_net_root_evasions_find_safer_defenses) {
    const struct {
        std::string_view fen;
        std::string_view bestmove;
    } cases[] = {
        {"2k4R/1p6/4p3/3pP3/8/2P1KB2/3Q4/1q4r1 b - - 16 53", "bestmove c8c7"},
        {"7R/1p6/8/4Q3/8/8/1k3K2/1q1r4 b - - 3 62", "bestmove b2c1"},
    };

    const int old_threads = cfgmgr.num_threads;
    cfgmgr.num_threads = 1;

    for (const auto & c : cases) {
        Board b{std::string{c.fen}};
        SearchInfo si{b, 18};
        si.board = b;
        si.nnue.refresh(si.board);
        si.starttime = std::chrono::high_resolution_clock::now();
        si.stoptime = si.starttime + std::chrono::hours(1);
        si.soft_stoptime = si.stoptime;

        tt::ttable.clear();
        testing::internal::CaptureStdout();
        thread::pool.init_threads(si, 1);
        thread::pool.wait();
        const auto out = testing::internal::GetCapturedStdout();

        EXPECT_NE(out.find(c.bestmove), std::string::npos) << c.fen << "\n" << out;
    }

    cfgmgr.num_threads = old_threads;
}

static std::optional<std::string> last_non_bound_pv_head(std::string_view out)
{
    std::optional<std::string> head;
    size_t line_start = 0;
    while (line_start < out.size()) {
        const size_t line_end = out.find('\n', line_start);
        const auto line = out.substr(
            line_start,
            line_end == std::string_view::npos ? out.size() - line_start : line_end - line_start);
        if (line.starts_with("info ")
            && line.find(" pv ") != std::string_view::npos
            && line.find(" lowerbound ") == std::string_view::npos
            && line.find(" upperbound ") == std::string_view::npos) {
            const size_t pv = line.find(" pv ") + 4;
            const size_t move_end = line.find(' ', pv);
            head = std::string(line.substr(pv, move_end == std::string_view::npos ? line.size() - pv : move_end - pv));
        }
        if (line_end == std::string_view::npos)
            break;
        line_start = line_end + 1;
    }
    return head;
}

static std::optional<std::string> final_bestmove(std::string_view out)
{
    const size_t pos = out.rfind("bestmove ");
    if (pos == std::string_view::npos)
        return std::nullopt;
    const size_t move = pos + 9;
    const size_t move_end = out.find_first_of(" \n\r\t", move);
    return std::string(out.substr(move, move_end == std::string_view::npos ? out.size() - move : move_end - move));
}

static int pv_move_count(std::string_view line)
{
    const size_t pv = line.find(" pv ");
    if (pv == std::string_view::npos)
        return 0;

    std::istringstream iss{std::string{line.substr(pv + 4)}};
    int count = 0;
    for (std::string token; iss >> token;)
        ++count;
    return count;
}

TEST(search, root_lmr_does_not_hide_c7c6_defense) {
    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    const auto old_nnue_file = cfgmgr.nnue_file;

    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = false;
    move_policy::clear_runtime_model();

    Board b;
    Uci uci{b};
    testing::internal::CaptureStdout();
    uci(fmt::format("setoption name nnue_file value {}", EVAL_FILE));
    (void)testing::internal::GetCapturedStdout();
    uci("position fen r1bq1b1r/1pp3pk/p2p3B/1P1Q4/8/6R1/1PP2PPP/4R1K1 b - - 0 18");

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    uci("go depth 20");
    const auto out = testing::internal::GetCapturedStdout();
    const auto best = final_bestmove(out);

    testing::internal::CaptureStdout();
    uci(fmt::format("setoption name nnue_file value {}", old_nnue_file));
    (void)testing::internal::GetCapturedStdout();
    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;

    ASSERT_TRUE(best.has_value()) << out;
    EXPECT_EQ(*best, "c7c6") << out;
    EXPECT_EQ(out.find("bestmove d8f6"), std::string::npos) << out;
}

TEST(search, root_lmr_does_not_hide_b4_defense) {
    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    const auto old_nnue_file = cfgmgr.nnue_file;

    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = false;
    move_policy::clear_runtime_model();

    Board b;
    Uci uci{b};
    testing::internal::CaptureStdout();
    uci(fmt::format("setoption name nnue_file value {}", EVAL_FILE));
    (void)testing::internal::GetCapturedStdout();
    uci("position fen 4r3/p4pk1/2P2bp1/2QB1q2/K1R4p/1P6/P7/8 w - - 2 42");

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    uci("go depth 20");
    const auto out = testing::internal::GetCapturedStdout();
    const auto best = final_bestmove(out);

    testing::internal::CaptureStdout();
    uci(fmt::format("setoption name nnue_file value {}", old_nnue_file));
    (void)testing::internal::GetCapturedStdout();
    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;

    ASSERT_TRUE(best.has_value()) << out;
    EXPECT_EQ(*best, "b3b4") << out;
}

TEST(search, pv_stops_at_fifty_move_rule) {
    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = false;

    Board b;
    Uci uci{b};
    uci("position fen 7k/8/8/8/8/8/8/R3K3 w - - 99 1");

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    uci("go depth 4");
    const auto out = testing::internal::GetCapturedStdout();

    int pv_lines = 0;
    size_t line_start = 0;
    while (line_start < out.size()) {
        const size_t line_end = out.find('\n', line_start);
        const auto line = std::string_view{out}.substr(
            line_start,
            line_end == std::string::npos ? out.size() - line_start : line_end - line_start);
        if (line.starts_with("info ")
            && line.find(" pv ") != std::string_view::npos
            && line.find(" lowerbound ") == std::string_view::npos
            && line.find(" upperbound ") == std::string_view::npos) {
            ++pv_lines;
            EXPECT_LE(pv_move_count(line), 1) << line << "\n" << out;
        }
        if (line_end == std::string::npos)
            break;
        line_start = line_end + 1;
    }

    EXPECT_GT(pv_lines, 0) << out;
    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(uci_root, checkmated_root_reports_score_and_bestmove_0000) {
    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = true;

    Board b;
    Uci uci{b};
    uci("position fen 7k/5KQ1/8/8/8/8/8/8 b - - 0 1");

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    uci("go depth 4");
    const auto out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("info depth 0 score cp -30000"), std::string::npos) << out;
    EXPECT_NE(out.find("bestmove 0000"), std::string::npos) << out;
    EXPECT_EQ(out.find("bestmove h1h1"), std::string::npos) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(uci_root, expired_root_still_reports_score_before_bestmove) {
    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = false;

    Board b{"rn1qkbnr/ppp1pppp/3p4/8/8/3P4/PPP1PPPP/RNBQKBNR w KQkq - 0 2"};
    SearchInfo si{b, MAX_PLY};
    si.board = b;
    si.nnue.refresh(si.board);
    const auto now = std::chrono::high_resolution_clock::now();
    si.starttime = now - std::chrono::milliseconds(2);
    si.stoptime = now - std::chrono::milliseconds(1);
    si.soft_stoptime = si.stoptime;

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    thread::pool.init_threads(si, 1);
    thread::pool.wait();
    const auto out = testing::internal::GetCapturedStdout();

    const auto score_pos = out.find("info depth 1 score cp 0");
    const auto bestmove_pos = out.find("bestmove ");
    ASSERT_NE(score_pos, std::string::npos) << out;
    ASSERT_NE(bestmove_pos, std::string::npos) << out;
    EXPECT_LT(score_pos, bestmove_pos) << out;
    EXPECT_EQ(out.find("bestmove 0000"), std::string::npos) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(search, timed_root_bestmove_matches_last_completed_pv) {
    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = false;

    Board b;
    Uci uci{b};
    uci("position fen r2qk1nr/ppp1b1p1/2n2p2/3pp1Pp/6bP/2PP1N2/PP2PP2/RNBQKB1R w KQkq - 2 8 moves e2e3");

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    uci("go movetime 20");
    const auto out = testing::internal::GetCapturedStdout();

    const auto pv_head = last_non_bound_pv_head(out);
    const auto best = final_bestmove(out);
    ASSERT_TRUE(pv_head.has_value()) << out;
    ASSERT_TRUE(best.has_value()) << out;
    EXPECT_EQ(*best, *pv_head) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(search, repeated_check_root_keeps_legal_previous_bestmove) {
    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = false;

    Board b;
    Uci uci{b};
    uci("position startpos moves e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6 f2f3 h7h5 c3d5 e6d5 e4d5 b8d7 d1d2 g7g6 e1c1 d7b6 c1b1 b6d5 e3g5 f8e7 f1d3 d8c7 h1e1 f6d7 f3f4 e7g5 f4g5 d5b6 d3e4 b6c4 d2b4 a8b8 b3d2 c4b6 d2f1 d7c5 f1e3 c5e4 b4e4 e8g8 e1f1 f8d8 c2c4 b8c8 h2h4 c7c6 e4d3 b6c4 e3d5 c6b5 f1f2 c8c5 d5e7 g8f8 d3f3 c4a3 b1a1 a3c2 a1b1 c2a3 f3a3 f8e7 a3f3 b5c4 f3f6 e7e8 f6h8 e8e7 h8f6");

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    uci("go nodes 200000");
    const auto out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("bestmove e7e8"), std::string::npos) << out;
    EXPECT_EQ(out.find("bestmove e7d7"), std::string::npos) << out;
    EXPECT_EQ(out.find(" pv h1g8"), std::string::npos) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(search, move_policy_root_guard_can_override_final_bestmove) {
    constexpr int to_offset = 14 + 64;
    constexpr int h3_python_square = 23;
    const auto path = fs::temp_directory_path() / "enyo_move_policy_h2h3_model.json";
    std::ofstream out(path);
    out << make_weighted_move_policy_model(to_offset + h3_python_square, 100.0).dump();
    out.close();

    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    const auto old_policy_file = cfgmgr.move_policy_file;
    const int old_max_drop = cfgmgr.move_policy_max_eval_drop;

    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = false;
    cfgmgr.move_policy_max_eval_drop = 1000;
    move_policy::clear_runtime_model();

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    {
        Board baseline_board;
        Uci baseline{baseline_board};
        baseline("position startpos");
        baseline("go depth 1 searchmoves e2e4 h2h3");
    }
    const auto baseline_out = testing::internal::GetCapturedStdout();
    EXPECT_NE(baseline_out.find("bestmove e2e4"), std::string::npos) << baseline_out;

    Board b;
    Uci uci{b};
    uci(fmt::format("setoption name move_policy_file value {}", path.string()));
    uci("position startpos");

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    uci("go depth 1 searchmoves e2e4 h2h3");
    const auto search_out = testing::internal::GetCapturedStdout();
    EXPECT_NE(search_out.find("bestmove h2h3"), std::string::npos) << search_out;

    move_policy::clear_runtime_model();
    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
    cfgmgr.move_policy_file = old_policy_file;
    cfgmgr.move_policy_max_eval_drop = old_max_drop;
    fs::remove(path);
}

// Qsearch-in-check regression. Pre-fix, qsearch unconditionally
// stand-patted — even when the side to move was in check — and the
// QSEARCH movepicker filtered out non-capturing / non-promoting moves
// (movepicker.hpp:85). Both together meant that a child qsearch node
// where the side to move was in check and mated would return the
// static eval instead of mated_in(ply).
//
// Concretely: from 7k/8/5KQ1/8/8/8/8/8 w - - 0 1, white's winning reply
// is g6g7 (mate in 1 — black is then in check with no legal move). At
// depth 1, negamax evaluates each root move by calling -qsearch on the
// child. After g6g7, the child position has black in check and mated;
// pre-fix qsearch stand-patted and returned a material-ahead eval that
// back-propagated as a finite score, not mate. Non-mating white moves
// like g6e4 looked better because they avoided losing the queen to a
// capture in the child's qsearch. The engine played g6e4 at depth 1
// and only found g6g7 once normal search depth reached 2.
//
// Post-fix: at depth 1 the root sees one move scoring mate-in-1 and
// picks g6g7.
TEST(search, qsearch_in_check_finds_mate_in_one_at_depth_one) {
    cfgmgr.num_threads = 1;

    Board b{"7k/8/5KQ1/8/8/8/8/8 w - - 0 1"};
    SearchInfo si{b, 1};
    si.board = b;
    si.nnue.refresh(si.board);
    // Leave starttime == stoptime (default-constructed) so time_expired()
    // is permanently false for this test. Setting only starttime would
    // make stoptime (still epoch) strictly less than now and expire
    // every call immediately.

    thread::pool.stop = false;
    tt::ttable.clear();
    tt::ttable.prepare();

    Worker worker{si, 0};
    worker.pvline.clear();

    // Bypass search_position() (which depends on thread::pool being
    // initialized for get_nps/get_nodes reporting). Call negamax<Root>
    // directly at depth 1 — the only thing that matters for this
    // regression is whether the in-check child-qsearch path back-
    // propagates mate correctly.
    Stack stack[MAX_PLY + 5];
    stack[0].ply = 4;
    stack[1].ply = 3;
    stack[2].ply = 2;
    stack[3].ply = 1;
    for (int i = 0; i <= MAX_PLY; i++)
        stack[i + 4].ply = i;
    Stack * ss = stack + 4;

    const auto value = negamax<white, NodeType::Root>(
        1, worker, ss, -Value::infinite, Value::infinite);

    // Expect the mate-in-1 move g6-g7 and a mate score.
    const Move best = worker.pvline.bestmove();
    EXPECT_EQ(best.src_sq(), g6);
    EXPECT_EQ(best.dst_sq(), g7);
    EXPECT_GE(static_cast<int>(value), static_cast<int>(Value::mate_in_max_ply));
}

TEST(search, root_ignores_empty_tt_cutoff_move) {
    Board b{"8/3k4/1P6/2K1P3/8/8/4b3/6B1 b - - 0 70"};
    SearchInfo si{b, 1};
    si.board = b;
    si.nnue.refresh(si.board);

    thread::pool.stop = false;
    tt::ttable.clear();
    tt::ttable.prepare();
    tt::ttable.store(
        si.board.hash,
        Move{},
        tt::value_to(Value::draw, 0),
        tt::type::ExactBound,
        64);

    Worker worker{si, 0};
    worker.pvline.clear();

    Stack stack[MAX_PLY + 5];
    stack[0].ply = 4;
    stack[1].ply = 3;
    stack[2].ply = 2;
    stack[3].ply = 1;
    for (int i = 0; i <= MAX_PLY; i++)
        stack[i + 4].ply = i;
    Stack * ss = stack + 4;

    (void)negamax<black, NodeType::Root>(
        1, worker, ss, -Value::infinite, Value::infinite);

    EXPECT_NE(worker.pvline.bestmove(), Move{});
    EXPECT_GT(worker.si.nodes, 0U);
}

// --- NNUE incremental audit -----------------------------------------------
// For each move type (quiet / capture / promotion / en-passant / castle),
// apply the move with UpdateNNUE=true (the live incremental/refresh path),
// snapshot the resulting accumulator, then do a full refresh() on the same
// post-move board and compare every element. Matching means the live path
// computed the correct accumulator; the whole HIDDEN_SIZE * 2 vector must
// agree.

namespace {

bool accumulators_match(const NNUE::Accumulator & a, const NNUE::Accumulator & b) {
    for (size_t i = 0; i < a.white_acc.size(); ++i)
        if (a.white_acc[i] != b.white_acc[i]) return false;
    for (size_t i = 0; i < a.black_acc.size(); ++i)
        if (a.black_acc[i] != b.black_acc[i]) return false;
    return true;
}

void ensure_network_mock_weights() {
    static std::vector<Network::acc_t> weights(
        static_cast<size_t>(Network::N_FEATURES) * Network::N_HIDDEN);
    static std::vector<Network::acc_t> biases(Network::N_HIDDEN);
    static bool initialized = false;

    if (!initialized) {
        for (size_t f = 0; f < static_cast<size_t>(Network::N_FEATURES); ++f) {
            for (size_t h = 0; h < static_cast<size_t>(Network::N_HIDDEN); ++h) {
                uint64_t x = (f + 0x9e3779b97f4a7c15ULL)
                           ^ ((h + 0xbf58476d1ce4e5b9ULL) << 1);
                x ^= x >> 30;
                x *= 0xbf58476d1ce4e5b9ULL;
                x ^= x >> 27;
                x *= 0x94d049bb133111ebULL;
                x ^= x >> 31;
                const int v = static_cast<int>(x % 255) - 127;
                weights[f * Network::N_HIDDEN + h] = static_cast<Network::acc_t>(v);
            }
        }
        for (size_t h = 0; h < static_cast<size_t>(Network::N_HIDDEN); ++h)
            biases[h] = static_cast<Network::acc_t>(static_cast<int>(h % 23) - 11);
        initialized = true;
    }

    Network::SetWeights(weights.data(), biases.data());
}

fs::path write_zero_network_blob(size_t size, std::string_view name) {
    const auto path = fs::temp_directory_path() / std::string(name);
    std::vector<char> bytes(size, 0);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}

void write_u32_le(std::vector<char> & bytes, size_t offset, uint32_t value) {
    for (size_t i = 0; i < 4; ++i)
        bytes[offset + i] = static_cast<char>((value >> (8 * i)) & 0xff);
}

void write_f32_le(std::vector<char> & bytes, size_t offset, float value) {
    static_assert(std::endian::native == std::endian::little);
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void write_i16_le(std::vector<char> & bytes, size_t offset, int16_t value) {
    const auto encoded = static_cast<uint16_t>(value);
    bytes[offset] = static_cast<char>(encoded & 0xff);
    bytes[offset + 1] = static_cast<char>((encoded >> 8) & 0xff);
}

fs::path write_quantized_test_network_blob(std::string_view name) {
    std::vector<char> bytes(Network::QuantizedNetworkSize(), 0);
    const size_t l1_bias_offset =
        sizeof(int16_t) * Network::FeatureCount(16, 12) * Network::N_HIDDEN
        + sizeof(int16_t) * Network::N_HIDDEN
        + sizeof(int8_t) * Network::N_L1 * Network::N_L2;
    const size_t l2_weight_offset =
        l1_bias_offset + sizeof(int32_t) * Network::N_L2;
    const size_t output_weight_offset =
        l2_weight_offset
        + sizeof(int16_t) * Network::N_L2 * Network::N_L3
        + sizeof(int32_t) * Network::N_L3;

    write_u32_le(bytes, l1_bias_offset, 64);
    write_i16_le(bytes, l2_weight_offset, 32);
    write_i16_le(bytes, output_weight_offset, 4096);

    const auto path = fs::temp_directory_path() / std::string(name);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}

fs::path write_zero_v2_network_blob(
    int input_buckets,
    int feature_channels,
    int trained_hidden,
    int output_buckets,
    std::string_view name)
{
    const size_t payload_size = Network::NetworkSize(
        input_buckets, output_buckets, 0, feature_channels);
    std::vector<char> bytes(Network::NETWORK_HEADER_SIZE + payload_size, 0);
    std::copy(
        Network::NETWORK_HEADER_MAGIC.begin(),
        Network::NETWORK_HEADER_MAGIC.end(),
        bytes.begin());
    write_u32_le(bytes, 8, Network::NETWORK_FORMAT_VERSION);
    write_u32_le(bytes, 12, Network::NETWORK_HEADER_SIZE);
    write_u32_le(bytes, 16, input_buckets);
    write_u32_le(bytes, 20, feature_channels);
    write_u32_le(bytes, 24, trained_hidden);
    write_u32_le(bytes, 28, Network::N_HIDDEN);
    write_u32_le(bytes, 32, Network::N_L2);
    write_u32_le(bytes, 36, Network::N_L3);
    write_u32_le(bytes, 40, output_buckets);
    write_u32_le(bytes, 44, 0);
    write_u32_le(bytes, 48, 0);
    write_u32_le(bytes, 52, static_cast<uint32_t>(payload_size));

    const auto path = fs::temp_directory_path() / std::string(name);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}

fs::path write_full_head_v3_network_blob(std::string_view name) {
    constexpr int output_buckets = 8;
    const size_t payload_size = Network::NetworkSize(
        16, output_buckets, 0, 12, false, true);
    std::vector<char> bytes(Network::NETWORK_HEADER_SIZE + payload_size, 0);
    std::copy(
        Network::NETWORK_V3_HEADER_MAGIC.begin(),
        Network::NETWORK_V3_HEADER_MAGIC.end(),
        bytes.begin());
    write_u32_le(bytes, 8, Network::NETWORK_V3_FORMAT_VERSION);
    write_u32_le(bytes, 12, Network::NETWORK_HEADER_SIZE);
    write_u32_le(bytes, 16, 16);
    write_u32_le(bytes, 20, 12);
    write_u32_le(bytes, 24, Network::N_HIDDEN);
    write_u32_le(bytes, 28, Network::N_HIDDEN);
    write_u32_le(bytes, 32, Network::N_L2);
    write_u32_le(bytes, 36, Network::N_L3);
    write_u32_le(bytes, 40, output_buckets);
    write_u32_le(bytes, 44, 0);
    write_u32_le(bytes, 48, Network::NETWORK_FLAG_FULL_HEADS);
    write_u32_le(bytes, 52, static_cast<uint32_t>(payload_size));

    const size_t payload = Network::NETWORK_HEADER_SIZE;
    const size_t input_weights = sizeof(int16_t)
        * Network::FeatureCount(16, 12) * Network::N_HIDDEN;
    const size_t input_biases = sizeof(int16_t) * Network::N_HIDDEN;
    const size_t l1_weights = output_buckets * sizeof(int8_t)
        * Network::N_L1 * Network::N_L2;
    const size_t l1_bias_offset = payload + input_weights + input_biases + l1_weights;
    const size_t l1_biases = output_buckets * sizeof(int32_t) * Network::N_L2;
    const size_t l2_weight_offset = l1_bias_offset + l1_biases;
    const size_t l2_weights = output_buckets * sizeof(float)
        * Network::N_L2 * Network::N_L3;
    const size_t l2_biases = output_buckets * sizeof(float) * Network::N_L3;
    const size_t output_weight_offset = l2_weight_offset + l2_weights + l2_biases;

    for (int bucket = 0; bucket < output_buckets; ++bucket) {
        write_u32_le(
            bytes,
            l1_bias_offset + static_cast<size_t>(bucket) * Network::N_L2 * sizeof(int32_t),
            32);
        write_f32_le(
            bytes,
            l2_weight_offset
                + static_cast<size_t>(bucket) * Network::N_L2 * Network::N_L3 * sizeof(float),
            1.0f);
        write_f32_le(
            bytes,
            output_weight_offset
                + static_cast<size_t>(bucket) * Network::N_L3 * sizeof(float),
            static_cast<float>(bucket + 1));
    }

    const auto path = fs::temp_directory_path() / std::string(name);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}

void overwrite_u32_le(const fs::path & path, size_t offset, uint32_t value) {
    std::array<char, sizeof(value)> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<char>((value >> (8 * i)) & 0xff);

    std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
    io.seekp(static_cast<std::streamoff>(offset));
    io.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

bool network_accumulators_match(const Network::Accumulator & a,
                              const Network::Accumulator & b) {
    for (int view = 0; view < 2; ++view) {
        for (size_t i = 0; i < static_cast<size_t>(Network::N_HIDDEN); ++i)
            if (a.values[view][i] != b.values[view][i]) return false;
    }
    return true;
}

constexpr int mirror_file(int sq) {
    return sq ^ 7;
}

constexpr std::array<int, 64> STOCKFISH_HALFKA_V2_HM_BUCKETS = {
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
     8,  9, 10, 11, 11, 10,  9,  8,
     4,  5,  6,  7,  7,  6,  5,  4,
     0,  1,  2,  3,  3,  2,  1,  0,
};

constexpr int stockfish_square_from_enyo(int sq) {
    return sq ^ 7;
}

constexpr int stockfish_halfka_v2_hm_orient(int stockfish_king_sq) {
    return (stockfish_king_sq & 4) == 0 ? 7 : 0;
}

constexpr int stockfish_halfka_v2_hm_channel(int pt, int color, int view) {
    if (pt == static_cast<int>(king))
        return 10;

    const int piece_type = pt - 1;
    const int enemy = color == view ? 0 : 1;
    return 2 * piece_type + enemy;
}

constexpr int enyo_channel_from_stockfish_halfka_v2_hm(int stockfish_channel) {
    if (stockfish_channel == 10)
        return 10;

    const int piece_type = stockfish_channel / 2;
    const int enemy = stockfish_channel & 1;
    return 5 * enemy + piece_type;
}

constexpr int stockfish_halfka_v2_hm_index_from_enyo(
    int pt,
    int color,
    int sq,
    int king_sq,
    int view)
{
    const int stockfish_sq = stockfish_square_from_enyo(sq);
    const int stockfish_king_sq = stockfish_square_from_enyo(king_sq);
    const int flip = 56 * view;
    const int orient = stockfish_halfka_v2_hm_orient(stockfish_king_sq);

    return STOCKFISH_HALFKA_V2_HM_BUCKETS[stockfish_king_sq ^ flip] * 11 * 64
        + stockfish_halfka_v2_hm_channel(pt, color, view) * 64
        + (stockfish_sq ^ orient ^ flip);
}

constexpr int enyo_order_from_stockfish_halfka_v2_hm_index(int stockfish_index) {
    const int stockfish_bucket = stockfish_index / (11 * 64);
    const int stockfish_channel = (stockfish_index / 64) % 11;
    const int stockfish_sq = stockfish_index % 64;

    return (31 - stockfish_bucket) * 11 * 64
        + enyo_channel_from_stockfish_halfka_v2_hm(stockfish_channel) * 64
        + (stockfish_sq ^ 56);
}

TEST(network_model, halfka_v2_feature_channels_merge_kings_only) {
    EXPECT_EQ(Network::FeatureCount(32, Network::HALFKA_V2_FEATURE_CHANNELS), 22528);
    EXPECT_EQ(Network::FeatureChannel(king, white, white, 11), 10);
    EXPECT_EQ(Network::FeatureChannel(king, black, white, 11), 10);
    EXPECT_EQ(Network::FeatureChannel(king, white, black, 11), 10);
    EXPECT_EQ(Network::FeatureChannel(king, black, black, 11), 10);
    EXPECT_EQ(Network::FeatureChannel(pawn, white, white, 11), 0);
    EXPECT_EQ(Network::FeatureChannel(queen, white, white, 11), 4);
    EXPECT_EQ(Network::FeatureChannel(pawn, black, white, 11), 5);
    EXPECT_EQ(Network::FeatureChannel(queen, black, white, 11), 9);
    EXPECT_EQ(Network::FeatureChannel(pawn, black, black, 11), 0);
    EXPECT_EQ(Network::FeatureChannel(queen, white, black, 11), 9);

    EXPECT_NE(
        Network::FeatureIdxFormula(king, white, 0, 4, white, 32, 12),
        Network::FeatureIdxFormula(king, black, 0, 4, white, 32, 12));
    EXPECT_EQ(
        Network::FeatureIdxFormula(king, white, 0, 4, white, 32, 11),
        Network::FeatureIdxFormula(king, black, 0, 4, white, 32, 11));
}

TEST(network_model, feature_index_uses_horizontal_mirroring) {
    for (const int feature_channels : {12, 11}) {
        const int input_buckets = feature_channels == 11 ? 32 : 16;
        for (int pt = static_cast<int>(pawn); pt <= static_cast<int>(king); ++pt) {
            for (int color = 0; color < 2; ++color) {
                for (int view = 0; view < 2; ++view) {
                    const int sq = 9;
                    const int king_sq = 3;
                    EXPECT_EQ(
                        Network::FeatureIdxFormula(pt, color, sq, king_sq, view,
                            input_buckets, feature_channels),
                        Network::FeatureIdxFormula(pt, color, mirror_file(sq),
                            mirror_file(king_sq), view, input_buckets,
                            feature_channels));
                }
            }
        }
    }
}

TEST(network_model, smaller_input_buckets_coarsen_16_bucket_layout) {
    for (int oriented_sq = 0; oriented_sq < 64; ++oriented_sq) {
        EXPECT_EQ(Network::KingBucketFor(1, oriented_sq), 0);
        EXPECT_EQ(
            Network::KingBucketFor(2, oriented_sq),
            Network::KingBucketFor(16, oriented_sq) / 8);
        EXPECT_EQ(
            Network::KingBucketFor(4, oriented_sq),
            Network::KingBucketFor(16, oriented_sq) / 4);
        EXPECT_EQ(
            Network::KingBucketFor(8, oriented_sq),
            Network::KingBucketFor(16, oriented_sq) / 2);
    }

    std::array<bool, 10> used_10{};
    for (int oriented_sq = 0; oriented_sq < 64; ++oriented_sq) {
        const int bucket = Network::KingBucketFor(10, oriented_sq);
        EXPECT_EQ(bucket, Network::KING_BUCKETS_10[oriented_sq]);
        ASSERT_GE(bucket, 0);
        ASSERT_LT(bucket, 10);
        used_10[static_cast<size_t>(bucket)] = true;
    }
    for (const bool used : used_10)
        EXPECT_TRUE(used);

    for (const int input_buckets : {1, 2, 4, 8, 10, 16}) {
        for (int pt = static_cast<int>(pawn); pt <= static_cast<int>(king); ++pt) {
            const int idx = Network::FeatureIdxFormula(
                pt, white, 9, 3, white, input_buckets, 12);
            EXPECT_GE(idx, 0);
            EXPECT_LT(idx, Network::FeatureCount(input_buckets, 12));
        }
    }
}

TEST(network_model, halfka_v2_hm_matches_stockfish_under_enyo_row_order) {
    for (int pt = static_cast<int>(pawn); pt <= static_cast<int>(king); ++pt) {
        for (int color = 0; color < 2; ++color) {
            for (int sq = 0; sq < 64; ++sq) {
                for (int king_sq = 0; king_sq < 64; ++king_sq) {
                    for (int view = 0; view < 2; ++view) {
                        const int enyo_index = Network::FeatureIdxFormula(
                            pt,
                            color,
                            sq,
                            king_sq,
                            view,
                            32,
                            Network::HALFKA_V2_FEATURE_CHANNELS);
                        const int stockfish_index = stockfish_halfka_v2_hm_index_from_enyo(
                            pt,
                            color,
                            sq,
                            king_sq,
                            view);

                        EXPECT_EQ(
                            enyo_index,
                            enyo_order_from_stockfish_halfka_v2_hm_index(stockfish_index));
                    }
                }
            }
        }
    }
}

TEST(network_model, detects_supported_bucket_counts) {
    for (const int input_buckets : {1, 2, 4, 8, 10, 16, 32}) {
        EXPECT_EQ(
            Network::DetectInputBuckets(Network::NetworkSize(input_buckets)),
            input_buckets);
        EXPECT_EQ(
            Network::DetectFeatureChannels(Network::NetworkSize(input_buckets)),
            12);
        EXPECT_EQ(Network::DetectOutputBuckets(Network::NetworkSize(input_buckets)), 1);
        EXPECT_TRUE(Network::IsSupportedNetworkSize(Network::NetworkSize(input_buckets)));
    }

    EXPECT_EQ(Network::DetectInputBuckets(Network::NetworkSize(32)), 32);
    EXPECT_EQ(
        Network::DetectFeatureChannels(Network::NetworkSize(32, 1, 0, 11)),
        11);
    EXPECT_EQ(Network::DetectInputBuckets(Network::NetworkSize(16, 4)), 16);
    EXPECT_EQ(Network::DetectOutputBuckets(Network::NetworkSize(16, 4)), 4);
    EXPECT_EQ(Network::DetectOutputHeadFeatures(Network::NetworkSize(16, 4)), 0);
    EXPECT_EQ(Network::DetectInputBuckets(Network::NetworkSize(16, 4, Network::N_HEAD_FEATURES)), 16);
    EXPECT_EQ(Network::DetectOutputBuckets(Network::NetworkSize(16, 4, Network::N_HEAD_FEATURES)), 4);
    EXPECT_EQ(
        Network::DetectOutputHeadFeatures(Network::NetworkSize(16, 4, Network::N_HEAD_FEATURES)),
        Network::N_HEAD_FEATURES);
    EXPECT_EQ(
        Network::DetectOutputHeadFeatures(Network::NetworkSize(
            16,
            4,
            Network::N_EXTENDED_HEAD_FEATURES)),
        Network::N_EXTENDED_HEAD_FEATURES);
    EXPECT_TRUE(Network::IsSupportedNetworkSize(Network::NetworkSize(32, 1, 0, 11)));
    EXPECT_TRUE(Network::IsSupportedNetworkSize(Network::NetworkSize(10, 1, 0, 11)));
    EXPECT_TRUE(Network::IsSupportedNetworkSize(Network::NetworkSize(16, 1, 0, 11)));
    EXPECT_TRUE(Network::IsSupportedNetworkSize(Network::NetworkSize(16, 4)));
    EXPECT_TRUE(Network::IsSupportedNetworkSize(Network::NetworkSize(16, 4, Network::N_HEAD_FEATURES)));
    EXPECT_TRUE(Network::IsSupportedNetworkSize(Network::NetworkSize(
        16,
        4,
        Network::N_EXTENDED_HEAD_FEATURES)));
    EXPECT_EQ(Network::DetectInputBuckets(Network::NetworkSize(16) + 1), 0);
    EXPECT_EQ(Network::DetectOutputBuckets(Network::NetworkSize(16) + 1), 0);
    EXPECT_LT(Network::NetworkSize(16), Network::NetworkSize(32));
}

TEST(network_model, rejects_oversized_legacy_networks) {
    EXPECT_TRUE(NNUE::IsSupportedLegacyNetworkSize(NNUE::LEGACY_NETWORK_SIZE));
    EXPECT_TRUE(NNUE::IsSupportedLegacyNetworkSize(NNUE::LEGACY_NETWORK_FILE_SIZE));
    EXPECT_FALSE(NNUE::IsSupportedLegacyNetworkSize(NNUE::LEGACY_NETWORK_SIZE - 1));
    EXPECT_FALSE(NNUE::IsSupportedLegacyNetworkSize(NNUE::LEGACY_NETWORK_FILE_SIZE + 1));
    EXPECT_FALSE(NNUE::IsSupportedLegacyNetworkSize(25201924));
}

TEST(network_model, loads_quantized_dense_network_blob) {
    EXPECT_EQ(Network::QuantizedNetworkSize(), 25201924U);
    const auto path = write_quantized_test_network_blob(
        "enyo_quantized_dense.nn");
    const auto path_string = path.string();

    ASSERT_TRUE(Network::IsSupportedQuantizedNetworkSize(fs::file_size(path)));
    ASSERT_TRUE(Network::LoadNetwork(path_string.c_str()));
    EXPECT_EQ(Network::DENSE_LAYER_FORMAT, Network::DenseLayerFormat::Quantized);
    EXPECT_EQ(Network::INPUT_BUCKETS, 16);
    EXPECT_EQ(Network::FEATURE_CHANNELS, 12);
    EXPECT_EQ(Network::OUTPUT_BUCKETS, 1);

    Network::Accumulator accumulator{};
    EXPECT_EQ(Network::Propagate(&accumulator, 0), 2);

    fs::remove(path);
    ensure_network_mock_weights();
}

TEST(network_model, loads_versioned_narrow_hidden_network_blob) {
    const auto path = write_zero_v2_network_blob(
        10,
        Network::HALFKA_V2_FEATURE_CHANNELS,
        512,
        8,
        "enyo_zero_v2_10x11x512_o8.nn");
    const auto path_string = path.string();
    ASSERT_TRUE(Network::IsSupportedNetworkSize(fs::file_size(path)));
    ASSERT_TRUE(Network::LoadNetwork(path_string.c_str()));
    EXPECT_EQ(Network::INPUT_BUCKETS, 10);
    EXPECT_EQ(Network::FEATURE_CHANNELS, 11);
    EXPECT_EQ(Network::TRAINED_HIDDEN, 512);
    EXPECT_EQ(Network::OUTPUT_BUCKETS, 8);
    EXPECT_NE(Network::INPUT_WEIGHTS, nullptr);
    fs::remove(path);
    ensure_network_mock_weights();
}

TEST(network_model, loads_1_bucket_network_blob) {
    const auto path = write_zero_network_blob(
        Network::NetworkSize(1),
        "enyo_zero_1_bucket.nn");
    const auto path_string = path.string();
    ASSERT_TRUE(Network::LoadNetwork(path_string.c_str()));
    EXPECT_EQ(Network::INPUT_BUCKETS, 1);
    EXPECT_EQ(Network::FEATURE_CHANNELS, 12);
    EXPECT_EQ(
        Network::FeatureIdx(pawn, white, 9, 3, white),
        Network::FeatureIdxFormula(pawn, white, 9, 3, white, 1, 12));
    EXPECT_NE(Network::INPUT_WEIGHTS, nullptr);
    fs::remove(path);
    ensure_network_mock_weights();
}

TEST(network_model, loads_32_bucket_network_blob) {
    const auto path = write_zero_network_blob(
        Network::NetworkSize(32),
        "enyo_zero_32_bucket.nn");
    const auto path_string = path.string();
    ASSERT_TRUE(Network::LoadNetwork(path_string.c_str()));
    EXPECT_EQ(Network::INPUT_BUCKETS, 32);
    EXPECT_NE(Network::INPUT_WEIGHTS, nullptr);
    fs::remove(path);
    ensure_network_mock_weights();
}

TEST(network_model, loads_halfka_v2_channel_network_blob) {
    const auto path = write_zero_network_blob(
        Network::NetworkSize(32, 1, 0, Network::HALFKA_V2_FEATURE_CHANNELS),
        "enyo_zero_32_bucket_11_channel.nn");
    const auto path_string = path.string();
    ASSERT_TRUE(Network::LoadNetwork(path_string.c_str()));
    EXPECT_EQ(Network::INPUT_BUCKETS, 32);
    EXPECT_EQ(Network::FEATURE_CHANNELS, 11);
    EXPECT_NE(Network::INPUT_WEIGHTS, nullptr);
    fs::remove(path);
    ensure_network_mock_weights();
}

TEST(network_model, loads_output_bucket_network_blob) {
    const auto path = write_zero_network_blob(
        Network::NetworkSize(16, 4),
        "enyo_zero_16_input_4_output_bucket.nn");
    const auto path_string = path.string();
    ASSERT_TRUE(Network::LoadNetwork(path_string.c_str()));
    EXPECT_EQ(Network::INPUT_BUCKETS, 16);
    EXPECT_EQ(Network::OUTPUT_BUCKETS, 4);
    EXPECT_EQ(Network::OUTPUT_HEAD_FEATURES, 0);
    EXPECT_EQ(Network::OUTPUT_WIDTH, Network::N_L3);
    EXPECT_NE(Network::OUTPUT_WEIGHTS, nullptr);
    EXPECT_NE(Network::OUTPUT_BIASES, nullptr);
    fs::remove(path);
    ensure_network_mock_weights();
}

TEST(network_model, full_head_v3_selects_every_dense_stack) {
    const auto path = write_full_head_v3_network_blob("enyo_full_head_v3.nn");
    const auto path_string = path.string();
    ASSERT_TRUE(Network::LoadNetwork(path_string.c_str()));
    ASSERT_TRUE(Network::FULL_HEADS_ENABLED);
    EXPECT_EQ(Network::OUTPUT_BUCKETS, 8);

    Network::Accumulator accumulator{};
    for (int bucket = 0; bucket < 8; ++bucket)
        EXPECT_EQ(Network::Propagate(&accumulator, 0, bucket), bucket + 1);

    fs::remove(path);
    ensure_network_mock_weights();
}

TEST(network_model, rejects_v3_without_full_head_flag) {
    const auto path = write_full_head_v3_network_blob("enyo_invalid_v3_missing_flag.nn");
    overwrite_u32_le(path, 48, 0);
    EXPECT_FALSE(Network::LoadNetwork(path.string().c_str()));
    fs::remove(path);
    ensure_network_mock_weights();
}

TEST(network_model, rejects_v2_with_full_head_flag) {
    const auto path = write_full_head_v3_network_blob("enyo_invalid_v2_full_head_flag.nn");
    {
        std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
        io.write(
            reinterpret_cast<const char*>(Network::NETWORK_HEADER_MAGIC.data()),
            static_cast<std::streamsize>(Network::NETWORK_HEADER_MAGIC.size()));
    }
    overwrite_u32_le(path, 8, Network::NETWORK_FORMAT_VERSION);
    EXPECT_FALSE(Network::LoadNetwork(path.string().c_str()));
    fs::remove(path);
    ensure_network_mock_weights();
}

TEST(network_model, loads_material_phase_head_network_blob) {
    const auto path = write_zero_network_blob(
        Network::NetworkSize(16, 4, Network::N_HEAD_FEATURES),
        "enyo_zero_16_input_4_output_bucket_head.nn");
    const auto path_string = path.string();
    ASSERT_TRUE(Network::LoadNetwork(path_string.c_str()));
    EXPECT_EQ(Network::INPUT_BUCKETS, 16);
    EXPECT_EQ(Network::OUTPUT_BUCKETS, 4);
    EXPECT_EQ(Network::OUTPUT_HEAD_FEATURES, Network::N_HEAD_FEATURES);
    EXPECT_EQ(Network::OUTPUT_WIDTH, Network::N_L3 + Network::N_HEAD_FEATURES);
    EXPECT_NE(Network::OUTPUT_WEIGHTS, nullptr);
    EXPECT_NE(Network::OUTPUT_BIASES, nullptr);
    fs::remove(path);
    ensure_network_mock_weights();
}

TEST(network_model, loads_extended_output_head_network_blob) {
    const auto path = write_zero_network_blob(
        Network::NetworkSize(16, 4, Network::N_EXTENDED_HEAD_FEATURES),
        "enyo_zero_16_input_4_output_bucket_extended_head.nn");
    const auto path_string = path.string();
    ASSERT_TRUE(Network::LoadNetwork(path_string.c_str()));
    EXPECT_EQ(Network::INPUT_BUCKETS, 16);
    EXPECT_EQ(Network::OUTPUT_BUCKETS, 4);
    EXPECT_EQ(Network::OUTPUT_HEAD_FEATURES, Network::N_EXTENDED_HEAD_FEATURES);
    EXPECT_EQ(Network::OUTPUT_WIDTH, Network::N_L3 + Network::N_EXTENDED_HEAD_FEATURES);
    EXPECT_NE(Network::OUTPUT_WEIGHTS, nullptr);
    EXPECT_NE(Network::OUTPUT_BIASES, nullptr);
    fs::remove(path);
    ensure_network_mock_weights();
}

TEST(network_model, material_count_bucket_matches_bullet_formula) {
    EXPECT_EQ(Network::OutputBucketForPieceCount(32, 4), 3);
    EXPECT_EQ(Network::OutputBucketForPieceCount(31, 4), 3);
    EXPECT_EQ(Network::OutputBucketForPieceCount(30, 4), 3);
    EXPECT_EQ(Network::OutputBucketForPieceCount(24, 4), 2);
    EXPECT_EQ(Network::OutputBucketForPieceCount(16, 4), 1);
    EXPECT_EQ(Network::OutputBucketForPieceCount(8, 4), 0);
    EXPECT_EQ(Network::OutputBucketForPieceCount(2, 4), 0);
    EXPECT_EQ(Network::OutputBucketForPieceCount(32, 1), 0);
}

TEST(network_model, material_head_features_are_normalized) {
    Network::MaterialSummary summary;
    summary.phase = 20;
    summary.piece_count = 32;
    summary.pawn_count = 16;
    summary.minor_count = 8;
    summary.rook_count = 4;
    summary.queen_count = 2;
    const auto features = Network::MaterialHeadFeatures(summary);
    EXPECT_FLOAT_EQ(features.values[Network::HEAD_PHASE_DELTA], 20.0f / 128.0f);
    EXPECT_FLOAT_EQ(features.values[Network::HEAD_PIECE_COUNT], 1.0f);
    EXPECT_FLOAT_EQ(features.values[Network::HEAD_PAWN_COUNT], 0.0f);
    EXPECT_FLOAT_EQ(features.values[Network::HEAD_MINOR_COUNT], 0.0f);
    EXPECT_FLOAT_EQ(features.values[Network::HEAD_ROOK_COUNT], 0.0f);
    EXPECT_FLOAT_EQ(features.values[Network::HEAD_QUEEN_COUNT], 0.0f);
    EXPECT_FLOAT_EQ(features.values[Network::HEAD_NON_PAWN_COUNT], 0.0f);
    EXPECT_FLOAT_EQ(features.values[Network::HEAD_PAWN_PHASE], 0.0f);
}

TEST(stockfish_nnue, current_architecture_hash) {
    EXPECT_EQ(NNUE::Stockfish::FeatureTransformerHash(), 0x6165ddc9U);
    EXPECT_EQ(NNUE::Stockfish::ArchitectureHash(), 0x63337116U);
    EXPECT_EQ(NNUE::Stockfish::NetworkHash(), 0x0256acdfU);
}

TEST(stockfish_nnue, format_probe_distinguishes_invalid_from_unknown) {
    const auto path = fs::temp_directory_path() / "enyo_stockfish_probe.nnue";
    std::vector<char> bytes(12, 0);
    write_u32_le(bytes, 0, NNUE::Stockfish::format_version);
    write_u32_le(bytes, 4, NNUE::Stockfish::NetworkHash() ^ 1U);
    write_u32_le(bytes, 8, 0);
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    auto result = NNUE::Stockfish::LoadNetwork(path.c_str());
    EXPECT_EQ(result.status, NNUE::LoadStatus::invalid);
    EXPECT_EQ(result.error, "unsupported Stockfish NNUE architecture hash");

    write_u32_le(bytes, 0, NNUE::Stockfish::format_version ^ 1U);
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    result = NNUE::Stockfish::LoadNetwork(path.c_str());
    EXPECT_EQ(result.status, NNUE::LoadStatus::not_recognized);
    fs::remove(path);
}

TEST(stockfish_nnue, production_net_matches_reference_evaluations) {
    const char * path = std::getenv("ENYO_STOCKFISH_NNUE_TEST_FILE");
    if (!path || !fs::exists(path))
        GTEST_SKIP() << "ENYO_STOCKFISH_NNUE_TEST_FILE is not available";

    const auto loaded = NNUE::Stockfish::LoadNetwork(path);
    ASSERT_EQ(loaded.status, NNUE::LoadStatus::loaded) << loaded.error;
    Network::enabled = false;

    struct PositionCase {
        const char * fen;
        int expected;
    };
    constexpr PositionCase cases[] = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6},
        {"r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPP1BPPP/R2Q1RK1 w kq - 0 10", -541},
        {"rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 2", 50},
        {"8/2p5/3p4/1P1Pp1k1/4P3/5K2/8/8 w - - 0 40", -242},
    };

    for (const auto & test : cases) {
        Board board(test.fen);
        NNUE::Stockfish::State state;
        EXPECT_EQ(state.Evaluate(board, 0), test.expected) << test.fen;
    }
    NNUE::Stockfish::Disable();
}

TEST(stockfish_nnue, current_net_matches_reference_evaluations) {
    const char * path = std::getenv("ENYO_STOCKFISH_NNUE_V15_TEST_FILE");
    if (!path || !fs::exists(path))
        GTEST_SKIP() << "ENYO_STOCKFISH_NNUE_V15_TEST_FILE is not available";

    const auto loaded = NNUE::Stockfish::LoadNetwork(path);
    ASSERT_EQ(loaded.status, NNUE::LoadStatus::loaded) << loaded.error;
    Network::enabled = false;

    constexpr std::pair<const char *, int> cases[] = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5},
        {"r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPP1BPPP/R2Q1RK1 w kq - 0 10", -461},
        {"rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 2", 83},
        {"8/2p5/3p4/1P1Pp1k1/4P3/5K2/8/8 w - - 0 40", -204},
    };

    for (const auto & [fen, expected] : cases) {
        Board board(fen);
        NNUE::Stockfish::State state;
        EXPECT_EQ(state.Evaluate(board, 0), expected) << fen;
    }
    NNUE::Stockfish::Disable();
}

template<Color Us>
void expect_stockfish_move_matches_refresh(Board & board, Move move) {
    NNUE::Net live;
    live.refresh(board);
    (void)live.EvaluateStockfish(board);

    apply_move<Us, true, true>(board, move, &live);
    const int incremental = live.EvaluateStockfish(board);

    NNUE::Net fresh;
    fresh.refresh(board);
    EXPECT_EQ(incremental, fresh.EvaluateStockfish(board))
        << board.fen() << " after " << fmt::format("{}", move);
}

TEST(stockfish_nnue, special_moves_match_fresh_accumulators) {
    const char * path = std::getenv("ENYO_STOCKFISH_NNUE_TEST_FILE");
    if (!path || !fs::exists(path))
        GTEST_SKIP() << "ENYO_STOCKFISH_NNUE_TEST_FILE is not available";

    const auto loaded = NNUE::Stockfish::LoadNetwork(path);
    ASSERT_EQ(loaded.status, NNUE::LoadStatus::loaded) << loaded.error;
    Network::enabled = false;

    {
        Board board{"r3k2r/pppqppbp/2np1np1/8/8/2NP1NP1/PPPQPPBP/R3K2R w KQkq - 0 1"};
        expect_stockfish_move_matches_refresh<white>(
            board, resolve_move<white>(board, king, e1, g1));
    }
    {
        Board board{"rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"};
        expect_stockfish_move_matches_refresh<white>(
            board, resolve_move<white>(board, pawn, e5, f6));
    }
    {
        Board board{"1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1"};
        Move promotion{};
        for (const Move move : generate_legal_moves<white>(board))
            if (move.flags() == Move::Flags::promote
                && move.dst_sq() == b8
                && move.promo_piece() == queen) {
                promotion = move;
                break;
            }
        ASSERT_TRUE(promotion);
        expect_stockfish_move_matches_refresh<white>(board, promotion);
    }

    NNUE::Stockfish::Disable();
}

TEST(stockfish_nnue, null_move_preserves_parent_accumulator) {
    const char * path = std::getenv("ENYO_STOCKFISH_NNUE_TEST_FILE");
    if (!path || !fs::exists(path))
        GTEST_SKIP() << "ENYO_STOCKFISH_NNUE_TEST_FILE is not available";

    const auto loaded = NNUE::Stockfish::LoadNetwork(path);
    ASSERT_EQ(loaded.status, NNUE::LoadStatus::loaded) << loaded.error;
    Network::enabled = false;

    Board board{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"};
    NNUE::Net live;
    live.refresh(board);
    (void)live.EvaluateStockfish(board);
    live.push(board);
    apply_null_move<white>(board);
    (void)live.EvaluateStockfish(board);
    revert_null_move<white>(board);
    live.pop();

    const Move move = resolve_move<white>(board, pawn, e2, e4);
    apply_move<white, true, true>(board, move, &live);
    const int incremental = live.EvaluateStockfish(board);
    NNUE::Net fresh;
    fresh.refresh(board);
    EXPECT_EQ(incremental, fresh.EvaluateStockfish(board));
    NNUE::Stockfish::Disable();
}

TEST(stockfish_nnue, incremental_accumulator_matches_refresh) {
    const char * path = std::getenv("ENYO_STOCKFISH_NNUE_TEST_FILE");
    if (!path || !fs::exists(path))
        GTEST_SKIP() << "ENYO_STOCKFISH_NNUE_TEST_FILE is not available";

    const auto loaded = NNUE::Stockfish::LoadNetwork(path);
    ASSERT_EQ(loaded.status, NNUE::LoadStatus::loaded) << loaded.error;
    Network::enabled = false;

    Board board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    NNUE::Net live;
    live.refresh(board);
    EXPECT_EQ(live.EvaluateStockfish(board), 6);

    const auto verify = [&] {
        NNUE::Net refreshed;
        refreshed.refresh(board);
        EXPECT_EQ(live.EvaluateStockfish(board), refreshed.EvaluateStockfish(board));
    };

    apply_move<white, true, true>(
        board, resolve_move<white>(board, pawn, e2, e4), &live);
    verify();
    apply_move<black, true, true>(
        board, resolve_move<black>(board, pawn, c7, c5), &live);
    verify();
    apply_move<white, true, true>(
        board, resolve_move<white>(board, knight, g1, f3), &live);
    verify();
    apply_move<black, true, true>(
        board, resolve_move<black>(board, pawn, d7, d6), &live);
    verify();
    apply_move<white, true, true>(
        board, resolve_move<white>(board, pawn, d2, d4), &live);
    verify();
    apply_move<black, true, true>(
        board, resolve_move<black>(board, pawn, c5, d4), &live);
    verify();

    NNUE::Stockfish::Disable();
}

void materialize_network(Board & b, NNUE::Net & net) {
    net.ensure_network(b);
}

template <Color Us>
void audit_network_lazy_tree(Board & b, NNUE::Net & live, int depth)
{
    if (depth == 0) {
        materialize_network(b, live);
        NNUE::Net fresh;
        fresh.refresh(b);
        EXPECT_TRUE(network_accumulators_match(
            live.network_accumulator_stack[live.currentAccumulator],
            fresh.network_accumulator_stack[fresh.currentAccumulator]))
            << "active Network lazy tree diverged; fen after: " << b.fen();
        return;
    }

    auto moves = generate_legal_moves<Us>(b);
    int searched = 0;
    for (auto move : moves) {
        apply_move<Us, true, true>(b, move, &live);
        audit_network_lazy_tree<~Us>(b, live, depth - 1);
        revert_move<Us, true, true>(b, &live);
        if (++searched >= 12)
            break;
    }
}

struct ScopedNetworkEnabled {
    bool previous;

    ScopedNetworkEnabled()
        : previous(Network::enabled)
    {
        Network::enabled = true;
    }

    ~ScopedNetworkEnabled() {
        Network::enabled = previous;
    }
};

void expect_network_matches_refresh(const char * label, Board & b, Move move, Color us) {
    ensure_network_mock_weights();

    NNUE::Net live;
    live.refresh(b);

    if (us == white)
        apply_move<white, true, true>(b, move, &live);
    else
        apply_move<black, true, true>(b, move, &live);
    materialize_network(b, live);

    const auto after_live = live.network_accumulator_stack[live.currentAccumulator];

    NNUE::Net fresh;
    fresh.refresh(b);
    const auto after_fresh = fresh.network_accumulator_stack[fresh.currentAccumulator];

    EXPECT_TRUE(network_accumulators_match(after_live, after_fresh))
        << label << ": Network incremental accumulator diverged from full refresh after "
        << fmt::format("{}", move) << " (fen after: " << b.fen() << ")";
}

template <Color Us>
void expect_incremental_matches_refresh(const char * label, Board & b, Move move) {
    NNUE::Net live;
    live.refresh(b);

    apply_move<Us, true, true>(b, move, &live);

    NNUE::Accumulator after_live = live.accumulator_stack[live.currentAccumulator];

    NNUE::Net fresh;
    fresh.refresh(b);
    NNUE::Accumulator after_fresh = fresh.accumulator_stack[fresh.currentAccumulator];

    EXPECT_TRUE(accumulators_match(after_live, after_fresh))
        << label << ": incremental accumulator diverged from full refresh after "
        << fmt::format("{}", move) << " (fen after: " << b.fen() << ")";
}

} // anon ns

TEST(nnue_audit, quiet_pawn_push) {
    Board b{"startpos"};
    auto move = resolve_move<white>(b, pawn, e2, e4);
    expect_incremental_matches_refresh<white>("quiet_pawn_push", b, move);
}

TEST(nnue_audit, quiet_knight_move) {
    Board b{"startpos"};
    auto move = resolve_move<white>(b, knight, g1, f3);
    expect_incremental_matches_refresh<white>("quiet_knight_move", b, move);
}

TEST(nnue_audit, capture) {
    // Scandinavian: 1.e4 d5 2.exd5 — white pawn captures black pawn
    Board b{"rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"};
    auto move = resolve_move<white>(b, pawn, e4, d5);
    expect_incremental_matches_refresh<white>("capture", b, move);
}

TEST(nnue_audit, promotion_no_capture) {
    // White pawn on a7 promotes to queen on a8; no capture.
    Board b{"4k3/P7/8/8/8/8/8/4K3 w - - 0 1"};
    auto moves = generate_legal_moves<white>(b);
    Move promo{};
    for (auto m : moves) {
        if (m.flags() == Move::Flags::promote
         && m.src_sq() == a7 && m.dst_sq() == a8
         && m.promo_piece() == queen) {
            promo = m; break;
        }
    }
    ASSERT_TRUE(promo) << "could not find promotion move";
    expect_incremental_matches_refresh<white>("promotion_no_capture", b, promo);
}

TEST(nnue_audit, promotion_with_capture) {
    // White pawn on a7 captures on b8 and promotes to queen.
    Board b{"1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1"};
    auto moves = generate_legal_moves<white>(b);
    Move promo{};
    for (auto m : moves) {
        if (m.flags() == Move::Flags::promote
         && m.src_sq() == a7 && m.dst_sq() == b8
         && m.promo_piece() == queen) {
            promo = m; break;
        }
    }
    ASSERT_TRUE(promo) << "could not find capture-promotion move";
    expect_incremental_matches_refresh<white>("promotion_with_capture", b, promo);
}

TEST(nnue_audit, enpassant) {
    // After 1.e4 d5 2.e5 f5 — white can ep-capture on f6.
    Board b{"rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"};
    auto moves = generate_legal_moves<white>(b);
    Move ep{};
    for (auto m : moves) {
        if (m.flags() == Move::Flags::enpassant) { ep = m; break; }
    }
    ASSERT_TRUE(ep) << "could not find en-passant move";
    expect_incremental_matches_refresh<white>("enpassant", b, ep);
}

TEST(nnue_audit, castle_kingside) {
    Board b{"r3k2r/pppqppbp/2np1np1/8/8/2NP1NP1/PPPQPPBP/R3K2R w KQkq - 0 1"};
    auto moves = generate_legal_moves<white>(b);
    Move castle{};
    for (auto m : moves) {
        if (m.flags() == Move::Flags::castle && m.dst_sq() < m.src_sq()) {
            castle = m; break;
        }
    }
    ASSERT_TRUE(castle) << "could not find kingside castle move";
    expect_incremental_matches_refresh<white>("castle_kingside", b, castle);
}

TEST(nnue_audit, castle_queenside) {
    Board b{"r3k2r/pppqppbp/2np1np1/8/8/2NP1NP1/PPPQPPBP/R3K2R w KQkq - 0 1"};
    auto moves = generate_legal_moves<white>(b);
    Move castle{};
    for (auto m : moves) {
        if (m.flags() == Move::Flags::castle && m.dst_sq() > m.src_sq()) {
            castle = m; break;
        }
    }
    ASSERT_TRUE(castle) << "could not find queenside castle move";
    expect_incremental_matches_refresh<white>("castle_queenside", b, castle);
}

TEST(nnue_audit, king_move_non_castle) {
    // Plain king move: the king-bucket index changes, so the live path
    // triggers refresh_with_cache. This test locks in correct behavior.
    Board b{"4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"};
    auto move = resolve_move<white>(b, king, e1, e2);
    expect_incremental_matches_refresh<white>("king_move_non_castle", b, move);
}

// Apply A, revert A, apply B (sibling) — verify sibling's accumulator
// matches a fresh refresh. This is the exact pattern a move loop uses.
TEST(nnue_audit, apply_revert_apply_sibling) {
    Board b{"startpos"};
    NNUE::Net live;
    live.refresh(b);

    auto a = resolve_move<white>(b, pawn, e2, e4);
    apply_move<white, true, true>(b, a, &live);
    revert_move<white, true, true>(b, &live);

    auto c = resolve_move<white>(b, pawn, d2, d4);
    apply_move<white, true, true>(b, c, &live);
    NNUE::Accumulator after_c_live = live.accumulator_stack[live.currentAccumulator];

    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(accumulators_match(after_c_live, fresh.accumulator_stack[fresh.currentAccumulator]))
        << "sibling apply after revert diverged; fen: " << b.fen();
}

// Deeper mimic of search: apply A, apply B, revert B, apply C — then
// unwind. Verify accumulators are correct at every visible step.
TEST(nnue_audit, nested_apply_revert_sibling) {
    Board b{"startpos"};
    NNUE::Net live;
    live.refresh(b);

    auto a = resolve_move<white>(b, pawn, e2, e4);
    apply_move<white, true, true>(b, a, &live);

    auto b1 = resolve_move<black>(b, pawn, e7, e5);
    apply_move<black, true, true>(b, b1, &live);
    revert_move<black, true, true>(b, &live);

    auto b2 = resolve_move<black>(b, pawn, c7, c5);
    apply_move<black, true, true>(b, b2, &live);
    NNUE::Accumulator live_b2 = live.accumulator_stack[live.currentAccumulator];

    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(accumulators_match(live_b2, fresh.accumulator_stack[fresh.currentAccumulator]))
        << "nested sibling diverged at ply 2; fen: " << b.fen();

    revert_move<black, true, true>(b, &live);
    revert_move<white, true, true>(b, &live);
    NNUE::Net root_fresh;
    root_fresh.refresh(b);
    EXPECT_TRUE(accumulators_match(live.accumulator_stack[live.currentAccumulator],
                                   root_fresh.accumulator_stack[root_fresh.currentAccumulator]))
        << "full unwind diverged; fen: " << b.fen();
}

// Deep recursion mimicking real search: at each ply, verify the live
// accumulator matches a fresh refresh. Tries several siblings per ply.
TEST(nnue_audit, deep_recursion_slots_consistent) {
    Board b{"r3k2r/pppqppbp/2np1np1/8/8/2NP1NP1/PPPQPPBP/R3K2R w KQkq - 0 1"};
    NNUE::Net live;
    live.refresh(b);

    auto verify_slot = [&](const char * tag) {
        NNUE::Net fresh;
        fresh.refresh(b);
        ASSERT_TRUE(accumulators_match(live.accumulator_stack[live.currentAccumulator],
                                       fresh.accumulator_stack[fresh.currentAccumulator]))
            << tag << " slot diverged; fen: " << b.fen()
            << " currentAccumulator=" << live.currentAccumulator;
    };

    // ply 0 (root): white
    auto w1 = resolve_move<white>(b, knight, c3, e4);
    apply_move<white, true, true>(b, w1, &live); verify_slot("w1");

    // ply 1 (black), try sibling B1
    auto b1 = resolve_move<black>(b, knight, f6, e4);
    apply_move<black, true, true>(b, b1, &live); verify_slot("b1");

    // ply 2 (white)
    auto w2 = resolve_move<white>(b, knight, f3, e5);
    apply_move<white, true, true>(b, w2, &live); verify_slot("w2");
    revert_move<white, true, true>(b, &live); verify_slot("revert w2");

    // sibling w2'
    auto w2b = resolve_move<white>(b, pawn, d3, e4);
    apply_move<white, true, true>(b, w2b, &live); verify_slot("w2'");

    // unwind all
    revert_move<white, true, true>(b, &live); verify_slot("revert w2'");
    revert_move<black, true, true>(b, &live); verify_slot("revert b1");

    // ply 1 sibling b1'
    auto b1b = resolve_move<black>(b, pawn, d6, d5);
    apply_move<black, true, true>(b, b1b, &live); verify_slot("b1'");
    revert_move<black, true, true>(b, &live); verify_slot("revert b1'");

    revert_move<white, true, true>(b, &live); verify_slot("revert w1");
}

// Mixed move types in a nested tree: capture → promotion → king-move →
// castle. Each of these hits a different apply/revert path.
TEST(nnue_audit, mixed_move_types_nested) {
    // Position set up so white can: (a) capture b4 with pawn a3xb4,
    // and also has castling rights for later.
    Board b{"r3k2r/pppqppbp/2np1np1/8/1p6/P1NP1NP1/1PPQPPBP/R3K2R w KQkq - 0 1"};
    NNUE::Net live;
    live.refresh(b);

    auto verify = [&](const char * tag) {
        NNUE::Net fresh;
        fresh.refresh(b);
        ASSERT_TRUE(accumulators_match(live.accumulator_stack[live.currentAccumulator],
                                       fresh.accumulator_stack[fresh.currentAccumulator]))
            << tag << "; fen: " << b.fen() << " cur=" << live.currentAccumulator;
    };

    // capture
    auto cap = resolve_move<white>(b, pawn, a3, b4);
    apply_move<white, true, true>(b, cap, &live); verify("capture");

    // black quiet
    auto bq = resolve_move<black>(b, pawn, e7, e6);
    apply_move<black, true, true>(b, bq, &live); verify("black quiet");

    // white castle kingside
    auto moves = generate_legal_moves<white>(b);
    Move castle{};
    for (auto m : moves) {
        if (m.flags() == Move::Flags::castle && m.dst_sq() < m.src_sq()) {
            castle = m; break;
        }
    }
    ASSERT_TRUE(castle);
    apply_move<white, true, true>(b, castle, &live); verify("castle");

    // unwind
    revert_move<white, true, true>(b, &live); verify("revert castle");
    revert_move<black, true, true>(b, &live); verify("revert black quiet");
    revert_move<white, true, true>(b, &live); verify("revert capture");
}

// Apply A, revert A, apply B — verify slot N (after revert) still matches
// a from-scratch refresh on the reverted board. Catches bugs where
// revert_move corrupts the popped slot.
TEST(nnue_audit, revert_restores_slot) {
    Board start{"startpos"};
    NNUE::Net live;
    live.refresh(start);

    const NNUE::Accumulator before = live.accumulator_stack[live.currentAccumulator];

    auto a = resolve_move<white>(start, pawn, e2, e4);
    apply_move<white, true, true>(start, a, &live);
    revert_move<white, true, true>(start, &live);

    const NNUE::Accumulator after_revert = live.accumulator_stack[live.currentAccumulator];
    EXPECT_TRUE(accumulators_match(before, after_revert))
        << "revert corrupted slot; fen: " << start.fen();

    NNUE::Net fresh;
    fresh.refresh(start);
    EXPECT_TRUE(accumulators_match(after_revert, fresh.accumulator_stack[fresh.currentAccumulator]))
        << "post-revert slot != fresh refresh; fen: " << start.fen();
}

// Deep-sequence audit: apply eight moves, each time verifying the live
// accumulator matches a full refresh. Flushes out any delta-stacking bug
// that a single-move test would miss.
TEST(nnue_audit, opening_sequence) {
    Board b{"startpos"};
    NNUE::Net live;
    live.refresh(b);

    struct Step { Color us; PieceType pt; square_t from; square_t to; };
    const Step seq[] = {
        { white, pawn,   e2, e4 },
        { black, pawn,   e7, e5 },
        { white, knight, g1, f3 },
        { black, knight, b8, c6 },
        { white, bishop, f1, b5 },
        { black, pawn,   a7, a6 },
        { white, bishop, b5, a4 },
        { black, knight, g8, f6 },
    };

    int step_idx = 0;
    for (auto const & s : seq) {
        Move m = (s.us == white)
            ? resolve_move<white>(b, s.pt, s.from, s.to)
            : resolve_move<black>(b, s.pt, s.from, s.to);

        if (s.us == white) apply_move<white, true, true>(b, m, &live);
        else               apply_move<black, true, true>(b, m, &live);

        NNUE::Accumulator after_live = live.accumulator_stack[live.currentAccumulator];
        NNUE::Net fresh;
        fresh.refresh(b);
        NNUE::Accumulator after_fresh = fresh.accumulator_stack[fresh.currentAccumulator];
        EXPECT_TRUE(accumulators_match(after_live, after_fresh))
            << "opening_sequence step " << step_idx << " ("
            << fmt::format("{}", m) << ") diverged; fen after: " << b.fen();
        ++step_idx;
    }
}
#endif

TEST(network_audit, opening_sequence_matches_refresh) {
    Board b{"startpos"};
    ensure_network_mock_weights();
    NNUE::Net live;
    live.refresh(b);

    struct Step { Color us; PieceType pt; square_t from; square_t to; };
    const Step seq[] = {
        { white, pawn,   e2, e4 },
        { black, pawn,   e7, e5 },
        { white, knight, g1, f3 },
        { black, knight, b8, c6 },
        { white, bishop, f1, b5 },
        { black, pawn,   a7, a6 },
        { white, bishop, b5, a4 },
        { black, knight, g8, f6 },
    };

    int step_idx = 0;
    for (auto const & s : seq) {
        Move m = (s.us == white)
            ? resolve_move<white>(b, s.pt, s.from, s.to)
            : resolve_move<black>(b, s.pt, s.from, s.to);

        if (s.us == white)
            apply_move<white, true, true>(b, m, &live);
        else
            apply_move<black, true, true>(b, m, &live);
        materialize_network(b, live);

        NNUE::Net fresh;
        fresh.refresh(b);
        EXPECT_TRUE(network_accumulators_match(
            live.network_accumulator_stack[live.currentAccumulator],
            fresh.network_accumulator_stack[fresh.currentAccumulator]))
            << "Network opening_sequence step " << step_idx << " ("
            << fmt::format("{}", m) << ") diverged; fen after: " << b.fen();
        ++step_idx;
    }
}

TEST(network_audit, active_quiet_delta_matches_refresh) {
    Board b{"startpos"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);

    struct Step { Color us; PieceType pt; square_t from; square_t to; };
    const Step seq[] = {
        { white, pawn,   e2, e4 },
        { black, pawn,   e7, e5 },
        { white, knight, g1, f3 },
        { black, knight, b8, c6 },
        { white, bishop, f1, b5 },
        { black, pawn,   a7, a6 },
        { white, bishop, b5, a4 },
        { black, knight, g8, f6 },
    };

    int step_idx = 0;
    for (auto const & s : seq) {
        Move m = s.us == white
            ? resolve_move<white>(b, s.pt, s.from, s.to)
            : resolve_move<black>(b, s.pt, s.from, s.to);

        if (s.us == white)
            apply_move<white, true, true>(b, m, &live);
        else
            apply_move<black, true, true>(b, m, &live);
        materialize_network(b, live);

        NNUE::Net fresh;
        fresh.refresh(b);
        EXPECT_TRUE(network_accumulators_match(
            live.network_accumulator_stack[live.currentAccumulator],
            fresh.network_accumulator_stack[fresh.currentAccumulator]))
            << "active Network quiet step " << step_idx << " ("
            << fmt::format("{}", m) << ") diverged; fen after: " << b.fen();
        ++step_idx;
    }
}

TEST(network_audit, active_lazy_chain_matches_refresh) {
    Board b{"startpos"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);

    struct Step { Color us; PieceType pt; square_t from; square_t to; };
    const Step seq[] = {
        { white, pawn,   e2, e4 },
        { black, pawn,   e7, e5 },
        { white, knight, g1, f3 },
        { black, knight, b8, c6 },
        { white, bishop, f1, b5 },
        { black, pawn,   a7, a6 },
        { white, bishop, b5, a4 },
        { black, knight, g8, f6 },
    };

    for (auto const & s : seq) {
        Move m = s.us == white
            ? resolve_move<white>(b, s.pt, s.from, s.to)
            : resolve_move<black>(b, s.pt, s.from, s.to);

        if (s.us == white)
            apply_move<white, true, true>(b, m, &live);
        else
            apply_move<black, true, true>(b, m, &live);
    }
    materialize_network(b, live);

    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(network_accumulators_match(
        live.network_accumulator_stack[live.currentAccumulator],
        fresh.network_accumulator_stack[fresh.currentAccumulator]))
        << "active Network lazy chain diverged; fen after: " << b.fen();
}

TEST(network_audit, active_lazy_tree_matches_refresh) {
    Board b{"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);
    audit_network_lazy_tree<white>(b, live, 3);
}

TEST(network_audit, active_kiwipete_pv_with_castle_matches_refresh) {
    Board b{"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);

    const std::string moves =
        "e2a6 b4c3 d2c3 e6d5 e4d5 h3g2 f3g2 f6d5 e1g1";
    std::istringstream iss{moves};
    std::string move_str;
    while (iss >> move_str) {
        const std::string src_str{move_str.substr(0, 2)};
        const std::string dst_str{move_str.substr(2, 2)};
        const auto src = str2sq(src_str.c_str());
        const auto dst = str2sq(dst_str.c_str());
        const auto pt = b.pt_mb[src];
        if (b.side == white) {
            auto move = resolve_move<white>(b, pt, src, dst);
            apply_move<white, true, true>(b, move, &live);
        } else {
            auto move = resolve_move<black>(b, pt, src, dst);
            apply_move<black, true, true>(b, move, &live);
        }
    }

    materialize_network(b, live);
    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(network_accumulators_match(
        live.network_accumulator_stack[live.currentAccumulator],
        fresh.network_accumulator_stack[fresh.currentAccumulator]))
        << "active Network Kiwipete PV diverged; fen after: " << b.fen();
}

TEST(network_audit, active_promotion_capture_king_chain_matches_refresh) {
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    const std::vector<std::string> moves = {
        "e2a6", "b4c3", "b2c3", "h3g2", "d5e6",
        "g2h1n", "e6f7", "e7f7", "e5f7", "e8f7",
    };

    for (size_t prefix = 1; prefix <= moves.size(); ++prefix) {
        Board b{"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"};
        NNUE::Net live;
        live.refresh(b);

        for (size_t i = 0; i < prefix; ++i) {
            const auto & move_str = moves[i];
            const std::string src_str{move_str.substr(0, 2)};
            const std::string dst_str{move_str.substr(2, 2)};
            const auto src = str2sq(src_str.c_str());
            const auto dst = str2sq(dst_str.c_str());
            const auto pt = b.pt_mb[src];
            if (b.side == white) {
                auto move = resolve_move<white>(b, pt, src, dst);
                apply_move<white, true, true>(b, move, &live);
            } else {
                auto move = resolve_move<black>(b, pt, src, dst);
                apply_move<black, true, true>(b, move, &live);
            }
        }

        materialize_network(b, live);
        NNUE::Net fresh;
        fresh.refresh(b);
        EXPECT_TRUE(network_accumulators_match(
            live.network_accumulator_stack[live.currentAccumulator],
            fresh.network_accumulator_stack[fresh.currentAccumulator]))
            << "active Network promotion/king chain diverged at prefix "
            << prefix << "; fen after: " << b.fen();
    }
}

TEST(network_audit, active_nonrefreshing_king_loop_matches_refresh) {
    Board b{"8/8/2k5/8/8/5Q2/1pq2PKP/8 b - - 13 60"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);

    const std::vector<std::string> moves = {
        "c6c5", "f3e3", "c5c6",
    };

    for (const auto & move_str : moves) {
        const std::string src_str{move_str.substr(0, 2)};
        const std::string dst_str{move_str.substr(2, 2)};
        const auto src = str2sq(src_str.c_str());
        const auto dst = str2sq(dst_str.c_str());
        const auto pt = b.pt_mb[src];
        if (b.side == white) {
            auto move = resolve_move<white>(b, pt, src, dst);
            apply_move<white, true, true>(b, move, &live);
        } else {
            auto move = resolve_move<black>(b, pt, src, dst);
            apply_move<black, true, true>(b, move, &live);
        }
    }

    materialize_network(b, live);
    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(network_accumulators_match(
        live.network_accumulator_stack[live.currentAccumulator],
        fresh.network_accumulator_stack[fresh.currentAccumulator]))
        << "active Network non-refreshing king loop diverged; fen after: " << b.fen();
}

TEST(network_audit, active_quiet_sibling_matches_refresh) {
    Board b{"startpos"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);

    auto e2e4 = resolve_move<white>(b, pawn, e2, e4);
    apply_move<white, true, true>(b, e2e4, &live);
    revert_move<white, true, true>(b, &live);

    auto d2d4 = resolve_move<white>(b, pawn, d2, d4);
    apply_move<white, true, true>(b, d2d4, &live);
    materialize_network(b, live);

    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(network_accumulators_match(
        live.network_accumulator_stack[live.currentAccumulator],
        fresh.network_accumulator_stack[fresh.currentAccumulator]))
        << "active Network sibling quiet move diverged; fen after: " << b.fen();
}

TEST(network_audit, active_generic_capture_matches_refresh) {
    Board b{"rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"};
    ensure_network_mock_weights();
    ScopedNetworkEnabled scoped;

    NNUE::Net live;
    live.refresh(b);

    auto move = resolve_move<white>(b, pawn, e4, d5);
    apply_move<white, true, true>(b, move, &live);
    materialize_network(b, live);

    NNUE::Net fresh;
    fresh.refresh(b);
    EXPECT_TRUE(network_accumulators_match(
        live.network_accumulator_stack[live.currentAccumulator],
        fresh.network_accumulator_stack[fresh.currentAccumulator]))
        << "active Network generic capture diverged; fen after: " << b.fen();
}

TEST(network_audit, special_moves_match_refresh) {
    ScopedNetworkEnabled scoped;

    {
        Board b{"rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"};
        auto moves = generate_legal_moves<white>(b);
        Move ep{};
        for (auto m : moves) {
            if (m.flags() == Move::Flags::enpassant) { ep = m; break; }
        }
        ASSERT_TRUE(ep);
        expect_network_matches_refresh("enpassant", b, ep, white);
    }
    {
        Board b{"1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1"};
        auto moves = generate_legal_moves<white>(b);
        Move promo{};
        for (auto m : moves) {
            if (m.flags() == Move::Flags::promote
             && m.src_sq() == a7 && m.dst_sq() == b8
             && m.promo_piece() == queen) {
                promo = m; break;
            }
        }
        ASSERT_TRUE(promo);
        expect_network_matches_refresh("promotion_with_capture", b, promo, white);
    }
    {
        Board b{"r3k2r/pppqppbp/2np1np1/8/8/2NP1NP1/PPPQPPBP/R3K2R w KQkq - 0 1"};
        auto moves = generate_legal_moves<white>(b);
        Move castle{};
        for (auto m : moves) {
            if (m.flags() == Move::Flags::castle && m.dst_sq() < m.src_sq()) {
                castle = m; break;
            }
        }
        ASSERT_TRUE(castle);
        expect_network_matches_refresh("castle_kingside", b, castle, white);
    }
    {
        Board b{"4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"};
        auto move = resolve_move<white>(b, king, e1, e2);
        expect_network_matches_refresh("king_move_non_castle", b, move, white);
    }
}

// Syzygy probing expects bitboards in A1=0 layout: a1=bit0, h1=bit7, a8=bit56.
// Enyo internally uses H1=0. board2pos() applies bbconv()/sqconv() to bridge
// the two. These tests pin down that conversion against hand-computed
// expected values for a handful of asymmetry-sensitive positions.
TEST(syzygy_bbconv, startpos_bitboards) {
    Board b{"startpos"};
    auto p = syzygy::board2pos(b);

    // Ranks 1 and 2 are all white, ranks 7 and 8 all black.
    EXPECT_EQ(p.white,   0x000000000000FFFFULL);
    EXPECT_EQ(p.black,   0xFFFF000000000000ULL);
    // Kings on e1, e8 → bits 4 and 60.
    EXPECT_EQ(p.kings,   (1ULL << 4) | (1ULL << 60));
    // Queens on d1, d8 → bits 3 and 59.
    EXPECT_EQ(p.queens,  (1ULL << 3) | (1ULL << 59));
    // Rooks on a1, h1, a8, h8.
    EXPECT_EQ(p.rooks,   (1ULL << 0) | (1ULL << 7) | (1ULL << 56) | (1ULL << 63));
    // Bishops on c1, f1, c8, f8.
    EXPECT_EQ(p.bishops, (1ULL << 2) | (1ULL << 5) | (1ULL << 58) | (1ULL << 61));
    // Knights on b1, g1, b8, g8.
    EXPECT_EQ(p.knights, (1ULL << 1) | (1ULL << 6) | (1ULL << 57) | (1ULL << 62));
    // Pawns span ranks 2 and 7.
    EXPECT_EQ(p.pawns,   0x00FF00000000FF00ULL);
    EXPECT_TRUE(p.turn);
}

TEST(syzygy_bbconv, asymmetric_a1_bishop) {
    // Only a white bishop on a1. If bbconv mirrored files incorrectly (a1↔h1)
    // this would come out as bit 7 instead of bit 0.
    Board b{"8/8/8/8/8/8/8/B6k w - - 0 1"};
    auto p = syzygy::board2pos(b);

    EXPECT_EQ(p.white,   1ULL << 0);
    EXPECT_EQ(p.bishops, 1ULL << 0);
    EXPECT_EQ(p.pawns, 0ULL);
}

TEST(syzygy_bbconv, asymmetric_h8_bishop) {
    // Black bishop on h8 (plus kings on a1, h1 to keep the FEN legal).
    Board b{"7b/8/8/8/8/8/8/K6k w - - 0 1"};
    auto p = syzygy::board2pos(b);

    // Black set: bishop on h8 (bit 63) + black king on h1 (bit 7).
    EXPECT_EQ(p.black,   (1ULL << 63) | (1ULL << 7));
    EXPECT_EQ(p.bishops, 1ULL << 63);
    // Kings: white on a1 (bit 0), black on h1 (bit 7).
    EXPECT_EQ(p.kings,   (1ULL << 0) | (1ULL << 7));
}

TEST(syzygy_bbconv, krvk_winning_position) {
    // Standard KR vs K sanity position: White K on e1, R on h1, black K on e8.
    Board b{"4k3/8/8/8/8/8/8/4K2R w - - 0 1"};
    auto p = syzygy::board2pos(b);

    EXPECT_EQ(p.white,   (1ULL << 4) | (1ULL << 7));   // e1 | h1
    EXPECT_EQ(p.black,   1ULL << 60);                  // e8
    EXPECT_EQ(p.kings,   (1ULL << 4) | (1ULL << 60));
    EXPECT_EQ(p.rooks,   1ULL << 7);
    EXPECT_EQ(p.queens,  0ULL);
    EXPECT_EQ(p.pawns,   0ULL);
    EXPECT_TRUE(p.turn);
}

TEST(syzygy_bbconv, ep_square_conversion) {
    // After 1.e4, the en-passant square is e3 (file e, rank 3) = A1-layout index
    // 4*8? No — rank 3 means rank_index=2, so bit = 2*8 + 4 = 20.
    Board b{"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"};
    auto p = syzygy::board2pos(b);

    // sqconv() maps the board's H1=0 square index to the tablebase A1=0 index.
    // We don't compare the raw square_t here — just the probe-ready uint8_t.
    EXPECT_EQ(p.ep, 20u);  // e3 in A1=0 layout
}

TEST(syzygy_bbconv, no_ep_square_stays_tablebase_sentinel) {
    Board b{"startpos"};
    auto p = syzygy::board2pos(b);

    EXPECT_EQ(p.ep, 0u);
}

TEST(syzygy_root, filters_root_promotions_that_lose_wdl) {
    if (!init_test_syzygy(6))
        GTEST_SKIP() << "6-man Syzygy tablebases not available";

    Board b{"8/8/8/8/1k6/2n1KP2/p7/5R2 b - - 0 77"};
    const auto legal = generate_legal_moves<black>(b);
    const auto winning = syzygy::root_WDL_filter(b, legal);

    auto knight_move = uci_to_move(b, "c3b1");
    auto promotion = uci_to_move(b, "a2a1q");
    ASSERT_TRUE(knight_move);
    ASSERT_TRUE(promotion);

    const auto contains = [&](Move move) {
        return std::ranges::any_of(winning, [&](Move candidate) {
            return candidate == move;
        });
    };

    EXPECT_TRUE(contains(*knight_move));
    EXPECT_FALSE(contains(*promotion));
    EXPECT_EQ(b.fen(), "8/8/8/8/1k6/2n1KP2/p7/5R2 b - - 0 77");
}

TEST(syzygy_root, dtz_returns_move_for_lost_root_position) {
    if (!init_test_syzygy(3))
        GTEST_SKIP() << "Syzygy tablebases not available";

    Board b{"6k1/2Q5/5K2/8/8/8/8/8 b - - 10 96"};
    syzygy::Status status = syzygy::Status::Error;
    const auto [score, move] = syzygy::DTZ_probe(b, status);

    EXPECT_EQ(status, syzygy::Status::Loss);
    EXPECT_EQ(score, Value::tb_loss_in_max_ply);
    ASSERT_TRUE(move);
    const auto legal = generate_legal_moves<black>(b);
    EXPECT_TRUE(std::ranges::any_of(legal, [&](Move candidate) {
        return candidate == move;
    }));
}

TEST(syzygy_root, root_moves_reports_complete_dtz_candidates) {
    if (!init_test_syzygy(3))
        GTEST_SKIP() << "Syzygy tablebases not available";

    Board b{"6k1/2Q5/5K2/8/8/8/8/8 b - - 10 96"};
    const auto legal = generate_legal_moves<black>(b);
    bool complete = false;
    const auto moves = syzygy::root_moves(b, legal, &complete);

    ASSERT_TRUE(complete);
    ASSERT_EQ(moves.size(), legal.size());
    EXPECT_TRUE(std::ranges::all_of(moves, [](const auto & move) {
        return move.dtz >= 0;
    }));
    EXPECT_TRUE(std::ranges::any_of(moves, [](const auto & move) {
        return move.status == syzygy::Status::Loss;
    }));
}

TEST(syzygy_root, piece_count_boundaries) {
    if (!init_test_syzygy(6))
        GTEST_SKIP() << "6-man Syzygy tablebases not available";
    ASSERT_LT(syzygy::largest(), 7u);

    {
        Board b{"8/8/8/3n4/1k6/5P2/4K3/R7 b - - 0 79"};
        ASSERT_EQ(count_bits(b.color_bb[white] | b.color_bb[black]), 5);

        syzygy::Status status = syzygy::Status::Error;
        const auto [score, move] = syzygy::DTZ_probe(b, status);

        EXPECT_EQ(syzygy::WDL_probe(b), syzygy::Status::Loss);
        EXPECT_EQ(status, syzygy::Status::Loss);
        EXPECT_EQ(score, Value::tb_loss_in_max_ply);
        EXPECT_TRUE(move);
    }

    {
        Board b{"8/8/8/8/1k6/2n1KP2/p7/5R2 b - - 0 77"};
        ASSERT_EQ(count_bits(b.color_bb[white] | b.color_bb[black]), 6);

        syzygy::Status status = syzygy::Status::Error;
        const auto [score, move] = syzygy::DTZ_probe(b, status);
        const auto legal = generate_legal_moves<black>(b);
        bool filter_complete = false;
        const auto filtered = syzygy::root_WDL_filter(b, legal, &filter_complete);

        EXPECT_EQ(syzygy::WDL_probe(b), syzygy::Status::Win);
        if (move) {
            EXPECT_EQ(status, syzygy::Status::Win);
            EXPECT_EQ(score, Value::tb_win_in_max_ply);
            EXPECT_TRUE(std::ranges::any_of(legal, [&](Move candidate) {
                return candidate == move;
            }));
        } else {
            EXPECT_EQ(status, syzygy::Status::Error);
            EXPECT_EQ(score, 0);
        }
        EXPECT_TRUE(filter_complete);
        EXPECT_EQ(filtered.size(), 1u);
    }

    {
        Board b{"8/8/8/8/1k6/2n1KP2/p7/4NR2 b - - 0 77"};
        ASSERT_EQ(count_bits(b.color_bb[white] | b.color_bb[black]), 7);

        syzygy::Status status = syzygy::Status::Error;
        const auto [score, move] = syzygy::DTZ_probe(b, status);
        const auto legal = generate_legal_moves<black>(b);
        bool filter_complete = true;
        const auto filtered = syzygy::root_WDL_filter(b, legal, &filter_complete);

        EXPECT_EQ(syzygy::WDL_probe(b), syzygy::Status::Error);
        EXPECT_EQ(status, syzygy::Status::Error);
        EXPECT_EQ(score, 0);
        EXPECT_FALSE(move);
        EXPECT_FALSE(filter_complete);
        EXPECT_EQ(filtered.size(), legal.size());
    }
}

TEST(syzygy_root, resolve_path_ignores_hidden_staging_dirs) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = fs::temp_directory_path() / fmt::format("enyo_tb_root_{}", stamp);
    const auto hidden = root / ".6-dtz";
    const auto visible = root / "6-wdl";
    const auto nested_visible = root / "6-dtz" / "6-dtz";
    fs::create_directories(hidden);
    fs::create_directories(visible);
    fs::create_directories(nested_visible);

    {
        std::ofstream file(hidden / "KQvK.rtbw", std::ios::binary);
        file << std::string(16, '\0');
    }
    {
        std::ofstream file(visible / "KRvK.rtbw", std::ios::binary);
        file << std::string(16, '\0');
    }
    {
        std::ofstream file(nested_visible / "KQvK.rtbz", std::ios::binary);
        file << std::string(16, '\0');
    }

    const auto resolved = syzygy::resolve_path(root.string());
    EXPECT_NE(resolved.find(visible.string()), std::string::npos) << resolved;
    EXPECT_NE(resolved.find(nested_visible.string()), std::string::npos) << resolved;
    EXPECT_EQ(resolved.find(hidden.string()), std::string::npos) << resolved;

    fs::remove_all(root);
}

TEST(syzygy_root, init_rejects_explicit_incomplete_tablebase_dir) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = fs::temp_directory_path() / fmt::format("enyo_incomplete_tb_{}", stamp);
    fs::create_directories(root);
    {
        std::ofstream file(root / "KQvK.rtbw", std::ios::binary);
        file.put('\0');
    }

    EXPECT_FALSE(syzygy::init(root.string()));

    fs::remove_all(root);
}

TEST(syzygy_root, init_rejects_nested_incomplete_tablebase_dir) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = fs::temp_directory_path() / fmt::format("enyo_nested_incomplete_tb_{}", stamp);
    const auto nested = root / "6-dtz" / "6-dtz";
    fs::create_directories(nested);
    {
        std::ofstream file(nested / "KQvK.rtbz", std::ios::binary);
        file.put('\0');
    }

    EXPECT_FALSE(syzygy::init(root.string()));

    fs::remove_all(root);
}

TEST(syzygy_root, six_piece_win_root_searches_winning_moves) {
    if (!init_test_syzygy(6))
        GTEST_SKIP() << "6-man Syzygy tablebases not available";

    Board b{"6k1/8/5K2/8/8/8/P1QBB3/8 w - - 0 1"};
    ASSERT_EQ(count_bits(b.color_bb[white] | b.color_bb[black]), 6);

    syzygy::Status status = syzygy::Status::Error;
    const auto [score, dtz_move] = syzygy::DTZ_probe(b, status);
    const auto legal = generate_legal_moves<white>(b);
    bool filter_complete = false;
    const auto filtered = syzygy::root_WDL_filter(b, legal, &filter_complete);

    if (dtz_move) {
        EXPECT_EQ(status, syzygy::Status::Win);
        EXPECT_EQ(score, Value::tb_win_in_max_ply);
    } else {
        EXPECT_EQ(status, syzygy::Status::Error);
        EXPECT_EQ(score, 0);
    }
    ASSERT_EQ(syzygy::WDL_probe(b), syzygy::Status::Win);
    ASSERT_TRUE(filter_complete);
    ASSERT_FALSE(filtered.empty());

    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = true;

    SearchInfo si{b, 2};
    si.board = b;
    si.nnue.refresh(si.board);
    si.starttime = std::chrono::high_resolution_clock::now();
    si.stoptime = si.starttime + std::chrono::hours(1);
    si.soft_stoptime = si.stoptime;

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    thread::pool.init_threads(si, 1);
    thread::pool.wait();
    const auto out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("string tbhit win"), std::string::npos) << out;
    EXPECT_NE(out.find("bestmove "), std::string::npos) << out;
    EXPECT_EQ(out.find("WDL root move"), std::string::npos) << out;
    EXPECT_TRUE(out.find("score cp ") != std::string::npos
             || out.find("score mate ") != std::string::npos) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(syzygy_root, available_six_piece_dtz_returns_root_move) {
    if (!init_test_syzygy(6))
        GTEST_SKIP() << "6-man Syzygy tablebases not available";

    Board b{"7k/8/5KQ1/8/2BB4/8/P7/8 w - - 0 1"};
    ASSERT_EQ(count_bits(b.color_bb[white] | b.color_bb[black]), 6);

    syzygy::Status status = syzygy::Status::Error;
    const auto [score, move] = syzygy::DTZ_probe(b, status);
    if (!move)
        GTEST_SKIP() << "selected 6-man DTZ material is not installed";

    EXPECT_EQ(status, syzygy::Status::Win);
    EXPECT_EQ(score, Value::tb_win_in_max_ply);
    const auto legal = generate_legal_moves<white>(b);
    EXPECT_TRUE(std::ranges::any_of(legal, [&](Move candidate) {
        return candidate == move;
    }));
}

TEST(syzygy_root, immediate_mate_reports_mate_before_tablebase_cp) {
    if (!init_test_syzygy(3))
        GTEST_SKIP() << "Syzygy tablebases not available";

    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = true;

    Board b{"7k/8/5KQ1/8/8/8/8/8 w - - 0 1"};
    SearchInfo si{b, 8};
    si.board = b;
    si.nnue.refresh(si.board);
    si.starttime = std::chrono::high_resolution_clock::now();
    si.stoptime = si.starttime + std::chrono::hours(1);
    si.soft_stoptime = si.stoptime;

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    thread::pool.init_threads(si, 1);
    thread::pool.wait();
    const auto out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("score mate 1"), std::string::npos) << out;
    EXPECT_NE(out.find("bestmove g6g7"), std::string::npos) << out;
    EXPECT_EQ(out.find("string tbhit win"), std::string::npos) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(uci_root, lost_tablebase_root_does_not_search_after_tbhit) {
    if (!init_test_syzygy(3))
        GTEST_SKIP() << "Syzygy tablebases not available";

    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = true;

    Board b;
    Uci uci{b};
    uci("position fen 6k1/2Q5/5K2/8/8/8/8/8 b - - 10 96");

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    uci("go wtime 82790 btime 68419 winc 5000 binc 5000");
    const auto out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("string tbhit loss"), std::string::npos) << out;
    EXPECT_NE(out.find("bestmove "), std::string::npos) << out;
    EXPECT_EQ(out.find("info depth 2 score"), std::string::npos) << out;
    EXPECT_NE(out.find("score cp -20000"), std::string::npos) << out;
    EXPECT_EQ(out.find("score mate "), std::string::npos) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(uci_root, wdl_only_lost_tablebase_root_keeps_cutoffs_fast) {
    // Node-count expectations assume the deterministic legacy eval.
    LegacyEvalScope legacy;

    const auto wdl_file = find_test_tablebase_file("KQRRvKQ.rtbw");
    if (wdl_file.empty())
        GTEST_SKIP() << "KQRRvKQ WDL tablebase not available";
    const auto child_wdl_file = find_test_tablebase_file("KQvK.rtbw");
    if (child_wdl_file.empty())
        GTEST_SKIP() << "3-5 piece WDL tablebases not available";

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = fs::temp_directory_path() / fmt::format("enyo_wdl_only_tb_{}", stamp);
    std::error_code ec;
    fs::create_directories(root, ec);
    if (ec)
        GTEST_SKIP() << "could not create temporary Syzygy directory: " << ec.message();
    if (!link_wdl_table_files(wdl_file.parent_path(), root)
        || !link_wdl_table_files(child_wdl_file.parent_path(), root)) {
        fs::remove_all(root);
        GTEST_SKIP() << "could not prepare WDL-only Syzygy directory";
    }

    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    const auto old_syzygy_path = cfgmgr.syzygy_path;

    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = true;
    cfgmgr.syzygy_path = root.string();
    if (!syzygy::init(root.string())) {
        cfgmgr.num_threads = old_threads;
        cfgmgr.use_syzygy = old_use_syzygy;
        cfgmgr.syzygy_path = old_syzygy_path;
        fs::remove_all(root);
        GTEST_SKIP() << "could not initialize WDL-only Syzygy directory";
    }

    const auto run_search = [](bool use_syzygy) {
        cfgmgr.use_syzygy = use_syzygy;
        Board b;
        Uci uci{b};
        uci.eval_loaded = true; // keep the LegacyEvalScope evaluator
        uci("position fen 8/2RQ4/1K6/4R3/8/5q2/k7/8 b - - 0 1");

        thread::pool.stop = false;
        tt::ttable.clear();
        testing::internal::CaptureStdout();
        uci("go depth 6");
        return testing::internal::GetCapturedStdout();
    };

    const auto tb_out = run_search(true);
    const auto plain_out = run_search(false);

    EXPECT_NE(tb_out.find("string tbhit loss"), std::string::npos) << tb_out;
    EXPECT_EQ(plain_out.find("string tbhit"), std::string::npos) << plain_out;

    const auto tb_nodes = uci_nodes_at_depth(tb_out, 6);
    const auto plain_nodes = uci_nodes_at_depth(plain_out, 6);
    EXPECT_TRUE(tb_nodes.has_value()) << tb_out;
    EXPECT_TRUE(plain_nodes.has_value()) << plain_out;
    if (tb_nodes && plain_nodes) {
        // Guards 271e335: suppressed TB cutoffs balloon the searched tree.
        EXPECT_LT(*tb_nodes, 20000u) << tb_out;
        EXPECT_LT(*tb_nodes, *plain_nodes) << "tb:\n" << tb_out << "\nplain:\n" << plain_out;
    }

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
    cfgmgr.syzygy_path = old_syzygy_path;
    if (old_use_syzygy && !old_syzygy_path.empty())
        syzygy::init(old_syzygy_path);
    fs::remove_all(root);
}

TEST(uci_root, wdl_only_root_does_not_repeat_first_filtered_move) {
    if (!init_test_syzygy(6))
        GTEST_SKIP() << "6-man Syzygy tablebases not available";

    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = true;

    Board b;
    Uci uci{b};
    uci("position startpos moves g1f3 d7d5 g2g3 c7c5 f1g2 a7a6 e1g1 b8c6 d2d4 h7h6 c2c4 e7e6 c1e3 g8e7 d4c5 e7f5 e3d4 f5d4 f3d4 f8c5 d4b3 c5a7 c4d5 e6d5 d1d5 d8d5 g2d5 c8g4 f1c1 e8g8 e2e3 f8d8 d5g2 a8b8 g2c6 b7c6 b1d2 g4e6 c1c6 a6a5 d2c4 a5a4 b3d2 d8c8 c6c8 b8c8 a1c1 c8b8 g1f1 g7g6 f1e2 a7c5 c1c2 c5f8 e3e4 b8c8 b2b3 a4b3 a2b3 c8b8 c4e3 g8g7 c2c1 f8b4 d2c4 b4c5 c1c3 c5d4 c3d3 d4c5 e3d5 b8a8 c4e3 a8a2 d3d2 a2a3 b3b4 c5e3 d5e3 a3b3 d2d4 b3b2 e2f3 g7f6 d4d6 f6e5 d6b6 h6h5 h2h4 b2b3 b6b5 e5d4 b5g5 d4c3 b4b5 c3d3 g3g4 h5g4 e3g4 d3d2 f3f4 b3h3 f4e5 h3h4 g4f6 h4h1 e5d6 d2e2 g5c5 e2f2 b5b6 h1b1 d6c7 e6c8 f6e8 c8g4 b6b7 f2f3 e8d6 b1b7 d6b7 f3e4 b7d6 e4f4 d6f7 g4e6 f7g5 e6f5 g5h7 g6g5 h7f6 f5h3 f6h5 f4g4");

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    uci("go depth 8");
    const auto out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("string tbhit win"), std::string::npos) << out;
    EXPECT_EQ(out.find("WDL root move"), std::string::npos) << out;
    EXPECT_NE(out.find("score cp "), std::string::npos) << out;
    EXPECT_NE(out.find("bestmove "), std::string::npos) << out;
    EXPECT_EQ(out.find("bestmove h5f6"), std::string::npos) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(search_root, reports_mate_distance_when_search_finds_mate) {
    const int old_threads = cfgmgr.num_threads;
    const bool old_use_syzygy = cfgmgr.use_syzygy;
    cfgmgr.num_threads = 1;
    cfgmgr.use_syzygy = false;

    Board b{"8/8/8/8/3Q4/k7/8/1K6 w - - 0 1"};
    SearchInfo si{b, 5};
    si.board = b;
    si.nnue.refresh(si.board);
    si.starttime = std::chrono::high_resolution_clock::now();
    si.stoptime = si.starttime + std::chrono::hours(1);
    si.soft_stoptime = si.stoptime;

    thread::pool.stop = false;
    tt::ttable.clear();
    testing::internal::CaptureStdout();
    thread::pool.init_threads(si, 1);
    thread::pool.wait();
    const auto out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("score mate 2"), std::string::npos) << out;
    EXPECT_NE(out.find("bestmove "), std::string::npos) << out;
    EXPECT_EQ(out.find("string tbhit"), std::string::npos) << out;

    cfgmgr.num_threads = old_threads;
    cfgmgr.use_syzygy = old_use_syzygy;
}

TEST(syzygy_root, game_tb_positions_filter_to_best_wdl) {
    if (!init_test_syzygy(6))
        GTEST_SKIP() << "6-man Syzygy tablebases not available";

    const std::vector<std::string> fens = {
        "8/8/8/8/1k6/2n1KP2/p7/5R2 b - - 0 77",
        "8/8/8/8/1k6/2n1KP2/8/R7 b - - 0 78",
        "8/8/8/3n4/1k6/5P2/4K3/R7 b - - 2 79",
        "8/8/8/8/1k3n2/5P2/8/R4K2 b - - 4 80",
        "8/8/8/8/5n2/1k3P2/5K2/R7 b - - 6 81",
        "8/8/8/8/1k3n2/5P2/5K2/5R2 b - - 8 82",
        "8/8/8/8/5n2/1k2KP2/8/5R2 b - - 10 83",
        "8/8/8/8/5n2/4KP2/1k6/2R5 b - - 12 84",
        "8/8/8/8/5K2/5P2/8/2k5 b - - 0 85",
        "8/8/8/8/8/4KP2/8/1k6 b - - 2 86",
        "8/8/8/8/5P2/4K3/k7/8 b - - 0 87",
        "8/8/8/5P2/8/k3K3/8/8 b - - 0 88",
        "8/8/5P2/8/1k6/4K3/8/8 b - - 0 89",
        "8/5P2/8/8/8/2k1K3/8/8 b - - 0 90",
        "5Q2/8/8/8/2k5/4K3/8/8 b - - 0 91",
        "3Q4/8/8/3k4/8/4K3/8/8 b - - 2 92",
        "3Q4/8/4k3/8/4K3/8/8/8 b - - 4 93",
        "3Q4/5k2/8/5K2/8/8/8/8 b - - 6 94",
        "8/2Q3k1/8/5K2/8/8/8/8 b - - 8 95",
        "6k1/2Q5/5K2/8/8/8/8/8 b - - 10 96",
    };

    const auto root_rank = [](syzygy::Status status) {
        switch (status) {
            case syzygy::Status::Win:  return 2;
            case syzygy::Status::Draw: return 1;
            case syzygy::Status::Loss: return 0;
            default:                   return -1;
        }
    };
    const auto child_to_root_rank = [&](syzygy::Status child_status) {
        return root_rank(
            child_status == syzygy::Status::Win  ? syzygy::Status::Loss
          : child_status == syzygy::Status::Loss ? syzygy::Status::Win
                                                 : syzygy::Status::Draw);
    };
    const auto tablebase_status = [](Board & board) {
        syzygy::Status status = syzygy::Status::Error;
        syzygy::DTZ_probe(board, status);
        if (status == syzygy::Status::Error)
            status = syzygy::WDL_probe(board);
        const int rule50 = board.half_moves + static_cast<int>(board.gamestate.half_moves);
        if (status == syzygy::Status::Error && rule50 <= 20) {
            Board zero_rule50 = board;
            zero_rule50.half_moves = 0;
            zero_rule50.gamestate.half_moves = 0;
            status = syzygy::WDL_probe(zero_rule50);
        }
        return status;
    };
    const auto apply_for_side = [](Board & board, Move move) {
        if (board.side == white)
            apply_move<white, true, false>(board, move);
        else
            apply_move<black, true, false>(board, move);
    };

    for (const auto & fen : fens) {
        Board b{fen};
        const auto original = b.fen();
        const auto legal = b.side == white
            ? generate_legal_moves<white>(b)
            : generate_legal_moves<black>(b);
        const auto filtered = syzygy::root_WDL_filter(b, legal);

        const auto root_status = tablebase_status(b);
        ASSERT_NE(root_status, syzygy::Status::Error) << fen;
        const int root_best_rank = root_rank(root_status);
        ASSERT_FALSE(filtered.empty()) << fen;

        for (auto move : filtered) {
            Board child = b;
            apply_for_side(child, move);
            const auto child_status = tablebase_status(child);
            ASSERT_NE(child_status, syzygy::Status::Error) << fen << " move " << mv2str(move);
            EXPECT_EQ(child_to_root_rank(child_status), root_best_rank)
                << fen << " move " << mv2str(move);
        }
        EXPECT_EQ(b.fen(), original);
    }
}

// --- AccumulatorCache pieces_hash regression ---------------------------------
// The original pieces_hash used `pt << (sq + color*6)`, which collides on
// pawn(=1)@N and knight(=2)@(N-1) — each contributes 2^N, their XORs cancel
// to zero. A board carrying such a pawn+knight pair therefore hashed
// identically to a board with those squares empty. With identical king
// squares (the cache's primary key) the stale-accumulator path was live.
//
// `compute_pieces_hash` is file-static, so we test the invariant that
// refresh_with_cache must uphold: two positions with DIFFERENT piece
// placements must produce DIFFERENT evals after refresh_with_cache, even
// when their old-hash contributions happen to cancel. If the cache returns
// a stale accumulator from the first position, the second position's eval
// will match the first.
TEST(nnue_cache, pieces_hash_pawn_knight_collision_does_not_stale_acc) {
    // Position A: kings on e8/e1 plus white pawn on f2 (sq=10) and
    //             white knight on g2 (sq=9). In H1=0 indexing:
    //             (pawn=1)<<10 ^ (knight=2)<<9 = 0x400 ^ 0x400 = 0.
    // Position B: same kings, pawn and knight absent. Old-hash contribution
    //             from those two pieces is likewise 0, so the stale hash
    //             functions compute the same value for both positions.
    Board a{"4k3/8/8/8/8/8/5PN1/4K3 w - - 0 1"};
    Board b{"4k3/8/8/8/8/8/8/4K3 w - - 0 1"};

    LegacyEvalScope legacy;

    NNUE::Net net;
    // Prime cache with position A, then ask for position B. Under the old
    // hash this was a false cache hit and B's eval == A's eval. Under the
    // Zobrist-based hash the cache misses and B refreshes correctly.
    net.refresh_with_cache(a);
    const auto eval_a = net.Evaluate(a.side);

    net.refresh_with_cache(b);
    const auto eval_b = net.Evaluate(b.side);

    EXPECT_NE(eval_a, eval_b)
        << "AccumulatorCache returned a stale accumulator across positions "
           "whose old XOR-based pieces_hash collide. The fix in 2672d35 "
           "uses Board::zbrs.psq_ randoms and must keep these distinct.\n"
           "Position A: " << a.fen() << " eval=" << eval_a << "\n"
           "Position B: " << b.fen() << " eval=" << eval_b;

    // Belt-and-suspenders: compare B's eval from the cached path against
    // a forced full refresh of B. If the cache is behaving, they match.
    NNUE::Net ref;
    ref.refresh(b);
    EXPECT_EQ(eval_b, ref.Evaluate(b.side))
        << "refresh_with_cache(B) after priming with A produced a different "
           "eval than a fresh refresh(B). Cache validator is unsound.";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Mirror the engine's embedded-default startup.
    const auto embedded = Network::LoadEnyoNetwork(
        NNUE::EmbeddedNetworkData(), NNUE::EmbeddedNetworkSize());
    if (embedded.status == NNUE::LoadStatus::loaded)
        Network::enabled = true;
    else
        NNUE::Init("");
    return RUN_ALL_TESTS();
}
