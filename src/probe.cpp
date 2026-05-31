#include "probe.hpp"
#if ENYO_USE_SYZYGY
extern "C" {
#include "3rdparty/Pyrrhic/tbprobe.h"
}
#endif

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "board.hpp"
#include "config.hpp"
#include "movegen.hpp"

using namespace enyo;

namespace {
bool initialized = false;
std::unordered_set<std::string> available_wdl;
std::unordered_set<std::string> available_dtz;

#ifdef _WIN32
constexpr char path_separator = ';';
#else
constexpr char path_separator = ':';
#endif

std::string expand_home_path(std::string path)
{
    if (path == "~" || path.starts_with("~/")) {
        if (const char * home = std::getenv("HOME")) {
            if (path.size() == 1)
                return home;
            return std::string(home) + path.substr(1);
        }
    }
    return path;
}

std::vector<std::string> split_path_list(const std::string & path)
{
    std::vector<std::string> parts;
    size_t begin = 0;
    while (begin <= path.size()) {
        const auto end = path.find(path_separator, begin);
        parts.emplace_back(path.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return parts;
}

constexpr int root_wdl_rank(syzygy::Status status)
{
    switch (status) {
        case syzygy::Status::Win:  return 2;
        case syzygy::Status::Draw: return 1;
        case syzygy::Status::Loss: return 0;
        default:                   return -1;
    }
}

int root_rank_from_child_status(syzygy::Status child_status)
{
    return root_wdl_rank(
        child_status == syzygy::Status::Win  ? syzygy::Status::Loss
      : child_status == syzygy::Status::Loss ? syzygy::Status::Win
                                             : syzygy::Status::Draw);
}

bool is_complete_tablebase_file(const std::filesystem::path & path)
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
        return false;
    const auto ext = path.extension();
    if (ext != ".rtbw" && ext != ".rtbz")
        return false;
    const auto size = std::filesystem::file_size(path, ec);
    return !ec && (size & 63U) == 16U;
}

bool is_tablebase_file(const std::filesystem::path & path)
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
        return false;
    const auto ext = path.extension();
    return ext == ".rtbw" || ext == ".rtbz";
}

bool has_any_tablebase_file(const std::filesystem::path & path)
{
    std::error_code ec;
    if (!std::filesystem::is_directory(path, ec))
        return false;
    for (std::filesystem::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
        if (is_tablebase_file(it->path()))
            return true;
    }
    return false;
}


bool is_hidden_directory(const std::filesystem::path & path)
{
    const auto name = path.filename().string();
    return !name.empty() && name.front() == '.';
}

void append_unique(std::vector<std::string> & paths, const std::filesystem::path & path);

void collect_tablebase_paths(std::vector<std::string> & paths, const std::filesystem::path & path)
{
    if (is_hidden_directory(path))
        return;

    std::error_code ec;
    if (!std::filesystem::is_directory(path, ec))
        return;

    if (has_any_tablebase_file(path))
        append_unique(paths, path);

    for (std::filesystem::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_directory(ec))
            collect_tablebase_paths(paths, it->path());
    }
}

bool index_tablebase_files(const std::string & resolved_path)
{
    available_wdl.clear();
    available_dtz.clear();
    for (const auto & segment : split_path_list(resolved_path)) {
        if (segment.empty())
            continue;
        std::error_code ec;
        const std::filesystem::path path(segment);
        if (!std::filesystem::is_directory(path, ec))
            continue;
        for (std::filesystem::directory_iterator it2(path, ec), end2; !ec && it2 != end2; it2.increment(ec)) {
            if (is_tablebase_file(it2->path()) && !is_complete_tablebase_file(it2->path())) {
                fmt::print(stderr, "Fatal: corrupt or incomplete tablebase file: {}\n",
                    it2->path().string());
                available_wdl.clear();
                available_dtz.clear();
                return false;
            }
        }
        for (std::filesystem::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
            if (!is_complete_tablebase_file(it->path()))
                continue;
            const auto ext = it->path().extension();
            if (ext == ".rtbw")
                available_wdl.insert(it->path().stem().string());
            else if (ext == ".rtbz")
                available_dtz.insert(it->path().stem().string());
        }
    }
    return true;
}

char table_piece_char(PieceType pt)
{
    switch (pt) {
        case king:   return 'K';
        case queen:  return 'Q';
        case rook:   return 'R';
        case bishop: return 'B';
        case knight: return 'N';
        case pawn:   return 'P';
        default:     return '?';
    }
}

std::string table_side_name(Board const & board, Color side)
{
    std::string out;
    for (int pt = static_cast<int>(king); pt >= static_cast<int>(pawn); --pt) {
        const auto piece = static_cast<PieceType>(pt);
        const int count = count_bits(board.pt_bb[side][piece]);
        out.append(static_cast<std::size_t>(count), table_piece_char(piece));
    }
    return out;
}

std::array<std::string, 2> table_material_names(Board const & board)
{
    const auto white_name = table_side_name(board, white);
    const auto black_name = table_side_name(board, black);
    return {
        white_name + "v" + black_name,
        black_name + "v" + white_name,
    };
}

bool has_table_for(Board const & board, const std::unordered_set<std::string> & tables)
{
    const auto names = table_material_names(board);
    return tables.contains(names[0]) || tables.contains(names[1]);
}

syzygy::Status status_from_wdl(int wdl)
{
    switch (wdl) {
        case TB_WIN:  return syzygy::Status::Win;
        case TB_LOSS: return syzygy::Status::Loss;
        default:      return syzygy::Status::Draw;
    }
}

bool move_matches_tablebase_result(Move move, unsigned result)
{
    if (sqconv(move.src_sq()) != TB_GET_FROM(result))
        return false;
    if (sqconv(move.dst_sq()) != TB_GET_TO(result))
        return false;

    const PieceType promo_translation[] = {no_piece_type, queen, rook, bishop, knight};
    const auto promo = TB_GET_PROMOTES(result);
    if (promo == PYRRHIC_FLAG_NONE)
        return move.flags() != Move::Flags::promote;
    if (static_cast<size_t>(promo) >= std::size(promo_translation))
        return false;
    return move.flags() == Move::Flags::promote
        && move.promo_piece() == promo_translation[promo];
}

void append_unique(std::vector<std::string> & paths, const std::filesystem::path & path)
{
    const auto str = path.string();
    if (std::find(paths.begin(), paths.end(), str) == paths.end())
        paths.push_back(str);
}

std::string join_path_list(const std::vector<std::string> & paths)
{
    std::string out;
    for (const auto & path : paths) {
        if (!out.empty())
            out += path_separator;
        out += path;
    }
    return out;
}
}

namespace syzygy {

std::string resolve_path(const std::string & tb_path)
{
    std::vector<std::string> resolved;
    for (const auto & segment : split_path_list(tb_path)) {
        if (segment.empty())
            continue;

        const std::filesystem::path path(expand_home_path(segment));
        const auto before = resolved.size();
        collect_tablebase_paths(resolved, path);
        if (resolved.size() == before)
            append_unique(resolved, path);
    }
    return join_path_list(resolved);
}

bool init(const std::string & tb_path)
{
#if ENYO_USE_SYZYGY
    const auto resolved_path = resolve_path(tb_path);
    const bool table_index_ok = index_tablebase_files(resolved_path);
    if (!table_index_ok || available_wdl.empty()) {
        initialized = false;
        return false;
    }
    // tb_init returns true even when no tables were found; the real signal
    // is TB_LARGEST (max piece count of any loaded table).
    initialized = tb_init(resolved_path.c_str()) && TB_LARGEST > 0;
    return initialized;
#else
    (void)tb_path;
    initialized = false;
    available_wdl.clear();
    available_dtz.clear();
    return false;
#endif
}

unsigned largest()
{
#if ENYO_USE_SYZYGY
    return TB_LARGEST;
#else
    return 0;
#endif
}

static inline constexpr uint8_t tablebase_ep_square(square_t ep)
{
    return ep ? sqconv(ep) : 0;
}

pos board2pos(Board & b)
{
    assert(b.color_bb[white] & b.pt_bb[white][king]);
    assert(b.color_bb[black] & b.pt_bb[black][king]);
    return pos {
        .white = bbconv(b.color_bb[white]),
        .black = bbconv(b.color_bb[black]),
        .kings = bbconv(b.pt_bb[white][king] | b.pt_bb[black][king]),
        .queens = bbconv(b.pt_bb[white][queen] | b.pt_bb[black][queen]),
        .rooks = bbconv(b.pt_bb[white][rook] | b.pt_bb[black][rook]),
        .bishops = bbconv(b.pt_bb[white][bishop] | b.pt_bb[black][bishop]),
        .knights = bbconv(b.pt_bb[white][knight] | b.pt_bb[black][knight]),
        .pawns = bbconv(b.pt_bb[white][pawn] | b.pt_bb[black][pawn]),
        .castling = uint8_t(b.gamestate.castling_rights),
        // Syzygy's rule50 is the FIDE halfmove clock. fen.cpp emits it as
        // (b.half_moves + b.gamestate.half_moves) — the dynamic counter
        // plus the base value parsed from the starting FEN — so mirror
        // that here. Passing 0 (or only the base) made WDL overclaim
        // wins that should have been cursed-draws near the 50-move limit,
        // and made DTZ score moves off a rule50 that lagged reality.
        .rule50 = static_cast<uint8_t>(std::min(255,
            b.half_moves + static_cast<int>(b.gamestate.half_moves))),
        .ep = tablebase_ep_square(b.gamestate.enpassant_square),
        .turn = b.side == white,
        .move = static_cast<uint16_t>(b.histply + 1),
    };
}

Status WDL_probe(Board &board)
{
#if !ENYO_USE_SYZYGY
    (void)board;
    return Status::Error;
#else
    if (!initialized)
        return Status::Error;
    if (!has_table_for(board, available_wdl))
        return Status::Error;

    auto pos = board2pos(board);
    if (pos.castling != 0)
        return Status::Error;

    auto ret =
        tb_probe_wdl(
            pos.white, pos.black,
            pos.kings, pos.queens, pos.rooks, pos.bishops, pos.knights, pos.pawns,
            pos.ep,
            pos.turn
    );
    switch (ret) {
        case TB_WIN:            return Status::Win;
        case TB_CURSED_WIN:     return Status::Draw;
        case TB_DRAW:           return Status::Draw;
        case TB_BLESSED_LOSS:   return Status::Draw;
        case TB_LOSS:           return Status::Loss;
        default:
            return Status::Error;
    }
#endif
}

std::pair<int, Move> DTZ_probe(Board & board, Status & status, int * dtz)
{
    status = Status::Error;
    if (dtz)
        *dtz = -1;
#if !ENYO_USE_SYZYGY
    (void)board;
    return {0, Move{}};
#else
    if (!initialized)
        return {0, Move{}};
    if (!has_table_for(board, available_dtz))
        return {0, Move{}};

    auto pos = board2pos(board);
    if (pos.castling != 0)
        return {0, Move{}};

    unsigned TBresult =
        tb_probe_root(
            pos.white, pos.black,
            pos.kings, pos.queens, pos.rooks, pos.bishops, pos.knights, pos.pawns,
            pos.rule50,
            pos.ep,
            pos.turn,
            nullptr // results
        );

    if (TBresult == TB_RESULT_FAILED
     || TBresult == TB_RESULT_CHECKMATE
     || TBresult == TB_RESULT_STALEMATE)
        return {0, Move{}};

    const int wdl = TB_GET_WDL(TBresult);
    if (dtz)
        *dtz = static_cast<int>(TB_GET_DTZ(TBresult));

    int score = 0;
    switch (wdl) {
        case TB_WIN:          score = Value::tb_win_in_max_ply;  status = Status::Win;  break;
        case TB_LOSS:         score = Value::tb_loss_in_max_ply; status = Status::Loss; break;
        case TB_CURSED_WIN:
        case TB_DRAW:
        case TB_BLESSED_LOSS: score = 0;                         status = Status::Draw; break;
        default:              return {0, Move{}};
    }

    const int promo = TB_GET_PROMOTES(TBresult);
    const PieceType promoTranslation[] = {no_piece_type, queen, rook, bishop, knight};
    if (promo > PYRRHIC_FLAG_NPROMO)
        return {0, Move{}};

    const auto tablebase_from = unsigned(TB_GET_FROM(TBresult));
    const auto tablebase_to   = unsigned(TB_GET_TO(TBresult));

    Movelist legalmoves;
    if (board.side == white)
        legalmoves = generate_legal_moves<white>(board);
    else
        legalmoves = generate_legal_moves<black>(board);

    for (auto move : legalmoves) {
        if (sqconv(move.src_sq()) != tablebase_from) continue;
        if (sqconv(move.dst_sq()) != tablebase_to)   continue;
        if (promoTranslation[promo] == no_piece_type) {
            if (move.flags() != Move::Flags::promote)
                return {score, move};
        } else if (promo <= PYRRHIC_FLAG_NPROMO) {
            if (move.flags() == Move::Flags::promote
             && move.promo_piece() == promoTranslation[promo])
                return {score, move};
        }
    }
    return {0, Move{}};
#endif
}

std::vector<RootMove> root_moves(Board & board, const Movelist & root_moves, bool * complete)
{
    if (complete)
        *complete = false;
#if !ENYO_USE_SYZYGY
    (void)board;
    (void)root_moves;
    return {};
#else
    if (!initialized)
        return {};
    if (!has_table_for(board, available_wdl))
        return {};

    auto pos = board2pos(board);
    if (pos.castling != 0)
        return {};

    std::vector<RootMove> out;
    out.reserve(root_moves.size());

    if (has_table_for(board, available_dtz)) {
        std::array<unsigned, TB_MAX_MOVES + 1> results {};
        const unsigned root_result = tb_probe_root(
            pos.white, pos.black,
            pos.kings, pos.queens, pos.rooks, pos.bishops, pos.knights, pos.pawns,
            pos.rule50,
            pos.ep,
            pos.turn,
            results.data());

        if (root_result != TB_RESULT_FAILED
         && root_result != TB_RESULT_CHECKMATE
         && root_result != TB_RESULT_STALEMATE) {
            for (const auto root_move : root_moves) {
                bool found = false;
                for (const auto tb_result : results) {
                    if (tb_result == TB_RESULT_FAILED)
                        break;
                    if (!move_matches_tablebase_result(root_move, tb_result))
                        continue;

                    out.emplace_back(
                        root_move,
                        status_from_wdl(static_cast<int>(TB_GET_WDL(tb_result))),
                        static_cast<int>(TB_GET_DTZ(tb_result)));
                    found = true;
                    break;
                }
                if (!found)
                    return {};
            }
            if (complete)
                *complete = true;
            return out;
        }
    }

    const bool use_rule50 = pos.rule50 != 0 && has_table_for(board, available_dtz);
    TbRootMoves results {};
    const int ok = tb_probe_root_wdl(
        pos.white, pos.black,
        pos.kings, pos.queens, pos.rooks, pos.bishops, pos.knights, pos.pawns,
        use_rule50 ? pos.rule50 : 0,
        pos.ep,
        pos.turn,
        use_rule50,
        &results);
    if (!ok)
        return {};

    const PieceType promo_translation[] = {no_piece_type, queen, rook, bishop, knight};
    const auto matches = [&](Move move, PyrrhicMove tb_move) {
        if (sqconv(move.src_sq()) != PYRRHIC_MOVE_FROM(tb_move))
            return false;
        if (sqconv(move.dst_sq()) != PYRRHIC_MOVE_TO(tb_move))
            return false;

        const auto promo = PYRRHIC_MOVE_FLAGS(tb_move) & PYRRHIC_MASK_PROMO_FLAGS;
        if (promo == PYRRHIC_FLAG_NONE)
            return move.flags() != Move::Flags::promote;
        if (static_cast<size_t>(promo) >= std::size(promo_translation))
            return false;
        return move.flags() == Move::Flags::promote
            && move.promo_piece() == promo_translation[promo];
    };

    for (const auto root_move : root_moves) {
        bool found = false;
        for (unsigned i = 0; i < results.size; ++i) {
            if (!matches(root_move, results.moves[i].move))
                continue;
            const auto status =
                results.moves[i].tbRank > 0 ? Status::Win
              : results.moves[i].tbRank < 0 ? Status::Loss
                                            : Status::Draw;
            out.emplace_back(root_move, status, -1);
            found = true;
            break;
        }
        if (!found)
            return {};
    }

    if (complete)
        *complete = true;
    return out;
#endif
}

Movelist root_WDL_filter(Board & board, const Movelist & root_moves, bool * complete)
{
#if !ENYO_USE_SYZYGY
    (void)board;
    if (complete)
        *complete = false;
    return root_moves;
#else
    if (complete)
        *complete = false;
    if (!initialized)
        return root_moves;
    if (!has_table_for(board, available_wdl))
        return root_moves;

    auto pos = board2pos(board);
    if (pos.castling != 0)
        return root_moves;

    const bool use_rule50 = pos.rule50 != 0 && has_table_for(board, available_dtz);
    TbRootMoves results {};
    const int ok = tb_probe_root_wdl(
        pos.white, pos.black,
        pos.kings, pos.queens, pos.rooks, pos.bishops, pos.knights, pos.pawns,
        use_rule50 ? pos.rule50 : 0,
        pos.ep,
        pos.turn,
        use_rule50,
        &results);
    if (!ok) {
        Movelist filtered;
        int best_rank = -1;
        for (const auto root_move : root_moves) {
            Board child = board;
            if (board.side == white)
                apply_move<white, true, false>(child, root_move);
            else
                apply_move<black, true, false>(child, root_move);

            const auto child_status = WDL_probe(child);
            if (child_status == Status::Error)
                return root_moves;

            const int rank = root_rank_from_child_status(child_status);
            if (rank > best_rank) {
                filtered.clear();
                best_rank = rank;
            }
            if (rank == best_rank)
                filtered.emplace(root_move);
        }
        if (complete)
            *complete = true;
        return filtered.empty() ? root_moves : filtered;
    }

    const PieceType promo_translation[] = {no_piece_type, queen, rook, bishop, knight};
    const auto matches = [&](Move move, PyrrhicMove tb_move) {
        if (sqconv(move.src_sq()) != PYRRHIC_MOVE_FROM(tb_move))
            return false;
        if (sqconv(move.dst_sq()) != PYRRHIC_MOVE_TO(tb_move))
            return false;

        const auto promo = PYRRHIC_MOVE_FLAGS(tb_move) & PYRRHIC_MASK_PROMO_FLAGS;
        if (promo == PYRRHIC_FLAG_NONE)
            return move.flags() != Move::Flags::promote;
        if (static_cast<size_t>(promo) >= std::size(promo_translation))
            return false;
        return move.flags() == Move::Flags::promote
            && move.promo_piece() == promo_translation[promo];
    };

    Movelist filtered;
    int best_rank = std::numeric_limits<int>::min();
    for (const auto root_move : root_moves) {
        for (unsigned i = 0; i < results.size; ++i) {
            if (!matches(root_move, results.moves[i].move))
                continue;
            if (results.moves[i].tbRank > best_rank) {
                filtered.clear();
                best_rank = results.moves[i].tbRank;
            }
            if (results.moves[i].tbRank == best_rank)
                filtered.emplace(root_move);
            break;
        }
    }

    if (filtered.empty())
        return root_moves;
    if (complete)
        *complete = true;
    return filtered;
#endif
}

}  // namespace syzygy
