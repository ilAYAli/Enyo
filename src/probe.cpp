#include "probe.hpp"
#if ENYO_USE_SYZYGY
#include "3rdparty/Fathom/src/tbprobe.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include "board.hpp"
#include "config.hpp"
#include "movegen.hpp"

using namespace enyo;

namespace {
bool initialized = false;

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

bool has_tablebase_files(const std::filesystem::path & path)
{
    std::error_code ec;
    if (!std::filesystem::is_directory(path, ec))
        return false;
    for (std::filesystem::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec))
            continue;
        const auto ext = it->path().extension();
        if (ext == ".rtbw" || ext == ".rtbz")
            return true;
    }
    return false;
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
        if (has_tablebase_files(path))
            append_unique(resolved, path);

        std::error_code ec;
        if (std::filesystem::is_directory(path, ec)) {
            for (std::filesystem::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
                if (it->is_directory(ec) && has_tablebase_files(it->path()))
                    append_unique(resolved, it->path());
            }
        }

        if (resolved.empty())
            append_unique(resolved, path);
    }
    return join_path_list(resolved);
}

bool init(const std::string & tb_path)
{
#if ENYO_USE_SYZYGY
    const auto resolved_path = resolve_path(tb_path);
    // tb_init returns true even when no tables were found; the real signal
    // is TB_LARGEST (max piece count of any loaded table).
    initialized = tb_init(resolved_path.c_str()) && TB_LARGEST > 0;
    return initialized;
#else
    (void)tb_path;
    initialized = false;
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

static inline constexpr uint8_t fathom_ep_square(square_t ep)
{
    // Enyo stores "no EP" as 0. Fathom also uses 0 as its no-EP sentinel,
    // so only convert real EP squares.
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
        .ep = fathom_ep_square(b.gamestate.enpassant_square),
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

    auto pos = board2pos(board);
    auto ret =
        tb_probe_wdl(
            pos.white, pos.black,
            pos.kings, pos.queens, pos.rooks, pos.bishops, pos.knights, pos.pawns,
            pos.rule50,
            pos.castling,
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

std::pair<int, Move> DTZ_probe(Board & board, Status & status)
{
    status = Status::Error;
#if !ENYO_USE_SYZYGY
    (void)board;
    return {0, Move{}};
#else
    if (!initialized)
        return {0, Move{}};

    auto pos = board2pos(board);

    // Fathom's root DTZ probe preserves WDL, but DTZ is not a practical
    // defensive policy for already-lost positions.
    unsigned TBresult =
        tb_probe_root(
            pos.white, pos.black,
            pos.kings, pos.queens, pos.rooks, pos.bishops, pos.knights, pos.pawns,
            pos.rule50,
            pos.castling,
            pos.ep,
            pos.turn,
            nullptr // results
        );

    if (TBresult == TB_RESULT_FAILED
     || TBresult == TB_RESULT_CHECKMATE
     || TBresult == TB_RESULT_STALEMATE)
        return {0, Move{}};

    const int wdl = TB_GET_WDL(TBresult);
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

    // Fathom hands back A1=0 squares; compare against legal moves in Enyo's
    // native H1=0 layout by translating the legal move into A1=0.
    const auto fathom_from = unsigned(TB_GET_FROM(TBresult));
    const auto fathom_to   = unsigned(TB_GET_TO(TBresult));

    Movelist legalmoves;
    if (board.side == white)
        legalmoves = generate_legal_moves<white>(board);
    else
        legalmoves = generate_legal_moves<black>(board);

    for (auto move : legalmoves) {
        if (sqconv(move.src_sq()) != fathom_from) continue;
        if (sqconv(move.dst_sq()) != fathom_to)   continue;
        if (promoTranslation[promo] == no_piece_type) {
            if (move.flags() != Move::Flags::promote)
                return {score, move};
        } else if (promo < 5) {
            if (move.flags() == Move::Flags::promote
             && move.promo_piece() == promoTranslation[promo])
                return {score, move};
        }
    }
    return {0, Move{}};
#endif
}

}  // namespace syzygy
