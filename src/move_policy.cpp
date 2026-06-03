#include "move_policy.hpp"

#include "movegen.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace enyo::move_policy {
namespace {

using json = nlohmann::json;

constexpr int compact_feature_count = 172;
constexpr int board_plane_count = 12 * 64;
constexpr int board_feature_count = compact_feature_count + 2 * board_plane_count;

int python_square(square_t sq)
{
    return static_cast<int>(sq ^ 7U);
}

int square_file_py(square_t sq)
{
    return python_square(sq) & 7;
}

int square_rank_py(square_t sq)
{
    return python_square(sq) >> 3;
}

int oriented_python_square(square_t sq, Color view)
{
    int out = python_square(sq);
    if (view == black)
        out ^= 56;
    return out;
}

int piece_index(PieceType pt)
{
    return pt == no_piece_type ? -1 : static_cast<int>(pt) - 1;
}

int capture_index(Move move)
{
    if (move.flags() & Move::Flags::enpassant)
        return piece_index(pawn);
    return piece_index(move.dst_piece());
}

int promotion_index(Move move)
{
    if ((move.flags() & Move::Flags::promote) == 0)
        return -1;
    return piece_index(move.promo_piece());
}

bool is_capture(Move move)
{
    return move.dst_piece() != no_piece_type || (move.flags() & Move::Flags::enpassant);
}

bool is_castling(Move move)
{
    return move.flags() & Move::Flags::castle;
}

int fullmove_number(Board const & board)
{
    const int ply_offset = board.gamestate.white_to_move ? 0 : 1;
    return static_cast<int>(board.gamestate.full_moves) + (board.histply + ply_offset) / 2;
}

int halfmove_clock(Board const & board)
{
    return board.half_moves + static_cast<int>(board.gamestate.half_moves);
}

template <Color Us>
Movelist legal_moves_for(Board const & board)
{
    Board copy = board;
    return generate_legal_moves<Us>(copy);
}

Movelist legal_moves_for(Board const & board)
{
    return board.side == white ? legal_moves_for<white>(board) : legal_moves_for<black>(board);
}

template <Color Us>
bool in_check(Board const & board)
{
    Board copy = board;
    return is_check<Us>(copy);
}

bool side_in_check(Board const & board)
{
    return board.side == white ? in_check<white>(board) : in_check<black>(board);
}

template <Color Us>
Board child_after(Board const & board, Move move)
{
    Board child = board;
    apply_move<Us, true, false>(child, move);
    return child;
}

Board child_after(Board const & board, Move move)
{
    return board.side == white ? child_after<white>(board, move) : child_after<black>(board, move);
}

bool move_gives_check(Board const & board, Move move)
{
    Board child = child_after(board, move);
    return side_in_check(child);
}

void append_one_hot(std::vector<double> & out, int index, int size)
{
    for (int i = 0; i < size; ++i)
        out.push_back(i == index ? 1.0 : 0.0);
}

void append_material(std::vector<double> & out, Board const & board)
{
    constexpr PieceType pieces[] = { pawn, knight, bishop, rook, queen, king };
    for (Color color : { white, black }) {
        for (PieceType pt : pieces)
            out.push_back(static_cast<double>(count_bits(board.pt_bb[color][pt])) / 8.0);
    }
}

void append_board_planes(std::vector<double> & out, Board const & board, Color view)
{
    const auto base = out.size();
    out.resize(base + board_plane_count, 0.0);

    for (Color color : { white, black }) {
        const int rel_color = color == view ? 0 : 1;
        for (PieceType pt : { pawn, knight, bishop, rook, queen, king }) {
            bitboard_t pieces = board.pt_bb[color][pt];
            while (pieces) {
                const square_t sq = pop_lsb(pieces);
                const int plane = rel_color * 6 + piece_index(pt);
                const int oriented = oriented_python_square(sq, view);
                out[base + static_cast<size_t>(plane * 64 + oriented)] = 1.0;
            }
        }
    }
}

FeatureSet parse_feature_set(std::string_view raw)
{
    if (raw == "compact")
        return FeatureSet::compact;
    if (raw == "board")
        return FeatureSet::board;
    throw std::runtime_error("unsupported move-policy feature_set");
}

std::vector<double> parse_vector(json const & value, std::string_view name)
{
    if (!value.is_array())
        throw std::runtime_error(std::string(name) + " must be an array");
    std::vector<double> out;
    out.reserve(value.size());
    for (auto const & item : value)
        out.push_back(item.get<double>());
    return out;
}

std::vector<std::vector<double>> parse_matrix(json const & value, std::string_view name)
{
    if (!value.is_array())
        throw std::runtime_error(std::string(name) + " must be an array");
    std::vector<std::vector<double>> out;
    out.reserve(value.size());
    for (auto const & row : value)
        out.push_back(parse_vector(row, name));
    return out;
}

std::vector<double> forward_layer(std::vector<double> const & input, Layer const & layer)
{
    std::vector<double> out(layer.bias.size(), 0.0);
    for (size_t i = 0; i < layer.bias.size(); ++i) {
        double value = layer.bias[i];
        for (size_t j = 0; j < input.size(); ++j)
            value += layer.weight[i][j] * input[j];
        if (layer.activation == "relu")
            value = std::max(0.0, value);
        out[i] = value;
    }
    return out;
}

} // namespace

const char * feature_set_name(FeatureSet feature_set)
{
    return feature_set == FeatureSet::board ? "board" : "compact";
}

std::vector<double> features(Board const & board, Move move, FeatureSet feature_set)
{
    const Movelist legal = legal_moves_for(board);
    const Board child = child_after(board, move);
    const Movelist child_legal = legal_moves_for(child);
    const Color view = board.side;

    std::vector<double> out;
    out.reserve(feature_set == FeatureSet::board ? board_feature_count : compact_feature_count);
    out.push_back(board.side == white ? 1.0 : -1.0);
    out.push_back(static_cast<double>(legal.size()) / 80.0);
    out.push_back(std::min(static_cast<double>(fullmove_number(board)), 120.0) / 120.0);
    out.push_back(static_cast<double>(halfmove_clock(board)) / 100.0);
    out.push_back(side_in_check(board) ? 1.0 : 0.0);
    out.push_back(is_capture(move) ? 1.0 : 0.0);
    out.push_back(move_gives_check(board, move) ? 1.0 : 0.0);
    out.push_back(is_castling(move) ? 1.0 : 0.0);
    out.push_back((move.flags() & Move::Flags::enpassant) ? 1.0 : 0.0);
    out.push_back((move.flags() & Move::Flags::promote) ? 1.0 : 0.0);
    out.push_back(static_cast<double>(square_file_py(move.dst_sq()) - square_file_py(move.src_sq())) / 7.0);
    out.push_back(static_cast<double>(square_rank_py(move.dst_sq()) - square_rank_py(move.src_sq())) / 7.0);
    out.push_back(static_cast<double>(child_legal.size()) / 80.0);
    out.push_back(side_in_check(child) ? 1.0 : 0.0);
    append_one_hot(out, python_square(move.src_sq()), 64);
    append_one_hot(out, python_square(move.dst_sq()), 64);
    append_one_hot(out, piece_index(move.src_piece()), 6);
    append_one_hot(out, capture_index(move), 6);
    append_one_hot(out, promotion_index(move), 6);
    append_material(out, board);

    if (feature_set == FeatureSet::board) {
        append_board_planes(out, board, view);
        append_board_planes(out, child, view);
    }
    return out;
}

bool Model::load(std::string_view path, std::string * error)
{
    try {
        std::ifstream in{std::string(path)};
        if (!in) {
            if (error)
                *error = "failed to open model";
            return false;
        }

        json payload;
        in >> payload;
        if (payload.value("schema", "") != "enyo.move_policy.v1")
            throw std::runtime_error("unsupported move-policy schema");

        const auto feature_set = parse_feature_set(payload.value("feature_set", "compact"));
        const int input_dim = payload.at("input_dim").get<int>();
        const double threshold = payload.value("threshold", 0.0);
        auto mean = parse_vector(payload.at("mean"), "mean");
        auto std = parse_vector(payload.at("std"), "std");
        if (input_dim <= 0 || mean.size() != static_cast<size_t>(input_dim)
            || std.size() != static_cast<size_t>(input_dim)) {
            throw std::runtime_error("normalization vectors do not match input_dim");
        }

        std::vector<Layer> layers;
        int previous_dim = input_dim;
        for (auto const & raw_layer : payload.at("layers")) {
            Layer layer;
            layer.activation = raw_layer.value("activation", "linear");
            layer.weight = parse_matrix(raw_layer.at("weight"), "weight");
            layer.bias = parse_vector(raw_layer.at("bias"), "bias");
            if (layer.weight.size() != layer.bias.size())
                throw std::runtime_error("layer weight rows do not match bias");
            for (auto const & row : layer.weight) {
                if (row.size() != static_cast<size_t>(previous_dim))
                    throw std::runtime_error("layer input dimension mismatch");
            }
            previous_dim = static_cast<int>(layer.bias.size());
            layers.push_back(std::move(layer));
        }
        if (previous_dim != 1)
            throw std::runtime_error("move-policy model must produce one output");

        feature_set_ = feature_set;
        input_dim_ = input_dim;
        threshold_ = threshold;
        mean_ = std::move(mean);
        std_ = std::move(std);
        layers_ = std::move(layers);
        loaded_ = true;
        return true;
    } catch (std::exception const & ex) {
        loaded_ = false;
        if (error)
            *error = ex.what();
        return false;
    }
}

double Model::score(Board const & board, Move move) const
{
    if (!loaded_)
        throw std::runtime_error("move-policy model is not loaded");
    auto x = features(board, move, feature_set_);
    if (x.size() != static_cast<size_t>(input_dim_))
        throw std::runtime_error("move-policy feature dimension mismatch");
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = (x[i] - mean_[i]) / std_[i];
    for (auto const & layer : layers_)
        x = forward_layer(x, layer);
    return x.front();
}

double Model::margin(Board const & board, Move best, Move played) const
{
    return score(board, best) - score(board, played);
}

void Model::clear()
{
    loaded_ = false;
    feature_set_ = FeatureSet::compact;
    input_dim_ = 0;
    threshold_ = 0.0;
    mean_.clear();
    std_.clear();
    layers_.clear();
}

Model & runtime_model()
{
    static Model model;
    return model;
}

bool load_runtime_model(std::string_view path, std::string * error)
{
    return runtime_model().load(path, error);
}

void clear_runtime_model()
{
    runtime_model().clear();
}

} // namespace enyo::move_policy
