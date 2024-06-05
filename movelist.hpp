#pragma once
#include <iterator>
#include "types.hpp"
#include "config.hpp"

struct Movelist {
    typedef enyo::Move *iterator;
    typedef const enyo::Move *const_iterator;

    constexpr static size_t max_size = 256;
    enyo::Move moves[max_size]; // presumably max # legal moves for a given position
    size_t size_ = 0;

    inline void emplace(enyo::Move move) {
        moves[size_++] = move;
    }

    inline void clear() {
        size_ = 0;
    }

    constexpr inline enyo::Move & operator[](int i) {
        return moves[i];
    }

    constexpr inline int find(enyo::Move m) {
        auto it = std::ranges::find_if(moves, [&](const auto move) {
            return move.src_sq() == m.src_sq()
                && move.dst_sq() == m.dst_sq();
        });
        return (it != moves + size_)
            ? static_cast<int>(std::distance(moves, it))
            : -1;
    }

    constexpr inline size_t size() const {
        return size_;
    }

    constexpr inline bool empty() const {
        return size_ == 0;
    }

    constexpr inline iterator begin() {
        return (std::begin(moves));
    }

    constexpr inline const_iterator begin() const {
        return (std::begin(moves));
    }

    inline iterator end() {
        auto it = std::begin(moves);
        std::advance(it, size_);
        return it;
    }

    inline const_iterator end() const {
        auto it = std::begin(moves);
        std::advance(it, size_);
        return it;
    }

    inline constexpr enyo::Move & operator[](size_t i) {
        return moves[i];
    }

    inline constexpr const enyo::Move & operator[](size_t i) const {
        return moves[i];
    }
};
