#ifndef MVLOOKUP_HPP
#define MVLOOKUP_HPP

#include <cstdint>

constexpr const char* mvlookup[64][64] = {
    {"h1h1", "h1g1", "h1f1", "h1e1", "h1d1", "h1c1", "h1b1", "h1a1", "h1h2", "h1g2", "h1f2", "h1e2", "h1d2", "h1c2", "h1b2", "h1a2", "h1h3", "h1g3", "h1f3", "h1e3", "h1d3", "h1c3", "h1b3", "h1a3", "h1h4", "h1g4", "h1f4", "h1e4", "h1d4", "h1c4", "h1b4", "h1a4", "h1h5", "h1g5", "h1f5", "h1e5", "h1d5", "h1c5", "h1b5", "h1a5", "h1h6", "h1g6", "h1f6", "h1e6", "h1d6", "h1c6", "h1b6", "h1a6", "h1h7", "h1g7", "h1f7", "h1e7", "h1d7", "h1c7", "h1b7", "h1a7", "h1h8", "h1g8", "h1f8", "h1e8", "h1d8", "h1c8", "h1b8", "h1a8"},
    {"g1h1", "g1g1", "g1f1", "g1e1", "g1d1", "g1c1", "g1b1", "g1a1", "g1h2", "g1g2", "g1f2", "g1e2", "g1d2", "g1c2", "g1b2", "g1a2", "g1h3", "g1g3", "g1f3", "g1e3", "g1d3", "g1c3", "g1b3", "g1a3", "g1h4", "g1g4", "g1f4", "g1e4", "g1d4", "g1c4", "g1b4", "g1a4", "g1h5", "g1g5", "g1f5", "g1e5", "g1d5", "g1c5", "g1b5", "g1a5", "g1h6", "g1g6", "g1f6", "g1e6", "g1d6", "g1c6", "g1b6", "g1a6", "g1h7", "g1g7", "g1f7", "g1e7", "g1d7", "g1c7", "g1b7", "g1a7", "g1h8", "g1g8", "g1f8", "g1e8", "g1d8", "g1c8", "g1b8", "g1a8"},
    {"f1h1", "f1g1", "f1f1", "f1e1", "f1d1", "f1c1", "f1b1", "f1a1", "f1h2", "f1g2", "f1f2", "f1e2", "f1d2", "f1c2", "f1b2", "f1a2", "f1h3", "f1g3", "f1f3", "f1e3", "f1d3", "f1c3", "f1b3", "f1a3", "f1h4", "f1g4", "f1f4", "f1e4", "f1d4", "f1c4", "f1b4", "f1a4", "f1h5", "f1g5", "f1f5", "f1e5", "f1d5", "f1c5", "f1b5", "f1a5", "f1h6", "f1g6", "f1f6", "f1e6", "f1d6", "f1c6", "f1b6", "f1a6", "f1h7", "f1g7", "f1f7", "f1e7", "f1d7", "f1c7", "f1b7", "f1a7", "f1h8", "f1g8", "f1f8", "f1e8", "f1d8", "f1c8", "f1b8", "f1a8"},
    {"e1h1", "e1g1", "e1f1", "e1e1", "e1d1", "e1c1", "e1b1", "e1a1", "e1h2", "e1g2", "e1f2", "e1e2", "e1d2", "e1c2", "e1b2", "e1a2", "e1h3", "e1g3", "e1f3", "e1e3", "e1d3", "e1c3", "e1b3", "e1a3", "e1h4", "e1g4", "e1f4", "e1e4", "e1d4", "e1c4", "e1b4", "e1a4", "e1h5", "e1g5", "e1f5", "e1e5", "e1d5", "e1c5", "e1b5", "e1a5", "e1h6", "e1g6", "e1f6", "e1e6", "e1d6", "e1c6", "e1b6", "e1a6", "e1h7", "e1g7", "e1f7", "e1e7", "e1d7", "e1c7", "e1b7", "e1a7", "e1h8", "e1g8", "e1f8", "e1e8", "e1d8", "e1c8", "e1b8", "e1a8"},
    {"d1h1", "d1g1", "d1f1", "d1e1", "d1d1", "d1c1", "d1b1", "d1a1", "d1h2", "d1g2", "d1f2", "d1e2", "d1d2", "d1c2", "d1b2", "d1a2", "d1h3", "d1g3", "d1f3", "d1e3", "d1d3", "d1c3", "d1b3", "d1a3", "d1h4", "d1g4", "d1f4", "d1e4", "d1d4", "d1c4", "d1b4", "d1a4", "d1h5", "d1g5", "d1f5", "d1e5", "d1d5", "d1c5", "d1b5", "d1a5", "d1h6", "d1g6", "d1f6", "d1e6", "d1d6", "d1c6", "d1b6", "d1a6", "d1h7", "d1g7", "d1f7", "d1e7", "d1d7", "d1c7", "d1b7", "d1a7", "d1h8", "d1g8", "d1f8", "d1e8", "d1d8", "d1c8", "d1b8", "d1a8"},
    {"c1h1", "c1g1", "c1f1", "c1e1", "c1d1", "c1c1", "c1b1", "c1a1", "c1h2", "c1g2", "c1f2", "c1e2", "c1d2", "c1c2", "c1b2", "c1a2", "c1h3", "c1g3", "c1f3", "c1e3", "c1d3", "c1c3", "c1b3", "c1a3", "c1h4", "c1g4", "c1f4", "c1e4", "c1d4", "c1c4", "c1b4", "c1a4", "c1h5", "c1g5", "c1f5", "c1e5", "c1d5", "c1c5", "c1b5", "c1a5", "c1h6", "c1g6", "c1f6", "c1e6", "c1d6", "c1c6", "c1b6", "c1a6", "c1h7", "c1g7", "c1f7", "c1e7", "c1d7", "c1c7", "c1b7", "c1a7", "c1h8", "c1g8", "c1f8", "c1e8", "c1d8", "c1c8", "c1b8", "c1a8"},
    {"b1h1", "b1g1", "b1f1", "b1e1", "b1d1", "b1c1", "b1b1", "b1a1", "b1h2", "b1g2", "b1f2", "b1e2", "b1d2", "b1c2", "b1b2", "b1a2", "b1h3", "b1g3", "b1f3", "b1e3", "b1d3", "b1c3", "b1b3", "b1a3", "b1h4", "b1g4", "b1f4", "b1e4", "b1d4", "b1c4", "b1b4", "b1a4", "b1h5", "b1g5", "b1f5", "b1e5", "b1d5", "b1c5", "b1b5", "b1a5", "b1h6", "b1g6", "b1f6", "b1e6", "b1d6", "b1c6", "b1b6", "b1a6", "b1h7", "b1g7", "b1f7", "b1e7", "b1d7", "b1c7", "b1b7", "b1a7", "b1h8", "b1g8", "b1f8", "b1e8", "b1d8", "b1c8", "b1b8", "b1a8"},
    {"a1h1", "a1g1", "a1f1", "a1e1", "a1d1", "a1c1", "a1b1", "a1a1", "a1h2", "a1g2", "a1f2", "a1e2", "a1d2", "a1c2", "a1b2", "a1a2", "a1h3", "a1g3", "a1f3", "a1e3", "a1d3", "a1c3", "a1b3", "a1a3", "a1h4", "a1g4", "a1f4", "a1e4", "a1d4", "a1c4", "a1b4", "a1a4", "a1h5", "a1g5", "a1f5", "a1e5", "a1d5", "a1c5", "a1b5", "a1a5", "a1h6", "a1g6", "a1f6", "a1e6", "a1d6", "a1c6", "a1b6", "a1a6", "a1h7", "a1g7", "a1f7", "a1e7", "a1d7", "a1c7", "a1b7", "a1a7", "a1h8", "a1g8", "a1f8", "a1e8", "a1d8", "a1c8", "a1b8", "a1a8"},
    {"h2h1", "h2g1", "h2f1", "h2e1", "h2d1", "h2c1", "h2b1", "h2a1", "h2h2", "h2g2", "h2f2", "h2e2", "h2d2", "h2c2", "h2b2", "h2a2", "h2h3", "h2g3", "h2f3", "h2e3", "h2d3", "h2c3", "h2b3", "h2a3", "h2h4", "h2g4", "h2f4", "h2e4", "h2d4", "h2c4", "h2b4", "h2a4", "h2h5", "h2g5", "h2f5", "h2e5", "h2d5", "h2c5", "h2b5", "h2a5", "h2h6", "h2g6", "h2f6", "h2e6", "h2d6", "h2c6", "h2b6", "h2a6", "h2h7", "h2g7", "h2f7", "h2e7", "h2d7", "h2c7", "h2b7", "h2a7", "h2h8", "h2g8", "h2f8", "h2e8", "h2d8", "h2c8", "h2b8", "h2a8"},
    {"g2h1", "g2g1", "g2f1", "g2e1", "g2d1", "g2c1", "g2b1", "g2a1", "g2h2", "g2g2", "g2f2", "g2e2", "g2d2", "g2c2", "g2b2", "g2a2", "g2h3", "g2g3", "g2f3", "g2e3", "g2d3", "g2c3", "g2b3", "g2a3", "g2h4", "g2g4", "g2f4", "g2e4", "g2d4", "g2c4", "g2b4", "g2a4", "g2h5", "g2g5", "g2f5", "g2e5", "g2d5", "g2c5", "g2b5", "g2a5", "g2h6", "g2g6", "g2f6", "g2e6", "g2d6", "g2c6", "g2b6", "g2a6", "g2h7", "g2g7", "g2f7", "g2e7", "g2d7", "g2c7", "g2b7", "g2a7", "g2h8", "g2g8", "g2f8", "g2e8", "g2d8", "g2c8", "g2b8", "g2a8"},
    {"f2h1", "f2g1", "f2f1", "f2e1", "f2d1", "f2c1", "f2b1", "f2a1", "f2h2", "f2g2", "f2f2", "f2e2", "f2d2", "f2c2", "f2b2", "f2a2", "f2h3", "f2g3", "f2f3", "f2e3", "f2d3", "f2c3", "f2b3", "f2a3", "f2h4", "f2g4", "f2f4", "f2e4", "f2d4", "f2c4", "f2b4", "f2a4", "f2h5", "f2g5", "f2f5", "f2e5", "f2d5", "f2c5", "f2b5", "f2a5", "f2h6", "f2g6", "f2f6", "f2e6", "f2d6", "f2c6", "f2b6", "f2a6", "f2h7", "f2g7", "f2f7", "f2e7", "f2d7", "f2c7", "f2b7", "f2a7", "f2h8", "f2g8", "f2f8", "f2e8", "f2d8", "f2c8", "f2b8", "f2a8"},
    {"e2h1", "e2g1", "e2f1", "e2e1", "e2d1", "e2c1", "e2b1", "e2a1", "e2h2", "e2g2", "e2f2", "e2e2", "e2d2", "e2c2", "e2b2", "e2a2", "e2h3", "e2g3", "e2f3", "e2e3", "e2d3", "e2c3", "e2b3", "e2a3", "e2h4", "e2g4", "e2f4", "e2e4", "e2d4", "e2c4", "e2b4", "e2a4", "e2h5", "e2g5", "e2f5", "e2e5", "e2d5", "e2c5", "e2b5", "e2a5", "e2h6", "e2g6", "e2f6", "e2e6", "e2d6", "e2c6", "e2b6", "e2a6", "e2h7", "e2g7", "e2f7", "e2e7", "e2d7", "e2c7", "e2b7", "e2a7", "e2h8", "e2g8", "e2f8", "e2e8", "e2d8", "e2c8", "e2b8", "e2a8"},
    {"d2h1", "d2g1", "d2f1", "d2e1", "d2d1", "d2c1", "d2b1", "d2a1", "d2h2", "d2g2", "d2f2", "d2e2", "d2d2", "d2c2", "d2b2", "d2a2", "d2h3", "d2g3", "d2f3", "d2e3", "d2d3", "d2c3", "d2b3", "d2a3", "d2h4", "d2g4", "d2f4", "d2e4", "d2d4", "d2c4", "d2b4", "d2a4", "d2h5", "d2g5", "d2f5", "d2e5", "d2d5", "d2c5", "d2b5", "d2a5", "d2h6", "d2g6", "d2f6", "d2e6", "d2d6", "d2c6", "d2b6", "d2a6", "d2h7", "d2g7", "d2f7", "d2e7", "d2d7", "d2c7", "d2b7", "d2a7", "d2h8", "d2g8", "d2f8", "d2e8", "d2d8", "d2c8", "d2b8", "d2a8"},
    {"c2h1", "c2g1", "c2f1", "c2e1", "c2d1", "c2c1", "c2b1", "c2a1", "c2h2", "c2g2", "c2f2", "c2e2", "c2d2", "c2c2", "c2b2", "c2a2", "c2h3", "c2g3", "c2f3", "c2e3", "c2d3", "c2c3", "c2b3", "c2a3", "c2h4", "c2g4", "c2f4", "c2e4", "c2d4", "c2c4", "c2b4", "c2a4", "c2h5", "c2g5", "c2f5", "c2e5", "c2d5", "c2c5", "c2b5", "c2a5", "c2h6", "c2g6", "c2f6", "c2e6", "c2d6", "c2c6", "c2b6", "c2a6", "c2h7", "c2g7", "c2f7", "c2e7", "c2d7", "c2c7", "c2b7", "c2a7", "c2h8", "c2g8", "c2f8", "c2e8", "c2d8", "c2c8", "c2b8", "c2a8"},
    {"b2h1", "b2g1", "b2f1", "b2e1", "b2d1", "b2c1", "b2b1", "b2a1", "b2h2", "b2g2", "b2f2", "b2e2", "b2d2", "b2c2", "b2b2", "b2a2", "b2h3", "b2g3", "b2f3", "b2e3", "b2d3", "b2c3", "b2b3", "b2a3", "b2h4", "b2g4", "b2f4", "b2e4", "b2d4", "b2c4", "b2b4", "b2a4", "b2h5", "b2g5", "b2f5", "b2e5", "b2d5", "b2c5", "b2b5", "b2a5", "b2h6", "b2g6", "b2f6", "b2e6", "b2d6", "b2c6", "b2b6", "b2a6", "b2h7", "b2g7", "b2f7", "b2e7", "b2d7", "b2c7", "b2b7", "b2a7", "b2h8", "b2g8", "b2f8", "b2e8", "b2d8", "b2c8", "b2b8", "b2a8"},
    {"a2h1", "a2g1", "a2f1", "a2e1", "a2d1", "a2c1", "a2b1", "a2a1", "a2h2", "a2g2", "a2f2", "a2e2", "a2d2", "a2c2", "a2b2", "a2a2", "a2h3", "a2g3", "a2f3", "a2e3", "a2d3", "a2c3", "a2b3", "a2a3", "a2h4", "a2g4", "a2f4", "a2e4", "a2d4", "a2c4", "a2b4", "a2a4", "a2h5", "a2g5", "a2f5", "a2e5", "a2d5", "a2c5", "a2b5", "a2a5", "a2h6", "a2g6", "a2f6", "a2e6", "a2d6", "a2c6", "a2b6", "a2a6", "a2h7", "a2g7", "a2f7", "a2e7", "a2d7", "a2c7", "a2b7", "a2a7", "a2h8", "a2g8", "a2f8", "a2e8", "a2d8", "a2c8", "a2b8", "a2a8"},
    {"h3h1", "h3g1", "h3f1", "h3e1", "h3d1", "h3c1", "h3b1", "h3a1", "h3h2", "h3g2", "h3f2", "h3e2", "h3d2", "h3c2", "h3b2", "h3a2", "h3h3", "h3g3", "h3f3", "h3e3", "h3d3", "h3c3", "h3b3", "h3a3", "h3h4", "h3g4", "h3f4", "h3e4", "h3d4", "h3c4", "h3b4", "h3a4", "h3h5", "h3g5", "h3f5", "h3e5", "h3d5", "h3c5", "h3b5", "h3a5", "h3h6", "h3g6", "h3f6", "h3e6", "h3d6", "h3c6", "h3b6", "h3a6", "h3h7", "h3g7", "h3f7", "h3e7", "h3d7", "h3c7", "h3b7", "h3a7", "h3h8", "h3g8", "h3f8", "h3e8", "h3d8", "h3c8", "h3b8", "h3a8"},
    {"g3h1", "g3g1", "g3f1", "g3e1", "g3d1", "g3c1", "g3b1", "g3a1", "g3h2", "g3g2", "g3f2", "g3e2", "g3d2", "g3c2", "g3b2", "g3a2", "g3h3", "g3g3", "g3f3", "g3e3", "g3d3", "g3c3", "g3b3", "g3a3", "g3h4", "g3g4", "g3f4", "g3e4", "g3d4", "g3c4", "g3b4", "g3a4", "g3h5", "g3g5", "g3f5", "g3e5", "g3d5", "g3c5", "g3b5", "g3a5", "g3h6", "g3g6", "g3f6", "g3e6", "g3d6", "g3c6", "g3b6", "g3a6", "g3h7", "g3g7", "g3f7", "g3e7", "g3d7", "g3c7", "g3b7", "g3a7", "g3h8", "g3g8", "g3f8", "g3e8", "g3d8", "g3c8", "g3b8", "g3a8"},
    {"f3h1", "f3g1", "f3f1", "f3e1", "f3d1", "f3c1", "f3b1", "f3a1", "f3h2", "f3g2", "f3f2", "f3e2", "f3d2", "f3c2", "f3b2", "f3a2", "f3h3", "f3g3", "f3f3", "f3e3", "f3d3", "f3c3", "f3b3", "f3a3", "f3h4", "f3g4", "f3f4", "f3e4", "f3d4", "f3c4", "f3b4", "f3a4", "f3h5", "f3g5", "f3f5", "f3e5", "f3d5", "f3c5", "f3b5", "f3a5", "f3h6", "f3g6", "f3f6", "f3e6", "f3d6", "f3c6", "f3b6", "f3a6", "f3h7", "f3g7", "f3f7", "f3e7", "f3d7", "f3c7", "f3b7", "f3a7", "f3h8", "f3g8", "f3f8", "f3e8", "f3d8", "f3c8", "f3b8", "f3a8"},
    {"e3h1", "e3g1", "e3f1", "e3e1", "e3d1", "e3c1", "e3b1", "e3a1", "e3h2", "e3g2", "e3f2", "e3e2", "e3d2", "e3c2", "e3b2", "e3a2", "e3h3", "e3g3", "e3f3", "e3e3", "e3d3", "e3c3", "e3b3", "e3a3", "e3h4", "e3g4", "e3f4", "e3e4", "e3d4", "e3c4", "e3b4", "e3a4", "e3h5", "e3g5", "e3f5", "e3e5", "e3d5", "e3c5", "e3b5", "e3a5", "e3h6", "e3g6", "e3f6", "e3e6", "e3d6", "e3c6", "e3b6", "e3a6", "e3h7", "e3g7", "e3f7", "e3e7", "e3d7", "e3c7", "e3b7", "e3a7", "e3h8", "e3g8", "e3f8", "e3e8", "e3d8", "e3c8", "e3b8", "e3a8"},
    {"d3h1", "d3g1", "d3f1", "d3e1", "d3d1", "d3c1", "d3b1", "d3a1", "d3h2", "d3g2", "d3f2", "d3e2", "d3d2", "d3c2", "d3b2", "d3a2", "d3h3", "d3g3", "d3f3", "d3e3", "d3d3", "d3c3", "d3b3", "d3a3", "d3h4", "d3g4", "d3f4", "d3e4", "d3d4", "d3c4", "d3b4", "d3a4", "d3h5", "d3g5", "d3f5", "d3e5", "d3d5", "d3c5", "d3b5", "d3a5", "d3h6", "d3g6", "d3f6", "d3e6", "d3d6", "d3c6", "d3b6", "d3a6", "d3h7", "d3g7", "d3f7", "d3e7", "d3d7", "d3c7", "d3b7", "d3a7", "d3h8", "d3g8", "d3f8", "d3e8", "d3d8", "d3c8", "d3b8", "d3a8"},
    {"c3h1", "c3g1", "c3f1", "c3e1", "c3d1", "c3c1", "c3b1", "c3a1", "c3h2", "c3g2", "c3f2", "c3e2", "c3d2", "c3c2", "c3b2", "c3a2", "c3h3", "c3g3", "c3f3", "c3e3", "c3d3", "c3c3", "c3b3", "c3a3", "c3h4", "c3g4", "c3f4", "c3e4", "c3d4", "c3c4", "c3b4", "c3a4", "c3h5", "c3g5", "c3f5", "c3e5", "c3d5", "c3c5", "c3b5", "c3a5", "c3h6", "c3g6", "c3f6", "c3e6", "c3d6", "c3c6", "c3b6", "c3a6", "c3h7", "c3g7", "c3f7", "c3e7", "c3d7", "c3c7", "c3b7", "c3a7", "c3h8", "c3g8", "c3f8", "c3e8", "c3d8", "c3c8", "c3b8", "c3a8"},
    {"b3h1", "b3g1", "b3f1", "b3e1", "b3d1", "b3c1", "b3b1", "b3a1", "b3h2", "b3g2", "b3f2", "b3e2", "b3d2", "b3c2", "b3b2", "b3a2", "b3h3", "b3g3", "b3f3", "b3e3", "b3d3", "b3c3", "b3b3", "b3a3", "b3h4", "b3g4", "b3f4", "b3e4", "b3d4", "b3c4", "b3b4", "b3a4", "b3h5", "b3g5", "b3f5", "b3e5", "b3d5", "b3c5", "b3b5", "b3a5", "b3h6", "b3g6", "b3f6", "b3e6", "b3d6", "b3c6", "b3b6", "b3a6", "b3h7", "b3g7", "b3f7", "b3e7", "b3d7", "b3c7", "b3b7", "b3a7", "b3h8", "b3g8", "b3f8", "b3e8", "b3d8", "b3c8", "b3b8", "b3a8"},
    {"a3h1", "a3g1", "a3f1", "a3e1", "a3d1", "a3c1", "a3b1", "a3a1", "a3h2", "a3g2", "a3f2", "a3e2", "a3d2", "a3c2", "a3b2", "a3a2", "a3h3", "a3g3", "a3f3", "a3e3", "a3d3", "a3c3", "a3b3", "a3a3", "a3h4", "a3g4", "a3f4", "a3e4", "a3d4", "a3c4", "a3b4", "a3a4", "a3h5", "a3g5", "a3f5", "a3e5", "a3d5", "a3c5", "a3b5", "a3a5", "a3h6", "a3g6", "a3f6", "a3e6", "a3d6", "a3c6", "a3b6", "a3a6", "a3h7", "a3g7", "a3f7", "a3e7", "a3d7", "a3c7", "a3b7", "a3a7", "a3h8", "a3g8", "a3f8", "a3e8", "a3d8", "a3c8", "a3b8", "a3a8"},
    {"h4h1", "h4g1", "h4f1", "h4e1", "h4d1", "h4c1", "h4b1", "h4a1", "h4h2", "h4g2", "h4f2", "h4e2", "h4d2", "h4c2", "h4b2", "h4a2", "h4h3", "h4g3", "h4f3", "h4e3", "h4d3", "h4c3", "h4b3", "h4a3", "h4h4", "h4g4", "h4f4", "h4e4", "h4d4", "h4c4", "h4b4", "h4a4", "h4h5", "h4g5", "h4f5", "h4e5", "h4d5", "h4c5", "h4b5", "h4a5", "h4h6", "h4g6", "h4f6", "h4e6", "h4d6", "h4c6", "h4b6", "h4a6", "h4h7", "h4g7", "h4f7", "h4e7", "h4d7", "h4c7", "h4b7", "h4a7", "h4h8", "h4g8", "h4f8", "h4e8", "h4d8", "h4c8", "h4b8", "h4a8"},
    {"g4h1", "g4g1", "g4f1", "g4e1", "g4d1", "g4c1", "g4b1", "g4a1", "g4h2", "g4g2", "g4f2", "g4e2", "g4d2", "g4c2", "g4b2", "g4a2", "g4h3", "g4g3", "g4f3", "g4e3", "g4d3", "g4c3", "g4b3", "g4a3", "g4h4", "g4g4", "g4f4", "g4e4", "g4d4", "g4c4", "g4b4", "g4a4", "g4h5", "g4g5", "g4f5", "g4e5", "g4d5", "g4c5", "g4b5", "g4a5", "g4h6", "g4g6", "g4f6", "g4e6", "g4d6", "g4c6", "g4b6", "g4a6", "g4h7", "g4g7", "g4f7", "g4e7", "g4d7", "g4c7", "g4b7", "g4a7", "g4h8", "g4g8", "g4f8", "g4e8", "g4d8", "g4c8", "g4b8", "g4a8"},
    {"f4h1", "f4g1", "f4f1", "f4e1", "f4d1", "f4c1", "f4b1", "f4a1", "f4h2", "f4g2", "f4f2", "f4e2", "f4d2", "f4c2", "f4b2", "f4a2", "f4h3", "f4g3", "f4f3", "f4e3", "f4d3", "f4c3", "f4b3", "f4a3", "f4h4", "f4g4", "f4f4", "f4e4", "f4d4", "f4c4", "f4b4", "f4a4", "f4h5", "f4g5", "f4f5", "f4e5", "f4d5", "f4c5", "f4b5", "f4a5", "f4h6", "f4g6", "f4f6", "f4e6", "f4d6", "f4c6", "f4b6", "f4a6", "f4h7", "f4g7", "f4f7", "f4e7", "f4d7", "f4c7", "f4b7", "f4a7", "f4h8", "f4g8", "f4f8", "f4e8", "f4d8", "f4c8", "f4b8", "f4a8"},
    {"e4h1", "e4g1", "e4f1", "e4e1", "e4d1", "e4c1", "e4b1", "e4a1", "e4h2", "e4g2", "e4f2", "e4e2", "e4d2", "e4c2", "e4b2", "e4a2", "e4h3", "e4g3", "e4f3", "e4e3", "e4d3", "e4c3", "e4b3", "e4a3", "e4h4", "e4g4", "e4f4", "e4e4", "e4d4", "e4c4", "e4b4", "e4a4", "e4h5", "e4g5", "e4f5", "e4e5", "e4d5", "e4c5", "e4b5", "e4a5", "e4h6", "e4g6", "e4f6", "e4e6", "e4d6", "e4c6", "e4b6", "e4a6", "e4h7", "e4g7", "e4f7", "e4e7", "e4d7", "e4c7", "e4b7", "e4a7", "e4h8", "e4g8", "e4f8", "e4e8", "e4d8", "e4c8", "e4b8", "e4a8"},
    {"d4h1", "d4g1", "d4f1", "d4e1", "d4d1", "d4c1", "d4b1", "d4a1", "d4h2", "d4g2", "d4f2", "d4e2", "d4d2", "d4c2", "d4b2", "d4a2", "d4h3", "d4g3", "d4f3", "d4e3", "d4d3", "d4c3", "d4b3", "d4a3", "d4h4", "d4g4", "d4f4", "d4e4", "d4d4", "d4c4", "d4b4", "d4a4", "d4h5", "d4g5", "d4f5", "d4e5", "d4d5", "d4c5", "d4b5", "d4a5", "d4h6", "d4g6", "d4f6", "d4e6", "d4d6", "d4c6", "d4b6", "d4a6", "d4h7", "d4g7", "d4f7", "d4e7", "d4d7", "d4c7", "d4b7", "d4a7", "d4h8", "d4g8", "d4f8", "d4e8", "d4d8", "d4c8", "d4b8", "d4a8"},
    {"c4h1", "c4g1", "c4f1", "c4e1", "c4d1", "c4c1", "c4b1", "c4a1", "c4h2", "c4g2", "c4f2", "c4e2", "c4d2", "c4c2", "c4b2", "c4a2", "c4h3", "c4g3", "c4f3", "c4e3", "c4d3", "c4c3", "c4b3", "c4a3", "c4h4", "c4g4", "c4f4", "c4e4", "c4d4", "c4c4", "c4b4", "c4a4", "c4h5", "c4g5", "c4f5", "c4e5", "c4d5", "c4c5", "c4b5", "c4a5", "c4h6", "c4g6", "c4f6", "c4e6", "c4d6", "c4c6", "c4b6", "c4a6", "c4h7", "c4g7", "c4f7", "c4e7", "c4d7", "c4c7", "c4b7", "c4a7", "c4h8", "c4g8", "c4f8", "c4e8", "c4d8", "c4c8", "c4b8", "c4a8"},
    {"b4h1", "b4g1", "b4f1", "b4e1", "b4d1", "b4c1", "b4b1", "b4a1", "b4h2", "b4g2", "b4f2", "b4e2", "b4d2", "b4c2", "b4b2", "b4a2", "b4h3", "b4g3", "b4f3", "b4e3", "b4d3", "b4c3", "b4b3", "b4a3", "b4h4", "b4g4", "b4f4", "b4e4", "b4d4", "b4c4", "b4b4", "b4a4", "b4h5", "b4g5", "b4f5", "b4e5", "b4d5", "b4c5", "b4b5", "b4a5", "b4h6", "b4g6", "b4f6", "b4e6", "b4d6", "b4c6", "b4b6", "b4a6", "b4h7", "b4g7", "b4f7", "b4e7", "b4d7", "b4c7", "b4b7", "b4a7", "b4h8", "b4g8", "b4f8", "b4e8", "b4d8", "b4c8", "b4b8", "b4a8"},
    {"a4h1", "a4g1", "a4f1", "a4e1", "a4d1", "a4c1", "a4b1", "a4a1", "a4h2", "a4g2", "a4f2", "a4e2", "a4d2", "a4c2", "a4b2", "a4a2", "a4h3", "a4g3", "a4f3", "a4e3", "a4d3", "a4c3", "a4b3", "a4a3", "a4h4", "a4g4", "a4f4", "a4e4", "a4d4", "a4c4", "a4b4", "a4a4", "a4h5", "a4g5", "a4f5", "a4e5", "a4d5", "a4c5", "a4b5", "a4a5", "a4h6", "a4g6", "a4f6", "a4e6", "a4d6", "a4c6", "a4b6", "a4a6", "a4h7", "a4g7", "a4f7", "a4e7", "a4d7", "a4c7", "a4b7", "a4a7", "a4h8", "a4g8", "a4f8", "a4e8", "a4d8", "a4c8", "a4b8", "a4a8"},
    {"h5h1", "h5g1", "h5f1", "h5e1", "h5d1", "h5c1", "h5b1", "h5a1", "h5h2", "h5g2", "h5f2", "h5e2", "h5d2", "h5c2", "h5b2", "h5a2", "h5h3", "h5g3", "h5f3", "h5e3", "h5d3", "h5c3", "h5b3", "h5a3", "h5h4", "h5g4", "h5f4", "h5e4", "h5d4", "h5c4", "h5b4", "h5a4", "h5h5", "h5g5", "h5f5", "h5e5", "h5d5", "h5c5", "h5b5", "h5a5", "h5h6", "h5g6", "h5f6", "h5e6", "h5d6", "h5c6", "h5b6", "h5a6", "h5h7", "h5g7", "h5f7", "h5e7", "h5d7", "h5c7", "h5b7", "h5a7", "h5h8", "h5g8", "h5f8", "h5e8", "h5d8", "h5c8", "h5b8", "h5a8"},
    {"g5h1", "g5g1", "g5f1", "g5e1", "g5d1", "g5c1", "g5b1", "g5a1", "g5h2", "g5g2", "g5f2", "g5e2", "g5d2", "g5c2", "g5b2", "g5a2", "g5h3", "g5g3", "g5f3", "g5e3", "g5d3", "g5c3", "g5b3", "g5a3", "g5h4", "g5g4", "g5f4", "g5e4", "g5d4", "g5c4", "g5b4", "g5a4", "g5h5", "g5g5", "g5f5", "g5e5", "g5d5", "g5c5", "g5b5", "g5a5", "g5h6", "g5g6", "g5f6", "g5e6", "g5d6", "g5c6", "g5b6", "g5a6", "g5h7", "g5g7", "g5f7", "g5e7", "g5d7", "g5c7", "g5b7", "g5a7", "g5h8", "g5g8", "g5f8", "g5e8", "g5d8", "g5c8", "g5b8", "g5a8"},
    {"f5h1", "f5g1", "f5f1", "f5e1", "f5d1", "f5c1", "f5b1", "f5a1", "f5h2", "f5g2", "f5f2", "f5e2", "f5d2", "f5c2", "f5b2", "f5a2", "f5h3", "f5g3", "f5f3", "f5e3", "f5d3", "f5c3", "f5b3", "f5a3", "f5h4", "f5g4", "f5f4", "f5e4", "f5d4", "f5c4", "f5b4", "f5a4", "f5h5", "f5g5", "f5f5", "f5e5", "f5d5", "f5c5", "f5b5", "f5a5", "f5h6", "f5g6", "f5f6", "f5e6", "f5d6", "f5c6", "f5b6", "f5a6", "f5h7", "f5g7", "f5f7", "f5e7", "f5d7", "f5c7", "f5b7", "f5a7", "f5h8", "f5g8", "f5f8", "f5e8", "f5d8", "f5c8", "f5b8", "f5a8"},
    {"e5h1", "e5g1", "e5f1", "e5e1", "e5d1", "e5c1", "e5b1", "e5a1", "e5h2", "e5g2", "e5f2", "e5e2", "e5d2", "e5c2", "e5b2", "e5a2", "e5h3", "e5g3", "e5f3", "e5e3", "e5d3", "e5c3", "e5b3", "e5a3", "e5h4", "e5g4", "e5f4", "e5e4", "e5d4", "e5c4", "e5b4", "e5a4", "e5h5", "e5g5", "e5f5", "e5e5", "e5d5", "e5c5", "e5b5", "e5a5", "e5h6", "e5g6", "e5f6", "e5e6", "e5d6", "e5c6", "e5b6", "e5a6", "e5h7", "e5g7", "e5f7", "e5e7", "e5d7", "e5c7", "e5b7", "e5a7", "e5h8", "e5g8", "e5f8", "e5e8", "e5d8", "e5c8", "e5b8", "e5a8"},
    {"d5h1", "d5g1", "d5f1", "d5e1", "d5d1", "d5c1", "d5b1", "d5a1", "d5h2", "d5g2", "d5f2", "d5e2", "d5d2", "d5c2", "d5b2", "d5a2", "d5h3", "d5g3", "d5f3", "d5e3", "d5d3", "d5c3", "d5b3", "d5a3", "d5h4", "d5g4", "d5f4", "d5e4", "d5d4", "d5c4", "d5b4", "d5a4", "d5h5", "d5g5", "d5f5", "d5e5", "d5d5", "d5c5", "d5b5", "d5a5", "d5h6", "d5g6", "d5f6", "d5e6", "d5d6", "d5c6", "d5b6", "d5a6", "d5h7", "d5g7", "d5f7", "d5e7", "d5d7", "d5c7", "d5b7", "d5a7", "d5h8", "d5g8", "d5f8", "d5e8", "d5d8", "d5c8", "d5b8", "d5a8"},
    {"c5h1", "c5g1", "c5f1", "c5e1", "c5d1", "c5c1", "c5b1", "c5a1", "c5h2", "c5g2", "c5f2", "c5e2", "c5d2", "c5c2", "c5b2", "c5a2", "c5h3", "c5g3", "c5f3", "c5e3", "c5d3", "c5c3", "c5b3", "c5a3", "c5h4", "c5g4", "c5f4", "c5e4", "c5d4", "c5c4", "c5b4", "c5a4", "c5h5", "c5g5", "c5f5", "c5e5", "c5d5", "c5c5", "c5b5", "c5a5", "c5h6", "c5g6", "c5f6", "c5e6", "c5d6", "c5c6", "c5b6", "c5a6", "c5h7", "c5g7", "c5f7", "c5e7", "c5d7", "c5c7", "c5b7", "c5a7", "c5h8", "c5g8", "c5f8", "c5e8", "c5d8", "c5c8", "c5b8", "c5a8"},
    {"b5h1", "b5g1", "b5f1", "b5e1", "b5d1", "b5c1", "b5b1", "b5a1", "b5h2", "b5g2", "b5f2", "b5e2", "b5d2", "b5c2", "b5b2", "b5a2", "b5h3", "b5g3", "b5f3", "b5e3", "b5d3", "b5c3", "b5b3", "b5a3", "b5h4", "b5g4", "b5f4", "b5e4", "b5d4", "b5c4", "b5b4", "b5a4", "b5h5", "b5g5", "b5f5", "b5e5", "b5d5", "b5c5", "b5b5", "b5a5", "b5h6", "b5g6", "b5f6", "b5e6", "b5d6", "b5c6", "b5b6", "b5a6", "b5h7", "b5g7", "b5f7", "b5e7", "b5d7", "b5c7", "b5b7", "b5a7", "b5h8", "b5g8", "b5f8", "b5e8", "b5d8", "b5c8", "b5b8", "b5a8"},
    {"a5h1", "a5g1", "a5f1", "a5e1", "a5d1", "a5c1", "a5b1", "a5a1", "a5h2", "a5g2", "a5f2", "a5e2", "a5d2", "a5c2", "a5b2", "a5a2", "a5h3", "a5g3", "a5f3", "a5e3", "a5d3", "a5c3", "a5b3", "a5a3", "a5h4", "a5g4", "a5f4", "a5e4", "a5d4", "a5c4", "a5b4", "a5a4", "a5h5", "a5g5", "a5f5", "a5e5", "a5d5", "a5c5", "a5b5", "a5a5", "a5h6", "a5g6", "a5f6", "a5e6", "a5d6", "a5c6", "a5b6", "a5a6", "a5h7", "a5g7", "a5f7", "a5e7", "a5d7", "a5c7", "a5b7", "a5a7", "a5h8", "a5g8", "a5f8", "a5e8", "a5d8", "a5c8", "a5b8", "a5a8"},
    {"h6h1", "h6g1", "h6f1", "h6e1", "h6d1", "h6c1", "h6b1", "h6a1", "h6h2", "h6g2", "h6f2", "h6e2", "h6d2", "h6c2", "h6b2", "h6a2", "h6h3", "h6g3", "h6f3", "h6e3", "h6d3", "h6c3", "h6b3", "h6a3", "h6h4", "h6g4", "h6f4", "h6e4", "h6d4", "h6c4", "h6b4", "h6a4", "h6h5", "h6g5", "h6f5", "h6e5", "h6d5", "h6c5", "h6b5", "h6a5", "h6h6", "h6g6", "h6f6", "h6e6", "h6d6", "h6c6", "h6b6", "h6a6", "h6h7", "h6g7", "h6f7", "h6e7", "h6d7", "h6c7", "h6b7", "h6a7", "h6h8", "h6g8", "h6f8", "h6e8", "h6d8", "h6c8", "h6b8", "h6a8"},
    {"g6h1", "g6g1", "g6f1", "g6e1", "g6d1", "g6c1", "g6b1", "g6a1", "g6h2", "g6g2", "g6f2", "g6e2", "g6d2", "g6c2", "g6b2", "g6a2", "g6h3", "g6g3", "g6f3", "g6e3", "g6d3", "g6c3", "g6b3", "g6a3", "g6h4", "g6g4", "g6f4", "g6e4", "g6d4", "g6c4", "g6b4", "g6a4", "g6h5", "g6g5", "g6f5", "g6e5", "g6d5", "g6c5", "g6b5", "g6a5", "g6h6", "g6g6", "g6f6", "g6e6", "g6d6", "g6c6", "g6b6", "g6a6", "g6h7", "g6g7", "g6f7", "g6e7", "g6d7", "g6c7", "g6b7", "g6a7", "g6h8", "g6g8", "g6f8", "g6e8", "g6d8", "g6c8", "g6b8", "g6a8"},
    {"f6h1", "f6g1", "f6f1", "f6e1", "f6d1", "f6c1", "f6b1", "f6a1", "f6h2", "f6g2", "f6f2", "f6e2", "f6d2", "f6c2", "f6b2", "f6a2", "f6h3", "f6g3", "f6f3", "f6e3", "f6d3", "f6c3", "f6b3", "f6a3", "f6h4", "f6g4", "f6f4", "f6e4", "f6d4", "f6c4", "f6b4", "f6a4", "f6h5", "f6g5", "f6f5", "f6e5", "f6d5", "f6c5", "f6b5", "f6a5", "f6h6", "f6g6", "f6f6", "f6e6", "f6d6", "f6c6", "f6b6", "f6a6", "f6h7", "f6g7", "f6f7", "f6e7", "f6d7", "f6c7", "f6b7", "f6a7", "f6h8", "f6g8", "f6f8", "f6e8", "f6d8", "f6c8", "f6b8", "f6a8"},
    {"e6h1", "e6g1", "e6f1", "e6e1", "e6d1", "e6c1", "e6b1", "e6a1", "e6h2", "e6g2", "e6f2", "e6e2", "e6d2", "e6c2", "e6b2", "e6a2", "e6h3", "e6g3", "e6f3", "e6e3", "e6d3", "e6c3", "e6b3", "e6a3", "e6h4", "e6g4", "e6f4", "e6e4", "e6d4", "e6c4", "e6b4", "e6a4", "e6h5", "e6g5", "e6f5", "e6e5", "e6d5", "e6c5", "e6b5", "e6a5", "e6h6", "e6g6", "e6f6", "e6e6", "e6d6", "e6c6", "e6b6", "e6a6", "e6h7", "e6g7", "e6f7", "e6e7", "e6d7", "e6c7", "e6b7", "e6a7", "e6h8", "e6g8", "e6f8", "e6e8", "e6d8", "e6c8", "e6b8", "e6a8"},
    {"d6h1", "d6g1", "d6f1", "d6e1", "d6d1", "d6c1", "d6b1", "d6a1", "d6h2", "d6g2", "d6f2", "d6e2", "d6d2", "d6c2", "d6b2", "d6a2", "d6h3", "d6g3", "d6f3", "d6e3", "d6d3", "d6c3", "d6b3", "d6a3", "d6h4", "d6g4", "d6f4", "d6e4", "d6d4", "d6c4", "d6b4", "d6a4", "d6h5", "d6g5", "d6f5", "d6e5", "d6d5", "d6c5", "d6b5", "d6a5", "d6h6", "d6g6", "d6f6", "d6e6", "d6d6", "d6c6", "d6b6", "d6a6", "d6h7", "d6g7", "d6f7", "d6e7", "d6d7", "d6c7", "d6b7", "d6a7", "d6h8", "d6g8", "d6f8", "d6e8", "d6d8", "d6c8", "d6b8", "d6a8"},
    {"c6h1", "c6g1", "c6f1", "c6e1", "c6d1", "c6c1", "c6b1", "c6a1", "c6h2", "c6g2", "c6f2", "c6e2", "c6d2", "c6c2", "c6b2", "c6a2", "c6h3", "c6g3", "c6f3", "c6e3", "c6d3", "c6c3", "c6b3", "c6a3", "c6h4", "c6g4", "c6f4", "c6e4", "c6d4", "c6c4", "c6b4", "c6a4", "c6h5", "c6g5", "c6f5", "c6e5", "c6d5", "c6c5", "c6b5", "c6a5", "c6h6", "c6g6", "c6f6", "c6e6", "c6d6", "c6c6", "c6b6", "c6a6", "c6h7", "c6g7", "c6f7", "c6e7", "c6d7", "c6c7", "c6b7", "c6a7", "c6h8", "c6g8", "c6f8", "c6e8", "c6d8", "c6c8", "c6b8", "c6a8"},
    {"b6h1", "b6g1", "b6f1", "b6e1", "b6d1", "b6c1", "b6b1", "b6a1", "b6h2", "b6g2", "b6f2", "b6e2", "b6d2", "b6c2", "b6b2", "b6a2", "b6h3", "b6g3", "b6f3", "b6e3", "b6d3", "b6c3", "b6b3", "b6a3", "b6h4", "b6g4", "b6f4", "b6e4", "b6d4", "b6c4", "b6b4", "b6a4", "b6h5", "b6g5", "b6f5", "b6e5", "b6d5", "b6c5", "b6b5", "b6a5", "b6h6", "b6g6", "b6f6", "b6e6", "b6d6", "b6c6", "b6b6", "b6a6", "b6h7", "b6g7", "b6f7", "b6e7", "b6d7", "b6c7", "b6b7", "b6a7", "b6h8", "b6g8", "b6f8", "b6e8", "b6d8", "b6c8", "b6b8", "b6a8"},
    {"a6h1", "a6g1", "a6f1", "a6e1", "a6d1", "a6c1", "a6b1", "a6a1", "a6h2", "a6g2", "a6f2", "a6e2", "a6d2", "a6c2", "a6b2", "a6a2", "a6h3", "a6g3", "a6f3", "a6e3", "a6d3", "a6c3", "a6b3", "a6a3", "a6h4", "a6g4", "a6f4", "a6e4", "a6d4", "a6c4", "a6b4", "a6a4", "a6h5", "a6g5", "a6f5", "a6e5", "a6d5", "a6c5", "a6b5", "a6a5", "a6h6", "a6g6", "a6f6", "a6e6", "a6d6", "a6c6", "a6b6", "a6a6", "a6h7", "a6g7", "a6f7", "a6e7", "a6d7", "a6c7", "a6b7", "a6a7", "a6h8", "a6g8", "a6f8", "a6e8", "a6d8", "a6c8", "a6b8", "a6a8"},
    {"h7h1", "h7g1", "h7f1", "h7e1", "h7d1", "h7c1", "h7b1", "h7a1", "h7h2", "h7g2", "h7f2", "h7e2", "h7d2", "h7c2", "h7b2", "h7a2", "h7h3", "h7g3", "h7f3", "h7e3", "h7d3", "h7c3", "h7b3", "h7a3", "h7h4", "h7g4", "h7f4", "h7e4", "h7d4", "h7c4", "h7b4", "h7a4", "h7h5", "h7g5", "h7f5", "h7e5", "h7d5", "h7c5", "h7b5", "h7a5", "h7h6", "h7g6", "h7f6", "h7e6", "h7d6", "h7c6", "h7b6", "h7a6", "h7h7", "h7g7", "h7f7", "h7e7", "h7d7", "h7c7", "h7b7", "h7a7", "h7h8", "h7g8", "h7f8", "h7e8", "h7d8", "h7c8", "h7b8", "h7a8"},
    {"g7h1", "g7g1", "g7f1", "g7e1", "g7d1", "g7c1", "g7b1", "g7a1", "g7h2", "g7g2", "g7f2", "g7e2", "g7d2", "g7c2", "g7b2", "g7a2", "g7h3", "g7g3", "g7f3", "g7e3", "g7d3", "g7c3", "g7b3", "g7a3", "g7h4", "g7g4", "g7f4", "g7e4", "g7d4", "g7c4", "g7b4", "g7a4", "g7h5", "g7g5", "g7f5", "g7e5", "g7d5", "g7c5", "g7b5", "g7a5", "g7h6", "g7g6", "g7f6", "g7e6", "g7d6", "g7c6", "g7b6", "g7a6", "g7h7", "g7g7", "g7f7", "g7e7", "g7d7", "g7c7", "g7b7", "g7a7", "g7h8", "g7g8", "g7f8", "g7e8", "g7d8", "g7c8", "g7b8", "g7a8"},
    {"f7h1", "f7g1", "f7f1", "f7e1", "f7d1", "f7c1", "f7b1", "f7a1", "f7h2", "f7g2", "f7f2", "f7e2", "f7d2", "f7c2", "f7b2", "f7a2", "f7h3", "f7g3", "f7f3", "f7e3", "f7d3", "f7c3", "f7b3", "f7a3", "f7h4", "f7g4", "f7f4", "f7e4", "f7d4", "f7c4", "f7b4", "f7a4", "f7h5", "f7g5", "f7f5", "f7e5", "f7d5", "f7c5", "f7b5", "f7a5", "f7h6", "f7g6", "f7f6", "f7e6", "f7d6", "f7c6", "f7b6", "f7a6", "f7h7", "f7g7", "f7f7", "f7e7", "f7d7", "f7c7", "f7b7", "f7a7", "f7h8", "f7g8", "f7f8", "f7e8", "f7d8", "f7c8", "f7b8", "f7a8"},
    {"e7h1", "e7g1", "e7f1", "e7e1", "e7d1", "e7c1", "e7b1", "e7a1", "e7h2", "e7g2", "e7f2", "e7e2", "e7d2", "e7c2", "e7b2", "e7a2", "e7h3", "e7g3", "e7f3", "e7e3", "e7d3", "e7c3", "e7b3", "e7a3", "e7h4", "e7g4", "e7f4", "e7e4", "e7d4", "e7c4", "e7b4", "e7a4", "e7h5", "e7g5", "e7f5", "e7e5", "e7d5", "e7c5", "e7b5", "e7a5", "e7h6", "e7g6", "e7f6", "e7e6", "e7d6", "e7c6", "e7b6", "e7a6", "e7h7", "e7g7", "e7f7", "e7e7", "e7d7", "e7c7", "e7b7", "e7a7", "e7h8", "e7g8", "e7f8", "e7e8", "e7d8", "e7c8", "e7b8", "e7a8"},
    {"d7h1", "d7g1", "d7f1", "d7e1", "d7d1", "d7c1", "d7b1", "d7a1", "d7h2", "d7g2", "d7f2", "d7e2", "d7d2", "d7c2", "d7b2", "d7a2", "d7h3", "d7g3", "d7f3", "d7e3", "d7d3", "d7c3", "d7b3", "d7a3", "d7h4", "d7g4", "d7f4", "d7e4", "d7d4", "d7c4", "d7b4", "d7a4", "d7h5", "d7g5", "d7f5", "d7e5", "d7d5", "d7c5", "d7b5", "d7a5", "d7h6", "d7g6", "d7f6", "d7e6", "d7d6", "d7c6", "d7b6", "d7a6", "d7h7", "d7g7", "d7f7", "d7e7", "d7d7", "d7c7", "d7b7", "d7a7", "d7h8", "d7g8", "d7f8", "d7e8", "d7d8", "d7c8", "d7b8", "d7a8"},
    {"c7h1", "c7g1", "c7f1", "c7e1", "c7d1", "c7c1", "c7b1", "c7a1", "c7h2", "c7g2", "c7f2", "c7e2", "c7d2", "c7c2", "c7b2", "c7a2", "c7h3", "c7g3", "c7f3", "c7e3", "c7d3", "c7c3", "c7b3", "c7a3", "c7h4", "c7g4", "c7f4", "c7e4", "c7d4", "c7c4", "c7b4", "c7a4", "c7h5", "c7g5", "c7f5", "c7e5", "c7d5", "c7c5", "c7b5", "c7a5", "c7h6", "c7g6", "c7f6", "c7e6", "c7d6", "c7c6", "c7b6", "c7a6", "c7h7", "c7g7", "c7f7", "c7e7", "c7d7", "c7c7", "c7b7", "c7a7", "c7h8", "c7g8", "c7f8", "c7e8", "c7d8", "c7c8", "c7b8", "c7a8"},
    {"b7h1", "b7g1", "b7f1", "b7e1", "b7d1", "b7c1", "b7b1", "b7a1", "b7h2", "b7g2", "b7f2", "b7e2", "b7d2", "b7c2", "b7b2", "b7a2", "b7h3", "b7g3", "b7f3", "b7e3", "b7d3", "b7c3", "b7b3", "b7a3", "b7h4", "b7g4", "b7f4", "b7e4", "b7d4", "b7c4", "b7b4", "b7a4", "b7h5", "b7g5", "b7f5", "b7e5", "b7d5", "b7c5", "b7b5", "b7a5", "b7h6", "b7g6", "b7f6", "b7e6", "b7d6", "b7c6", "b7b6", "b7a6", "b7h7", "b7g7", "b7f7", "b7e7", "b7d7", "b7c7", "b7b7", "b7a7", "b7h8", "b7g8", "b7f8", "b7e8", "b7d8", "b7c8", "b7b8", "b7a8"},
    {"a7h1", "a7g1", "a7f1", "a7e1", "a7d1", "a7c1", "a7b1", "a7a1", "a7h2", "a7g2", "a7f2", "a7e2", "a7d2", "a7c2", "a7b2", "a7a2", "a7h3", "a7g3", "a7f3", "a7e3", "a7d3", "a7c3", "a7b3", "a7a3", "a7h4", "a7g4", "a7f4", "a7e4", "a7d4", "a7c4", "a7b4", "a7a4", "a7h5", "a7g5", "a7f5", "a7e5", "a7d5", "a7c5", "a7b5", "a7a5", "a7h6", "a7g6", "a7f6", "a7e6", "a7d6", "a7c6", "a7b6", "a7a6", "a7h7", "a7g7", "a7f7", "a7e7", "a7d7", "a7c7", "a7b7", "a7a7", "a7h8", "a7g8", "a7f8", "a7e8", "a7d8", "a7c8", "a7b8", "a7a8"},
    {"h8h1", "h8g1", "h8f1", "h8e1", "h8d1", "h8c1", "h8b1", "h8a1", "h8h2", "h8g2", "h8f2", "h8e2", "h8d2", "h8c2", "h8b2", "h8a2", "h8h3", "h8g3", "h8f3", "h8e3", "h8d3", "h8c3", "h8b3", "h8a3", "h8h4", "h8g4", "h8f4", "h8e4", "h8d4", "h8c4", "h8b4", "h8a4", "h8h5", "h8g5", "h8f5", "h8e5", "h8d5", "h8c5", "h8b5", "h8a5", "h8h6", "h8g6", "h8f6", "h8e6", "h8d6", "h8c6", "h8b6", "h8a6", "h8h7", "h8g7", "h8f7", "h8e7", "h8d7", "h8c7", "h8b7", "h8a7", "h8h8", "h8g8", "h8f8", "h8e8", "h8d8", "h8c8", "h8b8", "h8a8"},
    {"g8h1", "g8g1", "g8f1", "g8e1", "g8d1", "g8c1", "g8b1", "g8a1", "g8h2", "g8g2", "g8f2", "g8e2", "g8d2", "g8c2", "g8b2", "g8a2", "g8h3", "g8g3", "g8f3", "g8e3", "g8d3", "g8c3", "g8b3", "g8a3", "g8h4", "g8g4", "g8f4", "g8e4", "g8d4", "g8c4", "g8b4", "g8a4", "g8h5", "g8g5", "g8f5", "g8e5", "g8d5", "g8c5", "g8b5", "g8a5", "g8h6", "g8g6", "g8f6", "g8e6", "g8d6", "g8c6", "g8b6", "g8a6", "g8h7", "g8g7", "g8f7", "g8e7", "g8d7", "g8c7", "g8b7", "g8a7", "g8h8", "g8g8", "g8f8", "g8e8", "g8d8", "g8c8", "g8b8", "g8a8"},
    {"f8h1", "f8g1", "f8f1", "f8e1", "f8d1", "f8c1", "f8b1", "f8a1", "f8h2", "f8g2", "f8f2", "f8e2", "f8d2", "f8c2", "f8b2", "f8a2", "f8h3", "f8g3", "f8f3", "f8e3", "f8d3", "f8c3", "f8b3", "f8a3", "f8h4", "f8g4", "f8f4", "f8e4", "f8d4", "f8c4", "f8b4", "f8a4", "f8h5", "f8g5", "f8f5", "f8e5", "f8d5", "f8c5", "f8b5", "f8a5", "f8h6", "f8g6", "f8f6", "f8e6", "f8d6", "f8c6", "f8b6", "f8a6", "f8h7", "f8g7", "f8f7", "f8e7", "f8d7", "f8c7", "f8b7", "f8a7", "f8h8", "f8g8", "f8f8", "f8e8", "f8d8", "f8c8", "f8b8", "f8a8"},
    {"e8h1", "e8g1", "e8f1", "e8e1", "e8d1", "e8c1", "e8b1", "e8a1", "e8h2", "e8g2", "e8f2", "e8e2", "e8d2", "e8c2", "e8b2", "e8a2", "e8h3", "e8g3", "e8f3", "e8e3", "e8d3", "e8c3", "e8b3", "e8a3", "e8h4", "e8g4", "e8f4", "e8e4", "e8d4", "e8c4", "e8b4", "e8a4", "e8h5", "e8g5", "e8f5", "e8e5", "e8d5", "e8c5", "e8b5", "e8a5", "e8h6", "e8g6", "e8f6", "e8e6", "e8d6", "e8c6", "e8b6", "e8a6", "e8h7", "e8g7", "e8f7", "e8e7", "e8d7", "e8c7", "e8b7", "e8a7", "e8h8", "e8g8", "e8f8", "e8e8", "e8d8", "e8c8", "e8b8", "e8a8"},
    {"d8h1", "d8g1", "d8f1", "d8e1", "d8d1", "d8c1", "d8b1", "d8a1", "d8h2", "d8g2", "d8f2", "d8e2", "d8d2", "d8c2", "d8b2", "d8a2", "d8h3", "d8g3", "d8f3", "d8e3", "d8d3", "d8c3", "d8b3", "d8a3", "d8h4", "d8g4", "d8f4", "d8e4", "d8d4", "d8c4", "d8b4", "d8a4", "d8h5", "d8g5", "d8f5", "d8e5", "d8d5", "d8c5", "d8b5", "d8a5", "d8h6", "d8g6", "d8f6", "d8e6", "d8d6", "d8c6", "d8b6", "d8a6", "d8h7", "d8g7", "d8f7", "d8e7", "d8d7", "d8c7", "d8b7", "d8a7", "d8h8", "d8g8", "d8f8", "d8e8", "d8d8", "d8c8", "d8b8", "d8a8"},
    {"c8h1", "c8g1", "c8f1", "c8e1", "c8d1", "c8c1", "c8b1", "c8a1", "c8h2", "c8g2", "c8f2", "c8e2", "c8d2", "c8c2", "c8b2", "c8a2", "c8h3", "c8g3", "c8f3", "c8e3", "c8d3", "c8c3", "c8b3", "c8a3", "c8h4", "c8g4", "c8f4", "c8e4", "c8d4", "c8c4", "c8b4", "c8a4", "c8h5", "c8g5", "c8f5", "c8e5", "c8d5", "c8c5", "c8b5", "c8a5", "c8h6", "c8g6", "c8f6", "c8e6", "c8d6", "c8c6", "c8b6", "c8a6", "c8h7", "c8g7", "c8f7", "c8e7", "c8d7", "c8c7", "c8b7", "c8a7", "c8h8", "c8g8", "c8f8", "c8e8", "c8d8", "c8c8", "c8b8", "c8a8"},
    {"b8h1", "b8g1", "b8f1", "b8e1", "b8d1", "b8c1", "b8b1", "b8a1", "b8h2", "b8g2", "b8f2", "b8e2", "b8d2", "b8c2", "b8b2", "b8a2", "b8h3", "b8g3", "b8f3", "b8e3", "b8d3", "b8c3", "b8b3", "b8a3", "b8h4", "b8g4", "b8f4", "b8e4", "b8d4", "b8c4", "b8b4", "b8a4", "b8h5", "b8g5", "b8f5", "b8e5", "b8d5", "b8c5", "b8b5", "b8a5", "b8h6", "b8g6", "b8f6", "b8e6", "b8d6", "b8c6", "b8b6", "b8a6", "b8h7", "b8g7", "b8f7", "b8e7", "b8d7", "b8c7", "b8b7", "b8a7", "b8h8", "b8g8", "b8f8", "b8e8", "b8d8", "b8c8", "b8b8", "b8a8"},
    {"a8h1", "a8g1", "a8f1", "a8e1", "a8d1", "a8c1", "a8b1", "a8a1", "a8h2", "a8g2", "a8f2", "a8e2", "a8d2", "a8c2", "a8b2", "a8a2", "a8h3", "a8g3", "a8f3", "a8e3", "a8d3", "a8c3", "a8b3", "a8a3", "a8h4", "a8g4", "a8f4", "a8e4", "a8d4", "a8c4", "a8b4", "a8a4", "a8h5", "a8g5", "a8f5", "a8e5", "a8d5", "a8c5", "a8b5", "a8a5", "a8h6", "a8g6", "a8f6", "a8e6", "a8d6", "a8c6", "a8b6", "a8a6", "a8h7", "a8g7", "a8f7", "a8e7", "a8d7", "a8c7", "a8b7", "a8a7", "a8h8", "a8g8", "a8f8", "a8e8", "a8d8", "a8c8", "a8b8", "a8a8"},
};

#endif // MVLOOKUP_HPP

