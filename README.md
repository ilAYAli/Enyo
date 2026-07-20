<p align="center">
  <img src="https://github.com/user-attachments/assets/50d0d944-a763-44c9-8e9c-af4112c2fc14" alt="Enyo logo">
</p>

<p align="center">
  <em>
    <a href="https://en.wikipedia.org/wiki/Enyo">Enyo</a>, the Greek goddess of war and destruction.
  </em>
  <br>
  Watch Enyo play live on <a href="https://lichess.org/@/EnyoBot/tv">Lichess</a>.
</p>

# Enyo

Enyo is a C++23 UCI chess engine combining iterative-deepening principal
variation search with custom HalfKAv2-style factorised NNUE, trained from
scratch.

It has been developed over several years — predating the generative-AI era.

## Evaluation

Enyo uses a HalfKAv2-style factorised NNUE trained from scratch for Enyo.
Training uses self-play/generated positions and binpack datasets, with
Stockfish used only as a labeling oracle.

No foreign weights are used.

Related project: [Enyo NNUE](https://github.com/ilAYAli/nnue).

## Board Representation

Enyo was originally written from first principles rather than modeled after
another engine. One early design choice remains: squares are indexed from `h1`.

Internally, `h1` is `0`, `g1` is `1`, and `a8` is `63`. This differs from the
A1-indexed layout used by many chess libraries and tablebase APIs, so Enyo
converts square and bitboard layouts at external boundaries such as Syzygy and
Pyrrhic probing, NNUE export/import, FEN, PGN, and UCI handling.

```text
H1 indexed: white king = 3, black king = 59

        A  B  C  D  E  F  G  H
      +------------------------+
8  63 | R  N  B  Q  K  B  N  R | 56
7  55 | P  P  P  P  P  P  P  P | 48
6  47 | -  -  -  -  -  -  -  - | 40
5  39 | -  -  -  -  -  -  -  - | 32
4  31 | -  -  -  -  -  -  -  - | 24
3  23 | -  -  -  -  -  -  -  - | 16
2  15 | p  p  p  p  p  p  p  p |  8
1   7 | r  n  b  q  k  b  n  r |  0
      +------------------------+
        7  6  5  4  3  2  1  0
```

## Features

- UCI engine with configurable options through `setoption` and `settings.json`.
- Iterative deepening, aspiration windows, quiescence search, and transposition
  table support.
- Selective search with razoring, null-move pruning, ProbCut, futility pruning,
  late-move pruning and reductions, singular extensions, and SEE.
- Move ordering using transposition-table moves, tactical ordering, killer
  moves, countermoves, history, and continuation history.
- Syzygy WDL/DTZ probing through Pyrrhic when tablebases are enabled.
- Enyo-owned HalfKAv2-style factorised NNUE evaluation.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Configuration

Enyo can read UCI options from JSON before accepting normal UCI commands. The
config file must contain a single `uci_options` object:

```json
{
  "uci_options": {
    "Threads": 4,
    "Hash": 1024,
    "nnue_file": "~/assets/nnue/default.nn",
    "logfile": "/tmp/enyo.log",
    "use_syzygy": true,
    "SyzygyPath": "~/assets/tablebases"
  }
}
```

Config lookup uses the explicit `--config` path when provided. Without
`--config`, Enyo checks:

- `~/.config/enyo/settings.json` for personal defaults.
- `settings.json` next to the engine binary when shipped with a build.

Every configured entry is validated and applied through the same path as a UCI
`setoption` command. Options absent from the file keep their compiled defaults,
and later UCI commands take precedence.

Example UCI override:

```text
setoption name nnue_file value ~/assets/nnue/enyo-1.30.0-rc3.nn
```


## Search

Enyo uses iterative deepening with aspiration windows and quiescence search for
tactical positions. Selective techniques focus effort on promising lines while
preserving tactical accuracy.

Move ordering combines transposition-table and tactical information with killer,
countermove, history, and continuation-history heuristics. The search handles
repetition and fifty-move draws, adjusts time usage when the root result is
unstable, and can use Syzygy tablebases when available.


## Development

See [README_dev.md](README_dev.md) for benchmarks, SPSA tuning, SPRT
validation, and development utilities.

## Acknowledgments

- Bluefever Software's [Vice](https://github.com/bluefeversoft/vice), from the
  Chess Engine In C series.
- The Chess Programming [BBC video series](https://www.youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs).
- The broader chess-engine community and the many engines that make serious
  engine development possible to study.
