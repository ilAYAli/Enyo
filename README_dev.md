# Enyo Development

This document contains workflows and tools intended for engine development.
For an overview of Enyo and instructions for building and configuring it, see
[README.md](README.md).

## Benchmarks

```sh
./build/enyo bench
./scripts/bench_avg.py ./build/enyo 10
./build/enyo bench perft
```

`bench` runs a deterministic 24-position search suite with the configured
evaluator. It defaults to depth 11, one thread, a 16 MB hash, and no
tablebases. The reported node total is the search signature; nodes per second
measures speed.

Use `bench_avg.py` for repeated timing and `bench perft` for the
move-generation-only benchmark.

## SPSA Tuning (Simultaneous Perturbation Stochastic Approximation)

SPSA tuning separates training, testing, and promotion:

```sh
./spsa/train --iterations 100
./spsa/sprt
./spsa/promote --dry-run
./spsa/promote
```

- `train` updates `spsa/state.json` and resumes previous work when possible.
- `sprt` launches a detached 1000-game Forge test against the last promoted
  defaults.
- `promote --dry-run` previews the promotion.
- `promote` updates `spsa/params.txt` and `src/config.hpp`; it does not commit
  or push.

Runtime CSV and lock files are kept under `spsa/.runtime`.

For alternate SPSA batches, pass both the matching params and state files:

```sh
./spsa/sprt --params spsa/params_node_tm.txt --state spsa/state_node_tm.json
```

## Utilities

- [Forge](https://github.com/ilAYAli/Forge): distributes tasks across
  configured workers.
- [Replay](https://github.com/ilAYAli/replay): evaluates games for
  inaccuracies, mistakes, blunders, and average centipawn loss.
- [sprt](https://github.com/ilAYAli/sprt): Python wrapper around fastchess.
- [Fastchess](https://github.com/Disservin/fastchess): CLI tool for SPRT
  validation.
- [Pyrrhic](https://github.com/AndyGrant/Pyrrhic): Syzygy WDL/DTZ probing.
