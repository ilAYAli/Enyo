![logo](https://github.com/ilAYAli/Enyo/assets/1106732/49026c53-0cf8-4256-b938-34e607876a2d)

<p align="center">
<i>
  <a href="https://en.wikipedia.org/wiki/Enyo" target="Enyo">Enyo</a> the greek godess of war and destruction.<br>
</i>
  Watch her play live at <a href="https://lichess.org/@/EnyoBot/tv" target="Lichess">Lichess</a>
</p>

<h3>Engine</h3>
Enyo is a C++23 UCI chess engine combining iterative-deepening principal variation search with NNUE evaluation.

<h3>Board representation</h3>
Enyo was originally written from first principles rather than modeled after another engine,<br>
and one of those early design choices stuck: squares are indexed from <code>h1</code>.<br>
Internally, <code>h1</code> is 0, <code>g1</code> is 1, ..., and <code>a8</code> is 63.
<br>
This differs from the A1-indexed layout used by many chess libraries and tablebase APIs,<br>
so Enyo converts square and bitboard layouts at external boundaries such as Syzygy/Pyrrhic probing,<br>
NNUE export/import, FEN, PGN, and UCI handling.
<br><br>
<pre>
H1 indexed: white king =  3, black king = 59
        A  B  C  D  E  F  G  H
      +------------------------+
8  63 | ♖  ♘  ♗  ♕  ♔  ♗  ♘  ♖ | 56
7  55 | ♙  ♙  ♙  ♙  ♙  ♙  ♙  ♙ | 48
6  47 | -  -  -  -  -  -  -  - | 40
5  39 | -  -  -  -  -  -  -  - | 32
4  31 | -  -  -  -  -  -  -  - | 24
3  23 | -  -  -  -  -  -  -  - | 16
2  15 | ♟  ♟  ♟  ♟  ♟  ♟  ♟  ♟ |  8
1   7 | ♜  ♞  ♝  ♛  ♚  ♝  ♞  ♜ |  0
      +------------------------+
        7  6  5  4  3  2  1  0
</pre>

<h3>Configuration</h3>
Enyo reads <code>~/.config/enyo/settings.json</code>.<br>
The file contains one <code>uci_options</code> object; every entry is validated and applied through the same <code>setoption</code><br>
path used by a UCI client.<br>
Options absent from the file keep their compiled defaults, and later UCI commands take precedence.<br>
</pre>


<h3>Move generation</h3>
Enyo represents positions with bitboards and uses magic-bitboard lookup tables for
fast sliding-piece attack generation.
</pre>

<h3>Benchmark</h3>
<pre>
./build/enyo bench
./scripts/bench_avg.py ./build/enyo 10
./build/enyo bench perft
</pre>
The <code>bench</code> argument runs a deterministic 24-position search suite
with the configured evaluator.<br>
It defaults to depth 11, one thread, a 16 MB hash, and no tablebases.<br>
The reported node total is the search signature; nodes per second measures speed.<br>
Use <code>bench_avg.py</code> for repeated timing.
Use <code>bench perft</code> for the move-generation-only benchmark.


<h3>SPSA (Simultaneous Perturbation Stochastic Approximation) tuning</h3>
Train, test, and promote are separate operations:
<pre>
./spsa/train --iterations 100        # train 100 iterations (default 500)
./spsa/sprt                          # test current values with Forge
./spsa/promote --dry-run             # inspect promotion
./spsa/promote                       # update params.txt and src/config.hpp
</pre>
<code>train</code> only updates <code>spsa/state.json</code> and resumes an
previous work. <code>sprt</code> launches a detached 1000-game
test against the last promoted defaults. <code>promote</code> only updates the
two tracked defaults files; it does not commit or push. Runtime CSV and lock
files are kept under <code>spsa/.runtime</code>.

<h3>Search</h3>
<p>
Enyo uses iterative deepening with aspiration windows, and quiescence search for
tactical positions.
Selective techniques such as razoring, null-move pruning, ProbCut, futility pruning,
late-move pruning and reductions, singular extensions, and static-exchange evaluation
focus effort on promising lines.
</p>
<p>
Move ordering combines transposition-table and tactical information with killer,
countermove, history, and continuation-history heuristics.
The search handles repetition and fifty-move draws, adjusts time usage when the root
result is unstable, and can use Syzygy tablebases when available.
</p>

<h3>Evaluation</h3>
<a href="https://github.com/ilAYAli/nnue" rel="nofollow">Enyo NNUE</a><br>
The Enyo NNUE is trained from scratch using selfplay and binpacks

<h3>Utilities</H3>
<a href="https://github.com/ilAYAli/Forge" rel="nofollow">Forge</a>
Distribute tasks between a set of configured workers<br>
<a href="https://github.com/ilAYAli/replay" rel="nofollow">Replay</a>
Game evaluation, used to resolve:
Inaccuracies, Mistakes, Blunders, and Average centipawn loss.<br>
<a href="https://github.com/ilAYAli/sprt" rel="nofollow">sprt</a>
fastchess python wrapper<br>
<a href="https://github.com/Disservin/fastchess" rel="nofollow">Fastchess</a>
CLI tool for running SPRT validation<br>
<a href="https://github.com/AndyGrant/Pyrrhic" rel="nofollow">Pyrrhic</a>
Syzygy WDL/DTZ probing<br>


<h3>Acknowledgments</h3>
Bluefever Software: Chess Engine In C <a href="https://github.com/bluefeversoft/vice" rel="nofollow">Vice</a><br>
Chess Programming <a href="https://www.youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs" rel="nofollow">BBC</a></br>
<br>
Enyo stands on the shoulders of giants and draws inspiration from many other chess engines.<br>
Building a strong engine without studying and understanding prior art is close to impossible<br>
