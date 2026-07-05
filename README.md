![logo](https://github.com/ilAYAli/Enyo/assets/1106732/49026c53-0cf8-4256-b938-34e607876a2d)

<p align="center">
<i>
  <a href="https://en.wikipedia.org/wiki/Enyo" target="Enyo">Enyo</a> the greek godess of war and destruction.<br>
</i>
  Watch her play live at <a href="https://lichess.org/@/EnyoBot/tv" target="Lichess">Lichess</a>
</p>

<h3>Engine</h3>
Enyo is a C++23 <a href="https://www.chessprogramming.org/UCI" rel="nofollow">UCI</a> chess engine built around iterative deepening negamax/PVS alpha-beta search and <a href="https://www.chessprogramming.org/NNUE" rel="nofollow">NNUE</a> evaluation.<br>

<h3>Board representation</h3>
Enyo's board representation was originally written from first principles rather
than modeled after another engine, and one of those early design choices stuck:
squares are indexed from <code>h1</code>.
Internally, <code>h1</code> is 0, <code>g1</code> is 1, ..., and <code>a8</code> is 63.

This differs from the A1-indexed layout used by many chess libraries and
tablebase APIs, so Enyo converts square and bitboard layouts at external
boundaries such as Syzygy/Pyrrhic probing, NNUE export/import, FEN, PGN, and
UCI handling.

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
<h3>Move generation</h3>
<a href="https://www.chessprogramming.org/Bitboards" rel="nofollow">Bitboards</a><br>
<a href="https://www.chessprogramming.org/Magic_Bitboards" rel="nofollow">Magic Bitboards</a><br>
</pre>

<h3>Benchmark</h3>
<pre>
./build/enyo bench
./scripts/bench_avg.py ./build/enyo 10
./build/enyo bench perft
</pre>
The <code>bench</code> argument runs a deterministic 24-position search suite
with the configured evaluator. It defaults to depth 11, one thread, a 16 MB
hash, and no tablebases. The reported node total is the search signature;
nodes per second measures speed. Use <code>bench_avg.py</code> for repeated timing.
Use <code>bench perft</code> for the move-generation-only benchmark.

<h3>Configuration</h3>
Enyo reads <code>~/.config/enyo/settings.json</code>. The file contains one
<code>uci_options</code> object; every entry is validated and applied through
the same <code>setoption</code> path used by a UCI client. Options absent from
the file keep their compiled defaults, and later UCI commands take precedence.
Keep machine-specific options here; validated search parameters belong in the
compiled defaults.

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
<a href="https://www.chessprogramming.org/Negamax" rel="nofollow">Negamax</a><br>
<a href="https://www.chessprogramming.org/Principal_Variation_Search" rel="nofollow">Principal Variation Search</a><br>
<a href="https://www.chessprogramming.org/Quiescence_Search" rel="nofollow">Quiescence Search</a><br>
<a href="https://www.chessprogramming.org/Iterative_Deepening" rel="nofollow">Iterative Deepening</a><br>
<a href="https://www.chessprogramming.org/Transposition_Table" rel="nofollow">Transposition Table</a><br>
<a href="https://www.chessprogramming.org/Aspiration_Windows" rel="nofollow">Aspiration Windows</a><br>
<a href="https://www.chessprogramming.org/Internal_Iterative_Reductions" rel="nofollow">Internal Iterative Reductions</a><br>
<a href="https://www.chessprogramming.org/Reverse_Futility_Pruning" rel="nofollow">Reverse Futility Pruning</a><br>
<a href="https://www.chessprogramming.org/Razoring" rel="nofollow">Razoring</a><br>
<a href="https://www.chessprogramming.org/Null_Move_Pruning" rel="nofollow">Null Move Pruning</a><br>
<a href="https://www.chessprogramming.org/ProbCut" rel="nofollow">ProbCut</a><br>
<a href="https://www.chessprogramming.org/Futility_Pruning" rel="nofollow">Futility Pruning</a><br>
<a href="https://www.chessprogramming.org/Move_Count_Based_Pruning" rel="nofollow">Late Move Pruning</a><br>
<a href="https://www.chessprogramming.org/Late_Move_Reductions" rel="nofollow">Late Move Reductions</a><br>
<a href="https://www.chessprogramming.org/Static_Exchange_Evaluation" rel="nofollow">Static Exchange Evaluation</a> for qsearch capture pruning<br>
<a href="https://www.chessprogramming.org/Killer_Heuristic" rel="nofollow">Killer Heuristic</a><br>
<a href="https://www.chessprogramming.org/Countermove_Heuristic" rel="nofollow">Countermove Heuristic</a><br>
<a href="https://www.chessprogramming.org/History_Heuristic" rel="nofollow">History Heuristic</a><br>
Continuation history / countermove history<br>
Root soft-time instability extension on best-move flips and score volatility<br>
Repetition and 50-move draw handling<br>
Root tablebase move selection when Syzygy is available<br>

<h3>Evaluation</h3>
<a href="https://github.com/ilAYAli/nnue" rel="nofollow">Enyo NNUE</a><br>
The network is trained with Bullet from Enyo self-play positions and Stockfish binpack's.

<h3>Tools</H3>
<a href="https://github.com/ilAYAli/Forge" rel="nofollow">Forge</a>
Distribute tasks between a set of configured workers
<a href="https://github.com/ilAYAli/sprt" rel="nofollow">sprt</a>
fastchess python wrapper
<a href="https://github.com/ilAYAli/replay" rel="nofollow">Replay</a>
Game evaluation, used to resolve:
Inaccuracies, Mistakes, Blunders, and Average centipawn loss.
<a href="https://github.com/ilAYAli/replay" rel="nofollow">Replay</a>
Evaluation validator<br>
<a href="https://github.com/Disservin/fastchess" rel="nofollow">Fastchess</a>
CLI tool for running SPRT validation<br>
<a href="https://github.com/AndyGrant/Pyrrhic" rel="nofollow">Pyrrhic</a>
Syzygy WDL/DTZ probing<br>


<h3>Acknowledgments</h3>
Bluefever Software: Chess Engine In C <a href="https://github.com/bluefeversoft/vice" rel="nofollow">Vice</a><br>
Chess Programming <a href="https://www.youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs" rel="nofollow">BBC</a></br>

Enyo stands on the shoulders of giants and draws inspiration from many other chess engines.
Building a strong engine without studying and understanding prior art is nearly impossible these
days
