![logo](https://github.com/ilAYAli/Enyo/assets/1106732/49026c53-0cf8-4256-b938-34e607876a2d)

<p align="center">
<i>
  <a href="https://en.wikipedia.org/wiki/Enyo" target="Enyo">Enyo</a> the greek godess of war and destruction.<br>
</i>
  Watch her play live at <a href="https://lichess.org/@/EnyoBot/tv" target="Lichess">Lichess</a>
</p>

<h3>Engine</h3>
Enyo is a C++23 <a href="https://www.chessprogramming.org/UCI" rel="nofollow">UCI</a> chess engine built around iterative deepening negamax/PVS
alpha-beta search and NNUE evaluation.<br>
Enyo's NNUE training work has used self-play positions and deeper offline teacher
labels, public evaluation data, and curated engine-training positions.<br>

<h3>Board representation</h3>
<br>
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
Legal move generation for standard chess<br>
Incremental make/unmake with hash, NNUE, castling, en-passant, halfmove and fullmove state<br>
<br>
</pre>

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
Root tablebase move selection when Syzygy DTZ is available<br>

<h3>Evaluation</h3>
<a href="https://www.chessprogramming.org/NNUE" rel="nofollow">NNUE</a><br>
Network training and exporter tooling lives in the sibling <code>nnue</code> repo<br>


<h3>Tools</H3>
<a href="https://github.com/Disservin/fastchess" rel="nofollow">Fastchess</a>
CLI tool for running SPRT validation<br>


<h3>Acknowledgments</h3>
<a href="https://github.com/jw1912/bullet" rel="nofollow">Bullet</a>
ML library for NNUE training<br>
<a href="https://github.com/Disservin/binpack-rust" rel="nofollow">Binpack</a>
Rust port of the Stockfish binpack reader<br>
<a href="https://github.com/AndyGrant/Pyrrhic" rel="nofollow">Pyrrhic</a>
Syzygy WDL/DTZ probing<br>
Bluefever Software: Chess Engine In C <a href="https://github.com/bluefeversoft/vice" rel="nofollow">Vice</a><br>
Chess Programming <a href="https://www.youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs" rel="nofollow">BBC</a></br>

<h3>Inspirations</h3>
* Stockfish<br>
* Rice<br>
* Smallbrain<br>
* Various open source engines.<br>
