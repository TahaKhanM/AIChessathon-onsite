# AI Chessathon

**Joint 2nd on points in London qualification. ~500 teams competed on the online ladder.**

The ladder had 465 teams. In the 334-team qualifying Swiss we scored 10/13.

London v8 was our qualifying entry. This is v9, the engine we built next. Each side started with two minutes and gained half a second per move. The engine had one CPU core, 2 GB of RAM and 30 seconds to start up. The whole submission, including the weights, had to fit in 50 MB.

## Engine

The neural network scores a position for the player to move. It keeps a running sum of piece contributions so that most moves only need a few additions and subtractions. Search compares legal moves against the opponent’s best replies, looking one turn further ahead on each pass. If the clock interrupts a deeper search, the engine can play the best move from its last completed pass.

We trained the v9 network from an earlier team checkpoint on 262,144,000 examples from generated positions and self-play, over 32,000 updates.

## Run locally

With Python 3.12 and uv installed:

```sh
uv venv --python 3.12
uv pip sync --python .venv/bin/python requirements.txt
.venv/bin/python -m demo
```

Open [the board](http://127.0.0.1:8767) to replay a recorded game or choose **Play v9** to play the engine yourself. The recording used Stockfish 18 at 8,000 nodes per move.

**New match** runs a live game against Stockfish. Install Stockfish on your PATH or pass its executable with `--stockfish`. In a series, each opening is played twice with colours reversed.

To search a single position, run:

```sh
.venv/bin/python agent.py --time-left-ms 3000
```

Add `--fen` to supply your own position.
