"""Local competition-clock games and colour-paired series."""

import time
from dataclasses import asdict, dataclass

import chess
import chess.engine


CLOCKS = {"competition": (120_000, 500), "quick": (5_000, 50)}
CURATED = (
    ("Sicilian Closed", "r1bqk2r/pp2ppbp/2np1np1/2p5/P3P3/2NPB1P1/1PP2PBP/R2QK1NR b KQkq - 0 7"),
    ("English Symmetrical", "r1bqk2r/pp1pppbp/2n2np1/2p5/2P4P/2N2NP1/PP1PPPB1/R1BQK2R b KQkq - 0 6"),
    ("Nimzo-Indian Defence", "rnbq1rk1/ppp2ppp/4pn2/3p4/2PP4/P1PBP3/5PPP/R1BQK1NR b KQ - 0 7"),
    ("French Advance", "r3kbnr/pp1b1ppp/1qn1p3/3pP3/3P4/5N1P/PP2BPP1/RNBQK2R b KQkq - 0 8"),
)
OPENINGS = {
    "curated": CURATED,
    "start": (("Initial position", chess.STARTING_FEN),),
    "tactics": (("Tactical middlegame", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),),
}


@dataclass(frozen=True)
class Settings:
    """Configuration shared by one local game or each game in a series."""
    opponent: str = "stockfish"
    color: str = "white"
    clock: str = "competition"
    opening: str = "curated"
    games: int = 1
    strength: int = 20
    nodes: int = 0

    def validate(self, stockfish_available):
        """Reject combinations the local referee or opponent cannot run."""
        if self.opponent not in ("stockfish", "human") or self.color not in ("white", "black"):
            raise ValueError("Choose an opponent and a colour.")
        if self.clock not in CLOCKS or self.opening not in OPENINGS:
            raise ValueError("Choose a clock and a starting position.")
        if type(self.games) is not int or self.games not in (1, 2, 4, 8):
            raise ValueError("Choose 1, 2, 4 or 8 games.")
        if type(self.strength) is not int or self.strength not in (0, 5, 10, 20):
            raise ValueError("Choose Stockfish skill 0, 5, 10 or 20.")
        if type(self.nodes) is not int or self.nodes not in (0, 8_000, 32_000, 128_000):
            raise ValueError("Choose a clock-managed or fixed-node opponent.")
        if self.opponent == "stockfish" and not stockfish_available:
            raise ValueError("Stockfish is unavailable. Choose Play v9 or install Stockfish.")
        if self.opponent == "human" and self.games != 1:
            raise ValueError("Choose one game when playing yourself.")


def game_plan(settings, index):
    """Use paired openings: play each position with colours reversed to balance any opening advantage."""
    openings = OPENINGS[settings.opening]
    name, fen = openings[(index // 2) % len(openings)]
    color = settings.color == "white"
    agent_color = not color if settings.opponent == "human" else color ^ bool(index % 2)
    return name, fen, agent_color


def outcome(board):
    """Match the referee order, including mate before draw rules."""
    decision = board.outcome()
    if decision:
        reason = decision.termination.name.replace("_", " ").capitalize()
        return decision.winner, reason
    if board.is_repetition(3):
        return None, "Threefold repetition"
    if board.is_fifty_moves():
        return None, "Fifty-move rule"
    if board.ply() >= 600:
        return None, "600-ply limit"
    return None


class Match:
    """Own the legal position, both clocks and replay snapshots for a local match."""
    def __init__(self, agent, stockfish=None, now=time.monotonic, settings=None, first_game=0):
        self.agent = agent
        self.stockfish = stockfish
        self.now = now
        self.stockfish_name = stockfish.id.get("name", "Stockfish") if stockfish else None
        idle = settings is None and stockfish is None
        if settings is None:
            settings = Settings() if stockfish else Settings(opponent="human", opening="start", games=1)
        self.new(settings, first_game)
        if idle:
            self.human_started = None  # Viewing the desk does not start a timed game.

    def new(self, settings, first_game=0):
        """Begin a fresh match with the selected settings and first game index."""
        settings.validate(self.stockfish is not None)
        if settings.opponent == "stockfish":
            self.stockfish.configure({"Skill Level": 20 if settings.nodes else settings.strength})
        self.settings = settings
        self.first_game = first_game
        self.history = []
        self._start_game()

    def _start_game(self):
        """Reset the board and clocks at the scheduled opening."""
        index = self.first_game + len(self.history)
        self.opening_name, fen, self.agent_color = game_plan(self.settings, index)
        self.board = chess.Board(fen)
        self.base_ms, self.increment_ms = CLOCKS[self.settings.clock]
        self.clocks = {chess.WHITE: float(self.base_ms), chess.BLACK: float(self.base_ms)}
        self.players = {
            self.agent_color: "Agent v9",
            not self.agent_color: "You" if self.settings.opponent == "human" else "Stockfish",
        }
        self.agent.reset_game(self.increment_ms)
        self.game = object()  # python-chess uses a different object to send ucinewgame
        self.result = self.search = self.last_agent_search = self.last_stockfish_search = None
        self.moves = []
        self.replay = []
        self.human_started = self.now() if self.human_turn else None

    @property
    def human_turn(self):
        return self.result is None and self.players[self.board.turn] == "You"

    @property
    def can_step(self):
        return self.result is None and not self.human_turn

    def _finish(self, winner, reason):
        score = "1/2-1/2" if winner is None else "1-0" if winner == chess.WHITE else "0-1"
        self.result = f"{score} · {reason}"
        self.history.append({
            "game": self.first_game + len(self.history) + 1,
            "agent_color": chess.COLOR_NAMES[self.agent_color],
            "result": score,
            "reason": reason,
            "plies": len(self.board.move_stack),
            "points": 0.5 if winner is None else int(winner == self.agent_color),
        })
        self.human_started = None

    def _flag(self, mover):
        """Adjudicate an expired clock, including draws against insufficient mating material."""
        winner = None if self.board.has_insufficient_material(not mover) else not mover
        self._finish(winner, "Time forfeit" if winner is not None else "Time forfeit with insufficient material")

    def _accept(self, move, elapsed_ms, search=None):
        """Check for flag fall (running out of time) before awarding the increment for a legal move."""
        mover = self.board.turn
        clocks_before = {chess.COLOR_NAMES[color]: int(milliseconds) for color, milliseconds in self.clocks.items()}
        self.clocks[mover] -= elapsed_ms
        if self.clocks[mover] < 0:
            self._flag(mover)
            return
        if move not in self.board.legal_moves:
            self._finish(not mover, "Illegal move")
            return
        self.search = search
        if search:
            if mover == self.agent_color:
                self.last_agent_search = search
            else:
                self.last_stockfish_search = search
        number = self.board.fullmove_number
        side = chess.COLOR_NAMES[mover]
        # Standard Algebraic Notation (SAN) needs the pre-move board, e.g. to distinguish two knights.
        san = search["san"] if search else self.board.san(move)
        self.replay.append({
            "fen": self.board.fen(),
            "number": number,
            "side": side,
            "uci": move.uci(),
            "san": san,
            "clocks": clocks_before,
            "search": search,
        })
        if mover == chess.WHITE or not self.moves:
            self.moves.append({"number": number, "white": None, "black": None})
        self.moves[-1][side] = san
        self.board.push(move)
        self.clocks[mover] += self.increment_ms
        finish = outcome(self.board)
        if finish:
            self._finish(*finish)
        else:
            self.human_started = self.now() if self.human_turn else None

    def _human_clock(self):
        """Adjudicate a human flag even while the browser is only polling for updates."""
        if self.human_started is not None:
            elapsed = (self.now() - self.human_started) * 1000
            if self.clocks[self.board.turn] - elapsed < 0:
                self.clocks[self.board.turn] = 0
                self._flag(self.board.turn)

    def human_move(self, uci):
        if not self.human_turn:
            raise ValueError("There is no human turn to play.")
        elapsed_ms = 0 if self.human_started is None else (self.now() - self.human_started) * 1000
        if elapsed_ms > self.clocks[self.board.turn]:
            self.clocks[self.board.turn] = 0
            self._flag(self.board.turn)
            return
        move = self.board.parse_uci(uci)
        if not move:
            raise ValueError("Choose a legal move.")
        self._accept(move, elapsed_ms)

    def _agent_move(self):
        """Time the competition call and attach its completed search iterations to the replay."""
        board = self.board
        fen = board.fen()
        started = self.now()
        uci = self.agent.get_move(fen, int(self.clocks[board.turn]))
        elapsed_ms = (self.now() - started) * 1000
        trace = self.agent.last_diagnostics
        notation_by_uci = {move.uci(): board.san(move) for move in board.legal_moves}
        iterations = [
            (depth, score, nodes, iteration_ms, notation_by_uci.get(self.agent.board_ops.move_to_uci(move)))
            for depth, score, nodes, iteration_ms, move in self.agent.last_iterations
        ]
        return uci, elapsed_ms, {
            "iterations": iterations,
            **{key: trace[key] for key in ("depth", "nodes", "score", "reason", "aborted", "partial")},
        }

    def _stockfish_move(self):
        """Run the UCI opponent with a clock or node budget and collect its analysis."""
        board = self.board
        self.stockfish.timeout = max(1.0, self.clocks[board.turn] / 1000 + 0.5)
        clocks = chess.engine.Limit(
            white_clock=self.clocks[chess.WHITE] / 1000,
            black_clock=self.clocks[chess.BLACK] / 1000,
            white_inc=self.increment_ms / 1000,
            black_inc=self.increment_ms / 1000,
        )
        limit = chess.engine.Limit(nodes=self.settings.nodes) if self.settings.nodes else clocks
        started = self.now()
        result = self.stockfish.play(board, limit, game=self.game, info=chess.engine.INFO_ALL)
        elapsed_ms = (self.now() - started) * 1000
        info = result.info
        principal_variation = info.get("pv") or []
        score = info["score"].pov(board.turn) if "score" in info else None
        uci = result.move.uci() if result.move else "0000"
        return uci, elapsed_ms, {
            "depth": info.get("depth"),
            "nodes": info.get("nodes"),
            "score": score.score() if score is not None else None,
            "mate": score.mate() if score is not None else None,
            "pv": board.variation_san(principal_variation[:8]) if principal_variation and principal_variation[0] == result.move else None,
            "iterations": [],
            "reason": "search",
            "aborted": False,
            "partial": False,
        }

    def step(self):
        """Ask the current engine for a move and translate timeouts or invalid responses into results."""
        self._human_clock()
        if not self.can_step:
            raise ValueError("No engine move is available.")
        mover = self.board.turn
        started = self.now()
        try:
            uci, elapsed_ms, search = self._agent_move() if mover == self.agent_color else self._stockfish_move()
        except (chess.engine.EngineError, TimeoutError, OSError):
            if (self.now() - started) * 1000 > self.clocks[mover]:
                self.clocks[mover] = 0
                self._flag(mover)
            else:
                self._finish(not mover, "Engine error")
            return
        try:
            move = self.board.parse_uci(uci)
        except ValueError:
            move = chess.Move.null()
        if move:
            search.update(
                actor=self.players[mover],
                side=chess.COLOR_NAMES[mover].capitalize(),
                san=self.board.san(move),
                ms=round(elapsed_ms, 1),
                capture=self.board.is_capture(move),
                check=self.board.gives_check(move),
            )
        self._accept(move, elapsed_ms, search)

    def state(self):
        """Build the browser snapshot from the referee’s authoritative position and clocks."""
        self._human_clock()
        board = self.board
        clocks = self.clocks.copy()
        if self.human_started is not None:
            clocks[board.turn] -= (self.now() - self.human_started) * 1000
        return {
            "build": "London v9",
            "evaluator": "Trained NNUE",
            "opening_name": self.opening_name,
            "pieces": {chess.square_name(square): piece.symbol() for square, piece in board.piece_map().items()},
            "legal": [] if self.result else [move.uci() for move in board.legal_moves],
            "turn": chess.COLOR_NAMES[board.turn].capitalize(),
            "check": board.is_check(),
            "result": self.result,
            "last_move": board.peek().uci() if board.move_stack else None,
            "plies": len(board.move_stack),
            "moves": self.moves,
            "search": self.search,
            "last_agent_search": self.last_agent_search,
            "last_stockfish_search": self.last_stockfish_search,
            "settings": asdict(self.settings),
            "stockfish": self.stockfish_name,
            "clock_control": {"base_ms": self.base_ms, "increment_ms": self.increment_ms},
            "clocks": {chess.COLOR_NAMES[color]: max(0, int(milliseconds)) for color, milliseconds in clocks.items()},
            "players": {chess.COLOR_NAMES[color]: name for color, name in self.players.items()},
            "human_turn": self.human_turn,
            "clock_running": self.human_started is not None,
            "can_step": self.can_step,
            "series": {
                "game": min(len(self.history) + int(self.result is None), self.settings.games),
                "total": self.settings.games,
                "wins": sum(row["points"] == 1 for row in self.history),
                "draws": sum(row["points"] == 0.5 for row in self.history),
                "losses": sum(row["points"] == 0 for row in self.history),
                "history": self.history,
            },
        }
