"""The referee imports this module once per game and calls get_move on each turn."""

from __future__ import annotations

import os
import time

_import_started_ns = time.monotonic_ns()
# Numba reads its optimisation level at import. Level 1 keeps startup compilation short.
os.environ["NUMBA_OPT"] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import chess  # noqa: E402

from engine import board as board_ops  # noqa: E402
from engine import evaluation  # noqa: E402
from engine import search as search_engine  # noqa: E402

V9_WEIGHTS_SHA256 = "6573ac452b30d517ce29b624571be4bb85d51136a2183fef2999514d7c7bd43f"
WARMUP_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

INCREMENT_MS = 500
RESERVE_MS = 120  # transport, JSON, python-chess and suspend/resume overhead (ms)
MOVES_HORIZON = 36.0
SOFT_FRACTION = 0.18
HARD_FACTOR = 2.0
HARD_FRACTION = 0.40
# A zero budget disables the corresponding stop inside search.search(), so the
# policy must always hand back strictly positive deadlines, however low the clock.
MIN_SOFT_MS = 1.0
MIN_HARD_MS = 5.0


def deadlines(time_left_ms: int) -> tuple[int, int]:
    """Return positive soft and hard duration limits in nanoseconds."""
    # The clock is for the rest of the game. Keep a small reserve, spread the
    # remainder over an assumed 36 moves and spend part of the next increment
    # (the time added after a move). The soft limit guides when to finish. The
    # hard limit also lets the engine interrupt a search already in progress.
    # With 3,000 ms left and the usual 500 ms increment, this gives 430 ms
    # soft and 860 ms hard: (3,000 - 120) / 36 + 0.7 × 500 = 430.
    available_ms = max(0.0, float(time_left_ms) - RESERVE_MS)
    base_budget_ms = available_ms / MOVES_HORIZON + 0.7 * _increment_ms
    soft_ms = min(base_budget_ms, available_ms * SOFT_FRACTION)
    hard_ms = min(soft_ms * HARD_FACTOR, available_ms * HARD_FRACTION)
    # Below three seconds, even a typical allocation could consume too much of
    # the clock. Cap the soft allowance against the increment as well.
    if time_left_ms < 3000:
        soft_ms = min(soft_ms, 0.45 * _increment_ms)
        hard_ms = min(hard_ms, available_ms * 0.35)
    soft_ms = max(soft_ms, MIN_SOFT_MS)
    hard_ms = max(hard_ms, MIN_HARD_MS, soft_ms)
    return int(soft_ms * 1e6), int(hard_ms * 1e6)


ASSETS = Path(__file__).with_name("assets")
_weights_path = ASSETS / "nnue_weights.npz"
_psqt_path = ASSETS / "psqt_tables.json"
# Two boards serve different purposes. _observed_board remembers moves
# actually played. Search can make and undo thousands of hypothetical moves
# in _board_state without adding any of them to the game's move history.
_board_state = board_ops.allocate_board_state()
_search_context = None
_observed_board: chess.Board | None = None
_turn_count = 0
_history_resets = 0
_fallbacks = 0
_model_status = "classical"
_increment_ms = INCREMENT_MS
last_diagnostics: dict = {}
last_iterations: list = []


def _log(record: dict) -> None:
    """Write optional diagnostics to stderr without interrupting play if the stream closes."""
    try:
        print(json.dumps(record, separators=(",", ":")), file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def _synchronize(authoritative: chess.Board) -> chess.Board:
    """Recover the opponent's single reply since our last committed move."""
    global _history_resets
    # FEN describes one position. It omits the sequence that produced it, which
    # matters for a repetition draw. On our first call we only know this snapshot.
    if _observed_board is None:
        return authoritative.copy(stack=False)
    target = authoritative.fen()
    if _observed_board.fen() == target:
        return _observed_board.copy(stack=True)
    # Our saved board ends just after our last move. Try each legal opponent
    # reply and compare the resulting FEN. Copy the matching line before the
    # finally block restores the saved board, including on a successful return.
    for reply in _observed_board.legal_moves:
        _observed_board.push(reply)
        try:
            if _observed_board.fen() == target:
                return _observed_board.copy(stack=True)
        finally:
            _observed_board.pop()
    # A mismatch leaves the path unknown. Start recording again from this FEN.
    _history_resets += 1
    _log({"event": "history_discontinuity"})
    return authoritative.copy(stack=False)


def _reversible_keys(board: chess.Board) -> list:
    """Keys of earlier positions inside the current fifty-move window."""
    # A ply is one player's turn. A pawn move or capture resets the halfmove
    # clock and prevents a position from before it recurring. We can look back
    # only as far as both that clock and our observed move stack allow.
    reversible_plies = min(board.halfmove_clock, len(board.move_stack))
    if reversible_plies == 0:
        return []
    probe = board.copy(stack=True)
    earlier_positions = []
    for _ in range(reversible_plies):
        probe.pop()
        earlier_positions.append(probe.fen())
    keys = []
    bitboards, mailbox, state, position_key, history, _, _ = board_ops.allocate_board_state()
    # Popping gave newest first. Feed the earlier positions to the engine in
    # game order, using the same Zobrist keys that search compares after moves.
    # load_fen will append the current position separately on entry to search.
    for fen in reversed(earlier_positions):
        board_ops.load_fen(fen, bitboards, mailbox, state, position_key, history)
        keys.append(position_key[0])
    return keys


def reset_game(increment_ms: int = INCREMENT_MS) -> None:
    """Start with fresh observed history, search tables and diagnostics."""
    global _observed_board, _turn_count, _history_resets, _fallbacks, _increment_ms
    _observed_board = None
    _turn_count = _history_resets = _fallbacks = 0
    _increment_ms = increment_ms
    search_engine.reset_game_tables(_search_context)
    last_diagnostics.clear()
    last_iterations.clear()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move, using observed history and the time available on entry."""
    # Start the hard clock before parsing. Time spent preparing the position
    # comes out of the same allowance as time spent comparing moves.
    entered_ns = time.monotonic_ns()
    global _turn_count, _observed_board, _fallbacks
    _turn_count += 1
    last_iterations.clear()
    # Forsyth-Edwards Notation (FEN) records the pieces, player to move, castling rights,
    # en passant target and move counters. Ask python-chess for legal answers
    # up front. With none, 0000 means "no move". Otherwise keep a fallback.
    authoritative = chess.Board(fen)
    legal_moves = list(authoritative.legal_moves)
    if not legal_moves:
        return "0000"
    chosen = legal_moves[0]
    reason = "search"
    if not authoritative.is_valid():
        # On an invalid board the move generator can allow a king capture.
        # The evaluator needs both kings to index its features.
        _log({"event": "invalid_position"})
        reason = "fallback:invalid_position"
        fresh_history = authoritative.copy(stack=False)
        fresh_history.push(chosen)
        _observed_board = fresh_history
        last_diagnostics.clear()
        last_diagnostics.update(event="move", turn=_turn_count, reason=reason)
        return chosen.uci()
    depth = seldepth = nodes = qnodes = score = 0
    aborted = partial = False
    try:
        history = _synchronize(authoritative)
        if len(legal_moves) == 1:
            reason = "forced"
        else:
            soft_ns, hard_ns = deadlines(time_left_ms)
            # load_fen fills the arrays already held by _search_context. Bitboards
            # tell search where sets of pieces are, the mailbox identifies a piece
            # by square, and the history keys let it recognise repeated positions.
            bitboards, mailbox, position_state, position_keys, position_history, _, _ = _board_state
            board_ops.load_fen(
                fen,
                bitboards,
                mailbox,
                position_state,
                position_keys,
                position_history,
                _reversible_keys(history),
            )
            # Search measures the soft budget from its own start. The hard
            # deadline includes the time already spent reconstructing the board.
            result = search_engine.search(
                _search_context,
                soft_ns=soft_ns,
                hard_ns=entered_ns + hard_ns,
            )
            # Decode the packed move to Universal Chess Interface (UCI) notation,
            # e.g. e7e8q for a promotion. Check it against the referee-facing board's legal moves.
            uci = board_ops.move_to_uci(result.move) if result.move else ""
            candidate = chess.Move.from_uci(uci) if uci else None
            if candidate is None or candidate not in legal_moves:
                raise ValueError("search returned a move outside the legal set: " + uci)
            chosen = candidate
            depth, seldepth = result.depth, result.seldepth
            nodes, qnodes, score = result.nodes, result.qnodes, result.score
            aborted, partial = result.aborted, result.partial
            last_iterations.extend(result.iterations)
        # Only the move we actually return belongs in next turn's history.
        history.push(chosen)
        _observed_board = history
    except Exception as error:  # fall back to a move from the validated list
        _fallbacks += 1
        reason = "fallback:" + type(error).__name__
        # Rebuild from the supplied position plus the fallback move so an
        # interrupted or inconsistent search cannot invent observed history.
        chosen = legal_moves[0]
        fresh_history = authoritative.copy(stack=False)
        fresh_history.push(chosen)
        _observed_board = fresh_history
    # depth describes the chosen iteration. seldepth is the deepest ply
    # recorded through the last completed iteration. nodes includes all visits
    # to qsearch (qnodes), as well as work spent in an aborted iteration.
    elapsed_ms = (time.monotonic_ns() - entered_ns) / 1e6
    last_diagnostics.clear()
    last_diagnostics.update(
        event="move",
        turn=_turn_count,
        ms=round(elapsed_ms, 1),
        left=time_left_ms,
        depth=depth,
        seldepth=seldepth,
        nodes=nodes,
        qnodes=qnodes,
        score=score,
        aborted=aborted,
        partial=partial,
        reason=reason,
        history_resets=_history_resets,
        fallback_count=_fallbacks,
        model=_model_status,
    )
    if __name__ != "__main__":
        _log(last_diagnostics)
    return chosen.uci()


def _initialize(use_weights=True):
    """Load evaluation data before compiling code that reads the tables."""
    global _search_context, _model_status
    evaluation.install_tables_from_json(_psqt_path)
    network = None
    # The SHA-256 digest identifies the exact London weights. An array with
    # the right shape from another run could still score positions differently.
    if use_weights:
        arrays, digest = evaluation.load_network(_weights_path)
        if digest != V9_WEIGHTS_SHA256:
            raise ValueError("assets/nnue_weights.npz does not match the London v9 checksum")
        network = evaluation.make_nn_context(arrays)
        _model_status = "nnue:" + digest[:12]
    _search_context = search_engine.make_context(_board_state, search_engine.new_search_tables(), network)
    search_engine.enable_network(_search_context, network is not None)
    # At cold start, Numba's just-in-time (JIT) compiler specialises the kernels
    # for these array types. Warm up the search here so the first move doesn't pay
    # for compilation, then clear the warm-up position's search results.
    compiled_kernel_count = search_engine.compile_all(_search_context)
    bitboards, mailbox, state, position_keys, history, _, _ = _board_state
    board_ops.load_fen(WARMUP_FEN, bitboards, mailbox, state, position_keys, history)
    warmup_result = search_engine.search(_search_context, max_depth=6)
    search_engine.reset_game_tables(_search_context)
    _log(
        {
            "event": "init",
            "model": _model_status,
            "compiled_kernels": compiled_kernel_count,
            "warm_nodes": warmup_result.nodes,
            "ms": round((time.monotonic_ns() - _import_started_ns) / 1e6, 1),
        }
    )


def _demo():
    """Run one supplied position through the same adapter the referee calls."""
    parser = argparse.ArgumentParser(description="Run one position through the AI Chessathon agent.")
    parser.add_argument("--fen", default=WARMUP_FEN, help="FEN to search")
    parser.add_argument("--time-left-ms", type=int, default=3_000, help="remaining game clock")
    parser.add_argument("--classical", action="store_true", help="use the fitted piece-square evaluator")
    args = parser.parse_args()
    try:
        board = chess.Board(args.fen)
    except ValueError as error:
        parser.error(str(error))
    if not board.is_valid() or not any(board.legal_moves):
        parser.error("provide a valid position with at least one legal move")
    if args.time_left_ms <= 0:
        parser.error("--time-left-ms must be positive")

    try:
        _initialize(use_weights=not args.classical)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    move = get_move(args.fen, args.time_left_ms)
    print(json.dumps({"move": move, **last_diagnostics}, indent=2))


if __name__ == "__main__":
    _demo()
else:
    _initialize()
