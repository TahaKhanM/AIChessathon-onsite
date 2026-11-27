"""Run independent, cold-started games alongside the interactive demo."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import chess
import chess.engine

from demo.match import Match, Settings, game_plan


INIT_SECONDS = 30


def run_game(stockfish_path, settings, index):
    """A worker owns one game. Send snapshots over stdout and engine logs over stderr."""
    import agent

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as stockfish:
        stockfish.configure({"Threads": 1, "Hash": 128})
        match = Match(agent, stockfish, settings=replace(settings, games=1), first_game=index)
        print(json.dumps(match.state()), flush=True)
        while match.can_step:
            before = len(match.replay)
            match.step()
            snapshot = match.state()
            if len(match.replay) > before:
                snapshot["review"] = match.replay[-1]
            print(json.dumps(snapshot), flush=True)


def terminate(process):
    """Signal the worker and its UCI subprocess on POSIX."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass


class Series:
    """Schedule independent game processes and retain their latest snapshots for review."""
    def __init__(self, settings, stockfish_path, workers):
        self.settings = settings
        self.stockfish_path = str(stockfish_path)
        self.workers = min(workers, settings.games)
        self.started = time.monotonic()
        self.finished = None
        self.stopped_at = None
        self.lock = threading.Lock()
        self.processes = {}
        self.views = {}
        self.replays = [[] for _ in range(settings.games)]
        self.jobs = []
        for index in range(settings.games):
            name, _, color = game_plan(settings, index)
            self.jobs.append({
                "game": index + 1,
                "opening": name,
                "agent_color": chess.COLOR_NAMES[color],
                "status": "Queued",
                "plies": 0,
            })
        self.pool = ThreadPoolExecutor(max_workers=self.workers)
        for index in range(settings.games):
            self.pool.submit(self._run, index)
        self.pool.shutdown(wait=False)

    def _run(self, index):
        """Launch a worker, enforce its startup budget and consume the snapshots it publishes."""
        job = self.jobs[index]
        with self.lock:
            if self.stopped_at is not None:
                return
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "demo.series", self.stockfish_path,
                     json.dumps(asdict(self.settings)), str(index)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    start_new_session=os.name == "posix",
                )
            except OSError as error:
                job.update(status="Error", reason=str(error))
                return
            job.update(status="Warming up")
            self.processes[index] = process

        expired = threading.Event()

        def expire():
            expired.set()
            terminate(process)

        watchdog = threading.Timer(INIT_SECONDS, expire)
        watchdog.daemon = True
        watchdog.start()
        issue = None
        try:
            with process.stdout:
                for line in process.stdout:
                    snapshot = json.loads(line)
                    review = snapshot.pop("review", None)
                    watchdog.cancel()
                    with self.lock:
                        if self.stopped_at is not None:
                            break
                        if review is not None:
                            self.replays[index].append(review)
                        self.views[index] = (snapshot, time.monotonic())
                        job["plies"] = snapshot["plies"]
                        if snapshot["result"]:
                            job.update(snapshot["series"]["history"][0], status="Finished")
                        else:
                            job["status"] = "Playing"
        except (OSError, ValueError, KeyError) as error:
            issue = str(error)
            terminate(process)
        finally:
            watchdog.cancel()
            exit_code = process.wait()
            with self.lock:
                self.processes.pop(index, None)
                if job["status"] != "Finished":
                    if self.stopped_at is not None:
                        job.update(status="Stopped")
                    else:
                        reason = (
                            "Initialization exceeded 30 s" if expired.is_set() else
                            issue or f"Worker exited before the game finished ({exit_code})"
                        )
                        job.update(status="Error", reason=reason)
                self._set_finish_time()

    def _set_finish_time(self):
        if not any(job["status"] in ("Queued", "Warming up", "Playing") for job in self.jobs):
            self.finished = self.finished or time.monotonic()

    def stop(self):
        """Stop scheduling further work and terminate live worker process groups."""
        with self.lock:
            self.stopped_at = self.stopped_at or time.monotonic()
            for job in self.jobs:
                if job["status"] == "Queued":
                    job.update(status="Stopped")
            for process in self.processes.values():
                terminate(process)
            self._set_finish_time()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def review(self, game, ply):
        with self.lock:
            if 0 < game <= len(self.replays) and 0 < ply <= len(self.replays[game - 1]):
                return self.replays[game - 1][ply - 1]
        return None

    def view(self, number, preview):
        """Return a consistent snapshot of the selected game, including queued and finished games."""
        with self.lock:
            self._set_finish_time()
            current, received_at = self.views.get(number - 1, (None, None))
            jobs = [job.copy() for job in self.jobs]
            active = self.finished is None
            stopped_at = self.stopped_at
            elapsed_seconds = (self.finished or time.monotonic()) - self.started
        state = (current or preview).copy()
        history = [job for job in jobs if job["status"] == "Finished"]
        state["series"] = {
            "game": number,
            "total": self.settings.games,
            "wins": sum(job["points"] == 1 for job in history),
            "draws": sum(job["points"] == 0.5 for job in history),
            "losses": sum(job["points"] == 0 for job in history),
            "history": history,
            "jobs": jobs,
            "workers": self.workers,
            "active": active,
            "elapsed_ms": round(elapsed_seconds * 1000),
        }
        state.update(can_step=False, human_turn=False, settings=asdict(self.settings))
        status = jobs[number - 1]["status"]
        state["clock_running"] = status == "Playing" and stopped_at is None
        if current and status in ("Playing", "Stopped"):
            side = state["turn"].lower()
            clocks = state["clocks"].copy()
            thinking_ms = int(((stopped_at or time.monotonic()) - received_at) * 1000)
            clocks[side] = max(0, clocks[side] - thinking_ms)
            state["clocks"] = clocks
        if current is None:
            name, fen, color = game_plan(self.settings, number - 1)
            board = chess.Board(fen)
            state.update(
                opening_name=name,
                pieces={chess.square_name(square): piece.symbol() for square, piece in board.piece_map().items()},
                players={
                    chess.COLOR_NAMES[color]: "Agent v9",
                    chess.COLOR_NAMES[not color]: "Stockfish",
                },
                turn=chess.COLOR_NAMES[board.turn].capitalize(),
                check=board.is_check(),
                clocks={"white": preview["clock_control"]["base_ms"],
                        "black": preview["clock_control"]["base_ms"]},
                last_move=None,
                plies=0,
                moves=[],
                search=None,
                last_agent_search=None,
                last_stockfish_search=None,
                result=None,
            )
        return state


if __name__ == "__main__":
    run_game(sys.argv[1], Settings(**json.loads(sys.argv[2])), int(sys.argv[3]))
