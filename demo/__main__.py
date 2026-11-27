"""Serve the local chessboard and match controls."""

import argparse
import json
import shutil
import webbrowser
from contextlib import ExitStack, suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

import chess.engine

from demo.match import Match, Settings
from demo.series import Series


def handler_for(match, stockfish_path, workers, lifetime):
    """Bind a local HTTP handler to the current referee and optional series under one match lock."""
    assets = {
        route: (Path(__file__).with_name(filename).read_bytes(), content_type)
        for route, filename, content_type in (
            ("/", "board.html", "text/html; charset=utf-8"),
            ("/board.css", "board.css", "text/css; charset=utf-8"),
            ("/board.js", "board.js", "text/javascript; charset=utf-8"),
            ("/recording", "recorded-game.json", "application/json"),
        )
    }
    series = None
    match_lock = Lock()  # Serialize the shared referee while allowing independent HTTP connections.

    def stop_series():
        if series:
            series.stop()

    lifetime.callback(stop_series)

    def state(game=1):
        preview = match.state()
        if series:
            game = max(1, min(game, series.settings.games))
            return series.view(game, preview)
        return preview

    def apply_action(path, data):
        """Validate a browser action and dispatch it to the referee or series controller."""
        nonlocal series
        if path == "/new":
            match.new(Settings(**data))
            stop_series()
            series = None
        elif path == "/series":
            settings = match.settings
            if settings.opponent != "stockfish" or settings.games < 2:
                raise ValueError("Set up at least two Stockfish games for a series.")
            if series is None or not series.view(1, match.state())["series"]["active"]:
                stop_series()
                series = Series(settings, stockfish_path, workers)
        elif path == "/stop" and series:
            stop_series()
        elif path == "/move" and not series and isinstance(data.get("uci"), str):
            match.human_move(data["uci"])
        elif path == "/step" and not series and match.settings.games == 1:
            match.step()
        else:
            raise ValueError("Unknown action.")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, body, content_type, status=HTTPStatus.OK):
            with suppress(BrokenPipeError, ConnectionResetError):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        def send_json(self, data, status=HTTPStatus.OK):
            self.reply(json.dumps(data).encode(), "application/json", status)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path in assets:
                self.reply(*assets[url.path])
            elif url.path in ("/state", "/review"):
                try:
                    query = parse_qs(url.query)
                    game = int(query.get("game", ["1"])[0])
                    with match_lock:
                        if url.path == "/state":
                            data = state(game)
                        else:
                            ply = int(query.get("ply", ["0"])[0])
                            if series:
                                data = series.review(game, ply)
                            else:
                                data = match.replay[ply - 1] if game == 1 and 0 < ply <= len(match.replay) else None
                        body = json.dumps(data).encode()
                    if data is None:
                        self.send_json({"error": "That move is unavailable."}, HTTPStatus.NOT_FOUND)
                    else:
                        self.reply(body, "application/json")
                except ValueError:
                    self.send_json({"error": "Invalid game or move number."}, HTTPStatus.BAD_REQUEST)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Send a JSON action.")
                body_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < body_length <= 1024:
                    raise ValueError("Invalid request length.")
                data = json.loads(self.rfile.read(body_length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object.")
                with match_lock:
                    apply_action(self.path, data)
                    body = json.dumps(state()).encode()
            except (ValueError, TypeError, chess.engine.EngineError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self.reply(body, "application/json")

        def log_request(self, code="-", size="-"):
            # A series can make hundreds of ordinary requests.
            if isinstance(code, int) and code >= 400 and self.path != "/favicon.ico":
                super().log_request(code, size)

    return Handler


def main():
    """Warm the engine, locate the optional UCI opponent and serve the desk on loopback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish", type=Path, help="path to the optional Stockfish executable")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    stockfish_path = args.stockfish.resolve() if args.stockfish else shutil.which("stockfish")
    with ExitStack() as lifetime:
        print("Warming up v9…", flush=True)
        try:
            import agent

            stockfish = None
            if stockfish_path:
                stockfish = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))
                lifetime.callback(stockfish.close)
                stockfish.configure({"Threads": 1, "Hash": 128})
            match = Match(agent, stockfish)
            handler = handler_for(match, stockfish_path, args.workers, lifetime)
            server = lifetime.enter_context(ThreadingHTTPServer(("127.0.0.1", args.port), handler))
        except (OSError, ValueError, chess.engine.EngineError) as error:
            parser.error(str(error))
        opponent = match.stockfish_name or "human play available"
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"London v9 · {opponent}. Open {url} (Ctrl+C to stop).", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
