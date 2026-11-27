const byId = (id) => document.getElementById(id);
const fileLetters = "abcdefgh";
const pieceNames = {
  k: "king",
  q: "queen",
  r: "rook",
  b: "bishop",
  n: "knight",
  p: "pawn",
};
const form = byId("form");
const fields = form.elements;
// The referee supplies the live position and clocks. Replay and square selection belong to the viewer.
let matchState,
  pendingPath = "",
  autoPlay = false,
  selectedSquare = "",
  orientation = "",
  nextActionTimer,
  snapshotTime = 0,
  viewedGame = 1,
  backgroundRequest;
let pausedAfterError = false;
let premove = "",
  premoveNote = "";
let recording,
  showingRecording = false,
  review,
  reviewPly = 0,
  reviewVersion = 0;
const formatNumber = (number) => Number(number).toLocaleString();
const setText = (element, text) => {
  if (element.textContent !== String(text)) element.textContent = text;
};
const setTextById = (id, text) => setText(byId(id), text);
const titleCase = (value) => value[0].toUpperCase() + value.slice(1);
const isHumanGame = () => matchState.settings.opponent === "human";
const isSeries = () => !isHumanGame() && matchState.series.total > 1;
const gameInView = () => (showingRecording ? recording : matchState);
const canPremove = () =>
  matchState &&
  !review &&
  isHumanGame() &&
  matchState.can_step &&
  (!pendingPath || pendingPath === "/step");

function piecesFromFen(fen) {
  const pieces = {};
  for (const [row, rank] of fen.split(" ")[0].split("/").entries()) {
    let file = 0;
    for (const char of rank) {
      if (pieceNames[char.toLowerCase()]) {
        pieces[fileLetters[file++] + (8 - row)] = char;
      } else {
        file += Number(char);
      }
    }
  }
  return pieces;
}
function coordinate(edge, value) {
  return `<span class="coord ${edge}" aria-hidden="true" data-label="${value}"></span>`;
}
// Keep square buttons mounted between moves so keyboard focus survives each refresh.
function makeBoard(flipped) {
  const key = flipped ? "black" : "white";
  if (orientation === key) return;
  orientation = key;
  const squares = [];
  for (let row = 0; row < 8; row++)
    for (let column = 0; column < 8; column++) {
      const file = fileLetters[flipped ? 7 - column : column],
        rank = flipped ? row + 1 : 8 - row;
      const cell = document.createElement("button");
      cell.className = "square" + ((row + column) % 2 ? " dark" : "");
      cell.dataset.square = file + rank;
      cell.tabIndex = file + rank === (flipped ? "e7" : "e2") ? 0 : -1;
      cell.innerHTML =
        '<svg viewBox="0 0 48 48" class="piece" aria-hidden="true"><use fill="var(--fill)" stroke="var(--edge)" stroke-width="1.65" stroke-linejoin="round" stroke-linecap="round"/></svg>' +
        (column === 0 ? coordinate("rank", rank) : "") +
        (row === 7 ? coordinate("file", file) : "");
      squares.push(cell);
    }
  byId("board").replaceChildren(...squares);
}
let boardKey;
function drawBoard() {
  makeBoard(!showingRecording && isHumanGame() && matchState.settings.color === "black");
  const interactive =
    (!review && matchState.human_turn && !pendingPath) || canPremove();
  const pieces = review?.pieces ?? matchState.pieces;
  const destinations = interactive && matchState.human_turn && selectedSquare
    ? matchState.legal.filter((move) => move.startsWith(selectedSquare)).map((move) => move.slice(2, 4))
    : [];
  const lastMove = review ? "" : matchState.last_move;
  const key = JSON.stringify([orientation, pieces, interactive, selectedSquare, destinations, premove, review?.uci, lastMove]);
  if (key === boardKey) return;
  boardKey = key;
  byId("board").classList.toggle("interactive", interactive);
  for (const cell of byId("board").children) {
    const square = cell.dataset.square,
      piece = pieces[square];
    const name = pieceNames[piece?.toLowerCase()];
    const white = piece && piece === piece.toUpperCase();
    const svg = cell.firstElementChild;
    svg.style.visibility = name ? "visible" : "hidden";
    svg.classList.toggle("white", !!white);
    if (name) svg.firstElementChild.setAttribute("href", "#" + name);
    const target = destinations.includes(square);
    cell.classList.toggle("target", target);
    cell.classList.toggle("occupied", !!piece);
    cell.classList.toggle("selected", interactive && square === selectedSquare);
    const queued =
      !review && [premove.slice(0, 2), premove.slice(2, 4)].includes(square);
    cell.classList.toggle("premove", queued);
    cell.classList.toggle("review-origin", review?.uci.slice(0, 2) === square);
    cell.classList.toggle("review-target", review?.uci.slice(2, 4) === square);
    cell.classList.toggle("last", lastMove?.slice(0, 2) === square || lastMove?.slice(2, 4) === square);
    cell.setAttribute(
      "aria-label",
      square +
        ", " +
        (piece ? (white ? "white " : "black ") + name : "empty") +
        (target ? ", legal destination" : "") +
        (queued ? ", queued premove" : ""),
    );
  }
}
byId("board").addEventListener("focusin", (event) => {
  const cell = event.target.closest("button");
  if (cell)
    for (const other of byId("board").children)
      other.tabIndex = other === cell ? 0 : -1;
});
byId("board").addEventListener("keydown", (event) => {
  const steps = {
    ArrowUp: [-1, 0],
    ArrowDown: [1, 0],
    ArrowLeft: [0, -1],
    ArrowRight: [0, 1],
  };
  if (event.key === "Escape") {
    clearPremove();
    return;
  }
  const step = steps[event.key],
    cells = [...byId("board").children],
    cellIndex = cells.indexOf(event.target);
  if (!step || cellIndex < 0) return;
  event.preventDefault();
  const clamp = (coordinateValue) => Math.min(7, Math.max(0, coordinateValue));
  cells[8 * clamp((cellIndex >> 3) + step[0]) + clamp((cellIndex % 8) + step[1])].focus();
});
function clearPremove() {
  premove = premoveNote = selectedSquare = "";
  render();
}
// Store the intended move locally. The next referee snapshot decides whether it is still legal.
function queuePremove(square) {
  premoveNote = "";
  if (square === selectedSquare) {
    clearPremove();
    return;
  }
  const piece = matchState.pieces[square];
  const white = matchState.settings.color === "white";
  if (piece && (piece === piece.toUpperCase()) === white) {
    selectedSquare = square;
  } else if (selectedSquare) {
    const pawn = matchState.pieces[selectedSquare].toLowerCase() === "p";
    const promotion = pawn && square[1] === (white ? "8" : "1");
    premove = selectedSquare + square + (promotion ? "q" : "");
    selectedSquare = "";
  }
  render();
}
byId("board").addEventListener("click", (event) => {
  const square = event.target.closest("button")?.dataset.square;
  if (!square || review || !matchState) return;
  if (canPremove()) {
    queuePremove(square);
    return;
  }
  if (!matchState.human_turn || pendingPath || matchState.result) return;
  premoveNote = "";
  const routes = selectedSquare
    ? matchState.legal.filter((move) => move.startsWith(selectedSquare + square))
    : [];
  if (routes.length === 1) return request("/move", { uci: routes[0] });
  if (routes.length > 1) {
    const from = selectedSquare,
      dialog = byId("promotion");
    dialog.returnValue = "";
    dialog.addEventListener(
      "close",
      () => {
        const move = from + square + dialog.returnValue;
        if (routes.includes(move) && !review && !matchState.result)
          request("/move", { uci: move });
      },
      { once: true },
    );
    dialog.showModal();
  } else {
    selectedSquare =
      square !== selectedSquare && matchState.legal.some((move) => move.startsWith(square))
        ? square
        : "";
    render();
  }
});
function clockText(milliseconds) {
  const ticks = Math.ceil(Math.max(0, milliseconds) / 100);
  return (
    Math.floor(ticks / 600) +
    ":" +
    String(Math.floor(ticks / 10) % 60).padStart(2, "0") +
    "." +
    (ticks % 10)
  );
}
function liveClockRunning() {
  return (
    matchState &&
    !matchState.result &&
    (matchState.clock_running || pendingPath === "/step")
  );
}
// Interpolate the visible clock between snapshots. Only the referee can declare a flag.
function clockValue(side) {
  const clockRunning =
    liveClockRunning() && matchState.turn.toLowerCase() === side;
  return (
    matchState.clocks[side] - (clockRunning ? performance.now() - snapshotTime : 0)
  );
}
function playerRow(slot, side) {
  const element = byId(slot);
  element.classList.toggle(
    "active",
    review
      ? review.side === side
      : !matchState.result && matchState.turn.toLowerCase() === side,
  );
  if (!element.firstElementChild) element.innerHTML =
    '<span class="identity"><i class="disc" aria-hidden="true"></i><strong></strong><span class="side"></span></span><span class="clock"></span>';
  element.querySelector("i").classList.toggle("white", side === "white");
  setText(element.querySelector("strong"), playerName(side));
  setText(element.querySelector(".side"), titleCase(side));
  const clock = element.querySelector(".clock");
  clock.dataset.side = side;
  setText(clock, clockText(review?.clocks[side] ?? clockValue(side)));
}
function playerName(side) {
  const player = gameInView().players[side];
  return player === "Stockfish" || player === "You" ? player : "London v9";
}
function scoreLabel(centipawns) {
  if (Math.abs(centipawns) >= 30000 - 128) {
    const moves = Math.ceil((30000 - Math.abs(centipawns)) / 2);
    return (centipawns < 0 ? "−" : "") + "M" + moves;
  }
  return (centipawns > 0 ? "+" : "") + (centipawns / 100).toFixed(2);
}
let traceKey;
// Convert the engine’s side-to-move score to White’s perspective for the rail and trace.
function drawAnalysis() {
  const stockfishReview =
    review && playerName(review.side) === "Stockfish";
  const searchReport = review ? review.search : matchState.last_agent_search;
  const sign = searchReport?.side?.toLowerCase() === "black" ? -1 : 1;
  const scored = searchReport && searchReport.depth > 0 && Number.isFinite(searchReport.score);
  const score = scored ? searchReport.score * sign : null;
  const mate =
    stockfishReview && Number.isFinite(searchReport?.mate) ? searchReport.mate * sign : null;
  const label =
    mate !== null
      ? (mate < 0 ? "−M" : "M") + Math.abs(mate)
      : scored
        ? scoreLabel(score) +
          (Math.abs(score) < 30000 - 128 ? " pawns" : "")
        : review
          ? "—"
          : "Awaiting score";
  setTextById("analysis-move", review ? moveLabel(review) : (searchReport?.san ?? ""));
  const flags = searchReport
    ? [
        searchReport.reason === "forced" ? "forced move" : "",
        searchReport.aborted ? "deadline reached" : "",
        searchReport.partial ? "partial depth" : "",
      ].filter(Boolean)
    : [];
  const source = review
    ? `${playerName(review.side)} playing ${titleCase(review.side)}`
    : searchReport
      ? "Latest agent move as " + searchReport.side
      : "Waiting for the agent’s first move.";
  setTextById("source", [source, ...flags].join(" · "));
  setTextById("score", label);
  byId("rail").setAttribute("aria-label", "White evaluation: " + label);
  const flipped = orientation === "black";
  byId("rail").classList.toggle("flipped", flipped);
  setTextById("rail-top", flipped ? "W" : "B");
  setTextById("rail-bottom", flipped ? "B" : "W");
  byId("fill").style.height =
    50 +
    47 * (mate !== null ? Math.sign(mate) : Math.tanh(score / 450)) +
    "%";
  setTextById("depth", searchReport?.depth ?? "—");
  setTextById("nodes", searchReport?.nodes == null ? "—" : formatNumber(searchReport.nodes));
  setTextById("elapsed", searchReport ? formatNumber(searchReport.ms) + " ms" : "—");
  drawSearchIterations(searchReport, sign, stockfishReview);
  drawStockfishReport(review ? stockfishReview && searchReport : matchState.last_stockfish_search, stockfishReview);
  setTextById("evaluator", matchState.evaluator);
}

function drawSearchIterations(searchReport, sign, stockfishReview) {
  // The clock and selection can change many times within a single search trace.
  // Rebuild the depth rows only when the displayed search changes.
  const iterations = searchReport?.iterations ?? [];
  const key = JSON.stringify([sign, iterations]);
  if (key !== traceKey) {
    traceKey = key;
    const rows = iterations
      .slice()
      .reverse()
      .map(([depth, centipawns, nodes, milliseconds, move]) => {
        const tableRow = document.createElement("tr");
        for (const value of [
          depth,
          move || "—",
          scoreLabel(centipawns * sign),
          formatNumber(nodes),
          formatNumber(milliseconds) + " ms",
        ])
          tableRow.insertCell().textContent = value;
        return tableRow;
      });
    byId("iterations").replaceChildren(...rows);
    byId("trace").hidden = !rows.length;
  }
  byId("depth-title").hidden = !!stockfishReview;
  byId("empty").hidden = !!iterations.length || !!stockfishReview;
  setTextById("empty", searchReport ? "No completed depth for this move." : review ? "Human move. No engine searched it." : "No search passes yet.");
  const earlier =
    review && iterations.findLast((row) => row[4] && row[4] !== searchReport.san);
  const switchTo =
    earlier &&
    iterations.find((row) => row[0] > earlier[0] && row[4] === searchReport.san);
  byId("search-change").hidden = !switchTo;
  if (switchTo) {
    setTextById(
      "search-change",
      `Depth ${earlier[0]} preferred ${earlier[4]}. Depth ${switchTo[0]} switched to ${switchTo[4]}.`,
    );
  }
}

function drawStockfishReport(stockfishReport, stockfishReview) {
  byId("opposing").hidden = !stockfishReport;
  if (stockfishReport) {
    const whiteSign = stockfishReport.side === "Black" ? -1 : 1;
    const mate = stockfishReport.mate * whiteSign;
    const value = Number.isFinite(stockfishReport.mate)
      ? (mate < 0 ? "−" : "") + "M" + Math.abs(mate)
      : Number.isFinite(stockfishReport.score)
        ? scoreLabel(stockfishReport.score * whiteSign)
        : "";
    setTextById(
      "opposing-summary",
      [
        stockfishReview
          ? "Final search report"
          : stockfishReport.side + " played " + stockfishReport.san,
        stockfishReport.depth != null ? "depth " + stockfishReport.depth : "",
        stockfishReport.nodes != null ? formatNumber(stockfishReport.nodes) + " nodes" : "",
        value ? value + " for White" : "",
        formatNumber(stockfishReport.ms) + " ms",
      ]
        .filter(Boolean)
        .join(" · "),
    );
    setTextById("pv", stockfishReport.pv ? "Principal variation " + stockfishReport.pv : "");
  }
}

function drawMatch() {
  const series = matchState.series;
  const game = gameInView();
  const config = game.settings;
  const showSeries = !showingRecording && isSeries();
  setTextById(
    "match-heading",
    showingRecording ? "Recorded game" : showSeries ? "Match series" : "Live game",
  );
  const result = !showSeries && game.result;
  byId("match-result").hidden = !result;
  setTextById("match-result", result || "");
  setTextById("game", `Game ${series.game} of ${series.total}`);
  const opponent =
    config.opponent === "human"
      ? "Human vs agent"
      : (game.stockfish || "Stockfish") +
        (config.nodes
          ? " · " + formatNumber(config.nodes) + " nodes/move"
          : " · clock-managed · skill " + config.strength);
  setTextById(
    "match-clock",
    config.clock === "quick" ? "5 s + 0.05 s" : "120 s + 0.5 s",
  );
  setTextById("match-opening", game.opening_name);
  setTextById("match-opponent", opponent);
  byId("series-pane").hidden = !showSeries;
  byId("record").hidden = !showSeries;
  byId("series-progress").hidden = !showSeries;
  byId("series-jobs").hidden = !showSeries || !series.jobs;
  if (!showSeries) return;
  setTextById(
    "record",
    `Agent record  ${series.wins} W  ·  ${series.draws} D  ·  ${series.losses} L`,
  );
  const active = series.jobs?.filter((job) =>
    ["Warming up", "Playing"].includes(job.status),
  ).length;
  setTextById(
    "series-progress",
    series.jobs
      ? `${series.history.length}/${series.total} complete · ${active} in progress · ${Math.floor(series.elapsed_ms / 1000)} s elapsed`
      : "Each opening is played once per colour.",
  );
  if (!series.jobs) return;
  const grid = byId("series-jobs");
  const openings = series.jobs.filter((_, index) => index % 2 === 0);
  const signature = JSON.stringify(openings.map((job) => job.opening));
  if (grid.dataset.openings !== signature) {
    grid.dataset.openings = signature;
    grid.replaceChildren(
      ...openings.map((job, index) => {
        const pair = document.createElement("section");
        pair.className = "opening-pair";
        const heading = document.createElement("h3");
        heading.textContent = job.opening;
        pair.append(heading);
        for (const member of series.jobs.slice(
          2 * index,
          2 * index + 2,
        )) {
          const button = document.createElement("button");
          button.className = "game-tile";
          button.dataset.game = member.game;
          button.append(
            document.createElement("strong"),
            document.createElement("span"),
          );
          pair.append(button);
        }
        return pair;
      }),
    );
  }
  const buttons = grid.querySelectorAll("[data-game]");
  for (const [index, job] of series.jobs.entries()) {
    const button = buttons[index];
    button.setAttribute("aria-pressed", job.game === series.game);
    const title = `Game ${job.game} · ${titleCase(job.agent_color)}`;
    const result = job.result
      ? `${job.result} · ${job.reason}`
      : job.status;
    const detail =
      job.status === "Playing"
        ? `Playing · ${job.plies} ${job.plies === 1 ? "ply" : "plies"}`
        : result;
    if (button.firstChild.textContent !== title)
      button.firstChild.textContent = title;
    if (button.lastChild.textContent !== detail)
      button.lastChild.textContent = detail;
    button.title = [job.opening, job.reason || job.status].join(" · ");
  }
}
let markedPly;
function drawMoves() {
  const list = byId("notation");
  const moves = gameInView().moves;
  const signature = JSON.stringify([showingRecording, matchState.series.game, moves]);
  if (list.dataset.moves !== signature) {
    list.dataset.moves = signature;
    markedPly = null;
    const focusedPly =
      list.contains(document.activeElement) &&
      document.activeElement.dataset.ply;
    let ply = 0;
    list.replaceChildren(
      ...moves.map(({ number, white, black }) => {
        const row = document.createElement("div");
        row.className = "move-row";
        const index = document.createElement("span");
        index.textContent = number + ".";
        row.append(index);
        for (const [side, san] of [
          ["white", white],
          ["black", black],
        ]) {
          const cell = document.createElement(san ? "button" : "span");
          cell.textContent = san || (side === "white" ? "…" : "");
          if (san) {
            cell.className = "move-choice";
            cell.dataset.ply = ++ply;
            cell.setAttribute(
              "aria-label",
              moveLabel({ number, side, san }),
            );
          }
          row.append(cell);
        }
        return row;
      }),
    );
    if (!ply) list.textContent = "No moves played.";
    if (!review) list.scrollTop = list.scrollHeight;
    moveButton(focusedPly)?.focus({ preventScroll: true });
  }
  if (markedPly === reviewPly) return;
  markedPly = reviewPly;
  for (const button of list.querySelectorAll("button")) {
    const ply = Number(button.dataset.ply);
    button.setAttribute("aria-pressed", ply === reviewPly);
    button.tabIndex = ply === (reviewPly || plyCount()) ? 0 : -1;
  }
}
function moveLabel({ number, side, san }) {
  return `${number}${side === "white" ? "." : "…"} ${san}`;
}
function plyCount() {
  return showingRecording ? recording.replay.length : matchState.plies;
}
function moveButton(ply) {
  return byId("notation").querySelector(`[data-ply="${ply}"]`);
}
async function readJson(path, body) {
  const options = body && {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch(path, options).catch(() => {
    throw Error("Couldn’t reach the local match desk.");
  });
  const data = await response.json().catch(() => null);
  if (!response.ok || !data) throw Error(data?.error || "The match desk could not complete the request.");
  return data;
}
function showReview(ply, move, focusMove) {
  setTextById("error", "");
  reviewPly = ply;
  review = { ...move, pieces: piecesFromFen(move.fen) };
  byId("opposing").open = playerName(review.side) === "Stockfish";
  render();
  const list = byId("notation");
  const button = moveButton(ply);
  if (button) {
    const row = button.parentElement;
    if (
      row.offsetTop < list.scrollTop ||
      row.offsetTop + row.offsetHeight >
        list.scrollTop + list.clientHeight
    )
      list.scrollTop =
        row.offsetTop - (list.clientHeight - row.offsetHeight) / 2;
    if (focusMove) button.focus({ preventScroll: true });
  }
}
// A slow review request can arrive after the user has selected a newer move.
async function inspect(ply, focusMove = false) {
  if (ply < 1 || ply > plyCount()) return;
  const version = ++reviewVersion;
  try {
    const move = showingRecording
      ? recording.replay[ply - 1]
      : await readJson(`/review?game=${matchState.series.game}&ply=${ply}`);
    if (version === reviewVersion) showReview(ply, move, focusMove);
  } catch (error) {
    if (version === reviewVersion) setTextById("error", error.message);
  }
}
async function openRecording() {
  const version = ++reviewVersion;
  try {
    recording ??= await readJson("/recording");
    if (version !== reviewVersion) return;
    showingRecording = true;
    showReview(
      recording.featured_ply,
      recording.replay[recording.featured_ply - 1],
    );
  } catch (error) {
    if (version === reviewVersion) setTextById("error", error.message);
  }
}
function leaveReview() {
  ++reviewVersion;
  showingRecording = false;
  review = null;
  reviewPly = 0;
  byId("opposing").open = false;
}
function render() {
  if (!matchState) return;
  drawBoard();
  const bottom = orientation;
  playerRow("top", bottom === "white" ? "black" : "white");
  playerRow("bottom", bottom);
  drawMatch();
  drawMoves();
  drawAnalysis();
  const series = matchState.series;
  const acting = !!pendingPath && !pendingPath.startsWith("/state");
  if (!matchState.can_step && !acting) autoPlay = false;
  byId("setup").disabled = byId("play").disabled = acting && pendingPath !== "/step";
  byId("play").hidden = isHumanGame() && !showingRecording;
  byId("recorded").hidden = showingRecording;
  byId("recorded").disabled = false;
  byId("replay-controls").hidden = !review;
  byId("previous").disabled = reviewPly <= 1;
  byId("next").disabled = reviewPly >= plyCount();
  byId("clear-premove").hidden = !premove;
  byId("step").hidden = !!review || isSeries() || (isHumanGame() && !pausedAfterError);
  byId("step").disabled = acting || !matchState.can_step;
  byId("run").hidden = !!review || isHumanGame();
  byId("run").disabled = isSeries()
    ? acting
    : !autoPlay && (acting || !matchState.can_step);
  setTextById(
    "run",
    isSeries()
      ? series.active
        ? "Stop series"
        : series.jobs
          ? "Run again"
          : "Run series"
      : autoPlay
        ? "Pause"
        : "Run match",
  );
  const job = series.jobs?.[series.game - 1];
  const thinking = pendingPath === "/step" || job?.status === "Playing";
  const player = playerName(matchState.turn.toLowerCase());
  const turn = thinking
    ? player + " thinking…"
    : matchState.turn +
      " to move" +
      (matchState.check ? " · Check" : "") +
      (matchState.human_turn ? " · Select a piece" : "");
  const premoveStatus = premove
    ? `Premove ${premove.slice(0, 2)}–${premove.slice(2)} queued`
    : canPremove()
      ? selectedSquare
        ? "Choose a premove destination"
        : "You can queue a premove"
      : premoveNote;
  const status = pendingPath === "/new" ? "Starting game…" : review
    ? "Before " + moveLabel(review)
    : matchState.result ||
      (job && job.status !== "Playing"
        ? job.reason || job.status
        : [turn, premoveStatus].filter(Boolean).join(" · "));
  setTextById("status", status);
}
function schedule() {
  clearTimeout(nextActionTimer);
  if (!matchState || pendingPath || pausedAfterError || byId("config").open) return;
  if (matchState.human_turn && premove) {
    const move = premove;
    premove = "";
    if (matchState.legal.includes(move)) {
      nextActionTimer = setTimeout(() => request("/move", { uci: move }), 0);
      return;
    }
    premoveNote = "Premove is no longer legal";
    render();
  }
  if (
    matchState.series.active ||
    (matchState.series.jobs && viewedGame !== matchState.series.game)
  ) {
    nextActionTimer = setTimeout(refresh, 270);
  } else if (matchState.can_step && (isHumanGame() || autoPlay) && !isSeries()) {
    nextActionTimer = setTimeout(step, isHumanGame() ? 50 : 320);
  }
}
function background(path, body) {
  return backgroundRequest ??= request(path, body).finally(() => { backgroundRequest = null; });
}
const refresh = () => background("/state?game=" + viewedGame);
const step = () => background("/step", {});
async function request(path, body) {
  // Let an in-flight poll or engine move finish before submitting the next action.
  if (body && backgroundRequest) await backgroundRequest;
  if (pendingPath) return false;
  clearTimeout(nextActionTimer);
  pendingPath = path;
  if (path === "/step") snapshotTime = performance.now();
  setTextById("error", "");
  if (!path.startsWith("/state")) render();
  let success = false;
  try {
    const data = await readJson(path, body);
    if (
      path === "/new" ||
      path === "/series" ||
      (!showingRecording && data.series.game !== matchState?.series.game)
    )
      leaveReview();
    matchState = data;
    if (path === "/new" || matchState.result) premove = premoveNote = "";
    if (selectedSquare && (path !== "/step" || !matchState.legal.some((move) => move.startsWith(selectedSquare)))) selectedSquare = "";
    snapshotTime = performance.now();
    pausedAfterError = false;
    byId("retry").hidden = true;
    success = true;
  } catch (error) {
    autoPlay = false;
    pausedAfterError = true;
    setTextById("error", error.message);
    byId("retry").hidden = false;
    if (!matchState) setTextById("status", "Waiting for the local match desk.");
  } finally {
    pendingPath = "";
    render();
    schedule();
  }
  return success;
}
byId("recorded").onclick = openRecording;
byId("clear-premove").onclick = clearPremove;
byId("live").onclick = () => {
  leaveReview();
  render();
  schedule();
  const control = byId("run").hidden ? byId("recorded") : byId("run");
  (control.disabled ? byId("recorded") : control).focus();
};
byId("previous").onclick = () => inspect(reviewPly - 1);
byId("next").onclick = () => inspect(reviewPly + 1);
byId("notation").onclick = (event) => {
  const button = event.target.closest("[data-ply]");
  if (button) inspect(Number(button.dataset.ply));
};
byId("notation").onkeydown = (event) => {
  const current = Number(event.target.dataset.ply);
  const direction = { ArrowLeft: -1, ArrowRight: 1 }[event.key];
  if (direction && current) {
    event.preventDefault();
    inspect(current + direction, true);
  }
};
byId("step").onclick = step;
byId("run").onclick = () => {
  if (isSeries()) {
    if (matchState.series.active) return request("/stop", {});
    viewedGame = 1;
    return request("/series", {});
  }
  autoPlay = !autoPlay;
  pausedAfterError = false;
  render();
  schedule();
};
byId("series-jobs").onclick = (event) => {
  const button = event.target.closest("[data-game]");
  if (!button) return;
  viewedGame = Number(button.dataset.game);
  if (!pendingPath) refresh();
};
function adjustSetup() {
  const isHuman = fields.opponent.value === "human";
  setTextById("color-label", isHuman ? "You play as" : "Agent starts as");
  byId("advanced").hidden = isHuman;
  for (const name of ["nodes", "strength", "games"]) {
    const group = byId("group-" + name);
    group.hidden = group.disabled = isHuman || (name === "strength" && Number(fields.nodes.value) > 0);
  }
  if (isHuman) fields.games.value = "1";
  const series = Number(fields.games.value) > 1;
  setTextById("start-game", series ? "Start series" : "Start game");
  setTextById("setup-note", series ? "Each pair uses the same opening with colours reversed." : "");
}
byId("setup").onclick = () => {
  clearTimeout(nextActionTimer);
  for (const [key, value] of Object.entries(matchState.settings))
    if (fields.namedItem(key)) fields.namedItem(key).value = value;
  form.querySelector('input[value="stockfish"]').disabled = !matchState.stockfish;
  byId("advanced").open = !isHumanGame() && (Number(fields.nodes.value) > 0 || Number(fields.strength.value) !== 20);
  adjustSetup();
  byId("config").showModal();
  render();
};
function humanDefaults() {
  if (fields.opponent.value === "human") {
    fields.clock.value = "competition";
    fields.opening.value = "start";
    fields.games.value = "1";
  }
  adjustSetup();
}
byId("play").onclick = () => {
  byId("setup").click();
  fields.opponent.value = "human";
  humanDefaults();
  form.querySelector('input[name="color"]:checked').focus();
};
form.onchange = (event) => {
  if (event.target.name === "opponent") humanDefaults();
  else adjustSetup();
};
byId("cancel").onclick = byId("close").onclick = () => byId("config").close();
byId("config").onclose = () => {
  schedule();
};
form.onsubmit = async (event) => {
  event.preventDefault();
  const settings = Object.fromEntries(new FormData(form));
  settings.games = Number(fields.games.value);
  settings.strength = Number(fields.strength.value);
  settings.nodes = Number(fields.nodes.value);
  viewedGame = 1;
  autoPlay = false;
  byId("config").close();
  if (await request("/new", settings)) {
    if (settings.games > 1) await request("/series", {});
    else if (settings.opponent === "human") byId("board").querySelector('[tabindex="0"]').focus({ preventScroll: true });
    else {
      autoPlay = true;
      schedule();
    }
  }
};
setInterval(() => {
  if (review || pausedAfterError || !liveClockRunning()) return;
  const side = matchState.turn.toLowerCase();
  const value = clockValue(side);
  setText(document.querySelector('.clock[data-side="' + side + '"]'), clockText(value));
  if (matchState.human_turn && value <= 0 && !pendingPath) refresh();
}, 100);
async function connect() {
  const first = !matchState;
  if (await refresh() && first && !matchState.plies && !matchState.result && !matchState.series.jobs && !review)
    openRecording();
}
byId("retry").onclick = connect;
connect();
