"""Principal variation search with iterative deepening and a per-move deadline."""

from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, objmode

from engine import board as board_ops
from engine import evaluation

# Scores favour the player whose turn it is. The scale uses roughly 100 units
# per pawn. A separate range for mate lets search prefer a forced win over
# any material gain, an earlier win over a later one, and a later loss over
# an immediate one.
MATE = 30000
INF = 32000
MATE_IN_MAX = MATE - board_ops.MAX_SEARCH_PLY
DRAW = 0
TT_EMPTY, TT_UPPER, TT_LOWER, TT_EXACT = 0, 1, 2, 3

# Mutable per-search counters, flags and limits in a single int64 array.
I_NODES, I_STOP, I_SELDEPTH, I_HARD_NS, I_ROOT_BEST, I_ROOT_SCORE, I_TT_AGE, I_ROOT_HIST = range(8)
I_NULL_MIN_PLY, I_ROOT_DEPTH, I_QNODES, I_NODE_LIMIT, I_USE_NN = range(8, 13)
I_BEST_NODES = 13  # nodes spent below the current root best move (time management)
EVAL_CLAMP = 20000
# Search stack rows have four leading slots for looking up preceding plies.
SS_MOVE, SS_PIECE, SS_EVAL, SS_NULL, SS_EXCL = range(5)

PIECE_VALUE = np.array([0, 100, 320, 330, 500, 950, 0, 100, 320, 330, 500, 950, 0, 0], dtype=np.int64)
PIECE_VALUE.setflags(write=False)
SEE_VALUE = np.array([0, 100, 320, 330, 500, 950, 20000], dtype=np.int64)  # by type
SEE_VALUE.setflags(write=False)

# Late move reductions (LMR) give lower-ranked moves a shorter first look.
# Precompute the reduction for each depth and move number once at startup.
LMR_TABLE = np.zeros((64, 64), dtype=np.int64)
for _d in range(1, 64):
    for _m in range(1, 64):
        LMR_TABLE[_d, _m] = int(0.75 + math.log(_d) * math.log(_m) / 2.25)
LMR_TABLE.setflags(write=False)

HISTORY_MAX = 16384
ZERO = np.int64(0)  # non-literal constants keep recursive call signatures identical
ONE = np.int64(1)
TT_ENTRIES = 1 << 23  # 128 MiB of (key, data) uint64 pairs
CORR_SIZE = 1 << 14
CORR_GRAIN = 256
CORR_LIMIT = 64 * CORR_GRAIN

# ordering score bands
S_TT = 1 << 30
S_GOOD_CAPTURE = 1 << 28
S_KILLER = 1 << 26
S_BAD_CAPTURE = -(1 << 26)

# Named slots for the typed arrays shared by recursive Numba calls.
(
    C_BITBOARDS,
    C_MAILBOX,
    C_POSITION,
    C_POSITION_KEYS,
    C_POSITION_HISTORY,
    C_UNDO,
    C_PREVIOUS_KEYS,
    C_MOVE_STACK,
    C_SCORE_STACK,
    C_QUIET_STACK,
    C_TRANSPOSITIONS,
    C_KILLERS,
    C_QUIET_HISTORY,
    C_CAPTURE_HISTORY,
    C_CONTINUATION_HISTORY,
    C_COUNTER,
    C_SEARCH_STACK,
    C_INFO,
    C_PAWN_CORRECTION,
    C_NETWORK,
) = range(20)


def new_search_tables(tt_entries=TT_ENTRIES):
    """Allocate reusable move buffers, transposition entries and ordering histories for one game."""
    # Each recursive call needs its own list of candidates. Index the buffers
    # by ply (turns from the root) so exploring a reply preserves the parent's
    # remaining choices. The calls all make and undo moves on the same board.
    move_stack = np.zeros((board_ops.MAX_SEARCH_PLY + 4, board_ops.MOVE_BUFFER_CAPACITY), dtype=np.int64)
    score_stack = np.zeros((board_ops.MAX_SEARCH_PLY + 4, board_ops.MOVE_BUFFER_CAPACITY), dtype=np.int64)
    quiet_stack = np.zeros((board_ops.MAX_SEARCH_PLY + 4, board_ops.MOVE_BUFFER_CAPACITY), dtype=np.int64)
    transposition_table = np.zeros((tt_entries, 2), dtype=np.uint64)
    killers = np.zeros((board_ops.MAX_SEARCH_PLY + 4, 2), dtype=np.int64)
    quiet_history = np.zeros((2, 64, 64), dtype=np.int64)
    capture_history = np.zeros((13, 64, 7), dtype=np.int64)
    continuation_history = np.zeros((13, 64, 13, 64), dtype=np.int32)
    countermoves = np.zeros((13, 64), dtype=np.int64)
    search_stack = np.zeros((board_ops.MAX_SEARCH_PLY + 8, 5), dtype=np.int64)
    search_info = np.zeros(I_BEST_NODES + 1, dtype=np.int64)
    pawn_correction = np.zeros((2, CORR_SIZE), dtype=np.int32)  # pawn-structure correction history
    return (
        move_stack,
        score_stack,
        quiet_stack,
        transposition_table,
        killers,
        quiet_history,
        capture_history,
        continuation_history,
        countermoves,
        search_stack,
        search_info,
        pawn_correction,
    )


def make_context(state, tables, network=None):
    """Group board and search storage in the fixed layout shared by compiled kernels. A zero network preserves its type in classical mode."""
    bitboards, mailbox, position_state, position_keys, position_history, undo_stack, previous_keys = state
    if network is None:
        network = evaluation.make_nn_context(evaluation.zero_network())
    return (
        (bitboards, mailbox, position_state, position_keys, position_history, undo_stack, previous_keys)
        + tuple(tables)
        + (network,)
    )


def enable_network(context, enabled=True):
    """Select the evaluator and invalidate scores accumulated under the previous mode."""
    context[C_INFO][I_USE_NN] = 1 if enabled else 0
    evaluation.invalidate(context[C_NETWORK])


def reset_game_tables(context):
    """Clear per-game learned state (call once per new game, not per move)."""
    context[C_TRANSPOSITIONS][:] = 0
    context[C_KILLERS][:] = 0
    context[C_QUIET_HISTORY][:] = 0
    context[C_CAPTURE_HISTORY][:] = 0
    context[C_CONTINUATION_HISTORY][:] = 0
    context[C_COUNTER][:] = 0
    context[C_PAWN_CORRECTION][:] = 0
    use_nn = context[C_INFO][I_USE_NN]
    context[C_INFO][:] = 0
    context[C_INFO][I_USE_NN] = use_nn
    evaluation.invalidate(context[C_NETWORK])


@njit(cache=False)
def static_eval(context, ply):
    """Side-to-move static evaluation, clamped away from mate scores."""
    # The evaluator estimates a position as it stands. Search supplies the
    # lookahead by making moves and evaluating the positions that result.
    if context[C_INFO][I_USE_NN]:
        score = evaluation.evaluate_nn(
            context[C_NETWORK],
            context[C_BITBOARDS],
            context[C_MAILBOX],
            context[C_POSITION],
            context[C_POSITION_KEYS],
            context[C_UNDO],
            context[C_PREVIOUS_KEYS],
            ply,
        )
    else:
        score = evaluation.evaluate_psqt(context[C_POSITION])
    if score > EVAL_CLAMP:
        score = EVAL_CLAMP
    elif score < -EVAL_CLAMP:
        score = -EVAL_CLAMP
    return score


@njit(cache=False)
def touch_eval(context, ply):
    """Keep incremental accumulators valid at nodes that skip static evaluation."""
    if context[C_INFO][I_USE_NN]:
        evaluation.ensure_accumulators(
            context[C_NETWORK],
            context[C_BITBOARDS],
            context[C_MAILBOX],
            context[C_POSITION],
            context[C_POSITION_KEYS],
            context[C_UNDO],
            context[C_PREVIOUS_KEYS],
            ply,
        )


@njit(cache=False)
def clock_ns():
    """Read a monotonic clock from a compiled search through a short Python callback."""
    with objmode(now="int64"):
        now = time.monotonic_ns()
    return now


# Transposition table: word0 = key, word1 = move16 | score16 | eval16 | depth8 | bound2 age6


@njit(cache=False)
def tt_probe(transposition_table, key, ply):
    """Read a matching entry from either bucket slot and convert mate distance to this ply."""
    # Different move sequences can reach the same board (a transposition).
    # Low bits of its Zobrist key choose a pair of slots. Clearing bit 0 aligns
    # to the first slot. Check the full key because other keys share this pair.
    bucket_start = np.int64(key & np.uint64(transposition_table.shape[0] - 1)) & ~np.int64(1)
    for slot in range(bucket_start, bucket_start + 2):
        if transposition_table[slot, 0] == key:
            entry = transposition_table[slot, 1]
            bound = np.int64(entry >> np.uint64(56)) & 3
            if bound == TT_EMPTY:
                continue
            # Shift a field down and mask off its neighbours. For signed scores,
            # subtracting 2^16 when the sign bit is set recovers negative values.
            move = np.int64(entry & np.uint64(0xFFFF))
            score = np.int64((entry >> np.uint64(16)) & np.uint64(0xFFFF))
            if score >= 0x8000:
                score -= 0x10000
            static_score = np.int64((entry >> np.uint64(32)) & np.uint64(0xFFFF))
            if static_score >= 0x8000:
                static_score -= 0x10000
            depth = np.int64((entry >> np.uint64(48)) & np.uint64(0xFF))
            # Convert distance from this position to distance from the current root.
            # Another move sequence may have reached the same board at a different
            # ply. tt_store performs the inverse conversion before saving.
            if score >= MATE_IN_MAX:
                score -= ply
            elif score <= -MATE_IN_MAX:
                score += ply
            return True, move, score, static_score, depth, bound
    return False, 0, 0, 0, 0, 0


@njit(cache=False)
def tt_store(transposition_table, key, move, score, static_score, depth, bound, age, ply):
    """Preserve useful deep entries and pack the new score, depth, bound and age in one word."""
    if score >= MATE_IN_MAX:
        score += ply
    elif score <= -MATE_IN_MAX:
        score -= ply
    if score > 32767:
        score = 32767
    elif score < -32768:
        score = -32768
    if static_score > 32767:
        static_score = 32767
    elif static_score < -32768:
        static_score = -32768
    if depth < 0:
        depth = 0
    bucket_start = np.int64(key & np.uint64(transposition_table.shape[0] - 1)) & ~np.int64(1)
    target = bucket_start
    lowest_priority = 1 << 30
    for slot in range(bucket_start, bucket_start + 2):
        entry = transposition_table[slot, 1]
        stored_bound = np.int64(entry >> np.uint64(56)) & 3
        if stored_bound == TT_EMPTY:
            target = slot
            break
        if transposition_table[slot, 0] == key:
            target = slot
            if move == 0:
                move = np.int64(entry & np.uint64(0xFFFF))
            stored_age = np.int64(entry >> np.uint64(58))
            stored_depth = np.int64((entry >> np.uint64(48)) & np.uint64(0xFF))
            # A shallow quiescence bound could overwrite a much deeper result
            # for this position. Keep the deeper entry while it is still fresh.
            if bound != TT_EXACT and stored_age == age and depth < stored_depth - 3:
                return
            break
        stored_age = np.int64(entry >> np.uint64(58))
        stored_depth = np.int64((entry >> np.uint64(48)) & np.uint64(0xFF))
        priority = stored_depth - 4 * ((age - stored_age) & 63)
        if priority < lowest_priority:
            lowest_priority = priority
            target = slot
    # Shift each field into its assigned bits, then OR the fields together.
    # A score, raw evaluation, move, depth, bound and age fit in one word.
    packed = (
        np.uint64(move & 0xFFFF)
        | (np.uint64(score & 0xFFFF) << np.uint64(16))
        | (np.uint64(static_score & 0xFFFF) << np.uint64(32))
        | (np.uint64(depth & 0xFF) << np.uint64(48))
        | (np.uint64(bound | (age << 2)) << np.uint64(56))
    )
    transposition_table[target, 0] = key
    transposition_table[target, 1] = packed


# Draw detection


@njit(cache=False)
def is_repetition(position_history, position_state, position_keys, root_history_index):
    """Scan same-side positions within the reversible window, separating search cycles from game repetitions."""
    if position_state[board_ops.STATE_HALFMOVE_CLOCK] < 4:
        return False
    history_length = position_state[board_ops.STATE_HISTORY_LENGTH]
    key = position_keys[0]
    oldest_index = history_length - 1 - position_state[board_ops.STATE_HALFMOVE_CLOCK]
    if oldest_index < 0:
        oldest_index = 0
    # The last entry is the current board. Step back two plies at a time to
    # compare positions with the same player to move. A return to the root or
    # a later position closes a search cycle. Threefold repetition in the
    # observed game needs two earlier matches as well as the current board.
    history_index = history_length - 3
    earlier_matches = 0
    while history_index >= oldest_index:
        if position_history[history_index] == key:
            if history_index >= root_history_index:
                return True  # repetition inside the search path
            earlier_matches += 1
            if earlier_matches >= 2:
                return True
        history_index -= 2
    return False


@njit(cache=False)
def has_insufficient_material(bitboards, color):
    """Check whether this side has mating material under the referee’s material rules."""
    base = color * 6
    if (
        bitboards[base + board_ops.PAWN]
        or bitboards[base + board_ops.ROOK]
        or bitboards[base + board_ops.QUEEN]
    ):
        return False
    opponent_offset = 6 - base
    if bitboards[base + board_ops.KNIGHT]:
        if board_ops.count_set_squares(bitboards[board_ops.WHITE_OCCUPANCY + color]) > 2:
            return False
        other_material = (
            bitboards[board_ops.WHITE_OCCUPANCY + 1 - color]
            & ~bitboards[opponent_offset + board_ops.KING]
            & ~bitboards[opponent_offset + board_ops.QUEEN]
        )
        return other_material == 0
    if bitboards[base + board_ops.BISHOP]:
        # python-chess (the referee's rule): ALL bishops on the board must share a
        # square colour, and no pawns or knights of either side may remain.
        light_squares = np.uint64(0x55AA55AA55AA55AA)
        bishops = bitboards[base + board_ops.BISHOP] | bitboards[opponent_offset + board_ops.BISHOP]
        same_color_complex = (bishops & light_squares) == 0 or (bishops & ~light_squares) == 0
        return (
            same_color_complex
            and bitboards[opponent_offset + board_ops.PAWN] == 0
            and bitboards[opponent_offset + board_ops.KNIGHT] == 0
        )
    return True


@njit(cache=False)
def is_material_draw(bitboards):
    """Return a draw by material only when neither player can mate."""
    return has_insufficient_material(bitboards, board_ops.WHITE) and has_insufficient_material(
        bitboards, board_ops.BLACK
    )


# Static exchange evaluation


@njit(cache=False)
def see_ge(bitboards, mailbox, move, threshold):
    """True if the exchange starting with move m is worth at least `threshold`."""
    # Static exchange evaluation (SEE) estimates the material outcome of
    # swapping pieces on one square. A queen taking a defended pawn may lose
    # far more than the pawn is worth. Test the required margin cheaply before
    # spending a recursive search on that candidate.
    # A move packs its origin into bits 0–5, destination into 6–11 and
    # move type into 12–15. Shift the wanted field down, then mask it off.
    from_square = move & 63
    to_square = (move >> 6) & 63
    flag = (move >> 12) & 15
    if (
        flag == board_ops.EN_PASSANT_CAPTURE_FLAG
        or (flag & 8)
        or flag == board_ops.KINGSIDE_CASTLE_FLAG
        or flag == board_ops.QUEENSIDE_CASTLE_FLAG
    ):
        # Castling, en passant and promotions get a neutral estimate here.
        return threshold <= 0
    victim = mailbox[to_square]
    exchange_margin = PIECE_VALUE[victim] - threshold
    if exchange_margin < 0:
        return False
    attacker = mailbox[from_square]
    exchange_margin = PIECE_VALUE[attacker] - exchange_margin
    if exchange_margin <= 0:
        return True
    occupancy = (
        (bitboards[board_ops.WHITE_OCCUPANCY] | bitboards[board_ops.BLACK_OCCUPANCY])
        ^ board_ops.square_mask(from_square)
        ^ board_ops.square_mask(to_square)
    )
    side_to_move = 0 if attacker > 6 else 1  # side to move after the capture
    attackers = board_ops.attackers_of_square(bitboards, to_square, occupancy) & occupancy
    meets_threshold = 1
    while True:
        side_attackers = attackers & bitboards[board_ops.WHITE_OCCUPANCY + side_to_move]
        if side_attackers == 0:
            break
        meets_threshold ^= 1
        base = side_to_move * 6
        # Try the least valuable available recapturer. Removing it can uncover
        # another attacker, such as a bishop behind a pawn. Count only exchanges
        # on this square. A threat elsewhere would need the actual search.
        attacker_type = 0
        for piece_type in range(board_ops.PAWN, board_ops.KING + 1):
            candidates = side_attackers & bitboards[base + piece_type]
            if candidates:
                square = board_ops.lowest_set_square(candidates)
                occupancy ^= board_ops.square_mask(square)
                attacker_type = piece_type
                break
        if attacker_type == board_ops.KING:
            if attackers & bitboards[board_ops.WHITE_OCCUPANCY + 1 - side_to_move] & occupancy:
                meets_threshold ^= 1
            break
        exchange_margin = SEE_VALUE[attacker_type] - exchange_margin
        if exchange_margin < meets_threshold:
            break
        if (
            attacker_type == board_ops.PAWN
            or attacker_type == board_ops.BISHOP
            or attacker_type == board_ops.QUEEN
        ):
            attackers |= board_ops.bishop_attacks(to_square, occupancy) & (
                bitboards[board_ops.WHITE_BISHOP]
                | bitboards[board_ops.BLACK_BISHOP]
                | bitboards[board_ops.WHITE_QUEEN]
                | bitboards[board_ops.BLACK_QUEEN]
            )
        if attacker_type == board_ops.ROOK or attacker_type == board_ops.QUEEN:
            attackers |= board_ops.rook_attacks(to_square, occupancy) & (
                bitboards[board_ops.WHITE_ROOK]
                | bitboards[board_ops.BLACK_ROOK]
                | bitboards[board_ops.WHITE_QUEEN]
                | bitboards[board_ops.BLACK_QUEEN]
            )
        attackers &= occupancy
        side_to_move = 1 - side_to_move
    return meets_threshold != 0


# Move ordering


@njit(cache=False)
def victim_type(mailbox, move):
    """Read the captured piece, accounting for the empty destination of en passant."""
    flag = (move >> 12) & 15
    if flag == board_ops.EN_PASSANT_CAPTURE_FLAG:
        return board_ops.PAWN
    if flag & 4:
        return (mailbox[(move >> 6) & 63] - 1) % 6 + 1
    return 0


@njit(cache=False)
def score_moves(context, moves, scores, move_count, ply, table_move, in_qsearch):
    """Rank the cached move, captures and quiet moves for an earlier cutoff."""
    bitboards = context[C_BITBOARDS]
    mailbox = context[C_MAILBOX]
    position_state = context[C_POSITION]
    killers = context[C_KILLERS]
    quiet_history = context[C_QUIET_HISTORY]
    capture_history = context[C_CAPTURE_HISTORY]
    continuation_history = context[C_CONTINUATION_HISTORY]
    countermoves = context[C_COUNTER]
    search_stack = context[C_SEARCH_STACK]
    side_to_move = position_state[board_ops.STATE_SIDE_TO_MOVE]
    last_piece = search_stack[ply + 3, SS_PIECE]
    last_target = (search_stack[ply + 3, SS_MOVE] >> 6) & 63
    earlier_piece = search_stack[ply + 2, SS_PIECE]
    earlier_target = (search_stack[ply + 2, SS_MOVE] >> 6) & 63
    countermove = countermoves[last_piece, last_target] if last_piece else 0
    # These scores decide which move to examine next. They measure priority,
    # not how good the resulting position is. Try the cached move first, even
    # when the search that found it was too shallow to reuse its score.
    for move_index in range(move_count):
        move = moves[move_index]
        from_square = move & 63
        to_square = (move >> 6) & 63
        flag = (move >> 12) & 15
        if move == table_move:
            scores[move_index] = S_TT
            continue
        piece = mailbox[from_square]
        if flag & 12:  # capture or promotion
            victim_code = victim_type(mailbox, move)
            ordering_score = (
                16 * PIECE_VALUE[victim_code] + capture_history[piece, to_square, victim_code] // 8
            )
            # Within the tactical group, favour valuable victims and queen
            # promotions. Main search sends costly ordinary captures behind quiet
            # moves. Quiescence applies its exchange filters later in its own loop.
            if flag & 8:
                ordering_score += 2400 if (flag & 3) == 3 else -3000
            if in_qsearch or (flag & 8) or see_ge(bitboards, mailbox, move, -PIECE_VALUE[piece] // 12):
                scores[move_index] = S_GOOD_CAPTURE + ordering_score
            else:
                scores[move_index] = S_BAD_CAPTURE + ordering_score
        else:
            # A quiet move neither captures nor promotes. A killer is a quiet
            # move that caused a cutoff at the same ply elsewhere. A countermove
            # is a reply that cut off after this particular piece moved to this
            # square. Quiet history rewards moves that cut off. Continuation history
            # also takes into account the piece moves one and two plies back.
            if move == killers[ply, 0]:
                scores[move_index] = S_KILLER + 2
            elif move == killers[ply, 1]:
                scores[move_index] = S_KILLER + 1
            elif move == countermove:
                scores[move_index] = S_KILLER
            else:
                ordering_score = quiet_history[side_to_move, from_square, to_square]
                if last_piece:
                    ordering_score += continuation_history[last_piece, last_target, piece, to_square]
                if earlier_piece:
                    ordering_score += continuation_history[earlier_piece, earlier_target, piece, to_square]
                scores[move_index] = ordering_score


@njit(cache=False)
def pick_move(moves, scores, next_index, move_count):
    """Swap the best remaining candidate into the next position of the move buffer."""
    # If the first move settles the window, the rest need never be sorted.
    # Select only the best remaining move each time the caller asks for one.
    best_index = next_index
    for candidate_index in range(next_index + 1, move_count):
        if scores[candidate_index] > scores[best_index]:
            best_index = candidate_index
    if best_index != next_index:
        saved_move = moves[next_index]
        moves[next_index] = moves[best_index]
        moves[best_index] = saved_move
        saved_score = scores[next_index]
        scores[next_index] = scores[best_index]
        scores[best_index] = saved_score
    return moves[next_index]


@njit(cache=False)
def hist_update(value, bonus):
    """Apply a bounded bonus or penalty to a move-ordering statistic."""
    updated_value = value + bonus - value * abs(bonus) // HISTORY_MAX
    if updated_value > HISTORY_MAX:
        updated_value = HISTORY_MAX
    elif updated_value < -HISTORY_MAX:
        updated_value = -HISTORY_MAX
    return updated_value


@njit(cache=False)
def stat_bonus(depth):
    """Weight a cutoff by searched depth, capping the largest rewards."""
    bonus = 16 * depth * depth + 32 * depth + 16
    return bonus if bonus < 1200 else 1200


@njit(cache=False)
def update_quiet_stats(context, ply, depth, best, quiets, quiet_count):
    """Reward the quiet cutoff and penalize the quiet alternatives already tried."""
    position_state = context[C_POSITION]
    mailbox = context[C_MAILBOX]
    killers = context[C_KILLERS]
    quiet_history = context[C_QUIET_HISTORY]
    continuation_history = context[C_CONTINUATION_HISTORY]
    countermoves = context[C_COUNTER]
    search_stack = context[C_SEARCH_STACK]
    side_to_move = position_state[board_ops.STATE_SIDE_TO_MOVE]
    bonus = stat_bonus(depth)
    if killers[ply, 0] != best:
        killers[ply, 1] = killers[ply, 0]
        killers[ply, 0] = best
    last_piece = search_stack[ply + 3, SS_PIECE]
    last_target = (search_stack[ply + 3, SS_MOVE] >> 6) & 63
    earlier_piece = search_stack[ply + 2, SS_PIECE]
    earlier_target = (search_stack[ply + 2, SS_MOVE] >> 6) & 63
    if last_piece:
        countermoves[last_piece, last_target] = best
    for move_index in range(quiet_count):
        move = quiets[move_index]
        from_square = move & 63
        to_square = (move >> 6) & 63
        piece = mailbox[from_square]
        adjustment = bonus if move == best else -bonus
        quiet_history[side_to_move, from_square, to_square] = hist_update(
            quiet_history[side_to_move, from_square, to_square], adjustment
        )
        if last_piece:
            continuation_history[last_piece, last_target, piece, to_square] = hist_update(
                continuation_history[last_piece, last_target, piece, to_square], adjustment
            )
        if earlier_piece:
            continuation_history[earlier_piece, earlier_target, piece, to_square] = hist_update(
                continuation_history[earlier_piece, earlier_target, piece, to_square], adjustment
            )


@njit(cache=False)
def check_time(search_info):
    """Poll the node budget and hard deadline periodically and propagate any earlier stop."""
    nodes = search_info[I_NODES]
    # 2047 has its lowest eleven bits set, so this test passes at multiples
    # of 2048 visited nodes. Polling in batches saves clock callbacks. Either
    # limit can be crossed between polls. Once set, the shared stop flag tells
    # every caller to unwind without treating the interrupted score as real.
    if (nodes & 2047) == 0:
        if search_info[I_NODE_LIMIT] > 0 and nodes >= search_info[I_NODE_LIMIT]:
            search_info[I_STOP] = 1
        elif search_info[I_HARD_NS] > 0 and clock_ns() >= search_info[I_HARD_NS]:
            search_info[I_STOP] = 1
    return search_info[I_STOP] != 0


@njit(cache=False)
def pawn_corr_index(bitboards):
    """Hash of both pawn bitboards into the correction table (splitmix-style mixing)."""
    # Mix the two sides' pawn locations into a small table index. Similar
    # evaluation errors can recur with the same pawn structure. Different
    # structures may share a slot, so each entry is only an estimated bias.
    pawn_hash = bitboards[board_ops.PAWN] * np.uint64(0x9E3779B97F4A7C15)
    pawn_hash ^= bitboards[6 + board_ops.PAWN] + np.uint64(0x632BE59BD9B4E019)
    pawn_hash ^= pawn_hash >> np.uint64(31)
    pawn_hash *= np.uint64(0xBF58476D1CE4E5B9)
    pawn_hash ^= pawn_hash >> np.uint64(29)
    return np.int64(pawn_hash & np.uint64(CORR_SIZE - 1))


@njit(cache=False)
def corrected_eval(context, raw_score):
    """Adjust the raw score using the learned pawn-structure error, staying outside mate scores."""
    side_to_move = context[C_POSITION][board_ops.STATE_SIDE_TO_MOVE]
    corrected_score = (
        raw_score
        + context[C_PAWN_CORRECTION][side_to_move, pawn_corr_index(context[C_BITBOARDS])] // CORR_GRAIN
    )
    if corrected_score > EVAL_CLAMP:
        corrected_score = EVAL_CLAMP
    elif corrected_score < -EVAL_CLAMP:
        corrected_score = -EVAL_CLAMP
    return corrected_score


@njit(cache=False)
def update_correction(context, depth, static, best):
    """Move the pawn-structure correction toward (search result - static evaluation)
    with a depth-weighted exponential average; callers exclude checks, tactical best
    moves and bounds that do not support the direction of the correction."""
    pawn_correction = context[C_PAWN_CORRECTION]
    side_to_move = context[C_POSITION][board_ops.STATE_SIDE_TO_MOVE]
    pawn_index = pawn_corr_index(context[C_BITBOARDS])
    target = (best - static) * CORR_GRAIN
    if target > CORR_LIMIT:
        target = CORR_LIMIT
    elif target < -CORR_LIMIT:
        target = -CORR_LIMIT
    weight = depth + 1
    if weight > 16:
        weight = 16
    previous_correction = np.int64(pawn_correction[side_to_move, pawn_index])
    updated_correction = ((256 - weight) * previous_correction + weight * target) // 256
    if updated_correction > CORR_LIMIT:
        updated_correction = CORR_LIMIT
    elif updated_correction < -CORR_LIMIT:
        updated_correction = -CORR_LIMIT
    pawn_correction[side_to_move, pawn_index] = updated_correction


@njit(cache=False)
def qsearch(context, alpha, beta, ply):
    """Quiescence search extends tactical lines past the normal depth limit."""
    # Suppose the planned depth ends just after a queen takes a defended pawn.
    # Scoring here could credit the pawn and miss the loss of the queen on
    # the next move. Quiescence follows ordinary captures, en passant and queen
    # promotions. In check, even a quiet block or king escape must be eligible.
    bitboards = context[C_BITBOARDS]
    mailbox = context[C_MAILBOX]
    position_state = context[C_POSITION]
    position_keys = context[C_POSITION_KEYS]
    position_history = context[C_POSITION_HISTORY]
    undo_stack = context[C_UNDO]
    previous_keys = context[C_PREVIOUS_KEYS]
    transposition_table = context[C_TRANSPOSITIONS]
    search_stack = context[C_SEARCH_STACK]
    search_info = context[C_INFO]
    search_info[I_NODES] += 1
    search_info[I_QNODES] += 1
    if ply > search_info[I_SELDEPTH]:
        search_info[I_SELDEPTH] = ply
    if check_time(search_info):
        return 0
    in_check = board_ops.in_check(bitboards, position_state)
    if (
        is_repetition(position_history, position_state, position_keys, search_info[I_ROOT_HIST])
        or position_state[board_ops.STATE_HALFMOVE_CLOCK] >= 100
        or position_state[board_ops.STATE_GAME_PLY] >= 600
        or is_material_draw(bitboards)
    ):
        # The referee resolves mate before draws, including on the last permitted ply.
        if in_check:
            evasions = context[C_MOVE_STACK][ply]
            if (
                board_ops.generate_legal_moves(
                    bitboards, mailbox, position_state, evasions, board_ops.GENERATE_ALL_MOVES
                )
                == 0
            ):
                return -MATE + ply
        return DRAW
    # Each call needs another row of move and undo storage. Even a long
    # sequence of checks or captures must stop before those rows run out.
    if ply >= board_ops.MAX_SEARCH_PLY - 1:
        return static_eval(context, ply)
    key = position_keys[0]
    table_hit, table_move, table_score, _, _, table_bound = tt_probe(transposition_table, key, ply)
    if table_hit:
        if (
            table_bound == TT_EXACT
            or (table_bound == TT_LOWER and table_score >= beta)
            or (table_bound == TT_UPPER and table_score <= alpha)
        ):
            return table_score
    best_move = 0
    if in_check:
        touch_eval(context, ply)
        static_score = -INF
        best_score = -INF
    else:
        # The stand-pat score uses the current evaluation as a baseline, on the
        # assumption that the mover can preserve it without another exchange.
        # In check the player must escape, so the branch above starts at -INF.
        # A cached bound can refine the baseline outside check.
        static_score = static_eval(context, ply)
        best_score = static_score
        if table_hit and (
            (table_bound == TT_LOWER and table_score > static_score)
            or (table_bound == TT_UPPER and table_score < static_score)
        ):
            best_score = table_score
        if best_score >= beta:
            if not table_hit:
                tt_store(
                    transposition_table,
                    key,
                    0,
                    best_score,
                    static_score,
                    0,
                    TT_LOWER,
                    search_info[I_TT_AGE],
                    ply,
                )
            return best_score
        if best_score > alpha:
            alpha = best_score
    moves = context[C_MOVE_STACK][ply]
    scores = context[C_SCORE_STACK][ply]
    move_count = board_ops.generate_legal_moves(
        bitboards,
        mailbox,
        position_state,
        moves,
        board_ops.GENERATE_ALL_MOVES if in_check else board_ops.GENERATE_TACTICAL_MOVES,
    )
    if in_check and move_count == 0:
        return -MATE + ply
    score_moves(context, moves, scores, move_count, ply, table_move, True)
    futility = static_score + 150
    for move_index in range(move_count):
        move = pick_move(moves, scores, move_index, move_count)
        flag = (move >> 12) & 15
        # Away from check, skip an ordinary capture when its victim plus a
        # margin cannot raise alpha and the exchange looks unprofitable. The
        # second test also drops moves likely to lose more than 50 score units.
        # Check evasions bypass both filters.
        if not in_check and best_score > -MATE_IN_MAX:
            if (flag & 4) and not (flag & 8):
                victim_code = victim_type(mailbox, move)
                gain = PIECE_VALUE[victim_code]
                if futility + gain <= alpha and not see_ge(bitboards, mailbox, move, 1):
                    if best_score < futility + gain:
                        best_score = futility + gain
                    continue
            if not see_ge(bitboards, mailbox, move, -50):
                continue
        search_stack[ply + 4, SS_MOVE] = move
        search_stack[ply + 4, SS_PIECE] = mailbox[move & 63]
        board_ops.make_move(
            bitboards,
            mailbox,
            position_state,
            position_keys,
            position_history,
            undo_stack,
            previous_keys,
            ply,
            move,
        )
        score = -qsearch(context, -beta, -alpha, ply + 1)
        board_ops.unmake_move(
            bitboards,
            mailbox,
            position_state,
            position_keys,
            position_history,
            undo_stack,
            previous_keys,
            ply,
        )
        if search_info[I_STOP]:
            return 0
        if score > best_score:
            best_score = score
            if score > alpha:
                best_move = move
                if score >= beta:
                    break
                alpha = score
    if in_check and best_score == -INF:
        best_score = -MATE + ply
    # Quiescence stores a lower bound on a beta cutoff and an upper bound
    # otherwise. These entries have depth zero.
    bound = TT_LOWER if best_score >= beta else TT_UPPER
    tt_store(
        transposition_table, key, best_move, best_score, static_score, 0, bound, search_info[I_TT_AGE], ply
    )
    return best_score


@njit(cache=False)
def alphabeta(context, depth, alpha, beta, ply, is_pv, cut_node):
    """Return a fail-soft score or bound from the side-to-move perspective.

    Use integer PV and cut-node flags to keep one recursive signature.
    """
    # One call represents one position (a node). ply counts turns already
    # played from the root. depth counts turns still budgeted on this branch.
    # Minimax assumes each player chooses the reply best for them. Negamax
    # expresses that with one maximising function: a child's +30 is our -30.
    # Negate its score on return and pass the window in reverse, negated order.

    bitboards = context[C_BITBOARDS]
    mailbox = context[C_MAILBOX]
    position_state = context[C_POSITION]
    position_keys = context[C_POSITION_KEYS]
    position_history = context[C_POSITION_HISTORY]
    undo_stack = context[C_UNDO]
    previous_keys = context[C_PREVIOUS_KEYS]
    transposition_table = context[C_TRANSPOSITIONS]
    search_stack = context[C_SEARCH_STACK]
    search_info = context[C_INFO]
    # In alpha-beta pruning, alpha is the score to beat and beta is the cutoff. Suppose White
    # already has +30 from move A. If one Black reply holds move B to +10,
    # White can keep A. From Black's side that reply scores -10, which clears
    # the cutoff of -30. Other Black replies cannot rescue B for White.
    # Fail-soft means we return the score we found, even outside the window.
    if depth <= 0:
        return qsearch(context, alpha, beta, ply)
    search_info[I_NODES] += 1
    if ply > search_info[I_SELDEPTH]:
        search_info[I_SELDEPTH] = ply
    root = ply == 0
    if check_time(search_info):
        return 0
    # Inside the tree a draw estimate can end a branch. At the root the
    # agent still needs a move to play. A zero from an interrupted search is
    # distinguished from a draw by the shared stop flag.
    if not root:
        if (
            is_repetition(position_history, position_state, position_keys, search_info[I_ROOT_HIST])
            or position_state[board_ops.STATE_HALFMOVE_CLOCK] >= 100
            or position_state[board_ops.STATE_GAME_PLY] >= 600
            or is_material_draw(bitboards)
        ):
            # Referee order: checkmate precedes every draw claim.
            if board_ops.in_check(bitboards, position_state):
                evasions = context[C_MOVE_STACK][ply]
                if (
                    board_ops.generate_legal_moves(
                        bitboards, mailbox, position_state, evasions, board_ops.GENERATE_ALL_MOVES
                    )
                    == 0
                ):
                    return -MATE + ply
            return DRAW
        if ply >= board_ops.MAX_SEARCH_PLY - 1:
            return static_eval(context, ply)
        # Mate-distance pruning tightens the window to reachable mate scores.
        # Even the fastest possible mate from this ply cannot exceed these bounds.
        lowest_mate_score = -MATE + ply
        if lowest_mate_score > alpha:
            alpha = lowest_mate_score
        highest_mate_score = MATE - ply - 1
        if highest_mate_score < beta:
            beta = highest_mate_score
        if alpha >= beta:
            return alpha
    key = position_keys[0]
    excluded_move = search_stack[ply + 4, SS_EXCL]
    table_hit, table_move, table_score, table_static, table_depth, table_bound = tt_probe(
        transposition_table, key, ply
    )
    # The principal variation (PV) is the sequence of best replies we are
    # trying to establish. Off that line, a deep enough entry can answer the
    # window directly: an exact score, a lower bound at beta or above, or an
    # upper bound at alpha or below. An exclusion search has a different set
    # of moves, so the ordinary board's cached score cannot settle it.
    if not is_pv and table_hit and table_depth >= depth and excluded_move == 0:
        if (
            table_bound == TT_EXACT
            or (table_bound == TT_LOWER and table_score >= beta)
            or (table_bound == TT_UPPER and table_score <= alpha)
        ):
            return table_score
    in_check = board_ops.in_check(bitboards, position_state)
    search_stack[ply + 4, SS_NULL] = 0
    raw_static = -INF
    if in_check:
        touch_eval(context, ply)
        static_score = -INF
        improving = False
    else:
        if table_hit and table_static != -INF:
            # The stored raw evaluation is model-identical; keep the accumulator
            # chain valid for the children without paying a full forward pass.
            touch_eval(context, ply)
            raw_static = table_static
        else:
            raw_static = static_eval(context, ply)
        static_score = corrected_eval(context, raw_static)
        # Two plies back it was this player's turn too. Compare scores from
        # the same perspective to decide how aggressively to prune.
        static_two_plies_ago = search_stack[ply + 2, SS_EVAL]
        improving = static_score > static_two_plies_ago if static_two_plies_ago != -INF else True
    search_stack[ply + 4, SS_EVAL] = static_score
    context[C_KILLERS][ply + 2, 0] = 0
    context[C_KILLERS][ply + 2, 1] = 0
    # Static evaluations can miss tactics. Reserve these shortcuts for
    # non-PV nodes outside check and outside the range of mate scores.
    if not is_pv and not in_check and beta < MATE_IN_MAX and beta > -MATE_IN_MAX:
        # Reverse futility pruning: if the evaluation still clears beta after subtracting
        # a depth-dependent safety margin, assume we already have enough.
        if depth <= 8 and static_score - (80 * depth - (50 if improving else 0)) >= beta:
            return static_score
        # Razoring: when a shallow position looks far below alpha, give its
        # tactics a chance in qsearch. A recovery there keeps the main search alive.
        if depth <= 3 and static_score + 250 * depth <= alpha:
            probe_score = qsearch(context, alpha, beta, ply)
            if probe_score <= alpha:
                return probe_score
        # Null-move pruning tries a hypothetical pass. If giving the opponent a free turn
        # still clears beta at reduced depth, a real move is likely good enough.
        # This can fail in zugzwang, where having to move hurts. Require a piece
        # other than a pawn or king, and disallow consecutive passes.
        side_to_move = position_state[board_ops.STATE_SIDE_TO_MOVE]
        base = side_to_move * 6
        non_pawn = (
            bitboards[base + board_ops.KNIGHT]
            | bitboards[base + board_ops.BISHOP]
            | bitboards[base + board_ops.ROOK]
            | bitboards[base + board_ops.QUEEN]
        )
        if (
            depth >= 3
            and excluded_move == 0
            and static_score >= beta
            and search_stack[ply + 3, SS_NULL] == 0
            and non_pawn != 0
            and ply >= search_info[I_NULL_MIN_PLY]
            and (not table_hit or table_bound != TT_UPPER or table_score >= beta)
        ):
            reduction = 3 + depth // 3 + min((static_score - beta) // 200, 3)
            search_stack[ply + 4, SS_NULL] = 1
            search_stack[ply + 4, SS_MOVE] = 0
            search_stack[ply + 4, SS_PIECE] = 0
            board_ops.make_null_move(
                position_state, position_keys, position_history, undo_stack, previous_keys, ply
            )
            score = -alphabeta(context, depth - reduction, -beta, -beta + 1, ply + 1, ZERO, 1 - cut_node)
            board_ops.unmake_null_move(
                position_state, position_keys, position_history, undo_stack, previous_keys, ply
            )
            search_stack[ply + 4, SS_NULL] = 0
            if search_info[I_STOP]:
                return 0
            if score >= beta:
                if score >= MATE_IN_MAX:
                    score = beta
                if depth < 14:
                    return score
                # At depth 14 or more, verify from the original board with actual
                # moves. Temporarily suppress more passes over the verification span
                # so the same assumption cannot certify itself.
                previous_null_limit = search_info[I_NULL_MIN_PLY]
                search_info[I_NULL_MIN_PLY] = ply + max(2, 3 * max(1, depth - reduction) // 4)
                probe_score = alphabeta(context, depth - reduction, beta - 1, beta, ply, ZERO, ZERO)
                search_info[I_NULL_MIN_PLY] = previous_null_limit
                if search_info[I_STOP]:
                    return 0
                if probe_score >= beta:
                    return score
    # Singular extension asks whether the cached move has a credible alternative.
    # Temporarily exclude it and search the same board at about half depth. If
    # the alternatives all fall well short of its cached score, spend one extra
    # ply on that move. Do this before filling this ply's move buffer because
    # the exclusion search uses the same row.
    singular = 0
    if (
        not root
        and excluded_move == 0
        and depth >= 8
        and table_hit
        and table_move != 0
        and table_bound != TT_UPPER
        and table_depth >= depth - 3
        and abs(table_score) < MATE_IN_MAX
        and ply < 2 * search_info[I_ROOT_DEPTH]
    ):
        singular_threshold = table_score - 2 * depth
        search_stack[ply + 4, SS_EXCL] = table_move
        probe_score = alphabeta(
            context, (depth - 1) // 2, singular_threshold - 1, singular_threshold, ply, ZERO, cut_node
        )
        search_stack[ply + 4, SS_EXCL] = 0
        if search_info[I_STOP]:
            return 0
        if probe_score < singular_threshold:
            singular = 1
    # Internal iterative reduction (IIR) saves a ply at PV or expected-cutoff
    # nodes when there is no cached move to guide the search.
    if depth >= 4 and table_move == 0 and (is_pv or cut_node):
        depth -= 1
    moves = context[C_MOVE_STACK][ply]
    scores = context[C_SCORE_STACK][ply]
    quiet_list = context[C_QUIET_STACK][ply]
    move_count = board_ops.generate_legal_moves(
        bitboards, mailbox, position_state, moves, board_ops.GENERATE_ALL_MOVES
    )
    # When an ordinary position has no legal moves, check means mate and no
    # check means stalemate. The mate score includes its distance from the root.
    if move_count == 0:
        if excluded_move != 0:
            return alpha
        return -MATE + ply if in_check else DRAW
    score_moves(context, moves, scores, move_count, ply, table_move, False)
    best_score = -INF
    best_move = 0
    moves_seen = 0
    quiet_count = 0
    quiet_move_limit = (3 + depth * depth) // (2 - (1 if improving else 0))
    for move_index in range(move_count):
        move = pick_move(moves, scores, move_index, move_count)
        if move == excluded_move:
            continue
        ordering_score = scores[move_index]
        flag = (move >> 12) & 15
        from_square = move & 63
        to_square = (move >> 6) & 63
        piece = mailbox[from_square]
        is_quiet = (flag & 12) == 0
        moves_seen += 1
        late_move_reduction = LMR_TABLE[min(depth, 63), min(moves_seen, 63)] if moves_seen > 1 else 0
        reduced_depth = depth - 1 - late_move_reduction
        # Once a candidate scores above the losing-mate range, we can afford
        # to skip unlikely alternatives. The root and check evasions stay exempt.
        if not root and best_score > -MATE_IN_MAX and not in_check:
            if is_quiet:
                # Late move pruning (LMP) skips further quiets once enough were tried.
                if quiet_count >= quiet_move_limit:
                    continue
                # Futility pruning skips a quiet whose estimated gain still falls short.
                if reduced_depth <= 8 and static_score + 100 + 120 * reduced_depth <= alpha:
                    continue
                if reduced_depth <= 6:
                    exchange_depth = reduced_depth if reduced_depth > 1 else 1
                    if not see_ge(bitboards, mailbox, move, -30 * exchange_depth * exchange_depth):
                        continue
            elif (
                depth <= 6
                and ordering_score < S_KILLER
                and not see_ge(bitboards, mailbox, move, -100 * depth)
            ):
                continue
        search_stack[ply + 4, SS_MOVE] = move
        search_stack[ply + 4, SS_PIECE] = piece
        # Play the move on the shared board. Its undo record at this ply lets
        # us restore the parent before comparing the next candidate.
        nodes_before = search_info[I_NODES]
        board_ops.make_move(
            bitboards,
            mailbox,
            position_state,
            position_keys,
            position_history,
            undo_stack,
            previous_keys,
            ply,
            move,
        )
        # A check can force a reply worth seeing beyond the horizon. A move
        # singled out by the exclusion test also gets an extra ply. Both reasons
        # together still add just one.
        gives_check = board_ops.in_check(bitboards, position_state)
        extension = 1 if (gives_check and depth < 12 and ply < 2 * search_info[I_ROOT_DEPTH]) else 0
        if singular and move == table_move:
            extension = 1
        new_depth = depth - 1 + extension
        score = -INF
        # Try late quiet moves and poorly ranked captures at reduced depth.
        # Reduce further when we expect a cutoff. If the move still beats
        # alpha, try it again at the intended depth.
        search_full_depth = True
        if depth >= 3 and moves_seen > (3 if is_pv else 2) and (is_quiet or ordering_score < 0):
            reduction = late_move_reduction
            if not improving:
                reduction += 1
            if cut_node:
                reduction += 1
            if is_pv:
                reduction -= 1
            if ordering_score >= S_KILLER:
                reduction -= 1
            elif is_quiet:
                reduction -= ordering_score // 8192
            if reduction < 0:
                reduction = 0
            if reduction > new_depth - 1:
                reduction = new_depth - 1
            if reduction > 0:
                score = -alphabeta(context, new_depth - reduction, -alpha - 1, -alpha, ply + 1, ZERO, ONE)
                search_full_depth = score > alpha
        # Principal variation search (PVS) gives later moves a null-window probe.
        # With integer scores [alpha, alpha + 1] asks "can this beat alpha?"
        # The opponent sees [-alpha - 1, -alpha]. The first PV move gets the full
        # window. A later candidate strictly between alpha and beta gets a wide
        # re-search to find its score.
        if search_full_depth and (not is_pv or moves_seen > 1):
            score = -alphabeta(context, new_depth, -alpha - 1, -alpha, ply + 1, ZERO, 1 - cut_node)
        if is_pv and (moves_seen == 1 or (score > alpha and score < beta)):
            score = -alphabeta(context, new_depth, -beta, -alpha, ply + 1, ONE, ZERO)
        board_ops.unmake_move(
            bitboards,
            mailbox,
            position_state,
            position_keys,
            position_history,
            undo_stack,
            previous_keys,
            ply,
        )
        # A stop can arrive several calls below us. Restore the parent first,
        # then propagate the flag. The zero on this path is a placeholder.
        if search_info[I_STOP]:
            return 0
        if is_quiet:
            quiet_list[quiet_count] = move
            quiet_count += 1
        if score > best_score:
            best_score = score
            if score > alpha:
                best_move = move
                # Record a root hint only after this candidate's search has returned.
                # It remains available if time runs out during the next candidate.
                if root:
                    search_info[I_ROOT_BEST] = move
                    search_info[I_ROOT_SCORE] = score
                    search_info[I_BEST_NODES] = search_info[I_NODES] - nodes_before
                # Reaching beta stops this node. Reward the cutoff move so similar
                # positions try it earlier. Otherwise raise alpha and make the
                # remaining candidates beat the new best score.
                if score >= beta:
                    if is_quiet:
                        update_quiet_stats(context, ply, depth, move, quiet_list, quiet_count)
                    else:
                        victim_code = victim_type(mailbox, move)
                        capture_history = context[C_CAPTURE_HISTORY]
                        capture_history[piece, to_square, victim_code] = hist_update(
                            capture_history[piece, to_square, victim_code], stat_bonus(depth)
                        )
                    break
                alpha = score
    # Commit only results from the ordinary search to tables used by future branches.
    if excluded_move != 0:
        # Exclusion search: no TT store, no correction update; "no alternative
        # move" means the bound, not a terminal result.
        return alpha if best_score == -INF else best_score
    # A beta cutoff establishes a lower bound. A PV improvement inside the
    # window supplies the searched score at this depth. If no move improved
    # alpha, an upper bound is enough. Later probes need this distinction.
    if best_score >= beta:
        bound = TT_LOWER
    elif is_pv and best_move != 0:
        bound = TT_EXACT
    else:
        bound = TT_UPPER
    # For a quiet result, adjust the static evaluation only when the bound
    # supports the direction of the adjustment. A lower bound below our
    # estimate, for example, gives no evidence that we overestimated.
    if (
        not in_check
        and abs(best_score) < MATE_IN_MAX
        and (best_move == 0 or ((best_move >> 12) & 12) == 0)
        and not (bound == TT_LOWER and best_score <= static_score)
        and not (bound == TT_UPPER and best_score >= static_score)
    ):
        update_correction(context, depth, static_score, best_score)
    tt_store(
        transposition_table, key, best_move, best_score, raw_static, depth, bound, search_info[I_TT_AGE], ply
    )
    return best_score


@njit(cache=False)
def search_iteration(context, depth, alpha, beta):
    """Clear per-ply search metadata and search the root at one depth and score window."""
    search_info = context[C_INFO]
    search_stack = context[C_SEARCH_STACK]
    # Start this attempt at the actual board (ply zero) with an empty path.
    # Keep the cache and ordering histories so earlier depths can help.
    for stack_index in range(search_stack.shape[0]):
        search_stack[stack_index, SS_EVAL] = -INF
        search_stack[stack_index, SS_MOVE] = 0
        search_stack[stack_index, SS_PIECE] = 0
        search_stack[stack_index, SS_NULL] = 0
        search_stack[stack_index, SS_EXCL] = 0
    search_info[I_ROOT_DEPTH] = depth
    search_info[I_NULL_MIN_PLY] = 0
    return alphabeta(context, depth, alpha, beta, ZERO, ONE, ZERO)


def _compile_exact(kernel, signature):
    """Compile one canonical signature and freeze the dispatcher.

    Callers then resolve literal-typed arguments (e.g. `is_square_attacked_by(bitboards, 5, ...)`)
    to this single specialization instead of compiling one variant per literal.
    """
    if signature not in kernel.signatures:
        kernel.compile(signature)
    kernel.disable_compile()


def compile_all(context):
    """Compile the kernel signatures in dependency order and return their count for the startup log."""
    from numba import boolean, int64, uint64, int8, typeof, types

    u64_1 = types.Array(uint64, 1, "C")
    i8_1 = types.Array(int8, 1, "C")
    i64_1 = types.Array(int64, 1, "C")
    i64_2 = types.Array(int64, 2, "C")
    u64_2 = types.Array(uint64, 2, "C")
    i16_2 = types.Array(types.int16, 2, "C")
    i16_3 = types.Array(types.int16, 3, "C")
    context_type = typeof(context)
    network_type = typeof(context[C_NETWORK])
    plan = [
        (board_ops.square_mask, (int64,)),
        (board_ops.lowest_set_square, (uint64,)),
        (board_ops.count_set_squares, (uint64,)),
        (board_ops.rook_attacks, (int64, uint64)),
        (board_ops.bishop_attacks, (int64, uint64)),
        (board_ops.queen_attacks, (int64, uint64)),
        (board_ops.attackers_of_square, (u64_1, int64, uint64)),
        (board_ops.is_square_attacked_by, (u64_1, int64, int64, uint64)),
        (board_ops.king_square, (u64_1, int64)),
        (board_ops.in_check, (u64_1, i64_1)),
        (board_ops.encode_move, (int64, int64, int64)),
        (board_ops._add_piece, (u64_1, i8_1, i64_1, u64_1, int64, int64)),
        (board_ops._remove_piece, (u64_1, i8_1, i64_1, u64_1, int64)),
        (board_ops._move_piece, (u64_1, i8_1, i64_1, u64_1, int64, int64)),
        (board_ops.has_legal_en_passant_capture, (u64_1, int64, int64)),
        (board_ops.make_move, (u64_1, i8_1, i64_1, u64_1, u64_1, i64_2, u64_1, int64, int64)),
        (board_ops.unmake_move, (u64_1, i8_1, i64_1, u64_1, u64_1, i64_2, u64_1, int64)),
        (board_ops.make_null_move, (i64_1, u64_1, u64_1, i64_2, u64_1, int64)),
        (board_ops.unmake_null_move, (i64_1, u64_1, u64_1, i64_2, u64_1, int64)),
        (board_ops._append_encoded_move, (i64_1, int64, int64, int64, int64)),
        (board_ops.generate_legal_moves, (u64_1, i8_1, i64_1, i64_1, int64)),
        (board_ops.compute_position_key, (u64_1, i8_1, i64_1)),
        (evaluation.evaluate_psqt, (i64_1,)),
        (evaluation.feature_index, (int64, int64, int64, int64)),
        (evaluation.refresh_perspective, (network_type, u64_1, i8_1, int64, int64)),
        (evaluation._add_feature, (i16_3, i16_2, int64, int64, int64)),
        (evaluation._sub_feature, (i16_3, i16_2, int64, int64, int64)),
        (evaluation.update_perspective, (network_type, u64_1, i8_1, i64_1, i64_2, int64, int64)),
        (evaluation.ensure_accumulators, (network_type, u64_1, i8_1, i64_1, u64_1, i64_2, u64_1, int64)),
        (evaluation.output_bucket, (u64_1,)),
        (evaluation.forward, (network_type, u64_1, i64_1, int64)),
        (evaluation.evaluate_nn, (network_type, u64_1, i8_1, i64_1, u64_1, i64_2, u64_1, int64)),
        (static_eval, (context_type, int64)),
        (touch_eval, (context_type, int64)),
        (clock_ns, ()),
        (tt_probe, (u64_2, uint64, int64)),
        (tt_store, (u64_2, uint64, int64, int64, int64, int64, int64, int64, int64)),
        (is_repetition, (u64_1, i64_1, u64_1, int64)),
        (has_insufficient_material, (u64_1, int64)),
        (is_material_draw, (u64_1,)),
        (see_ge, (u64_1, i8_1, int64, int64)),
        (victim_type, (i8_1, int64)),
        (pick_move, (i64_1, i64_1, int64, int64)),
        (hist_update, (int64, int64)),
        (stat_bonus, (int64,)),
        (check_time, (i64_1,)),
        (score_moves, (context_type, i64_1, i64_1, int64, int64, int64, boolean)),
        (update_quiet_stats, (context_type, int64, int64, int64, i64_1, int64)),
        (pawn_corr_index, (u64_1,)),
        (corrected_eval, (context_type, int64)),
        (update_correction, (context_type, int64, int64, int64)),
        (qsearch, (int64(context_type, int64, int64, int64),)),
        (alphabeta, (int64(context_type, int64, int64, int64, int64, int64, int64),)),
        (search_iteration, (context_type, int64, int64, int64)),
    ]
    from numba.core.typing.templates import Signature

    # The recursive kernels specify their return type as well as their inputs
    # so calls back into the same function resolve to one compiled signature.
    for kernel, kernel_types in plan:
        if len(kernel_types) == 1 and isinstance(kernel_types[0], Signature):
            signature = kernel_types[0]
            if not kernel.signatures:
                kernel.compile(signature)
            kernel.disable_compile()
        else:
            _compile_exact(kernel, kernel_types)
    return len(plan)


class SearchResult:
    """Hold the best completed root decision and diagnostics exposed to the adapter."""

    __slots__ = (
        "move",
        "score",
        "depth",
        "seldepth",
        "nodes",
        "qnodes",
        "aborted",
        "iterations",
        "partial",
    )

    def __init__(self):
        self.move = 0
        self.score = 0
        self.depth = 0
        self.seldepth = 0
        self.nodes = 0
        self.qnodes = 0
        self.aborted = False
        self.partial = False
        self.iterations = []


def search(context, soft_ns=0, hard_ns=0, max_depth=64, node_limit=0):
    """Search progressively deeper until the move budget runs out.

    soft_ns is a duration checked between iterations. hard_ns is an absolute
    monotonic deadline polled inside the tree. Zero disables the corresponding limit.
    """
    search_info = context[C_INFO]
    position_state = context[C_POSITION]
    started_ns = time.monotonic_ns()
    stable_iterations = 0
    previous_best_move = 0
    iteration_started_ns = started_ns
    # The counters and stop flag belong to this move. Cached positions survive
    # across turns. Their six-bit age wraps after 64 searches and helps decide
    # which old entry to replace when different keys share a bucket.
    search_info[I_NODES] = 0
    search_info[I_QNODES] = 0
    search_info[I_STOP] = 0
    search_info[I_SELDEPTH] = 0
    search_info[I_HARD_NS] = hard_ns
    search_info[I_NODE_LIMIT] = node_limit
    search_info[I_ROOT_HIST] = position_state[board_ops.STATE_HISTORY_LENGTH] - 1
    search_info[I_TT_AGE] = (search_info[I_TT_AGE] + 1) & 63
    context[C_KILLERS][:] = 0
    result = SearchResult()
    previous_score = 0
    # Iterative deepening searches the same root with one ply, then two and
    # so on. Each completed depth leaves a move we can play if time runs out.
    # Its table entry also gives the next iteration a promising first move.
    # max_depth is the ceiling even when the caller supplies no time limits.
    for depth in range(1, max_depth + 1):
        search_info[I_ROOT_BEST] = 0
        search_info[I_ROOT_SCORE] = -INF
        search_info[I_BEST_NODES] = 0
        nodes_at_iter_start = int(search_info[I_NODES])
        # From depth five, start near the previous score (an aspiration window).
        # For example, +40 gives [22, 58]. A return of +65 clears the ceiling, so
        # retry this depth with a higher ceiling before accepting the result.
        window_margin = 18
        if depth >= 5:
            alpha = max(previous_score - window_margin, -INF)
            beta = min(previous_score + window_margin, INF)
        else:
            alpha, beta = -INF, INF
        while True:
            score = int(search_iteration(context, depth, alpha, beta))
            if search_info[I_STOP]:
                break
            if score <= alpha:
                # Fail-low: the search returned an upper bound. Retry with a lower floor.
                beta = (alpha + beta) // 2
                alpha = max(score - window_margin, -INF)
                window_margin += window_margin // 2 + 5
            elif score >= beta:
                # Fail-high: the search returned a lower bound. Raise the ceiling and retry.
                beta = min(score + window_margin, INF)
                window_margin += window_margin // 2 + 5
            else:
                break
        now_ns = time.monotonic_ns()
        if search_info[I_STOP]:
            result.aborted = True
            # A partial iteration may have left stronger root moves unsearched.
            # Use its hint only when no full iteration has finished. If even
            # the first root candidate is unfinished, move stays zero and
            # get_move returns the legal fallback it selected before search.
            if result.move == 0 and search_info[I_ROOT_BEST] != 0 and search_info[I_ROOT_SCORE] > -INF:
                result.move = int(search_info[I_ROOT_BEST])
                result.score = int(search_info[I_ROOT_SCORE])
                result.depth = depth
                result.partial = True
            break
        # Publish a completed in-window result before making any decision to
        # deepen again. From depth six a mate-range score ends the search early.
        result.move = int(search_info[I_ROOT_BEST])
        result.score = score
        result.depth = depth
        result.seldepth = int(search_info[I_SELDEPTH])
        result.iterations.append(
            (depth, score, int(search_info[I_NODES]), (now_ns - started_ns) // 1_000_000, result.move)
        )
        previous_score = score
        if abs(score) >= MATE_IN_MAX and depth >= 6:
            break
        if result.move == previous_best_move:
            stable_iterations += 1
        else:
            stable_iterations = 0
            previous_best_move = result.move
        iteration_ns = now_ns - iteration_started_ns
        iteration_started_ns = now_ns
        elapsed_ns = now_ns - started_ns
        # A newly changed best move earns extra time from depth six. Repeated
        # agreement across depths earns less. Both signals adjust the soft budget.
        if soft_ns:
            if stable_iterations == 0 and depth >= 6:
                scale = 1.35
            elif stable_iterations >= 6:
                scale = 0.7
            elif stable_iterations >= 3:
                scale = 0.85
            else:
                scale = 1.0
            # From depth eight, also look at the fraction of this iteration's nodes
            # spent under the chosen root move. Concentrated effort lowers the time
            # allowance. These are estimates of how useful another depth might be.
            iteration_nodes = int(search_info[I_NODES]) - nodes_at_iter_start
            if depth >= 8 and iteration_nodes > 0:
                best_move_fraction = min(1.0, int(search_info[I_BEST_NODES]) / iteration_nodes)
                scale *= max(0.5, min(1.5, 2.0 * (1.0 - best_move_fraction) + 0.4))
            # Check the soft budget between completed depths. From depth eight
            # also forecast the next cost as 1.5 times the last iteration's time.
            # If that forecast is wrong, check_time still polls inside the tree.
            if elapsed_ns >= soft_ns * scale:
                break
            if hard_ns and depth >= 8 and now_ns + iteration_ns * 3 // 2 >= hard_ns:
                break
    # Counts include retries and any interrupted work. The move comes from
    # the saved result, which may belong to an earlier depth.
    result.nodes = int(search_info[I_NODES])
    result.qnodes = int(search_info[I_QNODES])
    return result
