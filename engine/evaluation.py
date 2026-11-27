"""NNUE (efficiently updatable neural network) and the tapered piece-square fallback.

  features   per perspective: 16 king buckets x 12 piece codes x 64 squares = 12288
             own-king-relative: ranks flipped for Black, files mirrored so the king
             sits on files a-d; piece code = type + 6 * (piece colour != perspective)
  layer 1    one int16 accumulator per side, 1,024 channels in the London v9 checkpoint
  activation squared clipped ReLU (SCReLU): clamp(acc, 0, QA)^2
  head       eight buckets by piece count, each with one 2,048-input affine head in v9
  output     centipawns from the side to move's view:
             (sum screlu(acc) * W1 + b1) * SCALE / (QA * QA * QB)

Quantization contract shared with the training export:
  W0: int16 = round(w * QA)     b0: int16 = round(b * QA)
  W1: int16 = round(w * QB)     b1: int32 = round(b * QA * QA * QB)
  The accumulator has at most 32 active rows plus its bias. Training must
  bound their sum to fit int16. The head accumulates in int64: each squared
  activation can reach 65,025 before multiplication by a signed int16 weight.

An accumulator belongs to a position key at a particular ply. A matching
parent key lets the evaluator apply the changes in its undo record. If the
parent is unavailable, it rebuilds both sums from the child board.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit

from engine import board as board_ops

PLACEHOLDER_WIDTH = 512  # Width of the zero-weight network in classical mode. The v9 checkpoint supplies 1,024.
KING_BUCKETS = 16
PIECE_CODES = 12
FEATURES = KING_BUCKETS * PIECE_CODES * 64
OUT_BUCKETS = 8
QA = 255
QB = 64
SCALE = 400
SCHEMA = "replacement-nnue-v2"

# Mirrored half-board (files a-d after mirroring) -> bucket. Finer near the back rank.
_BUCKET_ROWS = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 8, 9, 9),
    (10, 10, 11, 11),
    (12, 12, 13, 13),
    (12, 12, 13, 13),
    (14, 14, 15, 15),
    (14, 14, 15, 15),
)
KING_BUCKET = np.zeros(64, dtype=np.int64)
for rank_index in range(8):
    for file_index in range(8):
        KING_BUCKET[rank_index * 8 + file_index] = _BUCKET_ROWS[rank_index][
            file_index if file_index < 4 else 7 - file_index
        ]
KING_BUCKET.setflags(write=False)


def zero_network(hidden_width=PLACEHOLDER_WIDTH):
    """Untrained placeholder arrays with the exact runtime dtypes/shapes:
    (W0 int16 [FEATURES, hidden], b0 int16 [hidden], W1 int16 [OUT_BUCKETS, 2*hidden],
    b1 int32 [OUT_BUCKETS]). The runtime derives the width from W0, so trained
    networks of other widths (multiples of 64) load without code changes."""
    return (
        np.zeros((FEATURES, hidden_width), dtype=np.int16),
        np.zeros(hidden_width, dtype=np.int16),
        np.zeros((OUT_BUCKETS, 2 * hidden_width), dtype=np.int16),
        np.zeros(OUT_BUCKETS, dtype=np.int32),
    )


# Slots in the compiled network tuple, followed by the two mutable caches.
N_FEATURE_WEIGHTS, N_FEATURE_BIAS, N_OUTPUT_WEIGHTS, N_OUTPUT_BIAS, N_ACCUMULATORS, N_ACCUMULATOR_KEYS = (
    range(6)
)


def load_network(path):
    """Load exported arrays and return their SHA-256 for the run log."""
    # Quantization-aware training (QAT) used rounded weights in each forward pass.
    # At inference these four arrays already contain the scaled integers.
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = (
            np.ascontiguousarray(data["W0"], dtype=np.int16),
            np.ascontiguousarray(data["b0"], dtype=np.int16),
            np.ascontiguousarray(data["W1"], dtype=np.int16),
            np.ascontiguousarray(data["b1"], dtype=np.int32),
        )
        metadata = json.loads(str(data["meta"])) if "meta" in data else {}
    if arrays[0].ndim != 2:
        raise ValueError("W0 must have one row per feature and one column per channel")
    hidden_width = arrays[0].shape[1]
    expected = ((FEATURES, hidden_width), (hidden_width,), (OUT_BUCKETS, 2 * hidden_width), (OUT_BUCKETS,))
    if (
        tuple(array.shape for array in arrays) != expected
        or not 0 < hidden_width <= 4096
        or hidden_width % 64
    ):
        raise ValueError(f"Invalid network shapes: {[array.shape for array in arrays]}")
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"network schema {metadata.get('schema')!r} != {SCHEMA!r}")
    with path.open("rb") as weights_file:
        digest = hashlib.file_digest(weights_file, "sha256").hexdigest()
    return arrays, digest


def new_accumulators(hidden_width=PLACEHOLDER_WIDTH):
    """Allocate separate running feature sums for each perspective at each search ply."""
    accumulators = np.zeros((board_ops.MAX_SEARCH_PLY + 4, 2, hidden_width), dtype=np.int16)
    accumulator_keys = np.zeros(board_ops.MAX_SEARCH_PLY + 4, dtype=np.uint64)
    accumulator_keys[:] = np.uint64(0xFFFFFFFFFFFFFFFF)  # never equal to a real key in practice
    return accumulators, accumulator_keys


def make_nn_context(arrays):
    """Bundle immutable weights with the reusable per-ply accumulator storage."""
    return (*arrays, *new_accumulators(arrays[0].shape[1]))


def invalidate(network):
    """Mark every saved sum stale when a new game or evaluation mode starts."""
    network[N_ACCUMULATOR_KEYS][:] = np.uint64(0xFFFFFFFFFFFFFFFF)


@njit(cache=False)
def feature_index(perspective, king_sq, piece, square):
    """Orient the piece square to this king and return its row in the feature weights."""
    # Rank normalization flips Black's view. File mirroring puts each king on
    # files a–d. The king bucket then selects the piece-square feature table.
    if perspective == board_ops.BLACK:
        king_sq ^= 56
        square ^= 56
    if (king_sq & 7) >= 4:
        king_sq ^= 7
        square ^= 7
    colour = 1 if piece > 6 else 0
    piece_type = piece - 1 - 6 * colour
    code = piece_type + 6 * (colour != perspective)
    return (KING_BUCKET[king_sq] * PIECE_CODES + code) * 64 + square


@njit(cache=False)
def refresh_perspective(network, bitboards, mailbox, ply, perspective):
    """Rebuild one king’s running sum from the bias and all occupied feature rows."""
    # Sparse inputs: each occupied square activates one feature. Its embedding
    # is a learned row of weights. The accumulator sums those rows plus the bias.
    feature_weights = network[N_FEATURE_WEIGHTS]
    feature_bias = network[N_FEATURE_BIAS]
    accumulators = network[N_ACCUMULATORS]
    king_sq = board_ops.king_square(bitboards, perspective)
    hidden_width = accumulators.shape[2]
    for channel in range(hidden_width):
        accumulators[ply, perspective, channel] = feature_bias[channel]
    occupied_squares = bitboards[board_ops.WHITE_OCCUPANCY] | bitboards[board_ops.BLACK_OCCUPANCY]
    while occupied_squares:
        square = board_ops.lowest_set_square(occupied_squares)
        occupied_squares &= occupied_squares - board_ops.BITBOARD_ONE
        feature_id = feature_index(perspective, king_sq, mailbox[square], square)
        for channel in range(hidden_width):
            accumulators[ply, perspective, channel] += feature_weights[feature_id, channel]


@njit(cache=False)
def _add_feature(accumulators, feature_weights, ply, perspective, feature_id):
    """Add the feature row for a piece entering the position."""
    for channel in range(accumulators.shape[2]):
        accumulators[ply, perspective, channel] += feature_weights[feature_id, channel]


@njit(cache=False)
def _sub_feature(accumulators, feature_weights, ply, perspective, feature_id):
    """Subtract the feature row for a piece leaving the position."""
    for channel in range(accumulators.shape[2]):
        accumulators[ply, perspective, channel] -= feature_weights[feature_id, channel]


@njit(cache=False)
def update_perspective(network, bitboards, mailbox, position_state, undo_stack, ply, perspective):
    """Copy the parent sum and apply move deltas, rebuilding if this king changes feature frame."""
    feature_weights = network[N_FEATURE_WEIGHTS]
    accumulators = network[N_ACCUMULATORS]
    move = undo_stack[ply - 1, board_ops.UNDO_ENCODED_MOVE]
    for channel in range(accumulators.shape[2]):
        accumulators[ply, perspective, channel] = accumulators[ply - 1, perspective, channel]
    if move == 0:  # null move: identical position
        return
    from_square = move & 63
    to_square = (move >> 6) & 63
    flag = (move >> 12) & 15
    mover = 1 - position_state[board_ops.STATE_SIDE_TO_MOVE]  # side that made the move
    captured = undo_stack[ply - 1, board_ops.UNDO_CAPTURED_PIECE]
    piece_now = mailbox[to_square]
    if flag & 8:
        piece_before = mover * 6 + board_ops.PAWN
    else:
        piece_before = piece_now
    if piece_before == mover * 6 + board_ops.KING and perspective == mover:
        # A different king bucket or mirror changes the feature indices for the
        # whole perspective. An incremental update of the king alone would be wrong.
        previous_king_square = from_square ^ 56 if perspective == board_ops.BLACK else from_square
        current_king_square = to_square ^ 56 if perspective == board_ops.BLACK else to_square
        if ((previous_king_square & 7) >= 4) != ((current_king_square & 7) >= 4) or KING_BUCKET[
            previous_king_square
        ] != KING_BUCKET[current_king_square]:
            refresh_perspective(network, bitboards, mailbox, ply, perspective)
            return
    king_sq = board_ops.king_square(bitboards, perspective)
    _sub_feature(
        accumulators,
        feature_weights,
        ply,
        perspective,
        feature_index(perspective, king_sq, piece_before, from_square),
    )
    _add_feature(
        accumulators,
        feature_weights,
        ply,
        perspective,
        feature_index(perspective, king_sq, piece_now, to_square),
    )
    if captured:
        if flag == board_ops.EN_PASSANT_CAPTURE_FLAG:
            captured_square = to_square - 8 if mover == board_ops.WHITE else to_square + 8
        else:
            captured_square = to_square
        _sub_feature(
            accumulators,
            feature_weights,
            ply,
            perspective,
            feature_index(perspective, king_sq, captured, captured_square),
        )
    if flag == board_ops.KINGSIDE_CASTLE_FLAG:
        rook = mover * 6 + board_ops.ROOK
        _sub_feature(
            accumulators,
            feature_weights,
            ply,
            perspective,
            feature_index(perspective, king_sq, rook, from_square + 3),
        )
        _add_feature(
            accumulators,
            feature_weights,
            ply,
            perspective,
            feature_index(perspective, king_sq, rook, from_square + 1),
        )
    elif flag == board_ops.QUEENSIDE_CASTLE_FLAG:
        rook = mover * 6 + board_ops.ROOK
        _sub_feature(
            accumulators,
            feature_weights,
            ply,
            perspective,
            feature_index(perspective, king_sq, rook, from_square - 4),
        )
        _add_feature(
            accumulators,
            feature_weights,
            ply,
            perspective,
            feature_index(perspective, king_sq, rook, from_square - 1),
        )


@njit(cache=False)
def ensure_accumulators(
    network, bitboards, mailbox, position_state, position_keys, undo_stack, previous_keys, ply
):
    """Lazy accumulator update: materialise the two feature sums when evaluation needs them."""
    accumulator_keys = network[N_ACCUMULATOR_KEYS]
    key = position_keys[0]
    if accumulator_keys[ply] == key:
        return
    if ply > 0 and accumulator_keys[ply - 1] == previous_keys[ply - 1]:
        update_perspective(network, bitboards, mailbox, position_state, undo_stack, ply, board_ops.WHITE)
        update_perspective(network, bitboards, mailbox, position_state, undo_stack, ply, board_ops.BLACK)
    else:
        refresh_perspective(network, bitboards, mailbox, ply, board_ops.WHITE)
        refresh_perspective(network, bitboards, mailbox, ply, board_ops.BLACK)
    accumulator_keys[ply] = key


@njit(cache=False)
def output_bucket(bitboards):
    """Choose one of eight trained output heads from the number of pieces remaining."""
    piece_count = board_ops.count_set_squares(
        bitboards[board_ops.WHITE_OCCUPANCY] | bitboards[board_ops.BLACK_OCCUPANCY]
    )
    bucket = (piece_count - 1) // 4
    if bucket > OUT_BUCKETS - 1:
        bucket = OUT_BUCKETS - 1
    return bucket


@njit(cache=False)
def _screlu_dot(accumulators, ply, perspective, output_weights, bucket, offset):
    """sum_i clamp(activation[channel], 0, QA)^2 * w1[bucket, offset + i] in int64.

    The 1-D row views are taken once, outside the loop: multi-dimensional indexing
    inside the hot loop defeats LLVM's vectoriser on x86 (measured ~7x slower on
    EPYC), while a branch-free clamp over contiguous rows vectorises cleanly.
    """
    channels = accumulators[ply, perspective]
    head_weights = output_weights[bucket]
    weighted_sum = np.int64(0)
    for channel in range(channels.shape[0]):
        activation = np.int32(channels[channel])
        activation = min(max(activation, 0), QA)
        weighted_sum += np.int64(activation * activation) * np.int64(head_weights[offset + channel])
    return weighted_sum


@njit(cache=False)
def forward(network, bitboards, position_state, ply):
    """Forward inference: the selected output head combines both perspectives into one centipawn score."""
    accumulators = network[N_ACCUMULATORS]
    output_weights = network[N_OUTPUT_WEIGHTS]
    output_bias = network[N_OUTPUT_BIAS]
    side_to_move = position_state[board_ops.STATE_SIDE_TO_MOVE]
    bucket = output_bucket(bitboards)
    hidden_width = accumulators.shape[2]
    weighted_sum = _screlu_dot(accumulators, ply, side_to_move, output_weights, bucket, 0)
    weighted_sum += _screlu_dot(accumulators, ply, 1 - side_to_move, output_weights, bucket, hidden_width)
    weighted_sum += np.int64(output_bias[bucket])
    # Floor division matches the export on negative scores too. One pawn is 100 centipawns.
    return (weighted_sum * SCALE) // (QA * QA * QB)


@njit(cache=False)
def evaluate_nn(network, bitboards, mailbox, position_state, position_keys, undo_stack, previous_keys, ply):
    """Ensure both perspective sums match the current key before scoring them."""
    ensure_accumulators(
        network, bitboards, mailbox, position_state, position_keys, undo_stack, previous_keys, ply
    )
    return forward(network, bitboards, position_state, ply)


# Classical evaluation

TEMPO = 12
MAX_PHASE = 24


def install_tables_from_json(path):
    """Load the fitted middlegame and endgame tables before compiling the classical evaluator."""
    data = json.loads(Path(path).read_text())
    middlegame = np.array(data["mg"], dtype=np.int32).reshape(6, 64)
    endgame = np.array(data["eg"], dtype=np.int32).reshape(6, 64)
    board_ops.install_piece_square_scores(middlegame, endgame)
    return data


@njit(cache=False)
def evaluate_psqt(position_state):
    """Tapered evaluation interpolates between the two piece-square table (PSQT) totals by game phase."""
    phase = position_state[board_ops.STATE_GAME_PHASE]
    if phase > MAX_PHASE:
        phase = MAX_PHASE
    score = (
        position_state[board_ops.STATE_MIDDLEGAME_SCORE] * phase
        + position_state[board_ops.STATE_ENDGAME_SCORE] * (MAX_PHASE - phase)
    ) // MAX_PHASE
    if position_state[board_ops.STATE_SIDE_TO_MOVE] == board_ops.BLACK:
        score = -score
    return score + TEMPO  # Tempo bonus for the player who gets to move next.
