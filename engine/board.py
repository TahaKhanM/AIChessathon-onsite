"""Bitboards, legal move generation, and reversible updates.

Squares: a1 = 0 ... h8 = 63 (rank*8 + file), matching python-chess.
Pieces: 0 empty, 1..6 white P N B R Q K, 7..12 black P N B R Q K.

Arrays (all owned by the caller, no allocation inside kernels):
  bitboards       Piece sets at indices 1..12, then White and Black occupancy.
  mailbox         Piece code on each of the 64 squares.
  position_state  Turn, rights, clocks, classical score and active history length.
  position_keys   Slot zero holds the current Zobrist key.
  position_history  Earlier keys from the game followed by the search path.
  undo_stack and previous_keys  One saved transition and parent key per ply.

A bitboard uses one binary digit per square, from a1 (bit 0) to h8 (bit 63).
For example, 0b0101 has a1 and c1 set. A packed move uses bits 0–5 for its
origin, 6–11 for its destination, and 12–15 for the move kind.

For square sets, & keeps squares in both sets and | joins them. ^ toggles
the marked bits, ~ flips occupied and empty bits in a 64-bit mask, and
<< or >> shifts bits left or right. An equals sign after the operator
writes the result back into the same variable.
"""

from __future__ import annotations

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic, register_jitable

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING = 1, 2, 3, 4, 5, 6
BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING = 7, 8, 9, 10, 11, 12
WHITE_OCCUPANCY, BLACK_OCCUPANCY = 13, 14

# These are indices into position_state. For example, the en passant slot holds a square or -1.
(
    STATE_SIDE_TO_MOVE,
    STATE_CASTLING_RIGHTS,
    STATE_EN_PASSANT_SQUARE,
    STATE_HALFMOVE_CLOCK,
    STATE_GAME_PLY,
    STATE_MIDDLEGAME_SCORE,
    STATE_ENDGAME_SCORE,
    STATE_GAME_PHASE,
    STATE_HISTORY_LENGTH,
) = range(9)
# One undo row stores the parent values needed to reverse the move at that ply.
(
    UNDO_CASTLING_RIGHTS,
    UNDO_EN_PASSANT_SQUARE,
    UNDO_HALFMOVE_CLOCK,
    UNDO_CAPTURED_PIECE,
    UNDO_ENCODED_MOVE,
    UNDO_MIDDLEGAME_SCORE,
    UNDO_ENDGAME_SCORE,
    UNDO_GAME_PHASE,
) = range(8)
MAX_SEARCH_PLY = 128
POSITION_HISTORY_CAPACITY = 1024
MOVE_BUFFER_CAPACITY = 256

(
    QUIET_MOVE_FLAG,
    DOUBLE_PAWN_PUSH_FLAG,
    KINGSIDE_CASTLE_FLAG,
    QUEENSIDE_CASTLE_FLAG,
    CAPTURE_FLAG,
    EN_PASSANT_CAPTURE_FLAG,
) = 0, 1, 2, 3, 4, 5
# In the four flag bits, 0100 means capture and 1000 means promotion.
# For promotions the remaining two bits select knight, bishop, rook or queen in that order.
PROMOTION_FLAG = 8
PROMOTE_TO_KNIGHT, PROMOTE_TO_BISHOP, PROMOTE_TO_ROOK, PROMOTE_TO_QUEEN = range(8, 12)
(
    CAPTURE_AND_PROMOTE_TO_KNIGHT,
    CAPTURE_AND_PROMOTE_TO_BISHOP,
    CAPTURE_AND_PROMOTE_TO_ROOK,
    CAPTURE_AND_PROMOTE_TO_QUEEN,
) = range(12, 16)
WHITE_KINGSIDE_RIGHT, WHITE_QUEENSIDE_RIGHT, BLACK_KINGSIDE_RIGHT, BLACK_QUEENSIDE_RIGHT = 1, 2, 4, 8
ALL_CASTLING_RIGHTS = 15
GENERATE_ALL_MOVES, GENERATE_TACTICAL_MOVES, GENERATE_QUIET_MOVES = 0, 1, 2

BITBOARD_ONE = np.uint64(1)
ALL_SQUARES = np.uint64(0xFFFFFFFFFFFFFFFF)

# Static tables (built once at import in plain Python/NumPy)


def _square_index(rank, file_index):
    """Map a zero-based rank and file to the square index used by every board array."""
    return rank * 8 + file_index


def _build_jump_attack_table(deltas):
    """Build leaper attack tables. A knight's jumps and a king's steps ignore intervening pieces."""
    table = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        rank, file_index = divmod(square, 8)
        mask = 0
        for rank_step, file_step in deltas:
            target_rank, target_file = rank + rank_step, file_index + file_step
            if 0 <= target_rank < 8 and 0 <= target_file < 8:
                # Shift a 1 to the reachable square, then OR it into the attack set.
                mask |= 1 << _square_index(target_rank, target_file)
        table[square] = mask
    table.setflags(write=False)
    return table


KNIGHT_ATTACKS = _build_jump_attack_table(
    ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
)
KING_ATTACKS = _build_jump_attack_table(
    ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
)
PAWN_ATTACKS = np.stack(
    [_build_jump_attack_table(((1, 1), (1, -1))), _build_jump_attack_table(((-1, 1), (-1, -1)))]
)
PAWN_ATTACKS.setflags(write=False)


def _trace_sliding_attacks(square, occupancy, rook):
    """Walk each ray through the first blocker to build a reference attack mask."""
    start_rank, start_file = divmod(square, 8)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1)) if rook else ((1, 1), (1, -1), (-1, 1), (-1, -1))
    attacks = 0
    for rank_step, file_step in directions:
        rank, file_index = start_rank + rank_step, start_file + file_step
        while 0 <= rank < 8 and 0 <= file_index < 8:
            target_mask = 1 << _square_index(rank, file_index)  # Shift the 1 onto this square of the ray.
            attacks |= target_mask  # Include the square even if it contains the first blocker.
            if occupancy & target_mask:  # A shared bit means a piece blocks travel beyond this square.
                break
            rank += rank_step
            file_index += file_step
    return attacks


def _relevant_blocker_mask(square, rook):
    """Keep blockers that can change a sliding attack beyond their square."""
    start_rank, start_file = divmod(square, 8)
    excluded_edges = 0
    # A piece on the last square of a ray cannot block a square beyond it.
    if start_rank != 0:
        excluded_edges |= 0x00000000000000FF  # Include rank 1 in the edge mask.
    if start_rank != 7:
        excluded_edges |= 0xFF00000000000000  # Include rank 8.
    if start_file != 0:
        excluded_edges |= 0x0101010101010101  # Include file a (the first bit of every rank).
    if start_file != 7:
        excluded_edges |= 0x8080808080808080  # Include file h (the last bit of every rank).
    # Complement the excluded edges, then intersect with the ray to keep only relevant blockers.
    return _trace_sliding_attacks(square, 0, rook) & ~excluded_edges


# Each multiplier maps the blockers on a rook or bishop ray to an attack-table index.
# The table builder checks that blocker patterns sharing a slot also share an attack mask.
ROOK_INDEX_BITS = 12
BISHOP_INDEX_BITS = 9
ROOK_MAGIC_MULTIPLIERS = np.array(
    [
        0x3080004004603088,
        0x1040000860001000,
        0x0850002004200010,
        0x8009041210000800,
        0x0080800410081802,
        0x2004100080402100,
        0x2280004210800100,
        0x008001210000D080,
        0x0000800080104000,
        0x8C08408608044308,
        0x0108200060C40800,
        0x4204100400080022,
        0x442008C001008200,
        0x0002000082000804,
        0x0181000062000504,
        0xA490080084002008,
        0x0181092000106400,
        0x0128002000900088,
        0x0340040488808022,
        0x020E882104002008,
        0x2206402800240281,
        0x0002001401400080,
        0x8800012806101004,
        0x1000050804100300,
        0x2442100060002000,
        0x018010002008A100,
        0x0004618480100480,
        0x0801890040020400,
        0x0001012018004512,
        0x4219222020040881,
        0x0022400040010040,
        0x0000804080002100,
        0x00008103010022C0,
        0x0148400050810010,
        0x0010A00041180008,
        0x214418002020A010,
        0x0000215400281100,
        0x02D200440008D220,
        0x00820000C8200081,
        0x64000890C0200440,
        0x000F100860002002,
        0x0400802222090400,
        0x1020004801880100,
        0x4C85008030021014,
        0x0008020001002321,
        0x0410802240100300,
        0x0002082201004080,
        0x4081000029401280,
        0x0088B080C0002080,
        0x0200200010021822,
        0x0008000910200040,
        0x2050A840408002C0,
        0x0010080804460340,
        0x0044801A02014408,
        0x020100250C0B0208,
        0x8100014120810200,
        0x4080158004204103,
        0x1102088011000822,
        0x806000050045202D,
        0x0003204110920016,
        0x8208004426110001,
        0x0000008108500402,
        0x4022000206805401,
        0x8400041044810022,
    ],
    dtype=np.uint64,
)
BISHOP_MAGIC_MULTIPLIERS = np.array(
    [
        0x820C600140424900,
        0x6200C0808100C020,
        0x14E0C4D0100C0109,
        0x0000820804400004,
        0x00018A0060100400,
        0x02101200210C8440,
        0x0300225C0041C008,
        0x0000520010028040,
        0x009060002A104500,
        0x4200180880240204,
        0x4003014800100400,
        0xB000002026888010,
        0x00000080C0804000,
        0x1100051081010002,
        0x2208000A08140100,
        0x0040408A8E114020,
        0x0022004405A01101,
        0x0001062020140440,
        0x0001281020508004,
        0x0011C80401009240,
        0x2082847022010DC1,
        0x14101000908A2624,
        0x0015800942045020,
        0x0840040448440010,
        0x0081012121002100,
        0x0441080010280508,
        0x0004320401080200,
        0x3886080084004099,
        0x4401009001004000,
        0x480800880A110200,
        0x0001782000101000,
        0x0011061000020041,
        0x4204002000402882,
        0x0004001400880280,
        0x000420100008000F,
        0x1A80080800020A01,
        0x82C0044100025100,
        0x2002209500780420,
        0x0002101400020400,
        0x02A1034480022040,
        0x0200609250020902,
        0x1181204900202280,
        0x2128800988040100,
        0x0000022454000200,
        0x0C00400912000100,
        0x00012000A0400580,
        0x180409A168040C00,
        0x00202A20401020AC,
        0x08004C0020C09000,
        0x0100088644020000,
        0x000000200A10C000,
        0x0000286300880804,
        0x0001806022C82420,
        0x8104150040144003,
        0x20080090120A0010,
        0x2022A80010894200,
        0x0C82000405081800,
        0x4022140680240040,
        0x0018000200400800,
        0x0A02001008160042,
        0x0000200440004410,
        0x000A800682004104,
        0x0412004182001A00,
        0x0000900012102210,
    ],
    dtype=np.uint64,
)


ROOK_BLOCKER_MASKS = np.array([_relevant_blocker_mask(square, True) for square in range(64)], dtype=np.uint64)
BISHOP_BLOCKER_MASKS = np.array(
    [_relevant_blocker_mask(square, False) for square in range(64)], dtype=np.uint64
)
# Shifting 64 left by the index width reserves 2**width blocker patterns for each of the 64 origins.
ROOK_ATTACK_TABLE = np.zeros(64 << ROOK_INDEX_BITS, dtype=np.uint64)
# Each bishop square gets its own block of 2**BISHOP_INDEX_BITS attack masks.
BISHOP_ATTACK_TABLE = np.zeros(64 << BISHOP_INDEX_BITS, dtype=np.uint64)


def _build_sliding_attack_tables():
    """Build magic bitboards for sliding pieces (rooks and bishops).

    For each origin, keep the occupancy bits that can stop a ray. Multiplication by its
    magic multiplier mixes those bits into a small index. Two blocker patterns may
    share an index if they give identical attacks. A conflicting attack mask fails here.
    """
    for square in range(64):
        for rook, mask, magic, shift, table in (
            (True, ROOK_BLOCKER_MASKS, ROOK_MAGIC_MULTIPLIERS, ROOK_INDEX_BITS, ROOK_ATTACK_TABLE),
            (False, BISHOP_BLOCKER_MASKS, BISHOP_MAGIC_MULTIPLIERS, BISHOP_INDEX_BITS, BISHOP_ATTACK_TABLE),
        ):
            blocker_mask = int(mask[square])
            multiplier = int(magic[square])
            blockers = 0
            while True:
                # Mask to 64 bits to match unsigned overflow at runtime. Shift right to keep the high index bits.
                table_index = ((blockers * multiplier) & 0xFFFFFFFFFFFFFFFF) >> (64 - shift)
                attacks = _trace_sliding_attacks(square, blockers, rook)
                # Left shift selects this origin’s block of slots. The small index picks one blocker pattern.
                slot = (square << shift) + table_index
                if table[slot] and int(table[slot]) != attacks:
                    raise ValueError("magic collision")
                table[slot] = attacks
                # Subtract then mask to visit the next subset of relevant blockers. Returning to zero ends the cycle.
                blockers = (blockers - blocker_mask) & blocker_mask
                if blockers == 0:
                    break


_build_sliding_attack_tables()
ROOK_ATTACK_TABLE.setflags(write=False)
BISHOP_ATTACK_TABLE.setflags(write=False)
EMPTY_BOARD_ROOK_RAYS = np.array(
    [_trace_sliding_attacks(square, 0, True) for square in range(64)], dtype=np.uint64
)
EMPTY_BOARD_BISHOP_RAYS = np.array(
    [_trace_sliding_attacks(square, 0, False) for square in range(64)], dtype=np.uint64
)

SQUARES_BETWEEN = np.zeros((64, 64), dtype=np.uint64)
ALIGNED_SQUARES = np.zeros((64, 64), dtype=np.uint64)
for first_square in range(64):
    for second_square in range(64):
        if first_square == second_square:
            continue
        for is_rook in (True, False):
            rays = EMPTY_BOARD_ROOK_RAYS if is_rook else EMPTY_BOARD_BISHOP_RAYS
            first_bit = 1 << first_square  # Single-bit masks let us include either endpoint in a ray.
            second_bit = 1 << second_square  # Mark the far endpoint to stop the opposite ray.
            if int(rays[first_square]) & second_bit:  # The second square must lie on the first one's ray.
                # The intersection finds the common line. OR restores both origins, omitted from their own rays.
                ALIGNED_SQUARES[first_square, second_square] = (
                    (int(rays[first_square]) & int(rays[second_square])) | first_bit | second_bit
                )
                # Stop each ray at the opposite endpoint. Only the interior squares belong to both rays.
                SQUARES_BETWEEN[first_square, second_square] = _trace_sliding_attacks(
                    first_square, second_bit, is_rook
                ) & _trace_sliding_attacks(second_square, first_bit, is_rook)
SQUARES_BETWEEN.setflags(write=False)
ALIGNED_SQUARES.setflags(write=False)

CASTLING_RIGHTS_RETAINED_BY_SQUARE = np.full(64, ALL_CASTLING_RIGHTS, dtype=np.int64)
# Each corner belongs to a rook. Clear its right if a move leaves or lands there, even on a capture.
for corner, right in (
    (0, WHITE_QUEENSIDE_RIGHT),
    (7, WHITE_KINGSIDE_RIGHT),
    (56, BLACK_QUEENSIDE_RIGHT),
    (63, BLACK_KINGSIDE_RIGHT),
):
    # AND with the inverted right clears just this corner rook’s right.
    CASTLING_RIGHTS_RETAINED_BY_SQUARE[corner] &= ~right
# e1: OR joins both White rights. AND with the complement clears them when the king moves.
CASTLING_RIGHTS_RETAINED_BY_SQUARE[4] &= ~(WHITE_KINGSIDE_RIGHT | WHITE_QUEENSIDE_RIGHT)
# e8: clear both Black rights the same way and keep White's rights available.
CASTLING_RIGHTS_RETAINED_BY_SQUARE[60] &= ~(BLACK_KINGSIDE_RIGHT | BLACK_QUEENSIDE_RIGHT)

# Zobrist hashing gives each piece-square pair, rights state, en passant file and turn a 64-bit value.
# A fixed seed reproduces the same keys across runs. The specific seed has no chess meaning.
_hash_key_generator = np.random.default_rng(20260909)
PIECE_SQUARE_HASH_KEYS = _hash_key_generator.integers(0, 2**64, size=(13, 64), dtype=np.uint64)
PIECE_SQUARE_HASH_KEYS[0, :] = 0
CASTLING_HASH_KEYS = _hash_key_generator.integers(0, 2**64, size=16, dtype=np.uint64)
EN_PASSANT_FILE_HASH_KEYS = _hash_key_generator.integers(0, 2**64, size=8, dtype=np.uint64)
SIDE_TO_MOVE_HASH_KEY = np.uint64(_hash_key_generator.integers(0, 2**64, dtype=np.uint64))
for key_table in (PIECE_SQUARE_HASH_KEYS, CASTLING_HASH_KEYS, EN_PASSANT_FILE_HASH_KEYS):
    key_table.setflags(write=False)

# Tapered piece-square values from White's view; filled by evaluation module.
# Index [piece_code, square]; black pieces carry negative values.
MIDDLEGAME_PIECE_SQUARE_SCORES = np.zeros((13, 64), dtype=np.int32)
ENDGAME_PIECE_SQUARE_SCORES = np.zeros((13, 64), dtype=np.int32)
GAME_PHASE_BY_PIECE = np.array([0, 0, 1, 1, 2, 4, 0, 0, 1, 1, 2, 4, 0], dtype=np.int64)
GAME_PHASE_BY_PIECE.setflags(write=False)


def install_piece_square_scores(white_middlegame, white_endgame):
    """Install White scores and their rank-mirrored negatives for Black."""
    for piece_index in range(6):
        for square in range(64):
            MIDDLEGAME_PIECE_SQUARE_SCORES[1 + piece_index, square] = white_middlegame[piece_index, square]
            ENDGAME_PIECE_SQUARE_SCORES[1 + piece_index, square] = white_endgame[piece_index, square]
            # XOR 56 flips the three rank bits and preserves the file, so a1 and a8 share a fitted value.
            mirrored_square = square ^ 56
            MIDDLEGAME_PIECE_SQUARE_SCORES[7 + piece_index, square] = -white_middlegame[
                piece_index, mirrored_square
            ]
            ENDGAME_PIECE_SQUARE_SCORES[7 + piece_index, square] = -white_endgame[
                piece_index, mirrored_square
            ]


# Bit operations and attacks used during search


@njit(cache=False)
def square_mask(square):
    """Return the 64-bit mask containing just this square."""
    return BITBOARD_ONE << np.uint64(square)  # Shift the single 1 to this square's bit position.


@register_jitable(inline="always")
def _contains_square(squares, square):
    """Test membership of one board square in a set of squares."""
    # Shift the target bit into position zero. AND with 1 drops every other square.
    return ((squares >> np.uint64(square)) & BITBOARD_ONE) != 0


@register_jitable(inline="always")
def _without_lowest_square(squares):
    """Remove the lowest occupied square after a caller has visited it."""
    # Subtracting 1 flips the lowest 1 and the zeros below it. AND clears only that 1 (1010 → 1000).
    return squares & (squares - BITBOARD_ONE)


@intrinsic
def _count_trailing_zeros_uint64(typing_context, value):
    """Expose the CPU trailing-zero count to compiled move generation."""
    if value != types.uint64:
        raise TypeError("cttz requires uint64")

    def emit_instruction(context, builder, signature, arguments):
        return builder.cttz(arguments[0], ir.Constant(ir.IntType(1), 0))

    return types.int64(types.uint64), emit_instruction


@intrinsic
def _count_set_bits_uint64(typing_context, value):
    """Use LLVM's population count (popcount) to count the 1 bits in a bitboard."""
    if value != types.uint64:
        raise TypeError("ctpop requires uint64")

    def emit_instruction(context, builder, signature, arguments):
        intrinsic_function = builder.module.declare_intrinsic("llvm.ctpop", [ir.IntType(64)])
        return builder.call(intrinsic_function, [arguments[0]])

    return types.int64(types.uint64), emit_instruction


@njit(cache=False)
def lowest_set_square(bits):
    """Return the square of the lowest set bit in a nonempty mask."""
    return _count_trailing_zeros_uint64(bits)


@njit(cache=False)
def count_set_squares(bits):
    """Count the occupied squares in a mask."""
    return _count_set_bits_uint64(bits)


@njit(cache=False)
def rook_attacks(square, occupancy):
    """Use the relevant blockers to look up reachable rank and file squares."""
    # Keep relevant blockers with AND. The high bits of their wrapped product identify the attack pattern.
    table_index = ((occupancy & ROOK_BLOCKER_MASKS[square]) * ROOK_MAGIC_MULTIPLIERS[square]) >> np.uint64(
        64 - ROOK_INDEX_BITS
    )
    # Shift the origin to its block of rook entries and add the blocker-pattern index.
    return ROOK_ATTACK_TABLE[(square << ROOK_INDEX_BITS) + np.int64(table_index)]


@njit(cache=False)
def bishop_attacks(square, occupancy):
    """Use the relevant blockers to look up reachable diagonal squares."""
    # Ignore blockers off the diagonals. The high bits of the wrapped product select a bishop pattern.
    table_index = (
        (occupancy & BISHOP_BLOCKER_MASKS[square]) * BISHOP_MAGIC_MULTIPLIERS[square]
    ) >> np.uint64(64 - BISHOP_INDEX_BITS)
    # Shift the origin to its block of bishop entries and add the blocker-pattern index.
    return BISHOP_ATTACK_TABLE[(square << BISHOP_INDEX_BITS) + np.int64(table_index)]


@njit(cache=False)
def queen_attacks(square, occupancy):
    """Combine diagonal and orthogonal rays, stopping each ray at its first blocker."""
    # OR joins orthogonal and diagonal destinations.
    return rook_attacks(square, occupancy) | bishop_attacks(square, occupancy)


@njit(cache=False)
def attackers_of_square(bitboards, square, occupancy):
    """Find attackers of either color under the supplied occupancy."""
    # Trace backwards from the target. AND keeps White pawns on origins in Black's attack pattern.
    attacks = PAWN_ATTACKS[BLACK, square] & bitboards[WHITE_PAWN]
    # Reverse the colors and OR any Black pawn attackers into the same result.
    attacks |= PAWN_ATTACKS[WHITE, square] & bitboards[BLACK_PAWN]
    # Union both knight sets, keep those on an attacking square, then add them to the result.
    attacks |= KNIGHT_ATTACKS[square] & (bitboards[WHITE_KNIGHT] | bitboards[BLACK_KNIGHT])
    # Union both king sets, intersect with adjacent squares and add those attackers.
    attacks |= KING_ATTACKS[square] & (bitboards[WHITE_KING] | bitboards[BLACK_KING])
    # Union bishops and queens of both colors, intersect with open diagonals, and accumulate the attackers.
    attacks |= bishop_attacks(square, occupancy) & (
        bitboards[WHITE_BISHOP] | bitboards[BLACK_BISHOP] | bitboards[WHITE_QUEEN] | bitboards[BLACK_QUEEN]
    )
    # Union rooks and queens of both colors, intersect with open ranks/files, and add them too.
    attacks |= rook_attacks(square, occupancy) & (
        bitboards[WHITE_ROOK] | bitboards[BLACK_ROOK] | bitboards[WHITE_QUEEN] | bitboards[BLACK_QUEEN]
    )
    return attacks


@njit(cache=False)
def is_square_attacked_by(bitboards, square, attacking_color, occupancy):
    """Return early as soon as a piece of the requested color attacks the square."""
    base = attacking_color * 6
    # Reverse the pawn direction at the target. AND detects a pawn of the attacking color on an origin.
    if PAWN_ATTACKS[1 - attacking_color, square] & bitboards[base + PAWN]:
        return True
    if KNIGHT_ATTACKS[square] & bitboards[base + KNIGHT]:  # A shared bit is an enemy knight one jump away.
        return True
    if KING_ATTACKS[square] & bitboards[base + KING]:  # A shared bit is an adjacent attacking king.
        return True
    # Combine bishops and queens, then test whether any lies on an unblocked diagonal.
    if bishop_attacks(square, occupancy) & (bitboards[base + BISHOP] | bitboards[base + QUEEN]):
        return True
    # Combine rooks and queens, then test the open orthogonal rays.
    if rook_attacks(square, occupancy) & (bitboards[base + ROOK] | bitboards[base + QUEEN]):
        return True
    return False


@njit(cache=False)
def king_square(bitboards, color):
    """Locate the single king bit belonging to this color."""
    return lowest_set_square(bitboards[color * 6 + KING])


@njit(cache=False)
def in_check(bitboards, position_state):
    """Test the current player’s king against enemy attacks."""
    moving_side = position_state[STATE_SIDE_TO_MOVE]
    occupancy = bitboards[WHITE_OCCUPANCY] | bitboards[BLACK_OCCUPANCY]  # Both colors can block an attack.
    return is_square_attacked_by(bitboards, king_square(bitboards, moving_side), 1 - moving_side, occupancy)


@njit(cache=False)
def encode_move(from_square, to_square, flag):
    """Pack two six-bit square indices and a move flag into one integer."""
    # Shift the destination into bits 6–11 and the flag into 12–15. OR joins the three disjoint fields.
    return from_square | (to_square << 6) | (flag << 12)


@register_jitable(inline="always")
def decode_move(move):
    """Read the same origin, destination and flag fields written by encode_move."""
    from_square = move & 63  # Keep the low six bits: the origin, in the range 0–63.
    to_square = (move >> 6) & 63  # Drop the origin, then keep the six destination bits.
    flag = (move >> 12) & 15  # Drop both squares, then keep the four move-kind bits.
    return from_square, to_square, flag


@njit(cache=False)
def _add_piece(bitboards, mailbox, position_state, position_keys, square, piece):
    """Install a piece in both board views and add its hash and evaluation contributions."""
    changed_squares = square_mask(square)
    bitboards[piece] |= changed_squares  # Set this square in the piece-specific set.
    # Set the same square in its color’s occupancy.
    bitboards[WHITE_OCCUPANCY + (piece > 6)] |= changed_squares
    mailbox[square] = piece
    # XOR the piece-square contribution into the position hash.
    position_keys[0] ^= PIECE_SQUARE_HASH_KEYS[piece, square]
    position_state[STATE_MIDDLEGAME_SCORE] += MIDDLEGAME_PIECE_SQUARE_SCORES[piece, square]
    position_state[STATE_ENDGAME_SCORE] += ENDGAME_PIECE_SQUARE_SCORES[piece, square]
    position_state[STATE_GAME_PHASE] += GAME_PHASE_BY_PIECE[piece]


@njit(cache=False)
def _remove_piece(bitboards, mailbox, position_state, position_keys, square):
    """Remove the occupant from both board views and its cached hash and score."""
    piece = mailbox[square]
    changed_squares = square_mask(square)
    bitboards[piece] ^= changed_squares  # The square is occupied, so toggling its bit removes the piece.
    # Toggle off the matching bit in its color’s occupancy.
    bitboards[WHITE_OCCUPANCY + (piece > 6)] ^= changed_squares
    mailbox[square] = 0
    # XOR the same piece-square key again to cancel its contribution.
    position_keys[0] ^= PIECE_SQUARE_HASH_KEYS[piece, square]
    position_state[STATE_MIDDLEGAME_SCORE] -= MIDDLEGAME_PIECE_SQUARE_SCORES[piece, square]
    position_state[STATE_ENDGAME_SCORE] -= ENDGAME_PIECE_SQUARE_SCORES[piece, square]
    position_state[STATE_GAME_PHASE] -= GAME_PHASE_BY_PIECE[piece]


@njit(cache=False)
def _move_piece(bitboards, mailbox, position_state, position_keys, from_square, to_square):
    """Relocate one piece and update its two square-dependent contributions."""
    piece = mailbox[from_square]
    # OR marks exactly the two squares whose occupancy changes.
    changed_squares = square_mask(from_square) | square_mask(to_square)
    bitboards[piece] ^= changed_squares  # XOR empties the occupied origin and fills the empty destination.
    # Apply the same two toggles to the color occupancy.
    bitboards[WHITE_OCCUPANCY + (piece > 6)] ^= changed_squares
    mailbox[from_square] = 0
    mailbox[to_square] = piece
    # XOR cancels the origin key and adds the destination key.
    position_keys[0] ^= PIECE_SQUARE_HASH_KEYS[piece, from_square] ^ PIECE_SQUARE_HASH_KEYS[piece, to_square]
    position_state[STATE_MIDDLEGAME_SCORE] += (
        MIDDLEGAME_PIECE_SQUARE_SCORES[piece, to_square] - MIDDLEGAME_PIECE_SQUARE_SCORES[piece, from_square]
    )
    position_state[STATE_ENDGAME_SCORE] += (
        ENDGAME_PIECE_SQUARE_SCORES[piece, to_square] - ENDGAME_PIECE_SQUARE_SCORES[piece, from_square]
    )


@njit(cache=False)
def _king_safe_after_en_passant(
    bitboards, moving_side, king_location, occupancy, from_square, captured_square, en_passant_square
):
    """Test whether this particular en passant capture leaves the king safe.

    The moving pawn leaves its file and the victim disappears from a neighboring file.
    Simulating both changes can reveal a distant rook or bishop attack that a test of
    the destination square alone would miss. Both FEN loading and move generation use this rule.
    """
    enemy_offset = (1 - moving_side) * 6
    # XOR clears both occupied pawn squares. OR places the capturing pawn on the empty target.
    occupancy_after = (occupancy ^ square_mask(from_square) ^ square_mask(captured_square)) | square_mask(
        en_passant_square
    )
    # Join enemy bishops and queens. AND finds any now visible along the king’s diagonals.
    if bishop_attacks(king_location, occupancy_after) & (
        bitboards[enemy_offset + BISHOP] | bitboards[enemy_offset + QUEEN]
    ):
        return False
    # Join enemy rooks and queens. AND finds any now visible on its rank or file.
    if rook_attacks(king_location, occupancy_after) & (
        bitboards[enemy_offset + ROOK] | bitboards[enemy_offset + QUEEN]
    ):
        return False
    # Mask out the captured pawn with AND-NOT, then intersect with pawn origins that attack the king.
    if PAWN_ATTACKS[moving_side, king_location] & (
        bitboards[enemy_offset + PAWN] & ~square_mask(captured_square)
    ):
        return False
    # An empty intersection means no enemy knight attacks the king.
    return (KNIGHT_ATTACKS[king_location] & bitboards[enemy_offset + KNIGHT]) == 0


@njit(cache=False)
def has_legal_en_passant_capture(bitboards, en_passant_square, moving_side):
    """Keep an en passant right only if at least one pawn can exercise it with its king safe."""
    opponent = 1 - moving_side
    captured_square = en_passant_square - 8 if moving_side == WHITE else en_passant_square + 8
    # Reverse the pawn attack and intersect with our pawns to find candidate origins.
    pawns = PAWN_ATTACKS[opponent, en_passant_square] & bitboards[moving_side * 6 + PAWN]
    if pawns == 0:
        return False
    # Union both colors to simulate blockers after the capture.
    occupancy = bitboards[WHITE_OCCUPANCY] | bitboards[BLACK_OCCUPANCY]
    king_location = king_square(bitboards, moving_side)
    while pawns:
        from_square = lowest_set_square(pawns)
        pawns = _without_lowest_square(pawns)
        if _king_safe_after_en_passant(
            bitboards, moving_side, king_location, occupancy, from_square, captured_square, en_passant_square
        ):
            return True
    return False


@njit(cache=False)
def make_move(
    bitboards, mailbox, position_state, position_keys, position_history, undo_stack, previous_keys, ply, move
):
    """Play one generated move on the shared board and record how to reverse it.

    The invariant after make/unmake is that the mailbox, bitboards, score and hash all
    describe the same board. A capture overwrites its victim, so save the captured piece as well as
    the parent state and hash. En passant removes a pawn from a different square. Castling
    also moves a rook. Write the child key to history only after every change is complete.
    """
    from_square, to_square, flag = decode_move(move)
    moving_side = position_state[STATE_SIDE_TO_MOVE]
    opponent = 1 - moving_side
    piece = mailbox[from_square]
    captured = 0
    undo_stack[ply, UNDO_CASTLING_RIGHTS] = position_state[STATE_CASTLING_RIGHTS]
    undo_stack[ply, UNDO_EN_PASSANT_SQUARE] = position_state[STATE_EN_PASSANT_SQUARE]
    undo_stack[ply, UNDO_HALFMOVE_CLOCK] = position_state[STATE_HALFMOVE_CLOCK]
    undo_stack[ply, UNDO_ENCODED_MOVE] = move
    undo_stack[ply, UNDO_MIDDLEGAME_SCORE] = position_state[STATE_MIDDLEGAME_SCORE]
    undo_stack[ply, UNDO_ENDGAME_SCORE] = position_state[STATE_ENDGAME_SCORE]
    undo_stack[ply, UNDO_GAME_PHASE] = position_state[STATE_GAME_PHASE]
    previous_keys[ply] = position_keys[0]
    key = position_keys[0]
    if position_state[STATE_EN_PASSANT_SQUARE] >= 0:
        # The low three square bits identify the file. XOR removes its old en passant hash contribution.
        key ^= EN_PASSANT_FILE_HASH_KEYS[position_state[STATE_EN_PASSANT_SQUARE] & 7]
    # Cancel the hash of the old castling rights before applying the move.
    key ^= CASTLING_HASH_KEYS[position_state[STATE_CASTLING_RIGHTS]]
    position_keys[0] = key
    halfmove_clock = position_state[STATE_HALFMOVE_CLOCK] + 1
    if flag == EN_PASSANT_CAPTURE_FLAG:
        captured_square = to_square - 8 if moving_side == WHITE else to_square + 8
        captured = mailbox[captured_square]
        _remove_piece(bitboards, mailbox, position_state, position_keys, captured_square)
        _move_piece(bitboards, mailbox, position_state, position_keys, from_square, to_square)
        halfmove_clock = 0
    else:
        if flag & CAPTURE_FLAG:  # Test the capture bit, shared by ordinary and promotion captures.
            captured = mailbox[to_square]
            _remove_piece(bitboards, mailbox, position_state, position_keys, to_square)
            halfmove_clock = 0
        if flag & PROMOTION_FLAG:  # Test the promotion bit, shared by all four promoted pieces.
            _remove_piece(bitboards, mailbox, position_state, position_keys, from_square)
            _add_piece(
                bitboards,
                mailbox,
                position_state,
                position_keys,
                to_square,
                moving_side * 6 + ((flag & 3) + KNIGHT),  # Low two bits choose knight through queen.
            )
            halfmove_clock = 0
        else:
            _move_piece(bitboards, mailbox, position_state, position_keys, from_square, to_square)
            if piece == WHITE_PAWN or piece == BLACK_PAWN:
                halfmove_clock = 0
        if flag == KINGSIDE_CASTLE_FLAG:
            _move_piece(bitboards, mailbox, position_state, position_keys, from_square + 3, from_square + 1)
        elif flag == QUEENSIDE_CASTLE_FLAG:
            _move_piece(bitboards, mailbox, position_state, position_keys, from_square - 4, from_square - 1)
    undo_stack[ply, UNDO_CAPTURED_PIECE] = captured
    # Intersect existing rights with those surviving a move from the origin and onto the destination.
    castling_rights = (
        position_state[STATE_CASTLING_RIGHTS]
        & CASTLING_RIGHTS_RETAINED_BY_SQUARE[from_square]
        & CASTLING_RIGHTS_RETAINED_BY_SQUARE[to_square]
    )
    position_state[STATE_CASTLING_RIGHTS] = castling_rights
    en_passant_square = -1
    if flag == DOUBLE_PAWN_PUSH_FLAG:
        candidate_square = (from_square + to_square) // 2  # The pawn crossed the midpoint.
        if has_legal_en_passant_capture(bitboards, candidate_square, opponent):
            en_passant_square = candidate_square
    position_state[STATE_EN_PASSANT_SQUARE] = en_passant_square
    position_state[STATE_HALFMOVE_CLOCK] = halfmove_clock
    position_state[STATE_SIDE_TO_MOVE] = opponent
    position_state[STATE_GAME_PLY] += 1
    # XOR adds the new castling state and toggles the side-to-move key.
    key = position_keys[0] ^ CASTLING_HASH_KEYS[castling_rights] ^ SIDE_TO_MOVE_HASH_KEY
    if en_passant_square >= 0:
        # Use the low three bits for the file and XOR in the newly available en passant right.
        key ^= EN_PASSANT_FILE_HASH_KEYS[en_passant_square & 7]
    position_keys[0] = key
    history_length = position_state[STATE_HISTORY_LENGTH]
    position_history[history_length] = key
    position_state[STATE_HISTORY_LENGTH] = history_length + 1


@njit(cache=False)
def unmake_move(
    bitboards, mailbox, position_state, position_keys, position_history, undo_stack, previous_keys, ply
):
    """Reverse the piece changes and restore the saved scalar state and parent key."""
    move = undo_stack[ply, UNDO_ENCODED_MOVE]
    from_square, to_square, flag = decode_move(move)
    opponent = position_state[STATE_SIDE_TO_MOVE]
    moving_side = 1 - opponent
    captured = undo_stack[ply, UNDO_CAPTURED_PIECE]
    # Reverse the piece placements first. The undo row holds the original scores and hash.
    if flag & PROMOTION_FLAG:  # The promotion bit tells undo to put a pawn back on its starting square.
        piece = mailbox[to_square]
        changed_squares = square_mask(to_square)
        bitboards[piece] ^= changed_squares  # Toggle off the promoted piece at the destination.
        # Clear the same occupied bit from the mover’s color set.
        bitboards[WHITE_OCCUPANCY + moving_side] ^= changed_squares
        pawn = moving_side * 6 + PAWN
        bitboards[pawn] |= square_mask(from_square)  # Set the pawn’s original square again.
        # Restore that square to the mover’s occupancy.
        bitboards[WHITE_OCCUPANCY + moving_side] |= square_mask(from_square)
        mailbox[from_square] = pawn
        mailbox[to_square] = 0
    else:
        piece = mailbox[to_square]
        # OR the two endpoints into a single undo mask.
        changed_squares = square_mask(from_square) | square_mask(to_square)
        bitboards[piece] ^= changed_squares  # XOR moves the piece back by toggling both endpoints.
        # Apply those two toggles to the mover’s occupancy too.
        bitboards[WHITE_OCCUPANCY + moving_side] ^= changed_squares
        mailbox[from_square] = piece
        mailbox[to_square] = 0
        if flag == KINGSIDE_CASTLE_FLAG:
            rook = moving_side * 6 + ROOK
            # Mark the rook’s two castling squares (h-file and f-file).
            rook_squares = square_mask(from_square + 3) | square_mask(from_square + 1)
            bitboards[rook] ^= rook_squares  # Toggle the rook back from the f-file to the h-file.
            # Mirror the rook’s two toggles in the color occupancy.
            bitboards[WHITE_OCCUPANCY + moving_side] ^= rook_squares
            mailbox[from_square + 1] = 0
            mailbox[from_square + 3] = rook
        elif flag == QUEENSIDE_CASTLE_FLAG:
            rook = moving_side * 6 + ROOK
            # Mark the rook’s two castling squares (a-file and d-file).
            rook_squares = square_mask(from_square - 4) | square_mask(from_square - 1)
            bitboards[rook] ^= rook_squares  # Toggle the rook back from the d-file to the a-file.
            # Mirror the rook’s two toggles in the color occupancy.
            bitboards[WHITE_OCCUPANCY + moving_side] ^= rook_squares
            mailbox[from_square - 1] = 0
            mailbox[from_square - 4] = rook
    if captured:
        if flag == EN_PASSANT_CAPTURE_FLAG:
            captured_square = to_square - 8 if moving_side == WHITE else to_square + 8
        else:
            captured_square = to_square
        captured_mask = square_mask(captured_square)
        bitboards[captured] |= captured_mask  # Set the captured piece’s bit on its original square.
        # Set the same bit in the opponent’s occupancy.
        bitboards[WHITE_OCCUPANCY + opponent] |= captured_mask
        mailbox[captured_square] = captured
    position_state[STATE_CASTLING_RIGHTS] = undo_stack[ply, UNDO_CASTLING_RIGHTS]
    position_state[STATE_EN_PASSANT_SQUARE] = undo_stack[ply, UNDO_EN_PASSANT_SQUARE]
    position_state[STATE_HALFMOVE_CLOCK] = undo_stack[ply, UNDO_HALFMOVE_CLOCK]
    position_state[STATE_MIDDLEGAME_SCORE] = undo_stack[ply, UNDO_MIDDLEGAME_SCORE]
    position_state[STATE_ENDGAME_SCORE] = undo_stack[ply, UNDO_ENDGAME_SCORE]
    position_state[STATE_GAME_PHASE] = undo_stack[ply, UNDO_GAME_PHASE]
    position_state[STATE_SIDE_TO_MOVE] = moving_side
    position_state[STATE_GAME_PLY] -= 1
    position_state[STATE_HISTORY_LENGTH] -= 1
    position_keys[0] = previous_keys[ply]


@njit(cache=False)
def make_null_move(position_state, position_keys, position_history, undo_stack, previous_keys, ply):
    """Pass the turn for a search probe, clearing en passant without moving a piece."""
    undo_stack[ply, UNDO_EN_PASSANT_SQUARE] = position_state[STATE_EN_PASSANT_SQUARE]
    undo_stack[ply, UNDO_HALFMOVE_CLOCK] = position_state[STATE_HALFMOVE_CLOCK]
    undo_stack[ply, UNDO_CAPTURED_PIECE] = 0
    undo_stack[ply, UNDO_ENCODED_MOVE] = 0  # move 0 marks a null transition for incremental evaluators
    previous_keys[ply] = position_keys[0]
    key = position_keys[0] ^ SIDE_TO_MOVE_HASH_KEY  # Toggle the turn hash for the hypothetical pass.
    if position_state[STATE_EN_PASSANT_SQUARE] >= 0:
        # Keep the file bits and XOR away the en passant key that expires on the pass.
        key ^= EN_PASSANT_FILE_HASH_KEYS[position_state[STATE_EN_PASSANT_SQUARE] & 7]
    position_state[STATE_EN_PASSANT_SQUARE] = -1
    position_state[STATE_SIDE_TO_MOVE] = 1 - position_state[STATE_SIDE_TO_MOVE]
    position_state[STATE_GAME_PLY] += 1
    position_keys[0] = key
    history_length = position_state[STATE_HISTORY_LENGTH]
    position_history[history_length] = key
    position_state[STATE_HISTORY_LENGTH] = history_length + 1


@njit(cache=False)
def unmake_null_move(position_state, position_keys, position_history, undo_stack, previous_keys, ply):
    """Restore the metadata and key saved for a hypothetical pass."""
    position_state[STATE_EN_PASSANT_SQUARE] = undo_stack[ply, UNDO_EN_PASSANT_SQUARE]
    position_state[STATE_HALFMOVE_CLOCK] = undo_stack[ply, UNDO_HALFMOVE_CLOCK]
    position_state[STATE_SIDE_TO_MOVE] = 1 - position_state[STATE_SIDE_TO_MOVE]
    position_state[STATE_GAME_PLY] -= 1
    position_state[STATE_HISTORY_LENGTH] -= 1
    position_keys[0] = previous_keys[ply]


@njit(cache=False)
def _append_encoded_move(moves, move_count, from_square, to_square, flag):
    """Append a packed move to the caller’s buffer and return the next free index."""
    moves[move_count] = encode_move(from_square, to_square, flag)
    return move_count + 1


@njit(cache=False)
def generate_legal_moves(bitboards, mailbox, position_state, moves, mode):
    """Write legal moves into the caller’s buffer and return the count.

    Tactical generation includes ordinary captures, en passant and queen promotions.
    Quiet generation includes underpromotions, even when they capture. Generate king
    moves first because only a king move can escape double check. For the remaining
    pieces restrict destinations to moves that answer the check and respect any pin.
    A perft check counts paths with this generator and make/unmake. The initial
    position has 8,902 legal paths of exactly three plies.
    """
    moving_side = position_state[STATE_SIDE_TO_MOVE]
    opponent = 1 - moving_side
    friendly_offset = moving_side * 6
    enemy_offset = opponent * 6
    friendly_occupancy = bitboards[WHITE_OCCUPANCY + moving_side]
    enemy_occupancy = bitboards[WHITE_OCCUPANCY + opponent]
    occupancy = friendly_occupancy | enemy_occupancy  # Union the two sides to find all blockers.
    king_location = lowest_set_square(bitboards[friendly_offset + KING])
    move_count = 0
    # Intersect all attackers with enemy occupancy to find the pieces giving check.
    checkers = attackers_of_square(bitboards, king_location, occupancy) & enemy_occupancy
    # Restrict destinations before enumerating pieces. A check narrows these masks again below.
    if mode == GENERATE_TACTICAL_MOVES:
        capture_targets = enemy_occupancy
        quiet_targets = np.uint64(0)
    elif mode == GENERATE_QUIET_MOVES:
        capture_targets = np.uint64(0)
        quiet_targets = ~occupancy  # Invert occupied bits to obtain every empty square.
    else:
        capture_targets = enemy_occupancy
        quiet_targets = ~occupancy  # Empty squares are the 64-bit complement of occupancy.
    # king moves
    # Exclude friendly occupancy from the king’s adjacent squares.
    king_targets = KING_ATTACKS[king_location] & ~friendly_occupancy
    # Toggle off the king at its origin so it cannot block a ray behind itself.
    occupancy_without_king = occupancy ^ square_mask(king_location)
    while king_targets:
        to_square = lowest_set_square(king_targets)
        king_targets = _without_lowest_square(king_targets)
        is_capture = _contains_square(enemy_occupancy, to_square)
        if is_capture:
            if capture_targets == 0:
                continue
        elif quiet_targets == 0:
            continue
        if not is_square_attacked_by(bitboards, to_square, opponent, occupancy_without_king):
            move_count = _append_encoded_move(
                moves, move_count, king_location, to_square, CAPTURE_FLAG if is_capture else QUIET_MOVE_FLAG
            )
    if checkers and _without_lowest_square(checkers):
        return move_count  # double check: only king moves
    if checkers:
        checker_square = lowest_set_square(checkers)
        # OR gives two ways to answer a check: block the ray or capture the checker.
        evasion_targets = SQUARES_BETWEEN[king_location, checker_square] | checkers
    else:
        evasion_targets = ALL_SQUARES
    capture_targets &= evasion_targets  # Keep only captures that also answer the check.
    quiet_targets &= evasion_targets  # Keep only quiet moves that also answer the check.
    # Absolute pin: moving the only friendly blocker off this line would expose our king to check.
    pinned = np.uint64(0)
    # Union the enemy slider types per ray, intersect each with its ray, then join the two sets of possible pins.
    potential_pinners = (
        EMPTY_BOARD_ROOK_RAYS[king_location]
        & (bitboards[enemy_offset + ROOK] | bitboards[enemy_offset + QUEEN])
    ) | (
        EMPTY_BOARD_BISHOP_RAYS[king_location]
        & (bitboards[enemy_offset + BISHOP] | bitboards[enemy_offset + QUEEN])
    )
    while potential_pinners:
        slider_square = lowest_set_square(potential_pinners)
        potential_pinners = _without_lowest_square(potential_pinners)
        # Intersect the ray’s interior with occupancy to find interposed pieces.
        blockers = SQUARES_BETWEEN[king_location, slider_square] & occupancy
        # After removing the lowest blocker there must be none left. AND checks the lone blocker is friendly.
        if blockers and not _without_lowest_square(blockers) and (blockers & friendly_occupancy):
            pinned |= blockers  # Add the lone friendly blocker to the set of pinned pieces.
    # knights (pinned knights never move)
    # Exclude pinned knights because a knight jump always leaves the pin line.
    piece_squares = bitboards[friendly_offset + KNIGHT] & ~pinned
    while piece_squares:
        from_square = lowest_set_square(piece_squares)
        piece_squares = _without_lowest_square(piece_squares)
        attacks = KNIGHT_ATTACKS[from_square]
        destinations = attacks & capture_targets  # Keep knight attacks that land on eligible capture squares.
        while destinations:
            to_square = lowest_set_square(destinations)
            destinations = _without_lowest_square(destinations)
            move_count = _append_encoded_move(moves, move_count, from_square, to_square, CAPTURE_FLAG)
        destinations = attacks & quiet_targets  # Keep knight attacks that land on eligible empty squares.
        while destinations:
            to_square = lowest_set_square(destinations)
            destinations = _without_lowest_square(destinations)
            move_count = _append_encoded_move(moves, move_count, from_square, to_square, QUIET_MOVE_FLAG)
    # sliders
    for piece_type in range(BISHOP, QUEEN + 1):
        piece_squares = bitboards[friendly_offset + piece_type]
        while piece_squares:
            from_square = lowest_set_square(piece_squares)
            piece_squares = _without_lowest_square(piece_squares)
            if piece_type == BISHOP:
                attacks = bishop_attacks(from_square, occupancy)
            elif piece_type == ROOK:
                attacks = rook_attacks(from_square, occupancy)
            else:
                attacks = queen_attacks(from_square, occupancy)
            if _contains_square(pinned, from_square):
                # A pinned slider may move only along the line through its king.
                attacks &= ALIGNED_SQUARES[king_location, from_square]
            # Keep reachable slider squares that are eligible captures.
            destinations = attacks & capture_targets
            while destinations:
                to_square = lowest_set_square(destinations)
                destinations = _without_lowest_square(destinations)
                move_count = _append_encoded_move(moves, move_count, from_square, to_square, CAPTURE_FLAG)
            # Keep reachable slider squares that are eligible quiet moves.
            destinations = attacks & quiet_targets
            while destinations:
                to_square = lowest_set_square(destinations)
                destinations = _without_lowest_square(destinations)
                move_count = _append_encoded_move(moves, move_count, from_square, to_square, QUIET_MOVE_FLAG)
    # pawns
    pawns = bitboards[friendly_offset + PAWN]
    if moving_side == WHITE:
        pawn_step = 8
        promotion_rank = np.uint64(0x00FF000000000000)  # promotion source rank
        double_push_rank = np.uint64(0x000000000000FF00)  # double-push source rank
    else:
        pawn_step = -8
        promotion_rank = np.uint64(0x000000000000FF00)
        double_push_rank = np.uint64(0x00FF000000000000)
    empty = ~occupancy  # Complement all occupied bits to get empty pawn-push destinations.
    while pawns:
        from_square = lowest_set_square(pawns)
        pawns = _without_lowest_square(pawns)
        from_mask = square_mask(from_square)
        # An intersection means this pawn is pinned, so restrict its moves to the king’s line.
        line = ALIGNED_SQUARES[king_location, from_square] if (pinned & from_mask) else ALL_SQUARES
        # Intersect the origin bit with the promotion source rank to detect a promotion move.
        promotes = (from_mask & promotion_rank) != 0
        # pushes
        to_square = from_square + pawn_step
        if _contains_square(empty, to_square) and _contains_square(line, to_square):
            if promotes:
                if _contains_square(evasion_targets, to_square):
                    if mode != GENERATE_QUIET_MOVES:
                        move_count = _append_encoded_move(
                            moves, move_count, from_square, to_square, PROMOTE_TO_QUEEN
                        )
                    if mode != GENERATE_TACTICAL_MOVES:
                        move_count = _append_encoded_move(
                            moves, move_count, from_square, to_square, PROMOTE_TO_ROOK
                        )
                        move_count = _append_encoded_move(
                            moves, move_count, from_square, to_square, PROMOTE_TO_BISHOP
                        )
                        move_count = _append_encoded_move(
                            moves, move_count, from_square, to_square, PROMOTE_TO_KNIGHT
                        )
            else:
                if _contains_square(quiet_targets, to_square):
                    move_count = _append_encoded_move(
                        moves, move_count, from_square, to_square, QUIET_MOVE_FLAG
                    )
                # A double push must start on the pawn’s original rank.
                if (from_mask & double_push_rank) and mode != GENERATE_TACTICAL_MOVES:
                    double_push_square = to_square + pawn_step
                    if _contains_square(empty, double_push_square) and _contains_square(
                        quiet_targets, double_push_square
                    ):
                        move_count = _append_encoded_move(
                            moves, move_count, from_square, double_push_square, DOUBLE_PAWN_PUSH_FLAG
                        )
        # captures
        if mode != GENERATE_QUIET_MOVES or promotes:
            # Intersect attacked squares with enemy occupants, check evasions and the permitted pin line.
            pawn_captures = PAWN_ATTACKS[moving_side, from_square] & enemy_occupancy & evasion_targets & line
            while pawn_captures:
                to_square = lowest_set_square(pawn_captures)
                pawn_captures = _without_lowest_square(pawn_captures)
                if promotes:
                    if mode != GENERATE_QUIET_MOVES:
                        move_count = _append_encoded_move(
                            moves, move_count, from_square, to_square, CAPTURE_AND_PROMOTE_TO_QUEEN
                        )
                    if mode != GENERATE_TACTICAL_MOVES:
                        move_count = _append_encoded_move(
                            moves, move_count, from_square, to_square, CAPTURE_AND_PROMOTE_TO_ROOK
                        )
                        move_count = _append_encoded_move(
                            moves, move_count, from_square, to_square, CAPTURE_AND_PROMOTE_TO_BISHOP
                        )
                        move_count = _append_encoded_move(
                            moves, move_count, from_square, to_square, CAPTURE_AND_PROMOTE_TO_KNIGHT
                        )
                else:
                    move_count = _append_encoded_move(moves, move_count, from_square, to_square, CAPTURE_FLAG)
    # en passant (validated by occupancy simulation, covers pins and checks)
    en_passant_square = position_state[STATE_EN_PASSANT_SQUARE]
    if en_passant_square >= 0 and mode != GENERATE_QUIET_MOVES:
        captured_square = en_passant_square - 8 if moving_side == WHITE else en_passant_square + 8
        # Intersect the reversed attack pattern with our pawn set to find possible en passant captures.
        candidate_pawns = PAWN_ATTACKS[opponent, en_passant_square] & bitboards[friendly_offset + PAWN]
        while candidate_pawns:
            from_square = lowest_set_square(candidate_pawns)
            candidate_pawns = _without_lowest_square(candidate_pawns)
            if not _king_safe_after_en_passant(
                bitboards,
                moving_side,
                king_location,
                occupancy,
                from_square,
                captured_square,
                en_passant_square,
            ):
                continue
            move_count = _append_encoded_move(
                moves, move_count, from_square, en_passant_square, EN_PASSANT_CAPTURE_FLAG
            )
    # Castling also requires safe transit and destination squares.
    if checkers == 0 and mode != GENERATE_TACTICAL_MOVES:
        rights = position_state[STATE_CASTLING_RIGHTS]
        if moving_side == WHITE:
            # Test White’s kingside right and require zero occupied bits on f1 and g1.
            if (rights & WHITE_KINGSIDE_RIGHT) and (occupancy & np.uint64(0x60)) == 0:
                if not is_square_attacked_by(bitboards, 5, opponent, occupancy) and not is_square_attacked_by(
                    bitboards, 6, opponent, occupancy
                ):
                    move_count = _append_encoded_move(moves, move_count, 4, 6, KINGSIDE_CASTLE_FLAG)
            # Test White’s queenside right and require b1, c1 and d1 to be empty.
            if (rights & WHITE_QUEENSIDE_RIGHT) and (occupancy & np.uint64(0x0E)) == 0:
                if not is_square_attacked_by(bitboards, 3, opponent, occupancy) and not is_square_attacked_by(
                    bitboards, 2, opponent, occupancy
                ):
                    move_count = _append_encoded_move(moves, move_count, 4, 2, QUEENSIDE_CASTLE_FLAG)
        else:
            # Test Black’s kingside right and require f8 and g8 to be empty.
            if (rights & BLACK_KINGSIDE_RIGHT) and (occupancy & np.uint64(0x6000000000000000)) == 0:
                if not is_square_attacked_by(
                    bitboards, 61, opponent, occupancy
                ) and not is_square_attacked_by(bitboards, 62, opponent, occupancy):
                    move_count = _append_encoded_move(moves, move_count, 60, 62, KINGSIDE_CASTLE_FLAG)
            # Test Black’s queenside right and require b8, c8 and d8 to be empty.
            if (rights & BLACK_QUEENSIDE_RIGHT) and (occupancy & np.uint64(0x0E00000000000000)) == 0:
                if not is_square_attacked_by(
                    bitboards, 59, opponent, occupancy
                ) and not is_square_attacked_by(bitboards, 58, opponent, occupancy):
                    move_count = _append_encoded_move(moves, move_count, 60, 58, QUEENSIDE_CASTLE_FLAG)
    return move_count


@njit(cache=False)
def compute_position_key(bitboards, mailbox, position_state):
    """Recompute the 64-bit Zobrist key from the board rather than the move history.

    XOR combines one key per active feature. For example, moving a knight from b1 to c3
    cancels its b1 key and adds its c3 key. The incremental path can therefore update a
    move without scanning 64 squares. This full rebuild supplies a reference when loading a FEN.

    Turn, castling rights and a legally usable en passant file can change possible moves.
    Clocks and repetition counts are tracked separately. Different positions can share a
    64-bit key, so a hash match is a fast identity check with a small collision risk.
    """
    key = np.uint64(0)
    for square in range(64):
        piece = mailbox[square]
        if piece:
            # XOR in exactly one hash contribution for each occupied piece-square pair.
            key ^= PIECE_SQUARE_HASH_KEYS[piece, square]
    if position_state[STATE_SIDE_TO_MOVE] == BLACK:
        key ^= SIDE_TO_MOVE_HASH_KEY  # Toggle the turn key in positions with Black to move.
    # Mix the key for this exact combination of castling rights.
    key ^= CASTLING_HASH_KEYS[position_state[STATE_CASTLING_RIGHTS]]
    if position_state[STATE_EN_PASSANT_SQUARE] >= 0:
        # Mask to the file index and include its key so a legal en passant right changes position identity.
        key ^= EN_PASSANT_FILE_HASH_KEYS[position_state[STATE_EN_PASSANT_SQUARE] & 7]
    return key


def allocate_board_state():
    """Allocate one game’s board, redundant occupancy views and per-ply undo buffers.

    The mailbox answers “what is on this square?” while bitboards answer
    “which squares contain these pieces?” Make and unmake keep both in sync.
    """
    bitboards = np.zeros(15, dtype=np.uint64)
    mailbox = np.zeros(64, dtype=np.int8)
    position_state = np.zeros(16, dtype=np.int64)
    position_keys = np.zeros(4, dtype=np.uint64)
    position_history = np.zeros(POSITION_HISTORY_CAPACITY, dtype=np.uint64)
    undo_stack = np.zeros((MAX_SEARCH_PLY + 2, 8), dtype=np.int64)
    previous_keys = np.zeros(MAX_SEARCH_PLY + 2, dtype=np.uint64)
    return bitboards, mailbox, position_state, position_keys, position_history, undo_stack, previous_keys


FEN_PIECE_SYMBOLS = ".PNBRQKpnbrqk"


def load_fen(fen, bitboards, mailbox, position_state, position_keys, position_history, prior_keys=()):
    """Fill arrays from a FEN. prior_keys: keys of earlier game positions
    (oldest first) for repetition detection; the current position is appended."""
    bitboards[:] = 0
    mailbox[:] = 0
    position_state[:] = 0
    parts = fen.split()
    rows = parts[0].split("/")
    for rank, row in enumerate(rows):
        file_index = 0
        for symbol in row:
            if symbol.isdigit():
                file_index += int(symbol)
            else:
                square = (7 - rank) * 8 + file_index
                piece = FEN_PIECE_SYMBOLS.index(symbol)
                mailbox[square] = piece
                piece_bit = np.uint64(1 << square)  # Shift a 1 to the square named in the FEN.
                bitboards[piece] |= piece_bit  # Add this square to its piece set.
                bitboards[WHITE_OCCUPANCY + (piece > 6)] |= piece_bit  # Mark its color as occupying it too.
                position_state[STATE_MIDDLEGAME_SCORE] += MIDDLEGAME_PIECE_SQUARE_SCORES[piece, square]
                position_state[STATE_ENDGAME_SCORE] += ENDGAME_PIECE_SQUARE_SCORES[piece, square]
                position_state[STATE_GAME_PHASE] += GAME_PHASE_BY_PIECE[piece]
                file_index += 1
    position_state[STATE_SIDE_TO_MOVE] = WHITE if len(parts) < 2 or parts[1] == "w" else BLACK
    castling_rights = 0
    if len(parts) > 2 and parts[2] != "-":
        for symbol, right_mask, rook_location, king_location, rook_piece, king_piece in (
            ("K", WHITE_KINGSIDE_RIGHT, 7, 4, WHITE_ROOK, WHITE_KING),
            ("Q", WHITE_QUEENSIDE_RIGHT, 0, 4, WHITE_ROOK, WHITE_KING),
            ("k", BLACK_KINGSIDE_RIGHT, 63, 60, BLACK_ROOK, BLACK_KING),
            ("q", BLACK_QUEENSIDE_RIGHT, 56, 60, BLACK_ROOK, BLACK_KING),
        ):
            if (
                symbol in parts[2]
                and mailbox[rook_location] == rook_piece
                and mailbox[king_location] == king_piece
            ):
                # Set the verified castling flag alongside any rights already read.
                castling_rights |= right_mask
    position_state[STATE_CASTLING_RIGHTS] = castling_rights
    en_passant_square = -1
    if len(parts) > 3 and parts[3] != "-":
        candidate_square = (ord(parts[3][0]) - 97) + (int(parts[3][1]) - 1) * 8
        if has_legal_en_passant_capture(bitboards, candidate_square, position_state[STATE_SIDE_TO_MOVE]):
            en_passant_square = candidate_square
    position_state[STATE_EN_PASSANT_SQUARE] = en_passant_square
    position_state[STATE_HALFMOVE_CLOCK] = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1
    position_state[STATE_GAME_PLY] = 2 * (fullmove - 1) + (
        1 if position_state[STATE_SIDE_TO_MOVE] == BLACK else 0
    )
    position_keys[0] = compute_position_key(bitboards, mailbox, position_state)
    history_length = 0
    for prior_key in prior_keys:
        position_history[history_length] = prior_key
        history_length += 1
    position_history[history_length] = position_keys[0]
    position_state[STATE_HISTORY_LENGTH] = history_length + 1


def move_to_uci(move):
    """Write a move in UCI coordinate notation, such as e2e4 or a7a8q."""
    move = int(move)
    from_square, to_square, flag = decode_move(move)
    uci_move = (
        chr(97 + from_square % 8)
        + str(from_square // 8 + 1)
        + chr(97 + to_square % 8)
        + str(to_square // 8 + 1)
    )
    if flag & PROMOTION_FLAG:  # The promotion bit signals a fifth character in UCI notation.
        # The low two flag bits select the promoted piece in knight/bishop/rook/queen order.
        uci_move += "nbrq"[flag & 3]
    return uci_move
