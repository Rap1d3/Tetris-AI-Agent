"""
tetris_logic.py

Core Tetris engine, designed to be driven by an AI agent rather than a
human player. Contains zero rendering code — pure game state (lists,
enums, dicts) that any agent (heuristic, evolutionary, or a future
neural network) can read and act on directly.

Design principles:
- Gym-style interface: step(action) -> state, reward, done, info
- get_possible_placements() enumerates every reachable final position for
  the current piece (all rotations x all columns), which lets an agent
  choose a whole placement in one decision instead of driving the piece
  left/right/rotate one step at a time.
- Standard 7-bag randomizer, matching modern official Tetris (each of the
  7 pieces appears exactly once per bag, in random order).

Basic usage:
    game = TetrisGame(seed=42)
    state = game.reset()
    state, reward, done, info = game.step(Action.LEFT)
"""

import random
from copy import deepcopy
from enum import Enum


BOARD_WIDTH = 10
BOARD_HEIGHT = 20
# Extra hidden rows above the visible board where pieces spawn, matching
# real Tetris. Stripped out again in get_state() unless requested.
SPAWN_BUFFER = 4
TOTAL_HEIGHT = BOARD_HEIGHT + SPAWN_BUFFER


class Action(Enum):
    LEFT = "left"
    RIGHT = "right"
    ROTATE_CW = "rotate_cw"
    ROTATE_CCW = "rotate_ccw"
    SOFT_DROP = "soft_drop"
    HARD_DROP = "hard_drop"
    NOOP = "noop"


# Each piece has 4 rotation states, given as (row, col) cells inside a
# 4x4 bounding box. Standard Tetris Guideline shapes.
SHAPES = {
    "I": [
        [(1, 0), (1, 1), (1, 2), (1, 3)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(0, 1), (1, 1), (2, 1), (3, 1)],
    ],
    "O": [
        [(0, 1), (0, 2), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (1, 2)],
    ],
    "T": [
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    "S": [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 1), (1, 1), (1, 2), (2, 2)],
        [(1, 1), (1, 2), (2, 0), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 2), (1, 1), (1, 2), (2, 1)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (1, 2), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}

PIECE_TYPES = list(SHAPES.keys())
PIECE_IDS = {t: i + 1 for i, t in enumerate(PIECE_TYPES)}

# Simplified wall-kick offsets: if a rotation collides, try these
# (col, row) shifts in order before giving up. A lightweight stand-in
# for the full SRS kick table.
SIMPLE_KICKS = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]


class Piece:
    __slots__ = ("type", "rotation", "row", "col")

    def __init__(self, piece_type, rotation=0, row=0, col=3):
        self.type = piece_type
        self.rotation = rotation
        self.row = row
        self.col = col

    def cells(self, rotation=None, row=None, col=None):
        """Absolute (row, col) board cells occupied by this piece."""
        rotation = self.rotation if rotation is None else rotation
        row = self.row if row is None else row
        col = self.col if col is None else col
        shape = SHAPES[self.type][rotation % 4]
        return [(r + row, c + col) for (r, c) in shape]

    def clone(self):
        return Piece(self.type, self.rotation, self.row, self.col)


class SevenBag:
    """Standard 7-bag randomizer: each piece type appears exactly once
    per 7 pieces, shuffled within each bag."""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self._bag = []

    def next(self):
        if not self._bag:
            self._bag = PIECE_TYPES.copy()
            self.rng.shuffle(self._bag)
        return self._bag.pop()


class TetrisGame:
    """
    Main game class, exposing a Gym-style interface:

        game = TetrisGame(seed=42)
        state = game.reset()
        state, reward, done, info = game.step(Action.LEFT)

    get_state() returns a plain dict, ready to feed to an agent or use
    for heuristic evaluation.
    """

    def __init__(self, width=BOARD_WIDTH, height=BOARD_HEIGHT, seed=None,
                 preview_size=5):
        self.width = width
        self.height = height
        self.preview_size = preview_size
        self.rng = random.Random(seed)
        self.bag = SevenBag(self.rng)
        self.reset()

    def reset(self):
        self.board = [[0] * self.width for _ in range(TOTAL_HEIGHT)]
        self.next_queue = [self.bag.next() for _ in range(self.preview_size)]
        self.score = 0
        self.lines_cleared_total = 0
        self.pieces_placed = 0
        self.game_over = False
        self._spawn_next_piece()
        return self.get_state()

    def _spawn_next_piece(self):
        piece_type = self.next_queue.pop(0)
        self.next_queue.append(self.bag.next())
        spawn_col = (self.width - 4) // 2
        self.current = Piece(piece_type, rotation=0, row=0, col=spawn_col)
        if self._collision(self.current.cells()):
            self.game_over = True

    def _collision(self, cells):
        for r, c in cells:
            if c < 0 or c >= self.width:
                return True
            if r >= TOTAL_HEIGHT:
                return True
            if r >= 0 and self.board[r][c]:
                return True
        return False

    def step(self, action):
        """
        Applies one action and returns (state, reward, done, info).

        `reward` is currently always 0.0 — this is a placeholder for the
        future RL/evolutionary reward signal. `info["lines_cleared"]` and
        the board's heuristic features are the actual training signal
        used by the agents in this project right now.
        """
        if isinstance(action, str):
            action = Action(action)

        if self.game_over:
            return self.get_state(), 0.0, True, {"reason": "already_game_over"}

        reward = 0.0
        info = {"lines_cleared": 0, "locked": False}

        if action == Action.LEFT:
            self._try_move(0, -1)
        elif action == Action.RIGHT:
            self._try_move(0, 1)
        elif action == Action.SOFT_DROP:
            moved = self._try_move(1, 0)
            if not moved:
                # Piece hit the floor or another piece — lock it in place.
                lines_cleared = self._lock_piece()
                info["lines_cleared"] = lines_cleared
                info["locked"] = True
                self._score_for_lines(lines_cleared)
        elif action == Action.ROTATE_CW:
            self._try_rotate(1)
        elif action == Action.ROTATE_CCW:
            self._try_rotate(-1)
        elif action == Action.HARD_DROP:
            drop_dist = self._hard_drop_distance()
            self.current.row += drop_dist
            lines_cleared = self._lock_piece()
            info["lines_cleared"] = lines_cleared
            info["locked"] = True
            self._score_for_lines(lines_cleared)
        elif action == Action.NOOP:
            pass

        if self.game_over:
            info["reason"] = "game_over"

        return self.get_state(), reward, self.game_over, info

    def _try_move(self, d_row, d_col):
        new_cells = self.current.cells(
            row=self.current.row + d_row, col=self.current.col + d_col
        )
        if not self._collision(new_cells):
            self.current.row += d_row
            self.current.col += d_col
            return True
        return False

    def _try_rotate(self, direction):
        new_rotation = (self.current.rotation + direction) % 4
        for d_col, d_row in SIMPLE_KICKS:
            cells = self.current.cells(
                rotation=new_rotation,
                row=self.current.row + d_row,
                col=self.current.col + d_col,
            )
            if not self._collision(cells):
                self.current.rotation = new_rotation
                self.current.row += d_row
                self.current.col += d_col
                return True
        return False

    def _hard_drop_distance(self):
        dist = 0
        while not self._collision(self.current.cells(row=self.current.row + dist + 1)):
            dist += 1
        return dist

    def _lock_piece(self):
        """Locks the current piece onto the board, clears full lines,
        and spawns the next piece."""
        piece_id = PIECE_IDS[self.current.type]
        for r, c in self.current.cells():
            if 0 <= r < TOTAL_HEIGHT and 0 <= c < self.width:
                self.board[r][c] = piece_id

        lines_cleared = self._clear_lines()
        self.lines_cleared_total += lines_cleared
        self.pieces_placed += 1

        if not self.game_over:
            self._spawn_next_piece()
        return lines_cleared

    def _clear_lines(self):
        full_rows = [r for r in range(TOTAL_HEIGHT) if all(self.board[r])]
        for r in full_rows:
            del self.board[r]
            self.board.insert(0, [0] * self.width)
        return len(full_rows)

    def _score_for_lines(self, n):
        self.score += n * 40

    def get_possible_placements(self, piece_type=None):
        """
        Returns a list of every reachable final placement for a piece:
        {rotation, col, final_row, lines_cleared, resulting_board, features}.
        Operates on a copy of the board — does not mutate real game state.

        Defaults to the current piece; pass piece_type explicitly to
        evaluate a different piece (e.g. one from the next queue).
        """
        piece_type = piece_type or self.current.type
        results = []
        seen_shapes = set()

        for rotation in range(4):
            shape = tuple(SHAPES[piece_type][rotation])
            if shape in seen_shapes:
                continue  # e.g. the O piece looks identical in all 4 rotations
            seen_shapes.add(shape)

            min_c = min(c for _, c in shape)
            max_c = max(c for _, c in shape)
            for col in range(-min_c, self.width - max_c):
                test_piece = Piece(piece_type, rotation, row=0, col=col)
                if self._collision(test_piece.cells()):
                    continue
                drop = 0
                while not self._collision(test_piece.cells(row=drop + 1)):
                    drop += 1
                final_cells = test_piece.cells(row=drop)

                sim_board = deepcopy(self.board)
                piece_id = PIECE_IDS[piece_type]
                for r, c in final_cells:
                    if 0 <= r < TOTAL_HEIGHT:
                        sim_board[r][c] = piece_id
                lines_cleared, cleared_board = self._simulate_clear(sim_board)

                results.append({
                    "rotation": rotation,
                    "col": col,
                    "final_row": drop,
                    "lines_cleared": lines_cleared,
                    "resulting_board": cleared_board,
                    "features": compute_features(cleared_board, self.width),
                })
        return results

    @staticmethod
    def _simulate_clear(board):
        full_rows = [r for r in range(len(board)) if all(board[r])]
        n = len(full_rows)
        if n == 0:
            return 0, board
        new_board = [row for i, row in enumerate(board) if i not in full_rows]
        empty_rows = [[0] * len(board[0]) for _ in range(n)]
        return n, empty_rows + new_board

    def get_state(self, visible_only=True):
        """
        Returns the full game state as a plain dict:
        board, current_piece, next_queue, score, lines_cleared_total,
        pieces_placed, game_over. Set visible_only=False to include the
        hidden spawn buffer rows above the visible board.
        """
        if visible_only:
            board = [row[:] for row in self.board[SPAWN_BUFFER:]]
        else:
            board = [row[:] for row in self.board]

        return {
            "board": board,
            "current_piece": {
                "type": self.current.type,
                "rotation": self.current.rotation,
                "row": self.current.row - (SPAWN_BUFFER if visible_only else 0),
                "col": self.current.col,
                "cells": [
                    (r - (SPAWN_BUFFER if visible_only else 0), c)
                    for r, c in self.current.cells()
                ],
            },
            "next_queue": list(self.next_queue),
            "score": self.score,
            "lines_cleared_total": self.lines_cleared_total,
            "pieces_placed": self.pieces_placed,
            "game_over": self.game_over,
        }

    def __repr__(self):
        s = self.get_state()
        return (f"<TetrisGame score={s['score']} lines={s['lines_cleared_total']} "
                f"pieces={s['pieces_placed']} game_over={s['game_over']}>")


def compute_features(board, width):
    """
    Computes standard heuristic features used by most Tetris heuristic
    agents:
      - column_heights: height of each column
      - aggregate_height: sum of all column heights
      - holes: empty cells covered by a filled cell above them
      - bumpiness: total height difference between adjacent columns
      - max_height: tallest column
    """
    height = len(board)
    column_heights = [0] * width
    holes = 0

    for c in range(width):
        block_found = False
        for r in range(height):
            if board[r][c]:
                if not block_found:
                    column_heights[c] = height - r
                    block_found = True
            else:
                if block_found:
                    holes += 1

    aggregate_height = sum(column_heights)
    bumpiness = sum(
        abs(column_heights[i] - column_heights[i + 1]) for i in range(width - 1)
    )
    max_height = max(column_heights) if column_heights else 0

    return {
        "column_heights": column_heights,
        "aggregate_height": aggregate_height,
        "holes": holes,
        "bumpiness": bumpiness,
        "max_height": max_height,
    }


def render_state_ascii(state):
    """Quick ASCII rendering of a state dict, useful for console debugging."""
    board = [row[:] for row in state["board"]]
    for r, c in state["current_piece"]["cells"]:
        if 0 <= r < len(board) and 0 <= c < len(board[0]):
            board[r][c] = 8

    symbols = {0: ".", 8: "@"}
    lines = []
    lines.append("+" + "-" * (len(board[0]) * 2) + "+")
    for row in board:
        line = "|" + "".join(symbols.get(v, "#") + " " for v in row) + "|"
        lines.append(line)
    lines.append("+" + "-" * (len(board[0]) * 2) + "+")
    lines.append(f"Score: {state['score']}  Lines: {state['lines_cleared_total']}  "
                 f"Next: {state['next_queue'][:3]}")
    return "\n".join(lines)


if __name__ == "__main__":
    game = TetrisGame(seed=1)
    print(render_state_ascii(game.get_state()))
    print("\nPossible placements for current piece:", len(game.get_possible_placements()))