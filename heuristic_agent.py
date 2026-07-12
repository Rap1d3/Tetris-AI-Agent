"""
heuristic_agent.py

Heuristic Tetris agent. Not a learning system by itself — it evaluates
every reachable placement of the current piece using a weighted sum of
board features, then picks the highest-scoring one. The 4 weights below
are a hand-picked starting point; genetic_trainer.py searches for better
ones automatically by playing many games and evolving these numbers.

This module only contains the "brain" (scoring) and the "hands"
(turning a decision into real game actions) — it has no opinion on what
the weights should be. Both DEFAULT_WEIGHTS and any weights found by
genetic_trainer.py are used through the exact same functions here.
"""

from tetris_logic import TetrisGame, Action

# Negative weight = penalty (higher feature value -> worse placement).
# Positive weight = bonus (higher feature value -> better placement).
DEFAULT_WEIGHTS = {
    "aggregate_height": -0.51,
    "lines_cleared": 0.76,
    "holes": -0.36,
    "bumpiness": -0.18,
}


def score_placement(placement, weights=DEFAULT_WEIGHTS):
    """
    Scores one placement (an item from game.get_possible_placements())
    as a single number — higher is better. Just a weighted sum of the
    board features plus the number of lines that placement would clear.
    """
    features = placement["features"]
    score = (
        weights["aggregate_height"] * features["aggregate_height"]
        + weights["holes"] * features["holes"]
        + weights["bumpiness"] * features["bumpiness"]
        + weights["lines_cleared"] * placement["lines_cleared"]
    )
    return score


def choose_best_placement(game, weights=DEFAULT_WEIGHTS):
    """Evaluates every possible placement of the current piece and
    returns the highest-scoring one. Read-only — does not touch the game."""
    placements = game.get_possible_placements()
    best = max(placements, key=lambda p: score_placement(p, weights))
    return best


def execute_placement(game, placement, max_moves=40):
    """
    Turns a chosen placement into real game actions: rotate, move
    left/right into position, then hard drop.

    Deliberately does NOT pre-compute a fixed action list. Instead it
    re-checks the piece's actual position after every single action,
    because rotation can shift the piece sideways on its own (see
    SIMPLE_KICKS in tetris_logic.py) — a pre-computed move count could
    become wrong the moment a kick happens.
    """
    rotations_needed = (placement["rotation"] - game.current.rotation) % 4
    for _ in range(rotations_needed):
        game.step(Action.ROTATE_CW)

    moves_made = 0
    while game.current.col != placement["col"] and moves_made < max_moves:
        if game.current.col < placement["col"]:
            game.step(Action.RIGHT)
        else:
            game.step(Action.LEFT)
        moves_made += 1

    state, reward, done, info = game.step(Action.HARD_DROP)
    return state, reward, done, info


if __name__ == "__main__":
    game = TetrisGame(seed=1)
    placements = game.get_possible_placements()
    scored = [(score_placement(p), p["rotation"], p["col"]) for p in placements]
    scored.sort(reverse=True)

    print(f"Current piece: {game.current.type}")
    print(f"Total placements: {len(scored)}")
    print("\nTop 3 (score, rotation, col):")
    for s in scored[:3]:
        print(f"  {s}")
    print("\nBottom 3 (score, rotation, col):")
    for s in scored[-3:]:
        print(f"  {s}")