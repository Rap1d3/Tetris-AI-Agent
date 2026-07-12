"""
agent_watch.py

Watches the heuristic agent play live, in the same pygame window as
run.py — except decisions come from choose_best_placement() in
heuristic_agent.py instead of the keyboard.

Loads weights found by genetic_trainer.py / population_watch.py
(training_state.json) if available. If that file is missing (e.g. a
fresh clone of this repo with no training history), the agent gets a
brand new set of RANDOM, untrained weights instead — so out of the box
it plays as badly as generation 0 of the genetic algorithm, and only
plays well once you've actually trained it yourself.

Run:
    python3 agent_watch.py
"""

import sys
import time
import random
import pygame

from tetris_logic import TetrisGame, Action
from heuristic_agent import choose_best_placement
from genetic_trainer import load_training_state, SAVE_FILE, random_weights
from run import draw_board, SCREEN_W, SCREEN_H

# Pause between individual agent actions (rotate, move, drop), in seconds.
# Without it every move would happen instantly and be impossible to follow.
STEP_DELAY = 0.12


def load_weights_to_use():
    """Loads trained weights from SAVE_FILE if present. Otherwise
    generates a fresh random set of weights (untrained — deliberately
    plays badly) instead of falling back to any hand-picked numbers."""
    state = load_training_state(SAVE_FILE)
    if state is not None and state.get("best_weights"):
        print(f"Loaded trained weights from {SAVE_FILE} "
              f"(best fitness={state['best_fitness']})")
        return state["best_weights"]

    weights = random_weights(random.Random())
    print(f"{SAVE_FILE} not found — no training history, playing with "
          f"RANDOM untrained weights: {weights}")
    return weights


WEIGHTS = load_weights_to_use()


def render_frame(game, screen, font):
    draw_board(screen, game, font)
    # Keeps the window responsive to the close button while the agent
    # is mid-move, since we don't hit the normal event loop until later.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


def wait(clock):
    """Sleeps for STEP_DELAY seconds via clock.tick() rather than
    time.sleep(), so the OS doesn't flag the window as unresponsive."""
    waited = 0.0
    while waited < STEP_DELAY:
        dt = clock.tick(60) / 1000.0
        waited += dt


def agent_step_with_render(game, screen, font, clock):
    """
    One agent turn, driven step by step (unlike execute_placement() in
    heuristic_agent.py, which runs a whole placement instantly) — a
    frame is rendered and a short pause taken after every single action,
    so the motion is visible.
    """
    best = choose_best_placement(game, WEIGHTS)

    rotations_needed = (best["rotation"] - game.current.rotation) % 4
    for _ in range(rotations_needed):
        game.step(Action.ROTATE_CW)
        render_frame(game, screen, font)
        wait(clock)

    moves_made = 0
    while game.current.col != best["col"] and moves_made < 40:
        if game.current.col < best["col"]:
            game.step(Action.RIGHT)
        else:
            game.step(Action.LEFT)
        render_frame(game, screen, font)
        wait(clock)
        moves_made += 1

    game.step(Action.HARD_DROP)
    render_frame(game, screen, font)
    wait(clock)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Tetris - agent plays")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    game = TetrisGame()

    running = True
    while running:
        if game.game_over:
            render_frame(game, screen, font)
            # Auto-restart after a short pause, so it can be watched
            # indefinitely without manually re-running the script.
            time.sleep(1.5)
            game = TetrisGame()
            continue

        agent_step_with_render(game, screen, font, clock)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()