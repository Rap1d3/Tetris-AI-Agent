"""
population_watch.py

Shows the ENTIRE genetic_trainer.py population at once — a grid of
POPULATION_SIZE small boards in a single window, each playing with its
own weights. Displays the generation number and each individual's
running fitness. Selection between generations (elitism + crossover +
mutation) is the exact same logic as genetic_trainer.py, just with a
visualization layered on top.

Reads/writes the same training_state.json as genetic_trainer.py, so
training can be resumed regardless of which of the two scripts was used
last.

Run:
    python3 population_watch.py
"""

import sys
import random
import pygame

from tetris_logic import TetrisGame, Action, PIECE_IDS, BOARD_WIDTH, BOARD_HEIGHT
from heuristic_agent import choose_best_placement
from genetic_trainer import (random_weights, crossover, mutate,
                              save_training_state, load_training_state, SAVE_FILE)
from run import COLORS, GRID_LINE_COLOR, BG_COLOR, TEXT_COLOR

# ---------------------------------------------------------------------- #
# Settings
# ---------------------------------------------------------------------- #
POPULATION_SIZE = 20
ELITE_COUNT = 4
GENERATIONS = 8
MAX_PIECES_PER_EVAL = 100
STEP_DELAY = 0.03

CELL_SIZE_MINI = 9
COLS = 5                   # 5 x 4 grid = 20 boards
ROWS = POPULATION_SIZE // COLS
SLOT_PAD = 14
LABEL_HEIGHT = 16
TOP_MARGIN = 40

BOARD_PX_W = BOARD_WIDTH * CELL_SIZE_MINI
BOARD_PX_H = BOARD_HEIGHT * CELL_SIZE_MINI
SLOT_W = BOARD_PX_W + SLOT_PAD
SLOT_H = BOARD_PX_H + LABEL_HEIGHT + SLOT_PAD
WINDOW_W = COLS * SLOT_W + SLOT_PAD
WINDOW_H = TOP_MARGIN + ROWS * SLOT_H + SLOT_PAD


def agent_step_generator(game, weights):
    """
    A generator (`yield` instead of `return`): each call to next()
    resumes execution right after the previous yield instead of running
    the whole function at once. This lets us advance all 20 boards by
    exactly one small action per animation frame — call next() on each
    of the 20, render, wait, call next() again, and so on — so they
    appear to play simultaneously.
    """
    while not game.game_over:
        best = choose_best_placement(game, weights)

        rotations_needed = (best["rotation"] - game.current.rotation) % 4
        for _ in range(rotations_needed):
            game.step(Action.ROTATE_CW)
            yield
            if game.game_over:
                return

        moves_made = 0
        while game.current.col != best["col"] and moves_made < 40:
            if game.current.col < best["col"]:
                game.step(Action.RIGHT)
            else:
                game.step(Action.LEFT)
            yield
            if game.game_over:
                return
            moves_made += 1

        game.step(Action.HARD_DROP)
        yield
        if game.game_over:
            return


def draw_mini_board(surface, game, x, y, cell_size):
    state = game.get_state()
    board = state["board"]

    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            color = COLORS[board[r][c]] if board[r][c] else COLORS[0]
            rect = (x + c * cell_size, y + r * cell_size, cell_size, cell_size)
            pygame.draw.rect(surface, color, rect)

    piece_color = COLORS[PIECE_IDS[game.current.type]]
    for r, c in state["current_piece"]["cells"]:
        if 0 <= r < BOARD_HEIGHT:
            rect = (x + c * cell_size, y + r * cell_size, cell_size, cell_size)
            pygame.draw.rect(surface, piece_color, rect)

    pygame.draw.rect(surface, GRID_LINE_COLOR,
                      (x, y, BOARD_WIDTH * cell_size, BOARD_HEIGHT * cell_size), width=1)


def render_grid(screen, games, generation_num, font, small_font):
    screen.fill(BG_COLOR)
    gen_text = font.render(f"Generation {generation_num}", True, TEXT_COLOR)
    screen.blit(gen_text, (SLOT_PAD, 10))

    for i, game in enumerate(games):
        col = i % COLS
        row = i // COLS
        x = SLOT_PAD + col * SLOT_W
        y = TOP_MARGIN + row * SLOT_H

        draw_mini_board(screen, game, x, y, CELL_SIZE_MINI)

        fitness = game.pieces_placed + game.lines_cleared_total * 100
        label = small_font.render(f"#{i}  f={fitness}", True, TEXT_COLOR)
        screen.blit(label, (x, y + BOARD_PX_H + 2))

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


def wait(clock):
    waited = 0.0
    while waited < STEP_DELAY:
        dt = clock.tick(60) / 1000.0
        waited += dt


def run_generation_visual(population, screen, clock, font, small_font, generation_num, seed):
    """
    Runs one generation: every individual plays on the same seed
    (identical piece sequence, for a fair comparison) until it finishes
    (game over or MAX_PIECES_PER_EVAL pieces). Returns a fitness list in
    the same order as population.
    """
    games = [TetrisGame(seed=seed) for _ in population]
    runners = [agent_step_generator(games[i], population[i]) for i in range(len(population))]
    finished = [False] * len(population)

    while not all(finished):
        for i in range(len(population)):
            if finished[i]:
                continue
            try:
                next(runners[i])
            except StopIteration:
                finished[i] = True
            if games[i].game_over or games[i].pieces_placed >= MAX_PIECES_PER_EVAL:
                finished[i] = True

        render_grid(screen, games, generation_num, font, small_font)
        wait(clock)

    return [g.pieces_placed + g.lines_cleared_total * 100 for g in games]


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Tetris - whole population")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    small_font = pygame.font.SysFont("consolas", 12)

    rng = random.Random(0)

    saved_state = load_training_state(SAVE_FILE)
    if saved_state is not None:
        population = saved_state["population"]
        start_generation = saved_state["generation"] + 1
        best_weights_overall = saved_state["best_weights"]
        best_fitness_overall = saved_state["best_fitness"]
        print(f"Found saved progress — resuming from generation {start_generation}\n")
    else:
        population = [random_weights(rng) for _ in range(POPULATION_SIZE)]
        start_generation = 0
        best_weights_overall = None
        best_fitness_overall = float("-inf")
        print("No saved progress found — starting from scratch\n")

    for generation in range(start_generation, start_generation + GENERATIONS):
        eval_seed = rng.randint(0, 10_000)
        fitness_list = run_generation_visual(
            population, screen, clock, font, small_font, generation, eval_seed
        )

        scored = sorted(zip(fitness_list, population), key=lambda pair: pair[0], reverse=True)
        best_fitness, best_weights = scored[0]
        avg_fitness = sum(fitness_list) / len(fitness_list)
        print(f"Generation {generation}: best fitness={best_fitness}  "
              f"avg fitness={avg_fitness:.1f}")

        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness
            best_weights_overall = best_weights

        elites = [w for _, w in scored[:ELITE_COUNT]]
        next_population = list(elites)
        while len(next_population) < POPULATION_SIZE:
            parent_a, parent_b = rng.sample(elites, 2)
            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, rng)
            next_population.append(child)
        population = next_population

        save_training_state(population, generation, best_weights_overall,
                             best_fitness_overall, SAVE_FILE)

    print(f"\nTraining complete. Best fitness overall: {best_fitness_overall}")
    print(f"Weights: {best_weights_overall}")

    # Keep the window open after training so the last frame can be
    # inspected; closes only via the window's close button.
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        clock.tick(30)


if __name__ == "__main__":
    main()