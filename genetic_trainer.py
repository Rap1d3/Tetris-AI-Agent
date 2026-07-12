"""
genetic_trainer.py

Genetic algorithm that searches for good weights for the heuristic agent
(heuristic_agent.py).

Unlike heuristic_agent.DEFAULT_WEIGHTS (hand-picked — not learned), the
weights here start out RANDOM and improve purely through generational
selection. This is the actual learning process:

  1. Create a population — many individuals, each with 4 random numbers.
  2. Each individual plays a game of Tetris using its own weights (via
     choose_best_placement/execute_placement from heuristic_agent.py).
  3. Compute each individual's fitness — how well it played.
  4. Keep the best ("elite"), breed them via crossover + mutation to
     produce the next generation.
  5. Repeat for many generations, watching fitness rise from random
     noise to competent play.

Progress (population + generation count + best result so far) is saved
to SAVE_FILE after every generation, so training can be resumed across
runs instead of starting from scratch each time. population_watch.py
reads and writes the same file.
"""

import random
import json
import os
from tetris_logic import TetrisGame
from heuristic_agent import choose_best_placement, execute_placement, DEFAULT_WEIGHTS

# Reuses the weight names from heuristic_agent.py so the two files can't
# drift out of sync.
WEIGHT_NAMES = list(DEFAULT_WEIGHTS.keys())

# Range used to generate random weights for the very first generation.
RANDOM_WEIGHT_RANGE = (-1.0, 1.0)

SAVE_FILE = "training_state.json"


def save_training_state(population, generation, best_weights, best_fitness,
                         filepath=SAVE_FILE):
    """Writes the full training progress to a JSON file. Called after
    every generation, so at most one incomplete generation is ever lost
    if training is interrupted."""
    state = {
        "generation": generation,
        "population": population,
        "best_weights": best_weights,
        "best_fitness": best_fitness,
    }
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2)


def load_training_state(filepath=SAVE_FILE):
    """Loads saved progress if it exists, otherwise returns None (fresh start)."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def random_weights(rng):
    """Creates one individual: a dict of 4 random weights. A "newborn"
    agent that has no idea yet how to play well."""
    return {name: rng.uniform(*RANDOM_WEIGHT_RANGE) for name in WEIGHT_NAMES}


def evaluate_weights(weights, seed, max_pieces=150):
    """
    Plays one game of Tetris using the given weights and returns a
    fitness score — higher is better.

    fitness = pieces_placed + lines_cleared * 100

    Why not just use game.score? In early generations almost no
    individual clears a single line (they play like a random agent), so
    a fitness based only on lines cleared would be 0 for everyone,
    giving no signal to select on. pieces_placed varies immediately
    across individuals and rewards simply surviving longer. Once weights
    become competent and lines start clearing regularly, the *100 term
    dominates and becomes the main selection pressure.
    """
    game = TetrisGame(seed=seed)
    pieces_done = 0
    while not game.game_over and pieces_done < max_pieces:
        best = choose_best_placement(game, weights)
        execute_placement(game, best)
        pieces_done += 1

    fitness = game.pieces_placed + game.lines_cleared_total * 100
    return fitness


def crossover(weights_a, weights_b, rng):
    """Combines two parents: each of the 4 weights is taken from parent
    A or parent B with a 50/50 coin flip."""
    child = {}
    for name in WEIGHT_NAMES:
        child[name] = weights_a[name] if rng.random() < 0.5 else weights_b[name]
    return child


def mutate(weights, rng, mutation_rate=0.2, mutation_strength=0.3):
    """
    Adds small random noise to a child's weights. Without this, children
    would only ever be recombinations of existing numbers, and the
    population could never discover a value none of the parents had.

    mutation_rate: probability that any single weight mutates at all.
    mutation_strength: how large the change can be when it does mutate.
    """
    mutated = dict(weights)
    for name in WEIGHT_NAMES:
        if rng.random() < mutation_rate:
            mutated[name] += rng.uniform(-mutation_strength, mutation_strength)
    return mutated


def run_evolution(generations=15, population_size=20, elite_count=4, seed=0,
                   max_pieces=150, save_path=SAVE_FILE):
    """
    Main training loop. Per generation:

      1. Every individual plays a game (evaluate_weights) -> (fitness, weights).
      2. Sort best to worst.
      3. Print the generation's best fitness, so progress is visible.
      4. The top `elite_count` individuals carry over unchanged (elitism
         — protects an already-good solution from being mutated away).
      5. The rest of the population is filled with children: two random
         elite parents, crossover, then mutation.
      6. Repeat for the next generation.

    Loads saved progress from save_path if present and resumes from
    there instead of starting over; saves progress again after every
    generation.

    All individuals in a given generation are evaluated on the SAME
    seed (identical piece sequence) — otherwise an individual that
    happened to get an easy sequence could out-score better weights
    unfairly. The seed changes between generations so weights don't
    overfit to one specific sequence of pieces.
    """
    rng = random.Random(seed)

    saved_state = load_training_state(save_path)
    if saved_state is not None:
        population = saved_state["population"]
        start_generation = saved_state["generation"] + 1
        best_weights_overall = saved_state["best_weights"]
        best_fitness_overall = saved_state["best_fitness"]
        print(f"Found saved progress in {save_path} — "
              f"resuming from generation {start_generation}\n")
    else:
        population = [random_weights(rng) for _ in range(population_size)]
        start_generation = 0
        best_weights_overall = None
        best_fitness_overall = float("-inf")
        print("No saved progress found — starting from scratch\n")

    for generation in range(start_generation, start_generation + generations):
        eval_seed = rng.randint(0, 10_000)

        scored = [
            (evaluate_weights(w, seed=eval_seed, max_pieces=max_pieces), w)
            for w in population
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        best_fitness, best_weights = scored[0]
        avg_fitness = sum(f for f, _ in scored) / len(scored)
        print(f"Generation {generation:2d}: best fitness={best_fitness:6d}  "
              f"avg fitness={avg_fitness:8.1f}")

        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness
            best_weights_overall = best_weights

        elites = [w for _, w in scored[:elite_count]]
        next_population = list(elites)
        while len(next_population) < population_size:
            parent_a, parent_b = rng.sample(elites, 2)
            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, rng)
            next_population.append(child)
        population = next_population

        save_training_state(population, generation, best_weights_overall,
                             best_fitness_overall, save_path)

    print(f"\nBest result over the whole run: fitness={best_fitness_overall}")
    print(f"Weights: {best_weights_overall}")
    return best_weights_overall


if __name__ == "__main__":
    print("Starting training (weights start random)...\n")
    best_weights = run_evolution(generations=15, population_size=20,
                                  elite_count=4, seed=0, max_pieces=150)