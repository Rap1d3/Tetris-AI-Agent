# Tetris-AI-Agent
A Tetris engine and a self-learning AI agent for it. The agent uses a genetic algorithm — starting from random weights, it evolves through generations of selection, crossover and mutation until it learns to clear lines efficiently. Built in Python with a Pygame front-end.
# Evolutionary Tetris Agent

A Tetris engine built from scratch in Python, paired with an AI agent that
learns to play through a **genetic algorithm** — no pre-programmed strategy,
no neural network, no training data included in this repo. Clone it, run
the trainer, and watch it evolve from random noise into a competent player.

## How it works, in one paragraph

The agent doesn't decide moves one keypress at a time. Instead, for every
new piece it evaluates **every possible final placement** (every rotation
x every column) and scores each one using a weighted formula over four
board features (stack height, holes, surface bumpiness, lines cleared).
It picks the highest-scoring placement and executes it. The four weights
in that formula are not hand-tuned — a genetic algorithm starts with a
population of random weights, has each one play a game, keeps the best
performers, breeds them (crossover + mutation), and repeats for many
generations until the weights converge on something that plays well.

## Project structure

| File | Responsibility |
|---|---|
| `tetris_logic.py` | The game engine. Pure state — no rendering, no input handling. Exposes a Gym-style `step(action)` interface and `get_possible_placements()`, which enumerates every reachable placement of the current piece for the agent to evaluate. |
| `heuristic_agent.py` | The agent's "brain and hands." `score_placement()` scores a placement given a set of weights; `choose_best_placement()` picks the best one; `execute_placement()` turns that choice into real `LEFT`/`RIGHT`/`ROTATE`/`HARD_DROP` actions. Also defines `DEFAULT_WEIGHTS`, a hand-picked reference baseline (not used by training itself). |
| `genetic_trainer.py` | The genetic algorithm: creates a population of random weight sets, evaluates them by fitness, selects the best, breeds the next generation via crossover and mutation, and repeats. Saves progress to `training_state.json` after every generation so training can resume across runs. |
| `population_watch.py` | Same genetic algorithm as above, but with a live pygame visualization — a grid showing all individuals in the population playing simultaneously, generation by generation. Reads/writes the same `training_state.json`, so you can freely switch between this and `genetic_trainer.py`. |
| `agent_watch.py` | Watches **one** agent play live in a pygame window. Loads the best weights from `training_state.json` if it exists; if not (e.g. a fresh clone with no training history), it plays with fresh random weights instead — so it plays badly until you've actually trained it. |
| `run.py` | Manual, human-playable version of the game — keyboard controls, no AI involved. Also the source of the shared rendering functions (`draw_board`, colors, etc.) reused by `agent_watch.py` and `population_watch.py`. |
| `.gitignore` | Excludes `training_state.json` and `__pycache__/` from version control, so every fresh clone starts with an untrained agent. |

## Requirements

- Python 3.9+
- [pygame](https://www.pygame.org/) (only needed for the visual scripts — `tetris_logic.py`, `heuristic_agent.py`, and `genetic_trainer.py` have no graphics dependency at all)

```bash
pip install pygame
```

## Usage

Run everything from inside the project folder.

**Play the game yourself:**
```bash
python run.py
```
Arrow keys to move, `Up` to rotate clockwise, `Z` to rotate counter-clockwise,
`Down` for soft drop, `Space` for hard drop, `R` to restart after game over.

**Train the agent (fast, no graphics):**
```bash
python genetic_trainer.py
```
Prints one line of progress per generation. Safe to stop at any time
(`Ctrl+C`) — progress up to the last completed generation is already saved.
Running it again continues from where it left off instead of starting over.

**Train the agent (with a live visualization):**
```bash
python population_watch.py
```
Opens a window showing the entire population — every individual playing
at once, in a grid — updating generation by generation. Shares its
progress file with `genetic_trainer.py`, so you can alternate between the
two freely.

**Watch the trained agent play:**
```bash
python agent_watch.py
```
Loads whatever is currently the best result in `training_state.json`. If
that file doesn't exist yet (nothing has been trained), the agent plays
with random, untrained weights instead of a fake "instant win" — you have
to actually train it first to see it play well.

## The fitness function

```python
fitness = pieces_placed + lines_cleared_total * 100
```

Early in training, almost no individual clears a single line — if fitness
depended only on lines cleared, every individual in generation 0 would
score 0, giving the algorithm nothing to select on. `pieces_placed`
(how long an individual survived before losing) gives a useful signal
even before anyone starts clearing lines. As weights improve and lines
start clearing regularly, the `* 100` term takes over as the dominant
factor.

## Training progress and reproducibility

Training progress (the full population, current generation number, and
best result so far) is saved to `training_state.json` after every
generation. This file is intentionally excluded from version control
(see `.gitignore`) — every person who clones this repository starts
from a genuinely untrained agent and can watch it learn from scratch,
rather than inheriting someone else's already-evolved weights.

## Possible extensions

- Swap the genetic algorithm for a Deep RL approach (DQN / PPO) and
  compare results
- Expand the feature set the heuristic scores on (e.g. well depth,
  row transitions)
- Add a `hold` mechanic (removed early in development to keep the
  action space simple)