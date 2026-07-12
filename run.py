"""
Pygame front-end for the Tetris engine (tetris_logic.py).

This module contains zero game logic — it only reads state from
TetrisGame and draws it to the screen. It can be used either for manual
play (keyboard controls below) or to watch a bot play, since the render
functions here (draw_board, draw_mini_piece) are reused by the other
scripts in this project (agent_watch.py, population_watch.py).

Controls:
    Left / Right arrow  - move
    Up arrow             - rotate clockwise
    Z                     - rotate counter-clockwise
    Down arrow            - soft drop
    Space                 - hard drop
    R                     - restart after game over
    Esc / close window    - quit
"""

import sys
import pygame
from tetris_logic import TetrisGame, Action, PIECE_IDS, SHAPES

CELL_SIZE = 30
BOARD_W, BOARD_H = 10, 20
SIDEBAR_W = 200
SCREEN_W = BOARD_W * CELL_SIZE + SIDEBAR_W
SCREEN_H = BOARD_H * CELL_SIZE

# Piece colors — edit freely, format is (R, G, B), 0-255.
COLORS = {
    0: (25, 25, 25),
    PIECE_IDS["I"]: (0, 240, 240),
    PIECE_IDS["O"]: (240, 240, 0),
    PIECE_IDS["T"]: (160, 0, 240),
    PIECE_IDS["S"]: (0, 240, 0),
    PIECE_IDS["Z"]: (240, 0, 0),
    PIECE_IDS["J"]: (0, 0, 240),
    PIECE_IDS["L"]: (240, 160, 0),
}
GRID_LINE_COLOR = (50, 50, 50)
TEXT_COLOR = (230, 230, 230)
BG_COLOR = (15, 15, 15)
GHOST_ALPHA_COLOR = (255, 255, 255, 60)

GRAVITY_INTERVAL = 0.6

KEY_ACTION_MAP = {
    pygame.K_LEFT: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_UP: Action.ROTATE_CW,
    pygame.K_z: Action.ROTATE_CCW,
    pygame.K_DOWN: Action.SOFT_DROP,
    pygame.K_SPACE: Action.HARD_DROP,
}


def draw_cell(surface, row, col, color, alpha=255):
    x = col * CELL_SIZE
    y = row * CELL_SIZE
    rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
    if alpha < 255:
        cell_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cell_surface, (*color, alpha), cell_surface.get_rect())
        surface.blit(cell_surface, (x, y))
    else:
        pygame.draw.rect(surface, color, rect)
    pygame.draw.rect(surface, GRID_LINE_COLOR, rect, width=1)


def draw_mini_piece(surface, piece_type, x, y, cell_size):
    color = COLORS[PIECE_IDS[piece_type]]
    cells = SHAPES[piece_type][0]  # rotation 0 = default orientation
    for r, c in cells:
        rect = pygame.Rect(x + c * cell_size, y + r * cell_size, cell_size, cell_size)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, GRID_LINE_COLOR, rect, width=1)


def compute_ghost_row(game):
    """Row the current piece would land on with a hard drop."""
    piece = game.current
    drop = 0
    while not game._collision(piece.cells(row=piece.row + drop + 1)):
        drop += 1
    return piece.row + drop


def draw_board(screen, game, font):
    screen.fill(BG_COLOR)
    state = game.get_state()
    board = state["board"]

    for r in range(BOARD_H):
        for c in range(BOARD_W):
            if board[r][c]:
                draw_cell(screen, r, c, COLORS[board[r][c]])
            else:
                pygame.draw.rect(
                    screen, GRID_LINE_COLOR,
                    (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE), width=1
                )

    # -4 removes the engine's hidden spawn buffer above the visible board
    ghost_row = compute_ghost_row(game) - 4
    piece_color = COLORS[PIECE_IDS[game.current.type]]
    for r, c in game.current.cells():
        ghost_r = r - game.current.row + ghost_row
        if 0 <= ghost_r < BOARD_H:
            draw_cell(screen, ghost_r, c, piece_color, alpha=70)

    for r, c in state["current_piece"]["cells"]:
        if 0 <= r < BOARD_H:
            draw_cell(screen, r, c, piece_color)

    sidebar_x = BOARD_W * CELL_SIZE + 15
    lines = [
        f"Score: {state['score']}",
        f"Lines: {state['lines_cleared_total']}",
        f"Pieces: {state['pieces_placed']}",
    ]

    y = 15
    for line in lines:
        text_surface = font.render(line, True, TEXT_COLOR)
        screen.blit(text_surface, (sidebar_x, y))
        y += 26

    y += 10
    next_label = font.render("Next:", True, TEXT_COLOR)
    screen.blit(next_label, (sidebar_x, y))
    y += 30

    MINI_CELL_SIZE = 16
    MINI_PIECE_ROW_HEIGHT = 70
    for piece_type in state["next_queue"][:5]:
        draw_mini_piece(screen, piece_type, sidebar_x, y, cell_size=MINI_CELL_SIZE)
        y += MINI_PIECE_ROW_HEIGHT

    if state["game_over"]:
        over_text = font.render("GAME OVER (R to restart)", True, (255, 80, 80))
        screen.blit(over_text, (sidebar_x, y + 20))

    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Tetris")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # Makes held keys repeat: 180ms initial delay, then every 50ms
    pygame.key.set_repeat(180, 50)

    game = TetrisGame()
    gravity_timer = 0.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and game.game_over:
                    game = TetrisGame()
                    gravity_timer = 0.0
                elif not game.game_over and event.key in KEY_ACTION_MAP:
                    game.step(KEY_ACTION_MAP[event.key])

        if not game.game_over:
            gravity_timer += dt
            if gravity_timer >= GRAVITY_INTERVAL:
                gravity_timer = 0.0
                game.step(Action.SOFT_DROP)

        draw_board(screen, game, font)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()