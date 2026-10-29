"""Play a jaxtari game with the keyboard: python src/play.py freeway
Arrows move, space fires, R restarts, Esc quits. Keys the game has no action for do nothing."""

import sys

import jax
import jaxtari
import numpy as np
import pygame

SCALE, FPS = 4, 60
# ALE action ids, the numbering of jaxtari's ACTION_SET
ALE = {(0, 0, 0): 0, (0, 0, 1): 1, (-1, 0, 0): 2, (0, 1, 0): 3, (0, -1, 0): 4, (1, 0, 0): 5, (-1, 1, 0): 6, (-1, -1, 0): 7,
       (1, 1, 0): 8, (1, -1, 0): 9, (-1, 0, 1): 10, (0, 1, 1): 11, (0, -1, 1): 12, (1, 0, 1): 13, (-1, 1, 1): 14,
       (-1, -1, 1): 15, (1, 1, 1): 16, (1, -1, 1): 17}


def main(game: str):
    env = jaxtari.make(game)
    action_set = np.asarray(env.ACTION_SET).tolist()
    reset, step, render = jax.jit(env.reset), jax.jit(env.step), jax.jit(env.render)
    _, state = reset(jax.random.key(0))
    pygame.init()
    screen = pygame.display.set_mode((160 * SCALE, 210 * SCALE))
    clock, score = pygame.time.Clock(), 0.0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                _, state = reset(jax.random.key(int(score)))
                score = 0.0
        k = pygame.key.get_pressed()
        ale = ALE[(k[pygame.K_DOWN] - k[pygame.K_UP], k[pygame.K_RIGHT] - k[pygame.K_LEFT], k[pygame.K_SPACE])]
        action = action_set.index(ale) if ale in action_set else action_set.index(0)
        _, state, reward, done, _ = step(state, action)
        score += float(reward)
        frame = np.asarray(render(state)).swapaxes(0, 1)  # pygame surfaces are (width, height)
        screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(frame), screen.get_size()), (0, 0))
        pygame.display.set_caption(f"{game}  score {score:g}")
        pygame.display.flip()
        if done:
            print(f"episode over, score {score:g}")
            _, state = reset(jax.random.key(int(score) + 1))
            score = 0.0
        clock.tick(FPS)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "freeway")
