from numpy.random.mtrand import gamma
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from pygame.constants import K_SPACE, K_o

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 245)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 205, 0)
GREEN2 = (50, 255, 100)
RED1 = (255, 0, 0)
RED2 = (255, 100, 0)
YELLOW1 = (205, 205, 0)
YELLOW2 = (255, 255, 150)
BLACK = (0, 0, 0)
GRAY1 = (30, 30, 30)
GRAY2 = (40, 40, 40)

BLOCK_SIZE = 20

FAST_SPEED = 1000
WATCH_SPEED = 20
SPEED = FAST_SPEED


class Snake:
    def __init__(self, start_pos, color) -> None:
        self.direction = Direction.RIGHT

        self.head = start_pos
        self.body = [self.head,
                     Point(self.head.x-BLOCK_SIZE, self.head.y),
                     Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.color = color
        self.score = 0

        # Used to detect spinning strat
        self.last_action = None
        self.repeated_action = False


class SnakeGameAI:
    COLORS = ((BLUE1, BLUE2), (GREEN1, GREEN2),
              (RED1, RED2), (YELLOW1, YELLOW2))

    def __init__(self, w=640, h=480, n_players=1):
        self.w = w
        self.h = h
        self.n_players = n_players
        self.num_objects = 0
        self.snakes = []
        self.objects = []
        self.food = None
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def get_random_free_point(self):
        # TODO: This could freeze te game if no free spots
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        p = Point(x, y)

        while any(p in snake.body for snake in self.snakes) or p in self.objects or p == self.food:
            x = random.randint(0, (self.w-BLOCK_SIZE) //
                               BLOCK_SIZE)*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE) //
                               BLOCK_SIZE)*BLOCK_SIZE
            p = Point(x, y)

        return p

    def reset(self):
        # init game state
        self.snakes.clear()
        for i in range(self.n_players):
            self.snakes.append(
                Snake(self.get_random_free_point(), self.COLORS[i % len(self.COLORS)]))

        # Create random objects
        self.objects = []

        for _ in range(self.num_objects):
            self.objects.append(self.get_random_free_point())

        # self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if any(self.food in snake.body for snake in self.snakes) or self.food in self.objects:
            self._place_food()

    def play_step(self, action, snake_id):
        global SPEED

        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                print("KeyDown")
                if event.key == K_SPACE:
                    SPEED = FAST_SPEED if SPEED == WATCH_SPEED else WATCH_SPEED
                    print("Changed speed to:", SPEED)
                if event.key == K_o:
                    self.num_objects += 25
                    self.num_objects %= 100
                    print("Changed num objects to: ", self.num_objects)

        snake = self.snakes[snake_id]
        # 2. move
        self._move(snake_id, action)  # update the head
        snake.body.insert(0, snake.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision(snake_id) or self.frame_iteration > 100 * len(snake.body):
            game_over = True
            reward = -10
            return reward, game_over, snake.score

        for i, other_snake in enumerate(self.snakes):
            # Other snake killed by current snake
            if i != snake_id and other_snake.head in snake.body[1:]:
                reward = 20
                break
            # Other snake got the food
            elif other_snake.head == self.food:
                reward = -5
                break

        # 4. place new food or just move
        if snake.head == self.food:
            snake.score += 1
            reward = 10
            self._place_food()
        else:
            snake.body.pop()

        # Prevent spinning strat
        if snake.repeated_action:  # Set in _move
            reward -= 1

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, snake.score

    def is_collision(self, snake_id, pt=None):
        if pt is None:
            pt = self.snakes[snake_id].head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if any(pt in snake.body[1:] for snake in self.snakes):
            return True

        # hit objects
        if pt in self.objects:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for snake in self.snakes:
            for pt in snake.body:
                pygame.draw.rect(self.display, snake.color[0], pygame.Rect(
                    pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, snake.color[1],
                                 pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        for pt in self.objects:
            pygame.draw.rect(self.display, GRAY1, pygame.Rect(
                pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GRAY2,
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        for i, snake in enumerate(self.snakes):
            text = font.render(
                "Score: " + str(snake.score), True, WHITE)
            self.display.blit(text, [i * 150, 0])
        pygame.display.flip()

    def _move(self, snake_id, action):
        # action = [straight, right, left] bools
        snake = self.snakes[snake_id]

        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(snake.direction)

        if np.array_equal(action, [1, 0, 0]):  # Straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # Right
            nxt_idx = (idx + 1) % 4
            new_dir = clock_wise[nxt_idx]
        elif np.array_equal(action, [0, 0, 1]):  # Left
            nxt_idx = (idx - 1) % 4
            new_dir = clock_wise[nxt_idx]
        else:
            raise(ValueError("Unkown action"))

        snake.direction = new_dir

        x = snake.head.x
        y = snake.head.y
        if snake.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif snake.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif snake.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif snake.direction == Direction.UP:
            y -= BLOCK_SIZE

        snake.head = Point(x, y)

        snake.repeated_action = action != [
            1, 0, 0] and snake.last_action == action
        snake.last_action = action
