import torch
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self, name) -> None:
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 255,  3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

        # Data
        self.n_games = 0
        self.scores = []
        self.mean_scores = []
        self.total_score = 0
        self.record_score = 0
        self.name = name

    def get_state(self, game, snake_id):
        snake = game.snakes[snake_id]
        head = snake.body[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(snake_id, point_r)) or
            (dir_l and game.is_collision(snake_id, point_l)) or
            (dir_u and game.is_collision(snake_id, point_u)) or
            (dir_d and game.is_collision(snake_id, point_d)),

            # Danger right
            (dir_u and game.is_collision(snake_id, point_r)) or
            (dir_d and game.is_collision(snake_id, point_l)) or
            (dir_l and game.is_collision(snake_id, point_u)) or
            (dir_r and game.is_collision(snake_id, point_d)),

            # Danger left
            (dir_d and game.is_collision(snake_id, point_r)) or
            (dir_u and game.is_collision(snake_id, point_l)) or
            (dir_r and game.is_collision(snake_id, point_u)) or
            (dir_l and game.is_collision(snake_id, point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            import random
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: Exploration vs Exploitation
        # Eary model explore alot, later it will random less
        self.epsilon = 80 - self.n_games  # TODO: Fidle
        action = [0, 0, 0]
        if np.random.randint(0, 200) < self.epsilon:
            action[np.random.randint(0, len(action) - 1)] = 1
        else:
            prediction = self.model(torch.tensor(state, dtype=torch.float))
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


def train():
    agents = [Agent("Linear_QNet(11, 255,  3) #0"),
              Agent("Linear_QNet(11, 255,  3) #1")]
    game = SnakeGameAI(640, 480, 2)

    scores = [[]] * (len(agents) + 1)
    mean_scores = [[]] * len(scores)

    while True:
        for player_id, agent in enumerate(agents):
            # Get current state
            state_old = agent.get_state(game, player_id)

            # Get action
            action = agent.get_action(state_old)

            # Perform move
            reward, done, score = game.play_step(action, player_id)
            state_new = agent.get_state(game, player_id)

            # Train short memory
            agent.train_short_memory(
                state_old, action, reward, state_new, done)

            # Remember
            agent.remember(state_old, action, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > agent.record_score:
                    agent.record_score = score
                    agent.model.save(f"model_{agent.name}.pth")

                print("Agent", agent.name, "Game", agent.n_games,
                      "score", score, "record:", agent.record_score)

                agent.scores.append(score)
                agent.total_score += score
                mean_score = agent.total_score / agent.n_games
                agent.mean_scores.append(mean_score)

                # For plot
                print(len(scores), player_id)
                scores[player_id] = agent.scores
                mean_scores[player_id] = agent.mean_scores
                plot(scores, mean_scores, map(lambda x: x.name, agents))
