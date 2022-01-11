import torch
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class AgentBase:
    def __init__(self, model, name) -> None:
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = model
        self.trainer = QTrainer(self.model, LR, self.gamma)

        # Data
        self.done = False
        self.n_games = 0
        self.scores = []
        self.mean_scores = []
        self.total_score = 0
        self.record_score = 0
        self.name = f"{model}_{name}"

    def get_state(self, game, snake_id):
        raise(NotImplemented("Implement this in the child class"))

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

    def get_action(self, state, epsilon):
        # Random moves: Exploration vs Exploitation
        # Eary model explore alot, later it will random less
        action = [0, 0, 0]
        if np.random.randint(0, 200) < epsilon:
            action[np.random.randint(0, len(action) - 1)] = 1
        else:
            prediction = self.model(torch.tensor(state, dtype=torch.float))
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


class AgentBasic(AgentBase):
    def __init__(self, model, name) -> None:
        super().__init__(model, name)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        return super().get_action(state, self.epsilon)

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


class AgentLookAhead(AgentBase):
    def __init__(self, model, name) -> None:
        super().__init__(model, name)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games * \
            self.mean_scores[-1] if len(self.mean_scores) else 0 + \
            self.record_score
        return super().get_action(state, self.epsilon)

    def get_state(self, game, snake_id):
        # TODO: Tile size from game
        tile_size = 20

        snake = game.snakes[snake_id]
        head = snake.body[0]
        points_l = [Point(head.x - i * tile_size, head.y)
                    for i in range(game.w // tile_size)]
        points_r = [Point(head.x + i * tile_size, head.y)
                    for i in range(game.w // tile_size)]
        points_u = [Point(head.x, head.y - i * tile_size)
                    for i in range(game.h // tile_size)]
        points_d = [Point(head.x, head.y + i * tile_size)
                    for i in range(game.h // tile_size)]

        # TODO: Optimize
        collision_l = [game.is_collision(snake_id, p) for p in points_l]
        collision_r = [game.is_collision(snake_id, p) for p in points_r]
        collision_u = [game.is_collision(snake_id, p) for p in points_u]
        collision_d = [game.is_collision(snake_id, p) for p in points_d]

        empty_space_l = (collision_l.index(
            True) if True in collision_l else 0) / len(collision_l)
        empty_space_r = (collision_r.index(
            True) if True in collision_r else 0) / len(collision_r)
        empty_space_u = (collision_u.index(
            True) if True in collision_u else 0) / len(collision_u)
        empty_space_d = (collision_d.index(
            True) if True in collision_d else 0) / len(collision_d)

        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        if dir_r:
            empty_space_straight = empty_space_r
            empty_space_right = empty_space_d
            empty_space_left = empty_space_u
        elif dir_d:
            empty_space_straight = empty_space_d
            empty_space_right = empty_space_l
            empty_space_left = empty_space_r
        elif dir_l:
            empty_space_straight = empty_space_l
            empty_space_right = empty_space_u
            empty_space_left = empty_space_d
        elif dir_u:
            empty_space_straight = empty_space_u
            empty_space_right = empty_space_r
            empty_space_left = empty_space_l

        state = [
            # Danger straight [0..1] 1 = no space
            1 - empty_space_straight,

            # Danger right [0..1]
            1 - empty_space_right,

            # Danger left [0..1]
            1 - empty_space_left,

            # Move direction [True or False]
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # TODO: Distance to food
            # Food location [True or False]
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y  # food down
        ]

        return np.array(state, dtype=int)


def train():
    # TODO: Date to name to prevent overwrite
    agents = [
        AgentBasic(Linear_QNet((11, 255,  3)), "Basic_Blue"),
        AgentBasic(Linear_QNet((11, 20, 3)), "Basic_Green"),
        AgentBasic(Linear_QNet((11, 100, 3)), "Basic_Red"),
        AgentLookAhead(Linear_QNet((11, 255,  3)), "LookAhead_Yellow"),
    ]
    game = SnakeGameAI(640, 480, len(agents))

    scores = [[]] * (len(agents) + 1)
    mean_scores = [[]] * len(scores)

    while True:
        for player_id, agent in enumerate(agents):
            if agent.done:
                continue
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
                agent.n_games += 1
                agent.scores.append(score)
                agent.total_score += score
                mean_score = agent.total_score / agent.n_games
                agent.mean_scores.append(mean_score)
                agent.done = True

                # For plot
                scores[player_id] = agent.scores
                mean_scores[player_id] = agent.mean_scores

                print(
                    f"Agend {agent.name} died at game: {agent.n_games} with score: {score} (record: {agent.record_score})")

                if score > agent.record_score:
                    agent.record_score = score
                    agent.model.save(f"model_{agent.name}.pth")

        if all(agent.done for agent in agents):
            game.reset()

            for player_id, agent in enumerate(agents):
                agent.train_long_memory()
                agent.done = False

            plot(scores, mean_scores, map(lambda x: x.name, agents))
