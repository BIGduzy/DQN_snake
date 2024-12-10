import torch
import random
import numpy as np
import math
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using torch device:", device)


class AgentBase:
    def __init__(self, model, name) -> None:
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = model
        self.trainer = QTrainer(self.model, LR, self.gamma)

        # Data
        self.training = True
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
        if np.random.random() < epsilon and self.training:
            action[np.random.randint(0, len(action) - 1)] = 1
        else:
            # TODO: Move tensor / torch to model file, so we don't have to do the gpu thing
            # AKA model.predict function that returns action
            prediction = self.model(torch.tensor(
                state, dtype=torch.float).to(device))
            move = torch.argmax(prediction).to(device).item()
            action[move] = 1

        return action


class AgentBasic(AgentBase):
    def __init__(self, model, name) -> None:
        super().__init__(model, name)

    def get_action(self, state):
        self.epsilon = (80 - self.n_games) / 100
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
        self.gamma = 0.99
        super().__init__(model, name)

    def get_action(self, state):
        EPS_START = 0.9
        EPS_END = 0.1
        EPS_DECAY = 200
        # first 200 iterations with random, after that alot of full predict -> repeat
        # threshold = self.n_games % (EPS_DECAY * 5)
        threshold = self.n_games
        self.epsilon = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * threshold / EPS_DECAY)
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

        # empty_space_l = (collision_l.index(
        #     True) if True in collision_l else 0) / len(collision_l)
        # empty_space_r = (collision_r.index(
        #     True) if True in collision_r else 0) / len(collision_r)
        # empty_space_u = (collision_u.index(
        #     True) if True in collision_u else 0) / len(collision_u)
        # empty_space_d = (collision_d.index(
        #     True) if True in collision_d else 0) / len(collision_d)

        # empty_space_r -= (tile_size / 2) / game.w
        # empty_space_l -= (tile_size / 2) / game.w
        # empty_space_u -= (tile_size / 2) / game.h
        # empty_space_d -= (tile_size / 2) / game.h

        # empty_space_r = round(empty_space_r, 2)
        # empty_space_l = round(empty_space_l, 2)
        # empty_space_u = round(empty_space_u, 2)
        # empty_space_d = round(empty_space_d, 2)

        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        if dir_r:
            collisions_straight = collision_r
            collisions_right = collision_d
            collisions_left = collision_u
        elif dir_d:
            collisions_straight = collision_d
            collisions_right = collision_l
            collisions_left = collision_r
        elif dir_l:
            collisions_straight = collision_l
            collisions_right = collision_u
            collisions_left = collision_d
        elif dir_u:
            collisions_straight = collision_u
            collisions_right = collision_r
            collisions_left = collision_l

        if game.debugging:
            clock_wise = [Direction.RIGHT, Direction.DOWN,
                          Direction.LEFT, Direction.UP]
            idx = clock_wise.index(snake.direction)

            space_forward = collisions_straight.index(
                True) if True in collisions_straight else 0
            space_left = collisions_left.index(
                True) if True in collisions_left else 0
            space_right = collisions_right.index(
                True) if True in collisions_right else 0

            # Vision lines
            game._draw_line(space_forward,
                            clock_wise[idx], (255, 0, 0))
            game._draw_line(space_right,
                            clock_wise[(idx + 1) % 4], (0, 255, 0))
            game._draw_line(space_left,
                            clock_wise[(idx - 1) % 4], (0, 0, 255))

            # Text
            game._write_text((collisions_straight[2], (255, 0, 0)),
                             (collisions_right[2], (0, 255, 0)
                              ), (collisions_left[2], (0, 0, 255)))
            # game._write_text((space_forward, (255, 0, 0)),
            #                  (space_right, (0, 255, 0)
            #                   ), (space_left, (0, 0, 255)))

        # print(len(collisions_straight), len(
        #     collisions_left), len(collisions_right))
        # print()

        lookAhead = 6
        lookAhead = 21
        if self.name.endswith("Yellow).pth"):
            lookAhead = 6
        state = [
            # Look ahead 6 tiles
            *collisions_straight[:lookAhead],

            # 6 tiles to right
            *collisions_right[:lookAhead],

            # 6 tiles to left
            *collisions_left[:lookAhead],

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


class AgentCornerVision(AgentBase):
    def __init__(self, model, name) -> None:
        super().__init__(model, name)

    def get_action(self, state):
        self.epsilon = (80 - self.n_games) / 100
        return super().get_action(state, self.epsilon)

    def get_state(self, game, snake_id):
        snake = game.snakes[snake_id]
        head = snake.body[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        # Corners
        point_lu = Point(head.x - 20, head.y - 20)
        point_ru = Point(head.x + 20, head.y - 20)
        point_ld = Point(head.x - 20, head.y + 20)
        point_rd = Point(head.x + 20, head.y + 20)

        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        # Could be done in a loop d;)
        if dir_r:
            danger_straight = game.is_collision(snake_id, point_r)
            danger_left = game.is_collision(snake_id, point_u)
            danger_right = game.is_collision(snake_id, point_d)

            danger_left_corner = game.is_collision(snake_id, point_ru)
            danger_right_corner = game.is_collision(snake_id, point_rd)
        elif dir_d:
            danger_straight = game.is_collision(snake_id, point_d)
            danger_left = game.is_collision(snake_id, point_r)
            danger_right = game.is_collision(snake_id, point_l)

            danger_left_corner = game.is_collision(snake_id, point_rd)
            danger_right_corner = game.is_collision(snake_id, point_ld)
        elif dir_l:
            danger_straight = game.is_collision(snake_id, point_l)
            danger_left = game.is_collision(snake_id, point_d)
            danger_right = game.is_collision(snake_id, point_u)

            danger_left_corner = game.is_collision(snake_id, point_ld)
            danger_right_corner = game.is_collision(snake_id, point_lu)
        elif dir_u:
            danger_straight = game.is_collision(snake_id, point_u)
            danger_left = game.is_collision(snake_id, point_l)
            danger_right = game.is_collision(snake_id, point_r)

            danger_left_corner = game.is_collision(snake_id, point_lu)
            danger_right_corner = game.is_collision(snake_id, point_ru)

        state = [
            danger_straight,
            danger_left,
            danger_right,
            danger_left_corner,
            danger_right_corner,

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
    def __init__(self, model, distance, name) -> None:
        super().__init__(model, name)
        self.gamma = 0.99
        self.distance = distance

    def get_action(self, state):
        EPS_START = 0.9
        EPS_END = 0.001
        EPS_DECAY = 200
        # first 200 iterations with random, after that alot of full predict -> repeat
        # threshold = self.n_games % (EPS_DECAY * 5)
        threshold = self.n_games
        self.epsilon = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * threshold / EPS_DECAY)
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

        # empty_space_l = (collision_l.index(
        #     True) if True in collision_l else 0) / len(collision_l)
        # empty_space_r = (collision_r.index(
        #     True) if True in collision_r else 0) / len(collision_r)
        # empty_space_u = (collision_u.index(
        #     True) if True in collision_u else 0) / len(collision_u)
        # empty_space_d = (collision_d.index(
        #     True) if True in collision_d else 0) / len(collision_d)

        # empty_space_r -= (tile_size / 2) / game.w
        # empty_space_l -= (tile_size / 2) / game.w
        # empty_space_u -= (tile_size / 2) / game.h
        # empty_space_d -= (tile_size / 2) / game.h

        # empty_space_r = round(empty_space_r, 2)
        # empty_space_l = round(empty_space_l, 2)
        # empty_space_u = round(empty_space_u, 2)
        # empty_space_d = round(empty_space_d, 2)

        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        if dir_r:
            collisions_straight = collision_r
            collisions_right = collision_d
            collisions_left = collision_u
        elif dir_d:
            collisions_straight = collision_d
            collisions_right = collision_l
            collisions_left = collision_r
        elif dir_l:
            collisions_straight = collision_l
            collisions_right = collision_u
            collisions_left = collision_d
        elif dir_u:
            collisions_straight = collision_u
            collisions_right = collision_r
            collisions_left = collision_l

        debug = False
        if debug:
            clock_wise = [Direction.RIGHT, Direction.DOWN,
                          Direction.LEFT, Direction.UP]
            idx = clock_wise.index(snake.direction)

            space_forward = collisions_straight.index(
                True) if True in collisions_straight else 0
            space_left = collisions_left.index(
                True) if True in collisions_left else 0
            space_right = collisions_right.index(
                True) if True in collisions_right else 0

            # Vision lines
            game._draw_line(space_forward,
                            clock_wise[idx], (255, 0, 0))
            game._draw_line(space_right,
                            clock_wise[(idx + 1) % 4], (0, 255, 0))
            game._draw_line(space_left,
                            clock_wise[(idx - 1) % 4], (0, 0, 255))

            # Text
            game._write_text((collisions_straight[2], (255, 0, 0)),
                             (collisions_right[2], (0, 255, 0)
                              ), (collisions_left[2], (0, 0, 255)))
            # game._write_text((space_forward, (255, 0, 0)),
            #                  (space_right, (0, 255, 0)
            #                   ), (space_left, (0, 0, 255)))

        # print(len(collisions_straight), len(
        #     collisions_left), len(collisions_right))
        # print()

        lookAhead = self.distance
        state = [
            # Look ahead 6 tiles
            *collisions_straight[:lookAhead],

            # 6 tiles to right
            *collisions_right[:lookAhead],

            # 6 tiles to left
            *collisions_left[:lookAhead],

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


class AgentMushroomVision(AgentBase):
    def __init__(self, model, name) -> None:
        super().__init__(model, name)

    def get_action(self, state):
        EPS_START = 0.9
        EPS_END = 0.001
        # Number of games to reach ESP_END for epsilon, then we reset back to ESP_START
        NUMBER_OF_GAMES_FOR_END = 100
        # threshold = self.n_games
        power = 2 # You can adjust this power to control steepness
        # first 200 iterations with random, after that allot of full predict -> repeat
        threshold = (self.n_games % (NUMBER_OF_GAMES_FOR_END * 1.5)) + 1 # Polynomial decay function
        self.epsilon = EPS_END + (EPS_START - EPS_END) * (1 - threshold / NUMBER_OF_GAMES_FOR_END) ** power
        # print(f"{self.epsilon}")
         # Ensure that epsilon does not go below EPS_END 
        if self.epsilon < EPS_END:
            self.epsilon = EPS_END

        # print(self.epsilon)
        return super().get_action(state, self.epsilon)

    def get_state(self, game, snake_id):
        tile_size = 20
        snake = game.snakes[snake_id]
        head = snake.body[0]
        
        rotate_for_direction = {
            Direction.UP: lambda s: s,
            Direction.RIGHT: lambda s: np.rot90(s, -1),
            Direction.DOWN: lambda s: np.rot90(s, 2),
            Direction.LEFT: lambda s: np.rot90(s, 1),
        }
        rotate_back = {
            Direction.UP: lambda s: s[np.lexsort((s[:, 0], s[:, 1]))],
            Direction.RIGHT: lambda s: s[np.lexsort((-s[:, 1], s[:, 0]))],
            Direction.DOWN: lambda s: s[np.lexsort((-s[:, 0], -s[:, 1]))],
            Direction.LEFT: lambda s: s[np.lexsort((s[:, 1], -s[:, 0]))],
        }

        # Define collision detection box and rotate according to direction
        shape = np.array([
            [0, 0, 1 ,0, 0],
            [0, 1, 1 ,1, 0],
            [1, 1, 1 ,1, 1],
            [1, 1, 1 ,1, 1],
            [1, 1,'s',1, 1],
        # The snake is located at the s
        ], dtype=object)
        shape = rotate_for_direction[snake.direction](shape)

        # shape = map(lambda value, index: (value, index), shape)
        snake_position = np.where(shape == 's')
        rows, cols = np.indices(shape.shape)
        # Calculate the distance from the snake for each cell
        vision_points = np.dstack((cols - snake_position[1], snake_position[0] - rows))
        vision_points = vision_points
        # Use the shape as a mask to filter out non-one values, such as the snake and blind spots (0)
        mask = shape == 1
        filtered_vision_points = vision_points[mask]
        # print("Before")
        # print(filtered_vision_points)
        filtered_vision_points = rotate_back[snake.direction](filtered_vision_points)
        # print("After")
        # print(filtered_vision_points)
        # Calculate points
        coords = [Point(head.x + x * tile_size, head.y - y * tile_size) for x, y in filtered_vision_points]
        # Collisions
        collisions = [game.is_collision(snake_id, point) for point in coords]
        

        # print(shape)
        # print(mask)
        # print(vision_points)
        # print("Filtered", filtered_vision_points)
        # print("Coords")     
        # print(coords)
        # print("Collisions")     
        # print(collisions)
        # print(len(collisions))
        if game.debugging:
            i = 0
            for point, collision in zip(coords, collisions):
                # colors = [
                #     (255, 255, 255),
                #     (255, 255, 200),
                #     (255, 255, 150),
                #     (255, 255, 100),
                #     (255, 255, 50),
                #     (255, 255, 0),
                #     (255, 200, 0),
                #     (255, 150, 0),
                #     (255, 100, 0),                    
                #     (255, 50, 0),                    
                #     (255, 0, 0),                    
                #     (200, 0, 0),                     
                #     (150, 0, 0),                    
                #     (100, 0, 0),                    
                #     (50, 0, 0),                    
                #     (0, 0, 0),                    
                #     (255, 0, 255),                    
                #     (0, 255, 255),                ]
                # game._draw_rec(point, (255, 0, 0) if collision else colors[i])
                game._draw_rec(point, (255, 0, 0) if collision else (255, 255, 255))
                i += 1

            # Draw snek
            for point in snake.body:
                game._draw_rec(point, (0,0,255))


        state = [
            *collisions,

            # Move direction
            snake.direction == Direction.UP,
            snake.direction == Direction.RIGHT,
            snake.direction == Direction.DOWN,
            snake.direction == Direction.LEFT,

            # Food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y  # food down
        ]

        return np.array(state, dtype=int)

def loop(game, agent, player_id):
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

    return done, score


def train():
    # TODO: Date to name to prevent overwrite
    m = Linear_QNet((71, 255,  3))
    agents = [
        # TODO: Add input and output layers so we don't have to think about them here
        # AgentBasic(Linear_QNet((11, 150,  3)), "Basic(Deep Blue)[Many Rock]"),
        # AgentCornerVision(Linear_QNet((13, 150, 3)), "Corner(Green)[Many Rock]"),
        # AgentMushroomVision(Linear_QNet((26, 255,  3)), "Mushroom(Red)[Many Rock]"),
        # AgentLookAhead(Linear_QNet((71, 255,  3)), 21, "LookAhead(Large Yellow)[Many Rock]"),
        AgentMushroomVision(Linear_QNet((26, 255, 255,  3)), "Mushroom(Deep large)[Many Rock]"),
        AgentMushroomVision(Linear_QNet((26, 255, 26,  3)), "Mushroom(Deep small)[Many Rock]"),
        AgentMushroomVision(Linear_QNet((26, 155,  3)), "Mushroom(Small)[Many Rock]"),
        AgentMushroomVision(Linear_QNet((26, 255, 155, 50,  3)), "Mushroom(Very deep)[Many Rock]"),
    ]
    game = SnakeGameAI(640, 480, len(agents))

    scores = [[]] * (len(agents) + 1)
    mean_scores = [[]] * len(scores)

    # Pre train for x iterations or acceptable mean score
    pre_train_iterations = 3000
    acceptable_mean_score = 7
    increase_difficulty_at = 10
    import time
    for player_id, agent in enumerate(agents):
        game.num_objects = 150
        while agent.n_games < pre_train_iterations and \
                (agent.mean_scores[-1] if len(agent.mean_scores) > 0 else 0) < acceptable_mean_score:
            # TODO: Perf test (render seems to be non impact)
            now = time.time()
            done, score = loop(game, agent, player_id)
            game.render()

            if done:
                agent.n_games += 1
                agent.scores.append(score)
                agent.total_score += score
                mean_score = agent.total_score / agent.n_games
                agent.mean_scores.append(mean_score)

                # Increase difficulty
                if score > increase_difficulty_at:
                    game.num_objects += score
                    game.num_objects %= 200
                    print(f"Score above 10 achieved, Increasing number of object to {game.num_objects}")

                # For plot
                scores[player_id] = agent.scores
                mean_scores[player_id] = agent.mean_scores

                game.reset()
                plot(scores, mean_scores, map(lambda x: x.name, agents))

                agent.train_long_memory()
                if (agent.n_games % 50 == 0):
                    print("Num iterations:",agent.n_games)
                    print(time.time() - now)

    game.reset()
    game.reset_scores()
    game.num_objects = 150

    for agent in agents:
        agent.training = False

    train_for = 10_000
    infinite_play = True
    while not any([agent.n_games > train_for for agent in agents]) or infinite_play:
        for player_id, agent in enumerate(agents):
            if agent.done:
                continue

            done, score = loop(game, agent, player_id)
            game.render()

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

                if score > agent.record_score or score > 20:
                    agent.record_score = score
                    agent.model.save(f"model_{agent.name}.pth")

                if (agents[0].n_games % 50 == 0):
                    print("Number of matches:",agent.n_games)

        if all(agent.done for agent in agents):
            game.reset()

            for player_id, agent in enumerate(agents):
                agent.train_long_memory()
                agent.done = False

            plot(scores, mean_scores, map(lambda x: x.name, agents))

    print("Training done, saving models")
    for agent in agents:
        print(agent.name, "trained for", agent.n_games,
              "record score", agent.record_score)
        # agent.model.save(f"model_{agent.name}.pth")
