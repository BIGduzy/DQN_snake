from model import Linear_QNet
from agent import AgentBasic, AgentLookAhead, AgentCornerVision
from game import SnakeGameAI


def loop(game, agent, player_id):
    # Get current state
    state_old = agent.get_state(game, player_id)

    # Get action
    action = agent.get_action(state_old)

    # Perform move
    _, done, score = game.play_step(action, player_id)

    return done, score


def battle(agents):
    game = SnakeGameAI(640, 480, len(agents))

    scores = [[]] * (len(agents) + 1)
    mean_scores = [[]] * len(scores)

    game.reset()
    game.reset_scores()

    while True:
        game.render()
        for player_id, agent in enumerate(agents):
            if agent.done:
                continue

            done, _ = loop(game, agent, player_id)

            if done:
                agent.done = True

                # For plot
                scores[player_id] = agent.scores
                mean_scores[player_id] = agent.mean_scores

        if all(agent.done for agent in agents):
            game.reset()

            for player_id, agent in enumerate(agents):
                agent.done = False


if __name__ == "__main__":
    model1_name = "model_Linear_QNet(71,1500,1500,3)_LookAhead(LongTrain).pth"
    model1 = Linear_QNet((71, 1500, 1500, 3))
    model1.load(model1_name)

    model2_name = "model_Linear_QNet(26,500,3)_LookAhead(Yellow).pth"
    model2 = Linear_QNet((26, 500, 3))
    model2.load(model2_name)

    agents = [
        # TODO: Make a generic playback agent
        AgentLookAhead(model1, model1_name),
        AgentLookAhead(model2, model2_name),
        AgentLookAhead(model2, model2_name),
        AgentLookAhead(model2, model2_name),
    ]

    # TODO: Remove this after playback agent is created
    for agent in agents:
        # To remove epsilon
        agent.n_games = 1000

    battle(agents)
