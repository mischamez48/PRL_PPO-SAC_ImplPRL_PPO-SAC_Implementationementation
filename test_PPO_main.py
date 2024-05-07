import gym
import numpy as np
import matplotlib.pyplot as plt
from ppo_algorithm import Agent, PPOMemory

def plot_learning_curve(x, scores, figure_file):
    window_size = 100
    running_sum = np.cumsum(scores)
    running_avg = (running_sum[window_size:] - running_sum[:-window_size]) / window_size

    plt.plot(x[window_size - 1:], running_avg)
    plt.title('Running average of previous 100 scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(figure_file)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    N = 20
    batch_size=5
    agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
    n_games = 300

    figure_file = 'plots/cartpoleV0.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.select_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, prob, val, reward, done)
            observation = observation_
            
            if agent.memory.states:
                if len(agent.memory.states) >= batch_size:
                    agent.train()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

