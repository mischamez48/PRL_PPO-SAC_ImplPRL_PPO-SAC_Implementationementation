import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

class PPOMemory:
    def __init__(self, batch_size):
        self.clear_memory()
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return (np.array(self.states), np.array(self.actions), np.array(self.probs),
                np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches)

    def add_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, actor_alpha, n_epochs, gamma, fc1_dims=128, fc2_dims=128):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=actor_alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=n_epochs, gamma=gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = T.clamp(self.sigma(x), min=1e-3, max=1.0)  # Ensure positive std dev
        return mu, sigma

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, critic_alpha, n_epochs, gamma, fc1_dims=128, fc2_dims=128):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=critic_alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=n_epochs, gamma=gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class AgentConfig:
    gamma = 0.99
    alpha = 0.0003
    gae_lambda = 0.95
    eps_clip = 0.2
    batch_size = 5
    n_epochs = 4
    vf_coef = 0.5
    entropy_coef = 0.01
    target_kl_div = 0.01

class Agent(AgentConfig):
    def __init__(self, n_actions, input_dims, actor_alpha, critic_alpha):
        super(Agent, self).__init__()

        self.actor = ActorNetwork(n_actions, input_dims, actor_alpha, self.n_epochs, self.gamma)
        self.critic = CriticNetwork(input_dims, critic_alpha, self.n_epochs, self.gamma)
        self.memory = PPOMemory(self.batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.add_memory(state, action, probs, vals, reward, done)

    def select_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        
        mu, sigma = self.actor(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum()
        value = self.critic(state)
        
        action = T.tanh(action)  # To bound the actions between -1 and 1 for Pendulum

        return action.cpu().detach().numpy()[0], action_log_prob.cpu().detach().numpy(), value.cpu().detach().numpy()

    def train(self, target_kl_div=0.01):
        for _ in range(self.n_epochs):
            states, actions, old_probs, values, rewards, dones, batches = self.memory.generate_batches()
            advantages = self.calculate_advantages(rewards, values, dones)
            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                total_loss, states_batch, actions_batch = self.compute_losses(batch, states, actions, old_probs, values, advantages)
                self.update_networks(total_loss)

                # Compute KL divergence and break if it exceeds the target
                old_log_probs = T.tensor(old_probs[batch]).to(self.actor.device)
                dist = Normal(*self.actor(states_batch))
                new_log_probs = dist.log_prob(actions_batch).sum(1)
                kl_div = (old_log_probs - new_log_probs).mean()
                if kl_div >= target_kl_div:
                    break

        self.memory.clear_memory()

    def calculate_advantages(self, rewards, values, dones):
        advantage = np.zeros_like(rewards, dtype=np.float32)
        last_idx = len(rewards) - 1

        next_value = 0 if dones[last_idx] else values[last_idx]

        for t in reversed(range(last_idx)):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage[t] = delta + self.gamma * self.gae_lambda * advantage[t + 1]
            next_value = values[t]

        return T.tensor(advantage).to(self.actor.device)

    def compute_losses(self, batch, states, actions, old_probs, values, advantages):
        states_batch = T.tensor(states[batch], dtype=T.float).to(self.actor.device)
        old_probs_batch = T.tensor(old_probs[batch]).to(self.actor.device)
        actions_batch = T.tensor(actions[batch]).to(self.actor.device)

        dist = Normal(*self.actor(states_batch))
        critic_value = self.critic(states_batch).squeeze()

        new_probs = dist.log_prob(actions_batch).sum(1)
        prob_ratio = (new_probs.exp() / old_probs_batch.exp()).unsqueeze(1)
        weighted_probs = advantages[batch] * prob_ratio
        weighted_clipped_probs = T.clamp(prob_ratio,
                                        1-self.eps_clip,
                                        1+self.eps_clip) * advantages[batch]
        clip_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

        returns = advantages[batch] + self.gamma * values[batch]
        vf_loss = 0.5*((returns - critic_value) ** 2).mean()

        entropy = dist.entropy().mean()

        total_loss = clip_loss + self.vf_coef * vf_loss - self.entropy_coef * entropy
        return total_loss, states_batch, actions_batch

    def update_networks(self, total_loss):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make("Pendulum-v1")
    N = 25
    batch_size=7
    agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space.shape, actor_alpha=3e-4, critic_alpha=1e-5)

    n_games = 500

    figure_file = 'plots/pendulumV1.png'

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
            n_steps += 1
            score += reward
            agent.store_transition(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.train()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
