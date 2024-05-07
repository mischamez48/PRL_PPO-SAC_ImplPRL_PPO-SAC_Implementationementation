import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

class PPOMemory:
    def __init__(self, batch_size):
        self.clear_memory()
        self.batch_size = batch_size

    """List of int that correspond to the indices of our memories, and batch size chunks
     of those memories"""
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return (np.array(self.states), np.array(self.actions), np.array(self.probs),
                np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches)

    def add_memory(self, state, action, probs, vals, reward, done): # store the memory for each trajectory
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self): # clear the memory after each trajectory
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,n_epochs, gamma, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=n_epochs,
                                                   gamma=gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, n_epochs, gamma, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=n_epochs,
                                                   gamma=gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value
    
class AgentConfig:
    gamma = 0.99
    alpha = 0.0003
    gae_lambda = 0.95
    eps_clip = 0.2
    batch_size = 5
    n_epochs = 4
    gradient_clip = 0.5
    vf_coef = 1
    entropy_coef = 0.01

class Agent(AgentConfig):
    def __init__(self, n_actions, input_dims):
        super(Agent, self).__init__()

        self.actor = ActorNetwork(n_actions, input_dims, self.alpha, self.n_epochs, self.gamma)
        self.critic = CriticNetwork(input_dims, self.alpha, self.n_epochs, self.gamma)
        self.memory = PPOMemory(self.batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.add_memory(state, action, probs, vals, reward, done)

    def select_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        
        # Get the action probabilities and critic value estimate for the given state
        action_distribution = self.actor(state)
        critic_value = self.critic(state)
        action = action_distribution.sample()

        log_prob = T.squeeze(action_distribution.log_prob(action)).item()
        action = T.squeeze(action).item()
        critic_value = T.squeeze(critic_value).item()
        
        return action, log_prob, critic_value


    def train(self):
        for _ in range(self.n_epochs):
            # Retrieve batches of data from memory
            states, actions, old_probs, values, rewards, dones, batches = self.memory.generate_batches()

            # Calculate advantages
            advantages = self.calculate_advantages(rewards, values, dones)

            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                total_losss = self.compute_losses(batch, states, actions, old_probs,values, advantages)

                # Update actor and critic networks
                self.update_networks(total_losss)

        # Clear memory after each epoch
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

        dist = self.actor(states_batch)
        critic_value = self.critic(states_batch).squeeze()

        new_probs = dist.log_prob(actions_batch)
        prob_ratio = (new_probs.exp() / old_probs_batch.exp()).unsqueeze(1)
        weighted_probs = advantages[batch] * prob_ratio
        weighted_clipped_probs = T.clamp(prob_ratio,
                                        1-self.eps_clip,
                                        1+self.eps_clip) * advantages[batch]
        clip_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

        returns = advantages[batch] + self.gamma * values[batch]
        vf_loss = 0.5*((returns - critic_value) ** 2).mean()

        # Calculate entropy loss
        entropy = dist.entropy().mean()

        total_loss = clip_loss + self.vf_coef * vf_loss - self.entropy_coef * entropy
        return total_loss

    def update_networks(self, total_loss):
        # Zero out gradients
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        # Backpropagation and optimization
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.actor.optimizer.step()
        self.critic.optimizer.step()
