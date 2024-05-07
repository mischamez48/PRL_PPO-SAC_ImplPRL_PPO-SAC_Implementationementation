import torch
import torch.nn as nn
import torch.optim as optim
import gym

class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_space_size, action_space_size):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU())

        self.policy_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size))

        self.value_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def value(self, obs):
        z = self.shared_layers(obs)
        value = self.value_layers(z)
        return value

    def policy(self, obs):
        z = self.shared_layers(obs)
        policy_logits = self.policy_layers(z)
        return policy_logits

    def forward(self, obs):
        z = self.shared_layers(obs)
        policy_logits = self.policy_layers(z)
        value = self.value_layers(z)
        return policy_logits, value

class PPO:
    def __init__(self, state_dim, action_dim, lr, eps, gamma, GAE_lambda):
        self.policy_network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.eps = eps
        self.gamma = gamma
        self.GAE_lambda = GAE_lambda
        self.memory = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        policy_logits = self.policy_network.policy(state)
        action_probs = nn.functional.softmax(policy_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().squeeze(1)
        return action.item(), action_probs

    def store_transition(self, state, action, reward, next_state, done, action_probs):
        self.memory.append((state, action, reward, next_state, done, action_probs))

    def train(self, env, num_epochs):
        for epoch in range(num_epochs):
            state = env.reset()
            done = False
            total_reward = 0
            self.memory = []

            while not done:
                action, action_probs = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                self.store_transition(state, action, reward, next_state, done, action_probs)
                state = next_state

            # Compute advantages
            episode_states, episode_actions, episode_rewards = zip(*self.memory)
            episode_rewards = torch.tensor(episode_rewards, dtype=torch.float32)
            episode_values = []
            Gt = 0
            for reward in episode_rewards[::-1]:
                Gt = reward + self.gamma * Gt
                episode_values.insert(0, Gt)
            episode_values = torch.tensor(episode_values)

            deltas = episode_values[:-1] - episode_values[1:]
            advantages = torch.zeros_like(episode_values)
            A_t = 0
            for delta in deltas[::-1]:
                A_t = delta + self.gamma * self.GAE_lambda * A_t
                advantages[-1] = A_t
                A_t = A_t.detach()
            advantages = advantages[:-1]

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Optimize policy and value function
            for _ in range(10):  # multiple epochs of optimization
                state_batch = torch.cat(episode_states, dim=0)
                action_batch = torch.tensor(episode_actions, dtype=torch.long)
                old_action_probs, old_state_values = self.policy_network(state_batch)
                old_action_probs = old_action_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1)

                self.optimizer.zero_grad()
                action_probs, state_values = self.policy_network(state_batch)
                action_probs = action_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1)

                ratios = action_probs / (old_action_probs + 1e-8)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (state_values.squeeze() - episode_values[:-1])**2
                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()

            print(f"Epoch: {epoch}, Total Reward: {total_reward}")



# Environment settings
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# PPO settings
lr = 3e-4
eps = 0.2
gamma = 0.99
GAE_lambda = 0.95

# Create PPO agent and train
ppo = PPO(state_dim, action_dim, lr, eps, gamma, GAE_lambda)
ppo.train(env, num_epochs=1000)
