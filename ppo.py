import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import gym

class PPOMemory:
    def __init__(self):
        self.clear_memory()

    def add_memory(self, state, action, probs, values, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []

    def generate_mini_batches(self, states, actions, returns, log_probs, values, advantages, batch_size, num_epochs):
        n_states = len(states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        #we need to update the policy and value function using multiple epochs of training on mini-batches of data
        for _ in range(num_epochs): #The number of times the entire dataset will be iterated over during training.
            np.random.shuffle(indices) #ensure that the data is mixed differently in each epoch.
            batches = [indices[i:i + batch_size] for i in batch_start]
            for batch in batches:
                batch_states = states[batch].clone().detach()
                batch_actions = actions[batch].clone().detach()
                batch_returns = returns[batch].clone().detach()
                batch_log_probs = log_probs[batch].clone().detach()
                batch_values = values[batch].clone().detach()
                batch_advantages = advantages[batch].clone().detach()
                yield batch_states, batch_actions, batch_returns, batch_log_probs, batch_values, batch_advantages
                # 2 IMPORTANT remarks: (discovered and looked for when terminal sent warning flags)
                #   - clone is used to create a copy of the tensor, so as not to affect the original data.
                #   - detaches the tensor from the computation graph, meaning no gradients will be computed 
                #     for these operations. We only want to compute gradients during 
                #     backpropagation step in the training step, not when preparing mini-batches.

class ContinuousActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim1=64, hidden_dim2=64):
        super(ContinuousActor, self).__init__()
        
        self.hidden_layer1 = nn.Linear(in_dim, hidden_dim1)
        self.hidden_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.mu_layer = nn.Linear(hidden_dim2, out_dim)
        self.log_std_layer = nn.Linear(hidden_dim2, out_dim)

    def forward(self, state):
        x = torch.relu(self.hidden_layer1(state))
        x = torch.relu(self.hidden_layer2(x))
        mu = torch.tanh(self.mu_layer(x))
        log_std = torch.tanh(self.log_std_layer(x))
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.rsample()

        return action, dist
    
class DiscreteActor(nn.Module):
    #128 and 64 for mountain
    def __init__(self, in_dim, out_dim, hidden_dim1=64, hidden_dim2=64):
        super(DiscreteActor, self).__init__()
        
        self.hidden_layer1 = nn.Linear(in_dim, hidden_dim1)
        self.hidden_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, out_dim)

    def forward(self, state):
        x = torch.relu(self.hidden_layer1(state))
        x = torch.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        probs = torch.softmax(x, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist


class Critic(nn.Module):
    #64 and 32 for mountain
    def __init__(self, in_dim, hidden_dim1=64, hidden_dim2=64):
        super(Critic, self).__init__()

        self.hidden_layer1 = nn.Linear(in_dim, hidden_dim1)
        self.hidden_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, 1)

    def forward(self, state):
        x = torch.relu(self.hidden_layer1(state))
        x = torch.relu(self.hidden_layer2(x))
        value = self.out(x)
        return value


def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, np.sqrt(float(2)))
        if m.bias is not None:
            m.bias.data.fill_(0)


class PPOAgent():
    def __init__(self, make_env, continuous: bool, obs_dim: int, act_dim: int, gamma: float,
                 lamda: float, entropy_coef: float, epsilon: float, vf_coef: float, rollout_len: int,
                 total_rollouts: int, num_epochs: int, batch_size: int, plot_interval: int = 10, 
                 solved_reward: int = None, actor_lr: float = 3e-4, critic_lr: float = 1e-3):
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Device:", self.device)
        
        self.env = make_env()
        self.obs_dim = obs_dim

        # Hyperparameters
        self.gamma = gamma
        self.lamda = lamda
        self.entropy_coef = entropy_coef
        self.epsilon = epsilon
        self.vf_coef = vf_coef
        self.rollout_len = rollout_len
        self.total_rollouts = total_rollouts
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Initialize networks
        self.actor = (ContinuousActor if continuous else DiscreteActor)(self.obs_dim, act_dim).apply(init_weights).to(self.device)
        self.critic = Critic(self.obs_dim).apply(init_weights).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Learning Rate Schedulers
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.9)


        # Memory
        self.memory = PPOMemory()
        self.scores = []
        self.actor_losses = []  # Store actor losses
        self.critic_losses = []  # Store critic losses

        # Evaluation and plotting
        self.solved_reward = solved_reward
        self.plot_interval = plot_interval

    def select_action(self, state):
        """
        Get action from actor, and collect state value from critic, 
        collect elements of trajectory.
        """
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)

        value = self.critic(state)
        log_prob = dist.log_prob(action)
        
        # collect elements of trajectory
        self.memory.add_memory(state, action, log_prob, value, None, None)

        return list(action.detach().cpu().numpy()).pop()

    def step(self, action):
        """
        Make action in environment chosen by current policy,
        collect elements of trajectory.
        """
        next_state, reward, done, _ = self.env.step(action)

        # # Add additional reward for moving forward
        # position = next_state[0]
        # if position > -0.5:
        #     reward += 1

        # Convert elements to a torch tensor and add a batch dimension
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Collect elements of trajectory
        self.memory.rewards[-1] = reward
        self.memory.dones[-1] = 1 - done  # Use 1 - done to correctly mask future rewards and values for the advantage calculation

        return next_state, reward, done

    def train(self):
        """
        Interaction process in environment to collect trajectory,
        train process by agent nets after each rollout.
        """
        score = 0
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        episode_count = 0  # To track the number of episodes

        for n_step in range(self.total_rollouts):
            for _ in range(self.rollout_len):
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward.item()  # Extract scalar value from reward tensor

                if done.item():  # Extract scalar value from done tensor
                    self.scores.append(score)
                    score = 0
                    state = self.env.reset()
                    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    episode_count += 1  # Increment the episode count

                    if episode_count >= 100:
                        print("Training completed after 100 episodes")
                        self.env.close()
                        return

            if n_step % self.plot_interval == 0:
                self._plot_train_history(window=200)

            value = self.critic(next_state)
            self.memory.values.append(value)
            # Update policy
            self._update_weights()

        print("Training completed after reaching the total rollouts")
        self.env.close()

    def _update_weights(self):

        gamma, lamda = self.gamma, self.lamda

        rewards = self.memory.rewards
        values = self.memory.values
        dones = self.memory.dones

        # Compute the advantage function and ret
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * dones[i] - values[i]
            gae = delta + gamma * lamda * dones[i] * gae
            returns.insert(0, gae + values[i])

        # Flatten tensors for batch processing
        states = torch.cat(self.memory.states).view(-1, self.obs_dim)
        actions = torch.cat(self.memory.actions)
        returns = torch.cat(returns).detach()
        log_probs = torch.cat(self.memory.probs).detach()
        values = torch.cat(self.memory.values).detach()
        advantages = returns - values[:-1]
        #Advantage Normalizatio, it seems to help in some training, so I kept it:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mini_batches = self.memory.generate_mini_batches(
            states, actions, returns, log_probs, values, advantages, 
            self.batch_size, self.num_epochs
        )
        #Each iteration of this loop fetches the next mini-batch from the generator,
        #allowing the training code to process the data in chunks rather than all at once.  
        for state, action, return_, old_log_prob, old_value, advantage in mini_batches:
            _, dist = self.actor(state)
            new_log_prob = dist.log_prob(action)
            prob_ratio = torch.exp(new_log_prob - old_log_prob)

            # Compute entropy
            entropy = dist.entropy().mean()

            # Compute actor loss
            weighted_probs = advantage * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1. - self.epsilon, 1. + self.epsilon) * advantage
            actor_loss = -torch.mean(torch.min(weighted_probs, weighted_clipped_probs)) - self.entropy_coef * entropy #we are doing gradient ascent not descent on the policy, we want to maximize
            
            # Compute critic loss
            cur_value = self.critic(state)
            critic_loss = self.vf_coef * (return_ - cur_value).pow(2).mean() #here we minimize, for gradient descent now

            # Store losses
            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())

            # Perform optimization steps
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()

        # Step the learning rate schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # Clear memory
        self.memory.clear_memory()

        

    def _plot_train_history(self, window=200):
        data = [self.scores, self.actor_losses, self.critic_losses]

        # Ensure there are at least 10 elements in the scores for calculating the mean of the last 10
        score_label = f"score {int(np.mean(self.scores[-10:]))}" if len(self.scores) >= 10 else "score"

        labels = [score_label, "actor_loss", "critic_loss"]

        # Calculate the mean of actor_loss and critic_loss over a sliding window of the specified size
        if len(self.actor_losses) >= window:
            actor_mean = [np.mean(self.actor_losses[i:i+window]) for i in range(0, len(self.actor_losses), window)]
        else:
            actor_mean = [np.mean(self.actor_losses)]
        
        if len(self.critic_losses) >= window:
            critic_mean = [np.mean(self.critic_losses[i:i+window]) for i in range(0, len(self.critic_losses), window)]
        else:
            critic_mean = [np.mean(self.critic_losses)]

        mean_data = [actor_mean, critic_mean]
        mean_labels = ["Mean Actor Loss", "Mean Critic Loss"]

        clear_output(True)
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))  # Create 3 subplots

        for i, ax in enumerate(axes[:2]):
            ax.plot(mean_data[i])
            ax.set_title(mean_labels[i])
            ax.set_xlabel("Steps")
            ax.set_ylabel("Value")

        # Plot the original scores without window averaging
        axes[2].plot(data[0])
        axes[2].set_title(labels[0])
        axes[2].set_xlabel("Episodes")
        axes[2].set_ylabel("Value")

        plt.tight_layout()
        plt.show()
