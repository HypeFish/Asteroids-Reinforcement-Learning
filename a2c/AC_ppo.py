import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from torchvision.transforms import Resize
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class ActorCriticPPO(nn.Module):
    def __init__(self, state_dim, action_dim, lr_actor=0.0001, lr_critic=0.001):
        super(ActorCriticPPO, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.num_games = 0 # Number of games played
        self.scores = [] # List of scores obtained in each game

        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def build_actor(self):
        actor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.Flatten(),
            nn.Linear(3136, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_dim),
            nn.Softmax(dim=-1)
        )
        return actor

    def build_critic(self):
        critic = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.Flatten(),
            nn.Linear(3136, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        return critic

    def get_action(self, state):
        with torch.no_grad():
            state = state.clone().detach().requires_grad_(True).to(device).unsqueeze(0)
            action_probs = self.actor(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()    
        return action

    def train(self, states, actions, advantages, discounted_rewards):
        # Convert list of states into a single PyTorch tensor and move it to CPU
        states_tensor = torch.stack(states).cpu()

        # Convert the PyTorch tensor into a NumPy array
        states_np = states_tensor.numpy()
        # Convert the NumPy array into a PyTorch tensor
        states = torch.tensor(states_np, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
    
        mu = self.actor(states)
        value = self.critic(states)
        value_squeezed = value.squeeze()
        advantages = advantages.squeeze(1)

        advantage = advantages - value_squeezed
        old_mu = mu.gather(1, actions.unsqueeze(1)).squeeze()

        # Actor loss
        ratio = torch.exp(self.log_prob(mu, actions) - self.log_prob(old_mu, actions))
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantage
        actor_loss = -torch.mean(torch.min(surr1, surr2))

        # Critic loss
        critic_loss = nn.MSELoss()(value.squeeze(), advantages)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss.backward(retain_graph=True)
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def log_prob(self, mu, action):
        action_dist = torch.distributions.Categorical(mu)
        return action_dist.log_prob(action)

    def discounted_rewards(self, rewards, gamma):
        discounted_rewards = np.zeros(len(rewards))  # Initialize as 1D array
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards[:, np.newaxis]  # Reshape to have shape (batch_size, 1)


"""
Resizes image to 84x84
"""
def process_observation(observation):
    img = Image.fromarray(observation)
    img = img.resize((84, 84))
    img = img.convert("L")
    img = np.array(img)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img / 255.0
    img = img.to(device)
    return img

if __name__ == '__main__':
    pass

