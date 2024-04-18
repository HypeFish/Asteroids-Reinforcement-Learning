import copy
import random
import torch
import torch.optim as optim
import torch.nn.functional as f
from plot import LivePlot
import numpy as np
import time

"""
Class to help with memory management
"""
class ReplayMemory:

    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.device = device
        self.position = 0
        self.memory_max_report = 0

    def insert(self, transition):
        # slow down speed for memory to avoid issues
        transition = [item.to('cpu') for item in transition]
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.remove(self.memory[0])
            self.memory.append(transition)

    def sample(self, batch_size=32):
        assert self.can_sample(batch_size)
        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items).to(self.device) for items in batch]

    def can_sample(self, batch_size):
        # batch size should be 10 times (or more) less than memory
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)


class Agent:

    def __init__(self, model, device='cpu', epsilon=1.0, min_epsilon=0.1, nb_warmup=10000,
                 nb_actions=None, memory_capacity=10000, batch_size=32, learning_rate=0.00025):
        self.memory = ReplayMemory(device=device, capacity=memory_capacity)
        self.model = model
        self.device = device
        # model it can evaluate off of
        self.target_model = copy.deepcopy(model).eval()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (((epsilon - min_epsilon) / nb_warmup) * 2)
        self.batch_size = batch_size

        self.model.to(device)
        self.target_model.to(device)

        self.gamma = 0.99
        self.nb_actions = nb_actions

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print(f'Starting epsilon is {self.epsilon}')
        print(f'Epsilon decay is {self.epsilon_decay}')

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            # random action
            return torch.randint(self.nb_actions, (1, 1))
        else:
            ev = self.model(state).detach()  # list of probabilities of actions
            return torch.argmax(ev, dim=-1, keepdim=True)  # get best one

    def train(self, env, epochs):
        stats = {'Returns': [], 'AvgReturns': [], 'EpsilonCheckpoints': []}
        plotter = LivePlot()

        for epoch in range(1, epochs + 1):
            state = env.reset()
            done = False
            ep_return = 0

            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                self.memory.insert([state, action, reward, done, next_state])

                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)

                    qsa_b = self.model(state_b).gather(1, action_b)
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                    # ~done converts True -> 1
                    reward_b = reward_b.to(self.device)
                    done1 = ~done.to(self.device)
                    next_qsa_b = next_qsa_b.to(self.device)
                    qsa_b = qsa_b.to(self.device)
                    target_b = reward_b + done1 * self.gamma * next_qsa_b
                    loss = f.mse_loss(qsa_b, target_b)
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state
                ep_return += reward.item()

            stats['Returns'].append(ep_return)

            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay

            # Checkpoints every 10 epochs to save
            if epoch % 10 == 0:
                self.model.save_the_model()
                print(" ")
                average_returns = np.mean(stats['Returns'][-100:])  # average last 100 returns
                stats['AvgReturns'].append(average_returns)
                stats['EpsilonCheckpoints'].append(self.epsilon)

                if len(stats['Returns']) > 100:
                    print(f"Epoch: {epoch} - Average return {np.mean(stats['Returns'][-100:])} - epsilon {self.epsilon}")
                else:
                    print(f"Epoch: {epoch} - Episode return {np.mean(stats['Returns'])} - epsilon {self.epsilon}")

            # update the target model every 100 epochs
            if epoch % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                plotter.update_plot(stats)

            if epoch % 1000 == 0:
                self.model.save_the_model(f"models/model_iter_{epoch}.pt")

        return stats

    def test(self, env):
        for epoch in range(1, 3):
            state = env.reset()
            for _ in range(1000):
                time.sleep(0.01)
                action = self.get_action(state)
                state, reward, done, info = env.step(action)
                if done:
                    break




