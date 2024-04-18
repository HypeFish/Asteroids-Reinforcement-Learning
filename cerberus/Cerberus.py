import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchmetrics.classification import Accuracy
import torch.nn.functional as F
import gymnasium as gym
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class AngleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7*7*4, 16)

    def forward(self, x):
        x.to(device)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = x.to("cpu")
        return x

    def prep_input(self, data, x_coords, y_coords):
        angle_data = torch.zeros([data.shape[0], 1, 16, 16]).to(device)
        for idx, item in enumerate(data):
            if type(x_coords) == int or type(y_coords) == int:
                y_shift = -105-y_coords
                x_shift = -75-x_coords
            else:
                y_shift = -105-y_coords[idx]
                x_shift = -75-x_coords[idx]
            angle_data[idx, 0, :, :] = item[0].roll([y_shift, x_shift], [0, 1])[97:113, 80:96]
        return angle_data

class XNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.fc1 = nn.Linear(160*32, 160)
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = torch.amax(x, dim=2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = x.to("cpu")
        return x

class YNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.fc1 = nn.Linear(210*32, 210)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = torch.amax(x, dim=3)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = x.to("cpu")
        return x

class CerberusDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.env = gym.make('AsteroidsNoFrameskip-v4', obs_type="grayscale")
        obs, info = self.env.reset()
        angle_states = torch.zeros([16, 210, 160])
        for i in range(16):
            self.env.step(4)
            self.env.step(4)
            self.env.step(4)
            obs, reward, terminated, truncated, info = self.env.step(4)
            obs[0:25] = 0
            angle_states[i] = torch.roll(torch.from_numpy(obs)/256, [105, 75], dims=[0, 1])
        x_shifts = torch.randint(0, 160, [num_samples])
        y_shifts = torch.randint(0, 210, [num_samples])
        angles = torch.randint(0, 16, [num_samples])
        data = torch.zeros([num_samples, 1, 210, 160])
        for i in range(num_samples):
            data[i][0] = torch.roll(angle_states[angles[i]], [y_shifts[i], x_shifts[i]], [0, 1])
        self.data = data.to(device)
        self.x_shifts = F.one_hot(x_shifts, num_classes=160).to(torch.float32)
        self.y_shifts = F.one_hot(y_shifts, num_classes=210).to(torch.float32)
        self.angles = F.one_hot(angles, num_classes=16).to(torch.float32)
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return self.data[idx], self.x_shifts[idx], self.y_shifts[idx], self.angles[idx]

class Cerberus(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_net = XNet().to(device)
        self.y_net = YNet().to(device) 
        self.a_net = AngleNet().to(device)
        if not (os.path.isfile('XNET.pt') and os.path.isfile('YNET.pt') and os.path.isfile('ANET.pt')):
            print('Cerberus model not loaded, training now...')
            loss_fn = nn.CrossEntropyLoss()
            test_dataset = CerberusDataset(1000)
            x_net = XNet().to(device)
            y_net = YNet().to(device)
            angle_net = AngleNet().to(device)
            x_lr = 1e-2
            y_lr = 1e-2
            angle_lr = 1e-3
            epochs = 32
            batch_size = 32
            num_samples = 1024
            x_optim = Adam(x_net.parameters(), lr=x_lr)
            y_optim = Adam(y_net.parameters(), lr=y_lr)
            angle_optim = Adam(self.angle_net.parameters(), lr=angle_lr)
            for epoch in range(epochs):
                dataloader = DataLoader(CerberusDataset(num_samples), batch_size=batch_size)
                for step, [data, x_shifts, y_shifts, angles] in enumerate(dataloader):
                    x_pred = x_net(data)
                    x_optim.zero_grad()
                    x_loss = loss_fn(x_pred, x_shifts)
                    x_loss.backward()
                    x_optim.step()

                    y_pred = y_net(data)
                    y_optim.zero_grad()
                    y_loss = loss_fn(y_pred, y_shifts)
                    y_loss.backward()
                    y_optim.step()

                    x_coords = torch.argmax(x_net(data), axis=1)
                    y_coords = torch.argmax(y_net(data), axis=1)
                    angle_data = angle_net.prep_input(data, x_coords, y_coords)
                    angle_pred = angle_net(angle_data)
                    angle_optim.zero_grad()
                    angle_loss = loss_fn(angle_pred, angles)
                    angle_loss.backward()
                    angle_optim.step()
            # Test models
            x_accuracy = Accuracy(task="multiclass", num_classes=160).to(device)
            y_accuracy = Accuracy(task="multiclass", num_classes=210).to(device)
            angle_accuracy = Accuracy(task="multiclass", num_classes=16).to(device)

            test_dataloader = DataLoader(test_dataset, batch_size=32)
            for step, [data, x_shifts, y_shifts, angles] in enumerate(test_dataloader):
                x_pred = torch.argmax(x_net(data), axis=1)
                x_accuracy.update(x_pred, torch.argmax(x_shifts, axis=1))

                y_pred = torch.argmax(y_net(data), axis=1)
                y_accuracy.update(y_pred, torch.argmax(y_shifts, axis=1))

                x_coords = torch.argmax(x_net(data), axis=1)
                y_coords = torch.argmax(y_net(data), axis=1)
                angle_data = angle_net.prep_input(data, x_coords, y_coords)
                angle_pred = torch.argmax(angle_net(angle_data), axis=1)
                angle_accuracy.update(angle_pred, torch.argmax(angles, axis=1))
            print(f'Cerberus model trained with accuracies -> X:{x_accuracy.compute():.2f}, Y:{y_accuracy.compute():.2f}, Angle:{angle_accuracy.compute():.2f}')
            torch.save(x_net.state_dict(), f"XNET.pt")
            torch.save(y_net.state_dict(), f"YNET.pt")
            torch.save(angle_net.state_dict(), f"ANET.pt")
        self.x_net.load_state_dict(torch.load('XNET.pt'))
        self.y_net.load_state_dict(torch.load('YNET.pt'))
        self.a_net.load_state_dict(torch.load('ANET.pt'))
        print('Successfully loaded Cerberus model')
    
    def forward(self, x):
        if x.shape != torch.Size([2, 1, 210, 160]):
            print(f'Incorrect shape of {x.shape}, required shape of {torch.Size([2, 1, 210, 160])}')
            assert False
        x = x[1].reshape([1, 1, 210, 160])
        x_coord = torch.argmax(self.x_net(x)).item()
        y_coord = torch.argmax(self.y_net(x)).item()
        angle = torch.argmax(self.a_net(self.a_net.prep_input(x, x_coord, y_coord))).item()
        return [x_coord, y_coord, angle]