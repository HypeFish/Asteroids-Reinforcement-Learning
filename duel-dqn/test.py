import os
import torch
from breakout import DQNBreakout
from asteroids import DQNAsteroids
from model import AtariNet
from agent import Agent

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device {device}")
environment = DQNAsteroids(device=device)
model = AtariNet(nb_actions=14)
model.to(device)
model.load_the_model()

agent = Agent(model=model,
              device=device,
              epsilon=0.01,
              nb_warmup=5000,
              nb_actions=14,
              learning_rate=0.00001,
              memory_capacity=1000000,
              batch_size=64)

agent.test(env=environment)
