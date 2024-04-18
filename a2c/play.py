import gymnasium as gym
import torch
from AC_ppo import ActorCriticPPO, device, process_observation


# Environment settings
env_name = "ALE/Breakout-ram-v5"
env = gym.make(env_name, render_mode='human')
state_dim = 84 * 84  # Preprocessed frame dimensions
action_dim = env.action_space.n

# Hyperparameters
lr_actor = 0.0001
lr_critic = 0.001
gamma = 0.99
epochs = 1000
batch_size = 64

# Initialize Actor-Critic PPO model
agent = ActorCriticPPO(state_dim, action_dim, lr_actor, lr_critic)
agent.load_state_dict(torch.load("breakout.pth", map_location=device))
agent.to(device)

# Test the agent and render it
score_overtime = []
state, _ = env.reset()
state = process_observation(state)
done = False
while not done:
    env.render()
    action = agent.get_action(state)
    next_state, reward, done, _, _ = env.step(action)
    next_state = process_observation(next_state)
    state = next_state
    score_overtime.append(reward)
    

print(f"Score: {sum(score_overtime)}")

# # Play random agent
score_overtime = []
state, _ = env.reset()
state = process_observation(state)
done = False
while not done:
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, _, _ = env.step(action)
    next_state = process_observation(next_state)
    state = next_state
    score_overtime.append(reward)

print(f"Random agent score: {sum(score_overtime)}")
